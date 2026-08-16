// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// Mann-Whitney U (Wilcoxon rank-sum): a non-parametric test of whether one sample is stochastically
    /// greater than another.
    ///
    /// <para><b>Why rank-based rather than a z-score on the mean.</b> A z-score / standard-deviation gate
    /// assumes the data are normally distributed. Latency and per-request cost are not — they are heavy-tailed
    /// (roughly log-normal), so a σ-based threshold either fires constantly or misses real regressions. Ranking
    /// discards the distribution's shape and compares only ordering, which is exactly what "is the new version
    /// worse?" means. This is the same test canary-analysis systems such as Kayenta use.</para>
    ///
    /// <para><b>Significance is not enough.</b> With large samples an arbitrarily small difference reaches
    /// p &lt; 0.05, so the result also carries an effect size (<c>ProbabilitySuperior</c> / Cliff's delta). A
    /// decision gate should require both: statistically significant <i>and</i> materially large.</para>
    ///
    /// <para><b>Four entry points, in increasing order of what the caller already knows.</b> All four produce
    /// bit-identical results on the same data and none of them allocates on the GC heap:</para>
    /// <list type="bullet">
    /// <item><see cref="Compare(ReadOnlySpan{double}, ReadOnlySpan{double})"/> — unsorted input, scratch taken
    /// from the shared pool. O(N log N), 0 B allocated.</item>
    /// <item><see cref="Compare(ReadOnlySpan{double}, ReadOnlySpan{double}, Span{double})"/> — unsorted input,
    /// caller-owned scratch. Same cost, nothing rented either.</item>
    /// <item><see cref="CompareSorted"/> — both samples already ordered. O(n₁+n₂), a single merge, no sort.</item>
    /// <item><see cref="CompareHistograms"/> — bucketed counts. O(K) in the bucket count, independent of how
    /// many requests are behind them.</item>
    /// </list>
    ///
    /// <para>Ties are handled with mid-ranks and the variance is tie-corrected; a continuity correction is
    /// applied to the normal approximation. The normal approximation is appropriate once both samples hold
    /// roughly 8+ observations — below that, treat the p-value as indicative only.</para>
    /// </summary>
    public static class MannWhitneyU
    {
        /// <summary>
        /// Sample size at which an LSD radix sort over the doubles' order-preserving bit encoding overtakes
        /// the framework's introsort. Measured on a Ryzen 9 9950X3D: radix loses ~1.7x at 500 observations and
        /// wins ~3.9x at 5 000 and above, so the crossover is bracketed but not bisected — the threshold is set
        /// conservatively inside that bracket, where being wrong costs microseconds either way.
        /// </summary>
        private const int RadixSortThreshold = 2048;

        /// <summary>Smallest scratch that <see cref="Compare(ReadOnlySpan{double}, ReadOnlySpan{double}, Span{double})"/>
        /// will accept — enough to hold both samples sorted.</summary>
        public static int MinimumScratchLength(int baselineCount, int candidateCount)
        {
            return baselineCount + candidateCount;
        }

        /// <summary>
        /// Scratch that additionally unlocks the radix path on large windows. Below this the comparison still
        /// runs, on the framework's introsort — correct, just slower once a sample passes a few thousand
        /// observations. The pooled overload always rents this much.
        /// </summary>
        public static int RecommendedScratchLength(int baselineCount, int candidateCount)
        {
            return baselineCount + candidateCount + Math.Max(baselineCount, candidateCount);
        }

        /// <summary>
        /// Compares <paramref name="candidate"/> against <paramref name="baseline"/>, testing the one-sided
        /// hypothesis that the candidate is stochastically <b>greater</b> — which for a cost metric (latency,
        /// CPU per request, error rate) means <i>worse</i>. For a metric where larger is better, negate both
        /// samples before calling.
        ///
        /// <para>Scratch space is rented from the shared pool, so a canary evaluation costs no GC allocation
        /// and — importantly for a long-lived server — never puts a large-object-heap array in front of the
        /// collector, however wide the evaluation window is.</para>
        /// </summary>
        /// <exception cref="ArgumentException">Either sample is empty, or a value is NaN.</exception>
        public static MannWhitneyResult Compare(ReadOnlySpan<double> baseline, ReadOnlySpan<double> candidate)
        {
            // Only rent the radix ping-pong space when a sample is actually large enough to use it — renting
            // (and touching) scratch that the introsort path never reads just costs cache.
            var usesRadix = Math.Max(baseline.Length, candidate.Length) >= RadixSortThreshold;
            var required = usesRadix
                ? RecommendedScratchLength(baseline.Length, candidate.Length)
                : MinimumScratchLength(baseline.Length, candidate.Length);

            using var scratch = new PooledBuffer<double>(Math.Max(required, 1), clearMemory: false);

            return Compare(baseline, candidate, scratch.Span);
        }

        /// <summary>
        /// As <see cref="Compare(ReadOnlySpan{double}, ReadOnlySpan{double})"/>, but sorting into scratch the
        /// caller owns — nothing is rented and nothing is allocated. Use this when the evaluation runs on a
        /// timer and the buffer can live as long as the analyser.
        /// </summary>
        /// <param name="baseline">The reference sample; order does not matter.</param>
        /// <param name="candidate">The sample under test; order does not matter.</param>
        /// <param name="scratch">At least <see cref="MinimumScratchLength"/> elements, ideally
        /// <see cref="RecommendedScratchLength"/>; contents are overwritten and carry no meaning afterwards.</param>
        /// <exception cref="ArgumentException">A sample is empty, the scratch is too small, or a value is NaN.</exception>
        public static MannWhitneyResult Compare(
            ReadOnlySpan<double> baseline,
            ReadOnlySpan<double> candidate,
            Span<double> scratch)
        {
            if (baseline.IsEmpty || candidate.IsEmpty)
            {
                throw new ArgumentException("Both samples must contain at least one observation.");
            }

            var n1 = baseline.Length;
            var n2 = candidate.Length;

            if (scratch.Length < n1 + n2)
            {
                throw new ArgumentException($"Scratch must hold at least {n1 + n2} elements.", nameof(scratch));
            }

            // The two samples are sorted independently and then merged, rather than sorting one combined array
            // alongside a parallel "which sample did this come from" array. That removes the payload from the
            // sort and the allocation from the call — though measurement says the payload was worth only ~5%,
            // and the real win came later, from replacing the sort itself (see SortInto).
            var sortedBaseline = scratch.Slice(0, n1);
            var sortedCandidate = scratch.Slice(n1, n2);
            var spare = scratch.Slice(n1 + n2);

            SortInto(baseline, sortedBaseline, spare);
            SortInto(candidate, sortedCandidate, spare);

            return CompareSorted(sortedBaseline, sortedCandidate);
        }

        /// <summary>
        /// The test over two already-ordered samples: <b>O(n₁+n₂), one merge pass, zero allocation, no sort</b>.
        /// This is the cheapest exact path and the one the other overloads funnel into.
        ///
        /// <para>Both spans must be sorted ascending. NaN is rejected by inspecting the first element of each —
        /// the framework's sort hoists NaN to the front, so that check is exact for anything produced by
        /// <c>Sort()</c>, and O(1).</para>
        /// </summary>
        /// <exception cref="ArgumentException">Either sample is empty or begins with NaN.</exception>
        public static MannWhitneyResult CompareSorted(
            ReadOnlySpan<double> sortedBaseline,
            ReadOnlySpan<double> sortedCandidate)
        {
            if (sortedBaseline.IsEmpty || sortedCandidate.IsEmpty)
            {
                throw new ArgumentException("Both samples must contain at least one observation.");
            }

            if (double.IsNaN(sortedBaseline[0]) || double.IsNaN(sortedCandidate[0]))
            {
                throw new ArgumentException("Samples must not contain NaN.");
            }

            var n1 = sortedBaseline.Length;
            var n2 = sortedCandidate.Length;

            var i = 0;
            var j = 0;
            var baselineBelow = 0.0;
            var u = 0.0;
            var tieCorrection = 0.0;

            // Walk both samples in lockstep, one *distinct value* per iteration. Everything equal to that value
            // — from either sample — is one tie group, which is all the ranking machinery actually needs: U
            // gains a whole pair for each baseline observation strictly below and half a pair for each one tied.
            while (i < n1 || j < n2)
            {
                var takeBaseline = j >= n2 || (i < n1 && sortedBaseline[i] <= sortedCandidate[j]);
                var value = takeBaseline ? sortedBaseline[i] : sortedCandidate[j];

                var inBaseline = 0;

                while (i < n1 && sortedBaseline[i] == value)
                {
                    inBaseline++;
                    i++;
                }

                var inCandidate = 0;

                while (j < n2 && sortedCandidate[j] == value)
                {
                    inCandidate++;
                    j++;
                }

                u += inCandidate * (baselineBelow + (0.5 * inBaseline));

                var tied = (double)inBaseline + inCandidate;

                tieCorrection += (tied * tied * tied) - tied;
                baselineBelow += inBaseline;
            }

            return Finish(u, n1, n2, tieCorrection);
        }

        /// <summary>
        /// Runs many independent comparisons at once — one per metric — across the machine's cores.
        ///
        /// <para>This is the parallelism worth having. A canary evaluation does not compute one comparison; it
        /// computes a dozen (CPU per request, latency, error rate, working set, tokens per second …), and those
        /// are completely independent. Splitting <i>inside</i> a single comparison would mean forcing
        /// <c>Memory&lt;T&gt;</c> through the API to satisfy lambda capture, doubling the scratch, and fighting
        /// for roughly 1.5x on the two sorts; splitting <i>across</i> metrics needs none of that and scales with
        /// the core count.</para>
        ///
        /// <para>Each comparison rents its own scratch, and the shared pool keeps per-thread caches, so the
        /// workers do not contend on the buffer either.</para>
        ///
        /// <para><b>The one deliberate allocation in this file.</b> The delegate overload of
        /// <c>OverfitParallel.For</c> allocates a TPL closure — measured at ~5 KB for a 16-metric evaluation.
        /// The function-pointer overload is zero-allocation but takes its context as a <c>void*</c>, which
        /// would mean pinning managed arrays behind unsafe plumbing to save five kilobytes on a path that runs
        /// once every analysis window. Not worth it; the per-comparison scratch — megabytes, and the reason the
        /// GC used to see large-object-heap traffic — is pooled, which is where the pressure actually was.</para>
        /// </summary>
        /// <param name="baselines">Baseline sample per metric.</param>
        /// <param name="candidates">Candidate sample per metric, index-aligned with <paramref name="baselines"/>.</param>
        /// <param name="results">Receives one result per metric; must be at least as long as the inputs.</param>
        /// <exception cref="ArgumentException">The three collections do not line up.</exception>
        public static void CompareMany(
            IReadOnlyList<double[]> baselines,
            IReadOnlyList<double[]> candidates,
            MannWhitneyResult[] results)
        {
            ArgumentNullException.ThrowIfNull(baselines);
            ArgumentNullException.ThrowIfNull(candidates);
            ArgumentNullException.ThrowIfNull(results);

            if (baselines.Count != candidates.Count || results.Length < baselines.Count)
            {
                throw new ArgumentException("Baselines, candidates and results must line up one per metric.");
            }

            // OverfitParallel.For, not raw TPL (OVERFIT008): it honours SuppressParallelismOnCurrentThread, so
            // an analyser already running inside a parallel replica degrades to a sequential loop instead of
            // oversubscribing the box.
            OverfitParallel.For(0, baselines.Count, metric =>
            {
                results[metric] = Compare(baselines[metric], candidates[metric]);
            });
        }

        /// <summary>
        /// The same test computed directly from bucketed counts — <b>O(K) in the number of buckets, with no
        /// sorting and no allocation</b>, whatever the request volume behind them.
        ///
        /// <para>Every observation inside a bucket is indistinguishable, so a bucket is exactly one tie group;
        /// U falls out of a single prefix-sum sweep and the tie correction out of the group sizes. Feed it a
        /// Prometheus histogram (de-cumulated to per-bucket counts) and a canary window of any size costs the
        /// same handful of nanoseconds.</para>
        ///
        /// <para><b>The price is resolution, not speed.</b> Bucketing rounds every observation to its bucket,
        /// so a regression smaller than a bucket boundary is invisible — with 25/50/100 ms boundaries a 40 → 46
        /// ms shift moves nothing at all and this returns "no difference" truthfully but uselessly. Use it when
        /// raw per-request values are not available; prefer <see cref="Compare(ReadOnlySpan{double}, ReadOnlySpan{double})"/>
        /// when they are.</para>
        /// </summary>
        /// <param name="baselineCounts">Per-bucket observation counts for the baseline, in ascending bucket order.</param>
        /// <param name="candidateCounts">Per-bucket counts for the candidate, over the <b>same</b> boundaries.</param>
        /// <exception cref="ArgumentException">The lengths differ, a count is negative, or either side is empty.</exception>
        public static MannWhitneyResult CompareHistograms(
            ReadOnlySpan<long> baselineCounts,
            ReadOnlySpan<long> candidateCounts)
        {
            if (baselineCounts.Length != candidateCounts.Length)
            {
                throw new ArgumentException("Both histograms must use the same bucket boundaries.");
            }

            if (baselineCounts.IsEmpty)
            {
                throw new ArgumentException("Histograms must contain at least one bucket.");
            }

            var n1 = 0L;
            var n2 = 0L;
            var baselineBelow = 0.0;
            var u = 0.0;
            var tieCorrection = 0.0;

            for (var k = 0; k < baselineCounts.Length; k++)
            {
                var b = baselineCounts[k];
                var c = candidateCounts[k];

                if (b < 0 || c < 0)
                {
                    throw new ArgumentException("Bucket counts must not be negative.");
                }

                // Candidate observations in bucket k beat every baseline observation below k outright, and tie
                // with the b of them sharing the bucket — a tie is worth half a pair.
                u += c * (baselineBelow + (0.5 * b));

                var tied = (double)b + c;

                tieCorrection += (tied * tied * tied) - tied;
                baselineBelow += b;
                n1 += b;
                n2 += c;
            }

            if (n1 == 0 || n2 == 0)
            {
                throw new ArgumentException("Both histograms must contain at least one observation.");
            }

            return Finish(u, (int)n1, (int)n2, tieCorrection);
        }

        /// <summary>
        /// Produces an ascending copy of <paramref name="source"/> in <paramref name="destination"/>, choosing
        /// the sort by size: the framework's introsort for small samples, an LSD radix sort once the window is
        /// large enough for a linear pass count to beat O(n log n) comparisons (measured ~3.9x at 5 000+).
        /// Falls back to introsort whenever <paramref name="spare"/> cannot hold the radix ping-pong buffer, so
        /// a caller supplying only the minimum scratch still gets a correct — merely slower — answer.
        /// </summary>
        private static void SortInto(ReadOnlySpan<double> source, Span<double> destination, Span<double> spare)
        {
            if (source.Length < RadixSortThreshold || spare.Length < source.Length)
            {
                source.CopyTo(destination);
                destination.Sort();

                // Sort hoists NaN to the front, so one look settles it for the whole sample.
                if (double.IsNaN(destination[0]))
                {
                    throw new ArgumentException("Samples must not contain NaN.");
                }

                return;
            }

            RadixSortInto(source, destination, MemoryMarshal.Cast<double, ulong>(spare).Slice(0, source.Length));
        }

        /// <summary>
        /// Least-significant-digit radix sort, eight passes of eight bits, with every byte histogram built in
        /// one sweep so passes over a constant byte can be skipped — which for real latency data, whose
        /// exponents occupy a narrow band, skips most of the high passes.
        /// </summary>
        private static void RadixSortInto(ReadOnlySpan<double> source, Span<double> destination, Span<ulong> spare)
        {
            var n = source.Length;

            // The sorted keys are built in the destination's own memory, reinterpreted as ulong; `spare` is
            // only the alternate half of the ping-pong.
            var primary = MemoryMarshal.Cast<double, ulong>(destination).Slice(0, n);

            // Eight byte histograms, 8 KB, from the pool rather than the stack.
            //
            // NEGATIVE RESULT (2026-07-24, Ryzen 9 9950X3D): this was a `stackalloc int[8 * 256]`, argued for
            // on the grounds that the histogram is touched eight times per element and wants to stay in L1
            // while a recycled pooled array would arrive cold. Measured, that argument does not hold — pooled
            // versus stack came out 1.006x at 500 samples per arm, 0.992x at 2 000 and 1.025x at 10 000, which
            // is a tie across the whole range where the radix path actually runs. (At 100 000 the pooled arm
            // read 1.22x slower, but with 5% run-to-run spread against the stack arm's 0.08%, and 100 000
            // observations per arm is far outside any evaluation window this code is for.)
            //
            // Since it buys nothing measurable, it honours the OVERFIT025 budget instead: 8 KB is sixteen
            // times the 512 B ceiling, and a StackOverflowException in a hosted library cannot be caught,
            // cannot be logged, and kills the host.
            //
            // clearMemory is not optional. ArrayPool hands back dirty arrays, and the counting loop below
            // reads each counter before writing it — exactly the same requirement the stackalloc version had
            // under this assembly's [module: SkipLocalsInit].
            using var histogramBuffer = new PooledBuffer<int>(8 * 256, clearMemory: true);
            var counts = histogramBuffer.Span;

            for (var i = 0; i < n; i++)
            {
                var value = source[i];

                // Radix ordering sends NaN to the far end rather than the front, so unlike the introsort path
                // it has to be caught here, while the data is being read anyway.
                if (double.IsNaN(value))
                {
                    throw new ArgumentException("Samples must not contain NaN.");
                }

                var key = EncodeOrderPreserving(value);
                primary[i] = key;

                for (var pass = 0; pass < 8; pass++)
                {
                    counts[(pass * 256) + (int)((key >> (pass * 8)) & 0xFF)]++;
                }
            }

            var from = primary;
            var to = spare;
            var sortedInPrimary = true;

            for (var pass = 0; pass < 8; pass++)
            {
                var histogram = counts.Slice(pass * 256, 256);

                // Every key shares this byte: the pass would be an identity permutation.
                if (histogram[(int)((from[0] >> (pass * 8)) & 0xFF)] == n)
                {
                    continue;
                }

                var offset = 0;
                for (var bucket = 0; bucket < 256; bucket++)
                {
                    var count = histogram[bucket];
                    histogram[bucket] = offset;
                    offset += count;
                }

                for (var i = 0; i < n; i++)
                {
                    var key = from[i];
                    to[histogram[(int)((key >> (pass * 8)) & 0xFF)]++] = key;
                }

                // Deconstruction cannot swap ref structs, so this stays explicit.
                var previous = from;
                from = to;
                to = previous;
                sortedInPrimary = !sortedInPrimary;
            }

            if (!sortedInPrimary)
            {
                from.CopyTo(primary);
            }

            // Decode in place: primary aliases destination, and element i is read before it is overwritten.
            for (var i = 0; i < n; i++)
            {
                destination[i] = DecodeOrderPreserving(primary[i]);
            }
        }

        /// <summary>
        /// Maps a double onto a ulong whose unsigned ordering matches the double's ordering: flip the sign bit
        /// for non-negatives, invert every bit for negatives. Bijective, so the sort can run entirely on keys.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong EncodeOrderPreserving(double value)
        {
            var bits = BitConverter.DoubleToUInt64Bits(value);

            return (bits & (1UL << 63)) != 0 ? ~bits : bits | (1UL << 63);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double DecodeOrderPreserving(ulong key)
        {
            var bits = (key & (1UL << 63)) != 0 ? key & ~(1UL << 63) : ~key;

            return BitConverter.UInt64BitsToDouble(bits);
        }

        /// <summary>Shared tail: turn U plus the tie correction into a z score, a p-value and an effect size.</summary>
        private static MannWhitneyResult Finish(double u, int n1, int n2, double tieCorrection)
        {
            var pairs = (double)n1 * n2;
            var total = (double)n1 + n2;

            var probabilitySuperior = u / pairs;
            var cliffsDelta = (2.0 * probabilitySuperior) - 1.0;

            var meanU = pairs / 2.0;
            var variance = (pairs / 12.0) * ((total + 1.0) - (tieCorrection / (total * (total - 1.0))));

            // Every observation identical (or a degenerate single-pair comparison): the data carry no
            // ordering information, so report "no evidence" rather than dividing by zero.
            if (variance <= 0.0)
            {
                return new MannWhitneyResult(u, 0.0, 1.0, probabilitySuperior, cliffsDelta, n1, n2);
            }

            // Continuity correction toward the mean — without it the discrete U is tested against a
            // continuous normal and small samples come out anti-conservative.
            var deviation = u - meanU;
            var corrected = deviation > 0.0 ? deviation - 0.5 : deviation + 0.5;

            if (Math.Abs(deviation) < 0.5)
            {
                corrected = 0.0;
            }

            var z = corrected / Math.Sqrt(variance);

            return new MannWhitneyResult(
                u,
                z,
                1.0 - NormalDistribution.Cdf(z),
                probabilitySuperior,
                cliffsDelta,
                n1,
                n2);
        }
    }
}
