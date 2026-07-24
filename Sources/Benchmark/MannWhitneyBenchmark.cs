// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using DevOnBike.Overfit.Statistics;

namespace Benchmarks
{
    /// <summary>
    /// The cost of one canary evaluation, across every shape the same statistic can be computed in.
    ///
    /// <para><see cref="PairedSort_Original"/> is the first shipped implementation, kept verbatim as the
    /// baseline: it sorted one combined array with a parallel <c>bool[]</c> payload marking each element's
    /// sample. Everything below it is a candidate replacement measured against exactly that.</para>
    ///
    /// <para>Deliberately NOT on the shared <c>BenchmarkConfig</c>: that job pins
    /// <c>InvocationCount=1 / UnrollFactor=1</c>, which suits multi-millisecond model runs and leaves a
    /// microsecond-scale routine measuring timer noise. <see cref="SimpleJobAttribute"/> keeps BDN's default
    /// invocation counts.</para>
    /// </summary>
    [SimpleJob]
    [MemoryDiagnoser]
    public class MannWhitneyBenchmark
    {
        private const int Buckets = 24;

        private double[] _baseline = [];
        private double[] _candidate = [];
        private double[] _sortedBaseline = [];
        private double[] _sortedCandidate = [];
        private double[] _scratch = [];
        private long[] _baselineCounts = [];
        private long[] _candidateCounts = [];
        private ulong[] _radixKeys = [];
        private ulong[] _radixTemp = [];

        /// <summary>Observations per arm in the evaluation window — 10 000 is a minute of 10k rpm traffic.</summary>
        [Params(500, 2_000, 10_000, 100_000)]
        public int SamplesPerArm
        {
            get; set;
        }

        [GlobalSetup]
        public void Setup()
        {
            var rng = new Random(20260724);

            _baseline = SampleLatencies(rng, SamplesPerArm, 1.0);
            _candidate = SampleLatencies(rng, SamplesPerArm, 1.1);

            _sortedBaseline = (double[])_baseline.Clone();
            _sortedCandidate = (double[])_candidate.Clone();
            Array.Sort(_sortedBaseline);
            Array.Sort(_sortedCandidate);

            _scratch = new double[MannWhitneyU.RecommendedScratchLength(SamplesPerArm, SamplesPerArm)];
            _radixKeys = new ulong[2 * SamplesPerArm];
            _radixTemp = new ulong[2 * SamplesPerArm];

            _baselineCounts = Bucketize(_baseline);
            _candidateCounts = Bucketize(_candidate);
        }

        /// <summary>The original shape: one combined sort carrying a parallel <c>bool[]</c> payload.</summary>
        [Benchmark(Baseline = true)]
        public double PairedSort_Original()
        {
            var n1 = _baseline.Length;
            var n2 = _candidate.Length;
            var total = n1 + n2;

            var values = new double[total];
            var fromCandidate = new bool[total];

            _baseline.CopyTo(values.AsSpan(0, n1));
            _candidate.CopyTo(values.AsSpan(n1, n2));
            for (var i = n1; i < total; i++)
            {
                fromCandidate[i] = true;
            }

            Array.Sort(values, fromCandidate);

            var candidateRankSum = 0.0;
            var index = 0;
            while (index < total)
            {
                var last = index;
                while (last + 1 < total && values[last + 1] == values[index])
                {
                    last++;
                }

                var midRank = (index + last + 2) / 2.0;
                for (var k = index; k <= last; k++)
                {
                    if (fromCandidate[k])
                    {
                        candidateRankSum += midRank;
                    }
                }

                index = last + 1;
            }

            return candidateRankSum;
        }

        /// <summary>Shipping default: split sorts + merge, scratch rented from the shared pool.</summary>
        [Benchmark]
        public double Pooled()
        {
            return MannWhitneyU.Compare(_baseline, _candidate).PValueCandidateGreater;
        }

        /// <summary>Same, with scratch the caller keeps alive — nothing rented, nothing allocated.</summary>
        [Benchmark]
        public double CallerScratch()
        {
            return MannWhitneyU.Compare(_baseline, _candidate, _scratch).PValueCandidateGreater;
        }

        /// <summary>Merge only, on samples that arrive ordered: no sort at all.</summary>
        [Benchmark]
        public double PreSorted()
        {
            return MannWhitneyU.CompareSorted(_sortedBaseline, _sortedCandidate).PValueCandidateGreater;
        }

        /// <summary>
        /// Candidate: replace introsort with an LSD radix sort over the order-preserving 64-bit encoding of the
        /// doubles, skipping passes whose byte is constant. Hypothesis — linear beats comparison sorting once
        /// the window is large. Measured, not assumed.
        /// </summary>
        [Benchmark]
        public double RadixSort()
        {
            var n1 = _baseline.Length;
            var n2 = _candidate.Length;

            RadixSortInto(_baseline, _scratch.AsSpan(0, n1), _radixKeys, _radixTemp);
            RadixSortInto(_candidate, _scratch.AsSpan(n1, n2), _radixKeys, _radixTemp);

            return MannWhitneyU
                .CompareSorted(_scratch.AsSpan(0, n1), _scratch.AsSpan(n1, n2))
                .PValueCandidateGreater;
        }

        /// <summary>Bucketed counts, straight from a scraped histogram: O(K), independent of the window size.</summary>
        [Benchmark]
        public double Histograms()
        {
            return MannWhitneyU.CompareHistograms(_baselineCounts, _candidateCounts).PValueCandidateGreater;
        }

        /// <summary>Bucketing included, for a caller that has raw values and must reduce them first.</summary>
        [Benchmark]
        public double HistogramsIncludingBucketing()
        {
            var b = Bucketize(_baseline);
            var c = Bucketize(_candidate);

            return MannWhitneyU.CompareHistograms(b, c).PValueCandidateGreater;
        }

        // Maps a double onto a ulong whose unsigned ordering matches the double's ordering: flip the sign bit
        // for non-negatives, invert every bit for negatives.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong EncodeKey(double value)
        {
            var bits = BitConverter.DoubleToUInt64Bits(value);

            return (bits & (1UL << 63)) != 0 ? ~bits : bits | (1UL << 63);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double DecodeKey(ulong key)
        {
            var bits = (key & (1UL << 63)) != 0 ? key & ~(1UL << 63) : ~key;

            return BitConverter.UInt64BitsToDouble(bits);
        }

        private static void RadixSortInto(
            ReadOnlySpan<double> source,
            Span<double> destination,
            ulong[] keys,
            ulong[] temp)
        {
            var n = source.Length;

            for (var i = 0; i < n; i++)
            {
                keys[i] = EncodeKey(source[i]);
            }

            // All eight byte histograms in one pass over the keys; a pass whose byte never varies is skipped.
            Span<int> counts = stackalloc int[8 * 256];
            counts.Clear();

            for (var i = 0; i < n; i++)
            {
                var key = keys[i];
                for (var pass = 0; pass < 8; pass++)
                {
                    counts[(pass * 256) + (int)((key >> (pass * 8)) & 0xFF)]++;
                }
            }

            var from = keys;
            var to = temp;

            for (var pass = 0; pass < 8; pass++)
            {
                var histogram = counts.Slice(pass * 256, 256);
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

                (from, to) = (to, from);
            }

            for (var i = 0; i < n; i++)
            {
                destination[i] = DecodeKey(from[i]);
            }
        }

        private static double[] SampleLatencies(Random rng, int count, double scale)
        {
            var values = new double[count];

            for (var i = 0; i < count; i++)
            {
                var u1 = 1.0 - rng.NextDouble();
                var u2 = 1.0 - rng.NextDouble();
                var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                values[i] = scale * 40.0 * Math.Exp(0.35 * normal);
            }

            return values;
        }

        // Exponential boundaries at 5 ms x 1.35^k, the shape of a typical latency histogram.
        private static long[] Bucketize(double[] values)
        {
            var counts = new long[Buckets];

            for (var i = 0; i < values.Length; i++)
            {
                var slot = (int)(Math.Log(Math.Max(values[i], 5.0) / 5.0) / Math.Log(1.35));
                counts[Math.Min(slot, Buckets - 1)]++;
            }

            return counts;
        }
    }
}
