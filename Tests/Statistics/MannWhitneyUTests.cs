// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Statistics;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Statistics
{
    public sealed class MannWhitneyUTests
    {
        [Fact]
        public void PerfectSeparation_YieldsMaximumUAndEffect()
        {
            double[] baseline = [1, 2, 3, 4, 5];
            double[] candidate = [6, 7, 8, 9, 10];

            var result = MannWhitneyU.Compare(baseline, candidate);

            // Every one of the 5x5 pairs has the candidate larger.
            Assert.Equal(25.0, result.U, 9);
            Assert.Equal(1.0, result.ProbabilitySuperior, 9);
            Assert.Equal(1.0, result.CliffsDelta, 9);
            Assert.True(result.PValueCandidateGreater < 0.01, $"p = {result.PValueCandidateGreater}");
        }

        [Fact]
        public void PerfectSeparation_ProducesTheTextbookPValue()
        {
            // n1 = n2 = 10, U = 100: mean 50, var = (100/12) * 21 = 175, z = 49.5 / sqrt(175) = 3.742.
            double[] baseline = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
            double[] candidate = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];

            var result = MannWhitneyU.Compare(baseline, candidate);

            Assert.Equal(100.0, result.U, 9);
            Assert.Equal(3.742, result.Z, 3);
            Assert.InRange(result.PValueCandidateGreater, 5e-5, 2e-4);
        }

        [Fact]
        public void Ties_AreScoredAsHalfAPair()
        {
            // baseline {1,2,3} vs candidate {2,3,4}: pairwise wins 1.5 + 2.5 + 3 = 7 of 9.
            double[] baseline = [1, 2, 3];
            double[] candidate = [2, 3, 4];

            var result = MannWhitneyU.Compare(baseline, candidate);

            Assert.Equal(7.0, result.U, 9);
            Assert.Equal(7.0 / 9.0, result.ProbabilitySuperior, 9);
            Assert.Equal((2.0 * 7.0 / 9.0) - 1.0, result.CliffsDelta, 9);
        }

        [Fact]
        public void IdenticalSamples_ReportNoDifferenceAndNoSignificance()
        {
            double[] sample = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3];

            var result = MannWhitneyU.Compare(sample, sample);

            Assert.Equal(0.5, result.ProbabilitySuperior, 9);
            Assert.Equal(0.0, result.CliffsDelta, 9);
            Assert.True(result.PValueCandidateGreater > 0.4, $"p = {result.PValueCandidateGreater}");
        }

        [Fact]
        public void EveryObservationEqual_ReportsNoEvidenceInsteadOfDividingByZero()
        {
            double[] baseline = [7, 7, 7, 7];
            double[] candidate = [7, 7, 7, 7];

            var result = MannWhitneyU.Compare(baseline, candidate);

            Assert.Equal(0.0, result.Z, 9);
            Assert.Equal(1.0, result.PValueCandidateGreater, 9);
            Assert.Equal(0.5, result.ProbabilitySuperior, 9);
        }

        [Fact]
        public void CandidateFaster_ProducesNegativeEffectAndNoRegressionSignal()
        {
            double[] baseline = [40, 41, 42, 43, 44, 45, 46, 47];
            double[] candidate = [30, 31, 32, 33, 34, 35, 36, 37];

            var result = MannWhitneyU.Compare(baseline, candidate);

            Assert.Equal(0.0, result.U, 9);
            Assert.Equal(-1.0, result.CliffsDelta, 9);
            // The test is one-sided on "candidate worse", so a large improvement must NOT trip the gate.
            Assert.True(result.PValueCandidateGreater > 0.99, $"p = {result.PValueCandidateGreater}");
        }

        [Fact]
        public void SwappingTheSamples_ComplementsU()
        {
            double[] a = [1, 4, 4, 7, 9, 12];
            double[] b = [2, 4, 6, 8, 8, 11, 14];

            var forward = MannWhitneyU.Compare(a, b);
            var reversed = MannWhitneyU.Compare(b, a);

            Assert.Equal(a.Length * b.Length, forward.U + reversed.U, 9);
            Assert.Equal(1.0, forward.ProbabilitySuperior + reversed.ProbabilitySuperior, 9);
            Assert.Equal(0.0, forward.CliffsDelta + reversed.CliffsDelta, 9);
        }

        [Fact]
        public void HeavyTailedLatencies_WithNoShift_AreNotFlagged()
        {
            // Log-normal-ish latencies with a handful of extreme outliers on the candidate side: a
            // mean/sigma gate fires on this, a rank test does not, because the ordering barely moves.
            var rng = new Random(20260724);
            var baseline = SampleLatencies(rng, 400, spikes: 4);
            var candidate = SampleLatencies(rng, 400, spikes: 12);

            var result = MannWhitneyU.Compare(baseline, candidate);

            Assert.True(Mean(candidate) > Mean(baseline) * 1.15,
                "the outliers must move the mean, otherwise this test proves nothing");
            Assert.True(Math.Abs(result.CliffsDelta) < 0.15, $"delta = {result.CliffsDelta}");
            Assert.True(result.PValueCandidateGreater > 0.05, $"p = {result.PValueCandidateGreater}");
        }

        [Fact]
        public void RealShift_IsFlagged()
        {
            var rng = new Random(7);
            var baseline = SampleLatencies(rng, 300, spikes: 4);
            var candidate = SampleLatencies(rng, 300, spikes: 4);
            for (var i = 0; i < candidate.Length; i++)
            {
                candidate[i] *= 1.35;
            }

            var result = MannWhitneyU.Compare(baseline, candidate);

            Assert.True(result.PValueCandidateGreater < 0.001, $"p = {result.PValueCandidateGreater}");
            Assert.True(result.CliffsDelta > 0.25, $"delta = {result.CliffsDelta}");
        }

        [Fact]
        public void LargeSamples_MakeATinyShiftSignificantButTheEffectStaysNegligible()
        {
            // This is the reason the result carries an effect size at all: 1000 vs 1000 observations turn a
            // 4%-of-range shift into p ~ 0.001, while Cliff's delta correctly calls it negligible (<0.147).
            // U = 538720 strict wins + 960 exact ties scored as one half = 539200 of 1e6.
            var baseline = new double[1000];
            var candidate = new double[1000];
            for (var i = 0; i < baseline.Length; i++)
            {
                baseline[i] = i;
                candidate[i] = i + 40;
            }

            var result = MannWhitneyU.Compare(baseline, candidate);

            Assert.Equal(539200.0, result.U, 9);
            Assert.True(result.PValueCandidateGreater < 0.01, $"p = {result.PValueCandidateGreater}");
            Assert.Equal(0.0784, result.CliffsDelta, 4);
            Assert.True(Math.Abs(result.CliffsDelta) < 0.147, "a gate on p alone would reject this deploy");
        }

        [Fact]
        public void UnequalSampleSizes_AreHandled()
        {
            double[] baseline = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
            double[] candidate = [5, 6, 7];

            var result = MannWhitneyU.Compare(baseline, candidate);

            // candidate 5 beats {1,2,3,4} + ties 5 -> 4.5; 6 -> 5.5; 7 -> 6.5. Total 16.5 of 36.
            Assert.Equal(16.5, result.U, 9);
            Assert.Equal(12, result.BaselineCount);
            Assert.Equal(3, result.CandidateCount);
        }

        [Fact]
        public void EmptySample_Throws()
        {
            Assert.Throws<ArgumentException>(() => MannWhitneyU.Compare([], [1.0, 2.0]));
            Assert.Throws<ArgumentException>(() => MannWhitneyU.Compare([1.0, 2.0], []));
        }

        [Fact]
        public void NaN_Throws()
        {
            Assert.Throws<ArgumentException>(() => MannWhitneyU.Compare([1.0, double.NaN], [1.0, 2.0]));
        }

        [Fact]
        public void HistogramPath_MatchesTheRawPathExactly()
        {
            // Same data expressed both ways: bucket k holds b[k] baseline and c[k] candidate observations,
            // all of which are exactly equal, so the raw path sees one big tie group per bucket.
            long[] baselineCounts = [120, 340, 410, 90, 30, 8, 2];
            long[] candidateCounts = [80, 300, 420, 130, 50, 15, 5];

            var raw = MannWhitneyU.Compare(Expand(baselineCounts), Expand(candidateCounts));
            var bucketed = MannWhitneyU.CompareHistograms(baselineCounts, candidateCounts);

            Assert.Equal(raw.U, bucketed.U, 6);
            Assert.Equal(raw.Z, bucketed.Z, 9);
            Assert.Equal(raw.PValueCandidateGreater, bucketed.PValueCandidateGreater, 12);
            Assert.Equal(raw.CliffsDelta, bucketed.CliffsDelta, 12);
            Assert.Equal(raw.BaselineCount, bucketed.BaselineCount);
            Assert.Equal(raw.CandidateCount, bucketed.CandidateCount);
        }

        [Fact]
        public void HistogramPath_DetectsARightwardShiftOfMass()
        {
            long[] baselineCounts = [500, 400, 80, 15, 5];
            long[] candidateCounts = [200, 450, 250, 70, 30];

            var result = MannWhitneyU.CompareHistograms(baselineCounts, candidateCounts);

            Assert.True(result.PValueCandidateGreater < 1e-6, $"p = {result.PValueCandidateGreater}");
            Assert.True(result.CliffsDelta > 0.2, $"delta = {result.CliffsDelta}");
        }

        [Fact]
        public void HistogramPath_IsBlindToARegressionSmallerThanABucket()
        {
            // The documented price of the O(K) path: a real 15% shift that never crosses a boundary produces
            // an identical histogram, and the test correctly reports "no evidence" from data that has none.
            long[] identical = [100, 900, 200, 10];

            var result = MannWhitneyU.CompareHistograms(identical, identical);

            Assert.Equal(0.5, result.ProbabilitySuperior, 9);
            Assert.True(result.PValueCandidateGreater > 0.4, $"p = {result.PValueCandidateGreater}");
        }

        [Fact]
        public void HistogramPath_RejectsMismatchedOrEmptyInput()
        {
            Assert.Throws<ArgumentException>(() => MannWhitneyU.CompareHistograms([1L, 2L], [1L, 2L, 3L]));
            Assert.Throws<ArgumentException>(() => MannWhitneyU.CompareHistograms([], []));
            Assert.Throws<ArgumentException>(() => MannWhitneyU.CompareHistograms([0L, 0L], [1L, 2L]));
            Assert.Throws<ArgumentException>(() => MannWhitneyU.CompareHistograms([-1L, 3L], [1L, 2L]));
        }

        [Fact]
        public void AllPaths_AgreeOnTheSameData()
        {
            var rng = new Random(4242);
            var baseline = SampleLatencies(rng, 250, spikes: 3);
            var candidate = SampleLatencies(rng, 310, spikes: 3);

            var pooled = MannWhitneyU.Compare(baseline, candidate);

            var scratch = new double[baseline.Length + candidate.Length];
            var withScratch = MannWhitneyU.Compare(baseline, candidate, scratch);

            var sortedBaseline = (double[])baseline.Clone();
            var sortedCandidate = (double[])candidate.Clone();
            Array.Sort(sortedBaseline);
            Array.Sort(sortedCandidate);
            var preSorted = MannWhitneyU.CompareSorted(sortedBaseline, sortedCandidate);

            Assert.Equal(pooled, withScratch);
            Assert.Equal(pooled, preSorted);
            Assert.True(pooled.U > 0.0);
        }

        [Fact]
        public void Compare_DoesNotMutateItsInputs()
        {
            double[] baseline = [9, 3, 7, 1, 5];
            double[] candidate = [8, 2, 6];
            var baselineCopy = (double[])baseline.Clone();
            var candidateCopy = (double[])candidate.Clone();

            MannWhitneyU.Compare(baseline, candidate);

            Assert.Equal(baselineCopy, baseline);
            Assert.Equal(candidateCopy, candidate);
        }

        [Fact]
        public void Compare_WithCallerScratch_AllocatesNothing()
        {
            var rng = new Random(11);
            var baseline = SampleLatencies(rng, 2000, spikes: 0);
            var candidate = SampleLatencies(rng, 2000, spikes: 0);
            var scratch = new double[4000];

            // Warm up the JIT so the measurement sees steady state, not first-call codegen.
            for (var i = 0; i < 3; i++)
            {
                MannWhitneyU.Compare(baseline, candidate, scratch);
            }

            var before = GC.GetAllocatedBytesForCurrentThread();
            for (var i = 0; i < 20; i++)
            {
                MannWhitneyU.Compare(baseline, candidate, scratch);
            }

            Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        }

        [Fact]
        public void Compare_PooledOverload_AllocatesNothing()
        {
            var rng = new Random(12);
            var baseline = SampleLatencies(rng, 20_000, spikes: 0);
            var candidate = SampleLatencies(rng, 20_000, spikes: 0);

            for (var i = 0; i < 3; i++)
            {
                MannWhitneyU.Compare(baseline, candidate);
            }

            // 40 000 doubles is 320 KB — comfortably past the 85 KB large-object-heap threshold, which is the
            // exact shape the pooled scratch exists to keep off the GC's plate in a long-lived analyser.
            // Not "exactly zero on one attempt". ArrayPool<T>.Shared is trimmed on gen2 collections, so on a
            // loaded box a Rent that normally reuses a buffer allocates a fresh 320 KB one instead — a
            // guaranteed flake, and one that says nothing about this code. The minimum across attempts is the
            // steady-state figure, and it is still asserted at zero.
            var allocated = AllocationProbe.MinimumBytes(
                () => MannWhitneyU.Compare(baseline, candidate), calls: 10);

            Assert.Equal(0, allocated);
        }

        [Fact]
        public void Compare_ScratchTooSmall_Throws()
        {
            var scratch = new double[4];

            Assert.Throws<ArgumentException>(() =>
                MannWhitneyU.Compare([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], scratch));
        }

        [Fact]
        public void CompareSorted_RejectsNaNAndEmptyInput()
        {
            Assert.Throws<ArgumentException>(() => MannWhitneyU.CompareSorted([double.NaN, 1.0], [1.0, 2.0]));
            Assert.Throws<ArgumentException>(() => MannWhitneyU.CompareSorted([1.0, 2.0], [double.NaN, 1.0]));
            Assert.Throws<ArgumentException>(() => MannWhitneyU.CompareSorted([], [1.0]));
        }

        [Fact]
        public void CompareSorted_HandlesFullyInterleavedTies()
        {
            // Every value shared by both samples: the merge must consume both runs of each distinct value in
            // one step, or the tie correction and U both drift.
            double[] baseline = [1, 1, 2, 2, 3, 3];
            double[] candidate = [1, 2, 2, 3, 3, 3];

            var merged = MannWhitneyU.CompareSorted(baseline, candidate);
            var expanded = MannWhitneyU.Compare(baseline, candidate);

            Assert.Equal(expanded, merged);
            // Pairwise: candidate 1 -> 1.0; each 2 -> 2+1=3.0; each 3 -> 4+1=5.0. Total 1+3+3+5+5+5 = 22.
            Assert.Equal(22.0, merged.U, 9);
        }

        [Fact]
        public void SingletonSamples_AreHandled()
        {
            var worse = MannWhitneyU.Compare([1.0], [2.0]);
            Assert.Equal(1.0, worse.U, 9);
            Assert.Equal(1.0, worse.ProbabilitySuperior, 9);

            // n1 = n2 = 1 gives variance 1/6 but |U - mean| = 0.5, which the continuity correction zeroes:
            // a single pair can never be evidence of anything.
            Assert.Equal(0.0, worse.Z, 9);
            // 7 places, not more: the erf approximation is documented to ~1.5e-7 and Phi(0) lands on
            // 0.5000000005 rather than exactly one half.
            Assert.Equal(0.5, worse.PValueCandidateGreater, 7);
        }

        [Fact]
        public void RadixPath_MatchesTheIntrosortPath_IncludingNegativeValues()
        {
            // Past 2048 observations Compare switches to the radix sort. Mixed signs matter: the bit encoding
            // has to invert negatives, and getting that wrong reverses half the ordering silently.
            var rng = new Random(31337);
            var baseline = new double[3000];
            var candidate = new double[2500];
            for (var i = 0; i < baseline.Length; i++)
            {
                baseline[i] = (rng.NextDouble() * 200.0) - 100.0;
            }

            for (var i = 0; i < candidate.Length; i++)
            {
                candidate[i] = (rng.NextDouble() * 200.0) - 90.0;
            }

            var radix = MannWhitneyU.Compare(baseline, candidate);

            // Minimum scratch has no room for the radix ping-pong, so this takes the introsort fallback.
            var minimum = new double[MannWhitneyU.MinimumScratchLength(baseline.Length, candidate.Length)];
            var introsort = MannWhitneyU.Compare(baseline, candidate, minimum);

            Assert.Equal(introsort, radix);
            Assert.True(radix.PValueCandidateGreater < 0.05, $"p = {radix.PValueCandidateGreater}");
        }

        [Fact]
        public void RadixPath_SortsCorrectlyWithHeavyTiesAndExtremeValues()
        {
            var values = new double[4000];
            for (var i = 0; i < values.Length; i++)
            {
                // Only eight distinct values, plus zero and both infinities: maximum tie pressure, and the
                // exponent range that makes the high radix passes non-constant.
                values[i] = (i % 8) switch
                {
                    0 => 0.0,
                    1 => -0.0,
                    2 => double.Epsilon,
                    3 => -1e300,
                    4 => 1e300,
                    5 => double.NegativeInfinity,
                    6 => double.PositiveInfinity,
                    _ => 1.5,
                };
            }

            var other = new double[4000];
            values.CopyTo(other, 0);

            var radix = MannWhitneyU.Compare(values, other);
            var minimum = new double[MannWhitneyU.MinimumScratchLength(values.Length, other.Length)];
            var introsort = MannWhitneyU.Compare(values, other, minimum);

            Assert.Equal(introsort, radix);
            Assert.Equal(0.5, radix.ProbabilitySuperior, 9);
        }

        [Fact]
        public void RadixPath_RejectsNaN()
        {
            var baseline = new double[3000];
            var candidate = new double[3000];
            for (var i = 0; i < baseline.Length; i++)
            {
                baseline[i] = i;
                candidate[i] = i + 1;
            }

            baseline[1500] = double.NaN;

            // Radix ordering places NaN at the far end, so it cannot be caught by inspecting the sorted head —
            // this guards the check that runs during key encoding instead.
            Assert.Throws<ArgumentException>(() => MannWhitneyU.Compare(baseline, candidate));
        }

        [Fact]
        public void ScratchLengthHelpers_DescribeWhatTheOverloadsAccept()
        {
            Assert.Equal(300, MannWhitneyU.MinimumScratchLength(100, 200));
            Assert.Equal(500, MannWhitneyU.RecommendedScratchLength(100, 200));

            var baseline = new double[3000];
            var candidate = new double[3000];
            for (var i = 0; i < baseline.Length; i++)
            {
                baseline[i] = i;
                candidate[i] = i + 5;
            }

            var recommended = new double[MannWhitneyU.RecommendedScratchLength(3000, 3000)];
            var minimum = new double[MannWhitneyU.MinimumScratchLength(3000, 3000)];

            // Both are correct; only the speed differs.
            Assert.Equal(
                MannWhitneyU.Compare(baseline, candidate, minimum),
                MannWhitneyU.Compare(baseline, candidate, recommended));
        }

        [Fact]
        public void CompareMany_MatchesAMetricByMetricLoop()
        {
            var rng = new Random(909);
            const int metrics = 12;
            var baselines = new double[metrics][];
            var candidates = new double[metrics][];
            for (var m = 0; m < metrics; m++)
            {
                baselines[m] = SampleLatencies(rng, 2500 + m, spikes: 2);
                candidates[m] = SampleLatencies(rng, 2400 + m, spikes: 2);
            }

            var expected = new MannWhitneyResult[metrics];
            for (var m = 0; m < metrics; m++)
            {
                expected[m] = MannWhitneyU.Compare(baselines[m], candidates[m]);
            }

            var actual = new MannWhitneyResult[metrics];
            MannWhitneyU.CompareMany(baselines, candidates, actual);

            // Bit-identical, not merely close: the workers share nothing but the pool, so parallelism must not
            // be observable in the numbers.
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void CompareMany_RejectsMisalignedInput()
        {
            var one = new[] { new[] { 1.0, 2.0 } };
            var two = new[] { new[] { 1.0, 2.0 }, new[] { 3.0, 4.0 } };

            Assert.Throws<ArgumentException>(() => MannWhitneyU.CompareMany(one, two, new MannWhitneyResult[2]));
            Assert.Throws<ArgumentException>(() => MannWhitneyU.CompareMany(two, two, new MannWhitneyResult[1]));
        }

        [Fact]
        public void Comparer_ProjectsOntoTheTestAgnosticVerdict()
        {
            double[] baseline = [10, 11, 12, 13, 14, 15, 16, 17];
            double[] candidate = [20, 21, 22, 23, 24, 25, 26, 27];

            var raw = MannWhitneyU.Compare(baseline, candidate);
            var projected = MannWhitneyComparer.Instance.Compare(baseline, candidate);

            Assert.Equal("mann-whitney-u", MannWhitneyComparer.Instance.Name);
            Assert.Equal(raw.PValueCandidateGreater, projected.PValueCandidateWorse, 12);
            Assert.Equal(raw.CliffsDelta, projected.EffectSize, 12);
            Assert.Equal(raw.ProbabilitySuperior, projected.ProbabilityCandidateWorse, 12);
            Assert.Equal(raw.BaselineCount, projected.BaselineCount);
            Assert.Equal(raw.CandidateCount, projected.CandidateCount);
        }

        [Fact]
        public void Comparer_ExposesBothIngestionShapes()
        {
            long[] baselineCounts = [400, 300, 50, 5];
            long[] candidateCounts = [150, 350, 200, 40];

            ITwoSampleComparer comparer = MannWhitneyComparer.Instance;
            var bucketed = comparer.CompareHistograms(baselineCounts, candidateCounts);
            var raw = comparer.Compare(Expand(baselineCounts), Expand(candidateCounts));

            Assert.Equal(raw, bucketed);
            Assert.True(bucketed.EffectSize > 0.2, $"effect = {bucketed.EffectSize}");
        }

        [Fact]
        public void Gate_RequiresSignificanceAndEffectAndData()
        {
            // Significant but negligible: the 1000-vs-1000 case from LargeSamples_... above.
            var significantButTiny = new TwoSampleComparison(0.001, 0.0784, 0.5392, 1000, 1000);
            Assert.False(significantButTiny.IsRegression(0.05, 0.15, 100));

            // Large effect but no significance behind it.
            var largeButUnconvincing = new TwoSampleComparison(0.30, 0.60, 0.80, 400, 400);
            Assert.False(largeButUnconvincing.IsRegression(0.05, 0.15, 100));

            // Both, and enough data.
            var real = new TwoSampleComparison(0.0004, 0.42, 0.71, 400, 400);
            Assert.True(real.IsRegression(0.05, 0.15, 100));
        }

        [Fact]
        public void Gate_TreatsTooLittleDataAsNoVerdict_NotAsHealthy()
        {
            // Numbers that would otherwise scream regression, on eight requests.
            var thin = new TwoSampleComparison(0.0001, 0.95, 0.975, 8, 4);

            Assert.False(thin.IsRegression(0.05, 0.15, 100));
            Assert.False(thin.HasEnoughData(100));

            // The distinction the caller must act on: this is "no evidence", not "no regression".
            var healthy = new TwoSampleComparison(0.6, 0.01, 0.505, 4000, 4000);
            Assert.False(healthy.IsRegression(0.05, 0.15, 100));
            Assert.True(healthy.HasEnoughData(100));
        }

        private static double[] Expand(long[] counts)
        {
            var total = 0L;
            for (var k = 0; k < counts.Length; k++)
            {
                total += counts[k];
            }

            var values = new double[total];
            var at = 0;
            for (var k = 0; k < counts.Length; k++)
            {
                for (var i = 0L; i < counts[k]; i++)
                {
                    values[at++] = k;
                }
            }

            return values;
        }

        private static double[] SampleLatencies(Random rng, int count, int spikes)
        {
            var values = new double[count];
            for (var i = 0; i < count; i++)
            {
                // exp(N(0,1)) scaled to a ~40 ms median inter-token latency
                var u1 = 1.0 - rng.NextDouble();
                var u2 = 1.0 - rng.NextDouble();
                var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                values[i] = 40.0 * Math.Exp(0.25 * normal);
            }

            // Fixed, evenly-spread positions: random placement can collide and leave the fixture with fewer
            // effective outliers than the test assumes.
            for (var i = 0; i < spikes; i++)
            {
                values[i * count / spikes] = 25.0 * 40.0;
            }

            return values;
        }

        private static double Mean(double[] values)
        {
            var sum = 0.0;
            for (var i = 0; i < values.Length; i++)
            {
                sum += values[i];
            }

            return sum / values.Length;
        }
    }
}
