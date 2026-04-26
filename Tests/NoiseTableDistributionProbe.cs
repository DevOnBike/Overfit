// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Storage;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    ///     Sanity check that the new VectorizedRandom-backed PrecomputedNoiseTable fill
    ///     still produces standard-normal samples (mean≈0, variance≈1) with per-segment
    ///     statistics roughly matching what a Random-backed fill used to give.
    ///     Run only if OpenAiEsStrategyTests.Initialize_ResetsAdamMomentsAcrossTrainingRuns
    ///     starts failing after the RNG swap.
    /// </summary>
    public sealed class NoiseTableDistributionProbe
    {
        private readonly ITestOutputHelper _output;

        public NoiseTableDistributionProbe(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void FirstSamplesAreUnbiased()
        {
            // Matches the failing test exactly: length=1024, seed=1.
            var table = new PrecomputedNoiseTable(length: 1024, seed: 1);

            // The failing ES test touches ~32 noise values in its first generation.
            // Check the first 128 to smooth out pure sample luck while still probing
            // "what does the ES actually see on generation 1".
            ReportStats("first 128", table.GetSlice(0, 128));
            ReportStats("first 32", table.GetSlice(0, 32));
            ReportStats("first 8", table.GetSlice(0, 8));
            ReportStats("entire table", table.GetSlice(0, 1024));
        }

        [Fact]
        public void FirstSamplesPerDimension_NotSystematicallyPositive()
        {
            // Reproduces the slicing that OpenAiEsStrategy does for (populationSize=16,
            // parameterCount=4): eight 4-element slices at "random" offsets. For seed=1
            // the strategy's internal Random picks deterministic offsets; we don't
            // have those here but can at least verify that the first N slices don't
            // all have the same sign in any dimension — a catastrophic condition for
            // the gradient estimator.
            var table = new PrecomputedNoiseTable(length: 1024, seed: 1);

            const int pairs = 8;
            const int paramCount = 4;
            var sumsPerDim = new float[paramCount];

            for (var p = 0; p < pairs; p++)
            {
                var slice = table.GetSlice(p * paramCount, paramCount);
                for (var j = 0; j < paramCount; j++)
                {
                    sumsPerDim[j] += slice[j];
                }
            }

            for (var j = 0; j < paramCount; j++)
            {
                _output.WriteLine($"dim[{j}] sum over first {pairs} 4-slices = {sumsPerDim[j]:F4}");
            }
        }

        private void ReportStats(string label, ReadOnlySpan<float> values)
        {
            var sum = 0.0;
            var sumSq = 0.0;
            var positives = 0;

            for (var i = 0; i < values.Length; i++)
            {
                sum += values[i];
                sumSq += values[i] * values[i];
                if (values[i] > 0f) positives++;
            }

            var mean = sum / values.Length;
            var variance = (sumSq / values.Length) - (mean * mean);

            _output.WriteLine(
                $"[{label}] n={values.Length,4}  mean={mean,8:F4}  var={variance,8:F4}  positives={positives}/{values.Length} ({100.0 * positives / values.Length:F1}%)");
        }
    }
}