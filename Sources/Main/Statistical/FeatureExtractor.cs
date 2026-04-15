// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Statistical
{
    /// <summary>
    /// Wyodrębnia statystyki (Mean, Std, P95, Delta) z okna czasowego dla każdej cechy.
    /// Działa w trybie Zero-Allocation dla okien poniżej 256 próbek.
    /// </summary>
    public static class FeatureExtractor
    {
        public const int StatsPerFeature = 4;

        public const int MeanOffset = 0;
        public const int StdOffset = 1;
        public const int P95Offset = 2;
        public const int DeltaOffset = 3;

        private const int StackAllocThreshold = 256;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int OutputSize(int featureCount) => featureCount * StatsPerFeature;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool TryExtract(
            SlidingWindowBuffer buffer,
            Span<float> windowScratch,
            Span<float> output,
            out DateTime windowEnd)
        {
            if (!buffer.TryGetWindow(windowScratch, out windowEnd))
            {
                return false;
            }

            Extract(windowScratch, buffer.WindowSize, buffer.FeatureCount, output);
            return true;
        }

        public static void Extract(
            ReadOnlySpan<float> window,
            int windowSize,
            int featureCount,
            Span<float> output)
        {
            var expectedInput = windowSize * featureCount;
            var expectedOutput = featureCount * StatsPerFeature;

            if (window.Length != expectedInput)
            {
                throw new ArgumentException($"window.Length={window.Length}, oczekiwano {expectedInput}.", nameof(window));
            }

            if (output.Length < expectedOutput)
            {
                throw new ArgumentException($"output.Length={output.Length}, potrzeba min. {expectedOutput}.", nameof(output));
            }

            if (windowSize <= StackAllocThreshold)
            {
                Span<float> columnScratch = stackalloc float[windowSize];
                ExtractCore(window, windowSize, featureCount, output, columnScratch);
            }
            else
            {
                // ZMIANA: Zastosowanie PooledBuffer!
                using var rented = new PooledBuffer<float>(windowSize);
                ExtractCore(window, windowSize, featureCount, output, rented.Span);
            }
        }

        private static void ExtractCore(
            ReadOnlySpan<float> window,
            int windowSize,
            int featureCount,
            Span<float> output,
            Span<float> col)
        {
            for (var f = 0; f < featureCount; f++)
            {
                for (var t = 0; t < windowSize; t++)
                {
                    col[t] = window[t * featureCount + f];
                }

                var firstVal = col[0];
                var lastVal = col[windowSize - 1];

                var sum = TensorPrimitives.Sum(col);
                var mean = sum / windowSize;

                var sumSq = TensorPrimitives.Dot(col, col);
                var variance = sumSq / windowSize - mean * mean;
                var std = MathF.Sqrt(MathF.Max(0f, variance));

                col.Sort();
                var p95 = Percentile95(col, windowSize);

                var delta = lastVal - firstVal;

                var outBase = f * StatsPerFeature;
                output[outBase + MeanOffset] = mean;
                output[outBase + StdOffset] = std;
                output[outBase + P95Offset] = p95;
                output[outBase + DeltaOffset] = delta;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Percentile95(ReadOnlySpan<float> sorted, int length)
        {
            if (length == 1)
            {
                return sorted[0];
            }
            var index = (int)MathF.Ceiling(0.95f * length) - 1;
            return sorted[Math.Clamp(index, 0, length - 1)];
        }
    }
}