// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Monitoring
{
    /// <summary>
    /// Extracts per-feature statistics from a sliding window.
    ///
    /// Input:  window[windowSize × featureCount] — row-major, row = time step
    /// Output: stats[featureCount × StatsPerFeature] = [mean, std, p95, delta] per feature
    ///
    /// Output layout:
    ///   [mean_f0, std_f0, p95_f0, delta_f0, mean_f1, std_f1, p95_f1, delta_f1, ...]
    ///
    /// Zero-allocation path when windowSize <= StackAllocThreshold (256).
    /// For larger windows: a one-time lease from ArrayPool.
    ///
    /// Usage pattern in the pipeline:
    /// <code>
    ///   // Preallocate once outside the loop:
    ///   var windowScratch = new float[buffer.WindowFloats];
    ///   var statsScratch  = new float[FeatureExtractor.OutputSize(buffer.FeatureCount)];
    ///
    ///   // Inside the scraping loop:
    ///   if (FeatureExtractor.TryExtract(buffer, windowScratch, statsScratch))
    ///       robustScaler.Transform(statsScratch, normalizedScratch);
    /// </code>
    /// </summary>
    public static class FeatureExtractor
    {
        /// <summary>Number of statistics calculated for each feature.</summary>
        public const int StatsPerFeature = 4;

        // Indices within the statistics block of a single feature
        public const int MeanOffset = 0;
        public const int StdOffset = 1;
        public const int P95Offset = 2;
        public const int DeltaOffset = 3;

        // When windowSize > 256, we switch from stackalloc to ArrayPool.
        // 256 floats = 1 KB — safe stack limit.
        private const int StackAllocThreshold = 256;

        /// <summary>Output buffer size for the given number of features.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int OutputSize(int featureCount) => featureCount * StatsPerFeature;

        // -------------------------------------------------------------------------
        // Extract — main method
        // -------------------------------------------------------------------------

        /// <summary>
        /// Calculates window statistics and writes them to <paramref name="output"/>.
        /// </summary>
        /// <param name="window">
        ///   Flat buffer [windowSize × featureCount] from <see cref="SlidingWindowBuffer.TryGetWindow"/>.
        /// </param>
        /// <param name="windowSize">Number of time samples in the window.</param>
        /// <param name="featureCount">Number of features per sample.</param>
        /// <param name="output">
        ///   Caller-owned buffer with min. <see cref="OutputSize"/>(<paramref name="featureCount"/>) elements.
        /// </param>
        /// <exception cref="ArgumentException">Invalid window or output size.</exception>
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
                throw new ArgumentException($"window.Length={window.Length}, oczekiwano {expectedInput} ({windowSize}×{featureCount}).", nameof(window));
            }

            if (output.Length < expectedOutput)
            {
                throw new ArgumentException($"output.Length={output.Length}, potrzeba min. {expectedOutput}.", nameof(output));
            }

            if (windowSize <= StackAllocThreshold)
            {
                Span<float> col = stackalloc float[windowSize];

                ExtractCore(window, windowSize, featureCount, output, col);
            }
            else
            {
                using var rented = new FastBuffer<float>(windowSize);

                ExtractCore(window, windowSize, featureCount, output, rented.AsSpan());
            }
        }

        // -------------------------------------------------------------------------
        // TryExtract — integration with SlidingWindowBuffer
        // -------------------------------------------------------------------------

        /// <summary>
        /// Retrieves a window from the buffer and immediately calculates statistics.
        /// Returns false when the buffer is not ready yet — output remains unchanged.
        /// </summary>
        /// <param name="buffer">Sliding buffer.</param>
        /// <param name="windowScratch">
        ///   Reusable scratch buffer of size <see cref="SlidingWindowBuffer.WindowFloats"/>.
        ///   Preallocate once outside the loop.
        /// </param>
        /// <param name="output">
        ///   Reusable output buffer of size <see cref="OutputSize"/>(<c>buffer.FeatureCount</c>).
        ///   Preallocate once outside the loop.
        /// </param>
        /// <param name="windowEnd">Timestamp of the latest sample in the window (out).</param>
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

        // -------------------------------------------------------------------------
        // Computation core
        // -------------------------------------------------------------------------

        private static void ExtractCore(
            ReadOnlySpan<float> window,
            int windowSize,
            int featureCount,
            Span<float> output,
            Span<float> col) // scratch buffer for a single column
        {
            for (var f = 0; f < featureCount; f++)
            {
                // Extract column f to a contiguous buffer
                // (data is non-contiguous: stride = featureCount)
                for (var t = 0; t < windowSize; t++)
                {
                    col[t] = window[t * featureCount + f];
                }

                var first = col[0];
                var last = col[windowSize - 1];

                // --- mean (SIMD Sum) ---
                var mean = Sum(col, windowSize) / windowSize;

                // --- std (E[X²] - E[X]², numerically stable for typical metric values) ---
                var sumSq = Dot(col, windowSize);
                var variance = sumSq / windowSize - mean * mean;
                var std = MathF.Sqrt(MathF.Max(0f, variance)); // guard float rounding

                // --- p95 (requires sorting the copy) ---
                // col will be sorted in-place — fine, because first/last are already saved
                SortSpan(col.Slice(0, windowSize));
                var p95 = Percentile95(col, windowSize);

                // --- delta (last - first sample — trend direction) ---
                var delta = last - first;

                // Write statistics block for feature f
                var outBase = f * StatsPerFeature;
                output[outBase + MeanOffset] = mean;
                output[outBase + StdOffset] = std;
                output[outBase + P95Offset] = p95;
                output[outBase + DeltaOffset] = delta;
            }
        }

        // -------------------------------------------------------------------------
        // Private — operations on Span<float>
        // -------------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Sum(Span<float> span, int length)
        {
            return TensorPrimitives.Sum(span.Slice(0, length));
        }

        /// <summary>Dot product Σ(xi²) — sum of squares.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Dot(Span<float> span, int length)
        {
            var slice = span.Slice(0, length);

            return TensorPrimitives.Dot(slice, slice);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void SortSpan(Span<float> span)
        {
            // MemoryExtensions.Sort — available since .NET 6, zero allocation
            span.Sort();
        }

        /// <summary>
        /// 95th percentile using nearest-rank method from a sorted Span.
        /// For windowSize=6: ceil(0.95 × 6) - 1 = ceil(5.7) - 1 = 5 -> last element.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Percentile95(Span<float> sorted, int length)
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