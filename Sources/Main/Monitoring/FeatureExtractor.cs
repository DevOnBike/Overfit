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
    /// Wyodrębnia statystyki per-cecha z okna ślizgowego.
    ///
    /// Wejście:  window[windowSize × featureCount] — row-major, wiersz = krok czasowy
    /// Wyjście:  stats[featureCount × StatsPerFeature] = [mean, std, p95, delta] per cecha
    ///
    /// Układ wyjścia:
    ///   [mean_f0, std_f0, p95_f0, delta_f0, mean_f1, std_f1, p95_f1, delta_f1, ...]
    ///
    /// Ścieżka zero-alokacyjna gdy windowSize ≤ StackAllocThreshold (256).
    /// Dla większych okien: jednorazowe wypożyczenie z ArrayPool.
    ///
    /// Wzorzec użycia w pipeline:
    /// <code>
    ///   // Prealokuj raz poza pętlą:
    ///   var windowScratch = new float[buffer.WindowFloats];
    ///   var statsScratch  = new float[FeatureExtractor.OutputSize(buffer.FeatureCount)];
    ///
    ///   // W pętli scrapingu:
    ///   if (FeatureExtractor.TryExtract(buffer, windowScratch, statsScratch))
    ///       robustScaler.Transform(statsScratch, normalizedScratch);
    /// </code>
    /// </summary>
    public static class FeatureExtractor
    {
        /// <summary>Liczba statystyk obliczanych dla każdej cechy.</summary>
        public const int StatsPerFeature = 4;

        // Indeksy w bloku statystyk jednej cechy
        public const int MeanOffset = 0;
        public const int StdOffset = 1;
        public const int P95Offset = 2;
        public const int DeltaOffset = 3;

        // Przy windowSize > 256 przechodzimy z stackalloc na ArrayPool.
        // 256 float = 1 KB — bezpieczny limit stosu.
        private const int StackAllocThreshold = 256;

        /// <summary>Rozmiar bufora wyjściowego dla podanej liczby cech.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int OutputSize(int featureCount) => featureCount * StatsPerFeature;

        // -------------------------------------------------------------------------
        // Extract — główna metoda
        // -------------------------------------------------------------------------

        /// <summary>
        /// Oblicza statystyki okna i zapisuje je do <paramref name="output"/>.
        /// </summary>
        /// <param name="window">
        ///   Flat bufor [windowSize × featureCount] z <see cref="SlidingWindowBuffer.TryGetWindow"/>.
        /// </param>
        /// <param name="windowSize">Liczba próbek czasowych w oknie.</param>
        /// <param name="featureCount">Liczba cech per próbka.</param>
        /// <param name="output">
        ///   Caller-owned bufor min. <see cref="OutputSize"/>(<paramref name="featureCount"/>) elementów.
        /// </param>
        /// <exception cref="ArgumentException">Zły rozmiar window lub output.</exception>
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
                // Zero-alokacyjna ścieżka — typowy przypadek (windowSize ≤ 256)
                Span<float> col = stackalloc float[windowSize];

                ExtractCore(window, windowSize, featureCount, output, col);
            }
            else
            {
                // Duże okno — jedno wypożyczenie z ArrayPool
                using var rented = new FastBuffer<float>(windowSize);

                ExtractCore(window, windowSize, featureCount, output, rented.AsSpan());
            }
        }

        // -------------------------------------------------------------------------
        // TryExtract — integracja z SlidingWindowBuffer
        // -------------------------------------------------------------------------

        /// <summary>
        /// Pobiera okno z bufora i od razu oblicza statystyki.
        /// Zwraca false gdy bufor nie jest jeszcze gotowy — output pozostaje niezmieniony.
        /// </summary>
        /// <param name="buffer">Bufor ślizgowy.</param>
        /// <param name="windowScratch">
        ///   Reużywalny scratch o rozmiarze <see cref="SlidingWindowBuffer.WindowFloats"/>.
        ///   Prealokuj raz poza pętlą.
        /// </param>
        /// <param name="output">
        ///   Reużywalny bufor wyjściowy o rozmiarze <see cref="OutputSize"/>(<c>buffer.FeatureCount</c>).
        ///   Prealokuj raz poza pętlą.
        /// </param>
        /// <param name="windowEnd">Timestamp ostatniej próbki w oknie (out).</param>
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
        // Rdzeń obliczeń
        // -------------------------------------------------------------------------

        private static void ExtractCore(
            ReadOnlySpan<float> window,
            int windowSize,
            int featureCount,
            Span<float> output,
            Span<float> col) // scratch bufor na jedną kolumnę
        {
            for (var f = 0; f < featureCount; f++)
            {
                // Wyodrębnij kolumnę f do ciągłego bufora
                // (dane są non-contiguous: stride = featureCount)
                for (var t = 0; t < windowSize; t++)
                {
                    col[t] = window[t * featureCount + f];
                }

                var first = col[0];
                var last = col[windowSize - 1];

                // --- mean (SIMD Sum) ---
                var mean = Sum(col, windowSize) / windowSize;

                // --- std (E[X²] - E[X]², numerycznie stabilne dla typowych wartości metryk) ---
                var sumSq = Dot(col, windowSize);
                var variance = sumSq / windowSize - mean * mean;
                var std = MathF.Sqrt(MathF.Max(0f, variance)); // guard float rounding

                // --- p95 (wymaga sortowania kopii) ---
                // col będzie posortowane in-place — ok, bo first/last już zapisane
                SortSpan(col.Slice(0, windowSize));
                var p95 = Percentile95(col, windowSize);

                // --- delta (ostatnia - pierwsza próbka — kierunek trendu) ---
                var delta = last - first;

                // Zapisz blok statystyk dla cechy f
                var outBase = f * StatsPerFeature;
                output[outBase + MeanOffset] = mean;
                output[outBase + StdOffset] = std;
                output[outBase + P95Offset] = p95;
                output[outBase + DeltaOffset] = delta;
            }
        }

        // -------------------------------------------------------------------------
        // Prywatne — operacje na Span<float>
        // -------------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Sum(Span<float> span, int length)
        {
            return TensorPrimitives.Sum(span.Slice(0, length));
        }

        /// <summary>Dot product Σ(xi²) — suma kwadratów.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Dot(Span<float> span, int length)
        {
            var slice = span.Slice(0, length);
            
            return TensorPrimitives.Dot(slice, slice);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void SortSpan(Span<float> span)
        {
            // MemoryExtensions.Sort — dostępne od .NET 6, bez alokacji
            span.Sort();
        }

        /// <summary>
        /// Percentyl 95 metodą nearest-rank z posortowanego Span.
        /// Dla windowSize=6: ceil(0.95 × 6) - 1 = ceil(5.7) - 1 = 5 → ostatni element.
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