// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Statistical
{
    /// <summary>
    /// Wyodrębnia statystyki (Mean, Std, P95, Delta) z okna czasowego dla każdej cechy.
    /// Działa w trybie Zero-Allocation dla okien poniżej 256 próbek.
    /// </summary>
    public static class FeatureExtractor
    {
        /// <summary>Liczba statystyk obliczanych dla każdej cechy z MetricSnapshot.</summary>
        public const int StatsPerFeature = 4;

        // Offsety w wygenerowanym wektorze wyjściowym
        public const int MeanOffset = 0;
        public const int StdOffset = 1;
        public const int P95Offset = 2;
        public const int DeltaOffset = 3;

        // Jeśli okno ma <= 256 próbek (np. okno 60 sekund), używamy stosu.
        // 256 floatów to równo 1 KB pamięci — absolutnie bezpieczne dla C#.
        private const int StackAllocThreshold = 256;

        /// <summary>
        /// Zwraca rozmiar wektora, który wpadnie do modelu AI/HMM.
        /// (Liczba cech * 4 statystyki).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int OutputSize(int featureCount) => featureCount * StatsPerFeature;

        /// <summary>
        /// Próbuje pobrać okno z bufora i od razu dokonać ekstrakcji.
        /// Zwraca false, jeśli bufor nie ma jeszcze wystarczającej ilości danych (Cold Start).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool TryExtract(
            SlidingWindowBuffer buffer,
            Span<float> windowScratch,
            Span<float> output,
            out DateTime windowEnd)
        {
            // Próba pobrania płaskiego okna [windowSize * featureCount]
            if (!buffer.TryGetWindow(windowScratch, out windowEnd))
            {
                return false;
            }

            Extract(windowScratch, buffer.WindowSize, buffer.FeatureCount, output);
            return true;
        }

        /// <summary>
        /// Główny silnik ekstrakcji.
        /// </summary>
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
                // Ścieżka optymalna: Brak alokacji (L1 Cache)
                Span<float> columnScratch = stackalloc float[windowSize];
                ExtractCore(window, windowSize, featureCount, output, columnScratch);
            }
            else
            {
                // Ścieżka awaryjna: Bardzo duże okno, wypożyczamy z ArrayPool
                using var rented = new FastBuffer<float>(windowSize);
                ExtractCore(window, windowSize, featureCount, output, rented.AsSpan());
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
                // Krok 1: Kopiowanie pojedynczej cechy (striding) do ciągłego bloku pamięci (col)
                for (var t = 0; t < windowSize; t++)
                {
                    col[t] = window[t * featureCount + f];
                }

                var firstVal = col[0];
                var lastVal = col[windowSize - 1];

                // Krok 2: Średnia (Mean) za pomocą SIMD
                var sum = TensorPrimitives.Sum(col);
                var mean = sum / windowSize;

                // Krok 3: Odchylenie standardowe (Std) 
                // Używamy wzoru: E[X^2] - (E[X])^2
                var sumSq = TensorPrimitives.Dot(col, col);
                var variance = sumSq / windowSize - mean * mean;
                var std = MathF.Sqrt(MathF.Max(0f, variance)); // Max(0) chroni przed błędami precyzji float

                // Krok 4: Percentyl 95 (P95)
                // Sortowanie Span w miejscu nie alokuje pamięci w .NET 6+
                col.Sort();
                var p95 = Percentile95(col, windowSize);

                // Krok 5: Kierunek trendu (Delta)
                var delta = lastVal - firstVal;

                // Zapis do wynikowego wektora
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
            if (length == 1) return sorted[0];

            // Metoda Nearest-Rank
            var index = (int)MathF.Ceiling(0.95f * length) - 1;
            return sorted[Math.Clamp(index, 0, length - 1)];
        }
    }
}