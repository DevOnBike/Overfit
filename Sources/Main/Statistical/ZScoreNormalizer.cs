using System;
using System.Buffers;
using System.Numerics.Tensors;

namespace DevOnBike.Overfit.Statistical
{
    /// <summary>
    /// Klasa realizująca normalizację Z-Score (Standaryzację).
    /// Posiada dwa tryby pracy: szybki wsadowy (SIMD) oraz precyzyjny strumieniowy (Welford).
    /// </summary>
    public sealed class ZScoreNormalizer
    {
        private long _count;
        private double _mean;
        private double _m2; // Agregator sumy kwadratów różnic (potrzebny do wariancji)

        public float Mean => (float)_mean;

        // Zależnie od tego, czy traktujemy dane jako populację czy próbkę:
        // Tutaj używamy wariancji populacyjnej (dzielenie przez N). Dla próbki użyj (Count - 1).
        public float StandardDeviation => _count > 1 ? (float)Math.Sqrt(_m2 / _count) : 0f;

        public long Count => _count;

        /// <summary>
        /// Opcja 1: Turbowydajne przetwarzanie kolekcji (Batch) z użyciem SIMD (TensorPrimitives).
        /// </summary>
        /// <param name="data">Ciągła pamięć z wartościami zmiennoprzecinkowymi.</param>
        public void FitBatch(ReadOnlySpan<float> data)
        {
            if (data.Length == 0)
            {
                return;
            }

            // 1. Sprzętowo akcelerowana suma
            var sum = TensorPrimitives.Sum(data);
            var mean = sum / data.Length;

            // 2. Sprzętowe liczenie sumy kwadratów odchyleń (Variance)
            var buffer = ArrayPool<float>.Shared.Rent(data.Length);
            var diffs = buffer.AsSpan(0, data.Length);

            try
            {
                // diffs = data - mean
                TensorPrimitives.Subtract(data, mean, diffs);

                // sumOfSquares = diffs * diffs (Iloczyn skalarny z samym sobą)
                var sumSq = TensorPrimitives.Dot(diffs, diffs);

                // Zapisujemy parametry dla całej paczki
                _count = data.Length;
                _mean = mean;
                _m2 = sumSq;
            }
            finally
            {
                ArrayPool<float>.Shared.Return(buffer);
            }
        }

        /// <summary>
        /// Opcja 2: Algorytm sekwencyjny (Welford's Algorithm). 
        /// Niezawodny dla ogromnych strumieni danych o nieznanej z góry wielkości.
        /// </summary>
        /// <param name="value">Pojedyncza próbka wpadająca z generatora/strumienia.</param>
        public void FitIncremental(float value)
        {
            _count++;

            // Logika Welforda - matematycznie zapobiega utracie precyzji w liczbach floating-point
            var delta = value - _mean;
            _mean += delta / _count;
            var delta2 = value - _mean;

            _m2 += delta * delta2;
        }

        /// <summary>
        /// Aplikuje wyuczone parametry (Z-Score) bezpośrednio na podanych danych.
        /// Nadpisuje przekazaną kolekcję w celu oszczędzania pamięci.
        /// </summary>
        public void TransformInPlace(Span<float> data)
        {
            if (data.Length == 0)
            {
                return;
            }

            var stdDev = StandardDeviation;

            // Zabezpieczenie przed dzieleniem przez zero dla stałych danych
            if (stdDev < 1e-8f)
            {
                stdDev = 1e-8f;
            }

            // Sprzętowo akcelerowana aplikacja Z-Score: z = (x - mean) / stdDev
            TensorPrimitives.Subtract(data, Mean, data);

            // Mnożenie jest szybsze dla procesora niż dzielenie
            var invStdDev = 1.0f / stdDev;
            
            TensorPrimitives.Multiply(data, invStdDev, data);
        }
        
        public void Reset()
        {
            _count = 0;
            _mean = 0;
            _m2 = 0;
        }
    }
}