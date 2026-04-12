using System.Buffers;
using System.Numerics.Tensors;

namespace DevOnBike.Overfit.Statistical
{
    public sealed class ZScoreNormalizer
    {
        private long _count;
        private double _mean;
        private double _m2; // Agregator sumy kwadratów różnic

        public float Mean => (float)_mean;

        // Wariancja populacyjna. Dla próbkowej zmień na: _count > 1 ? _m2 / (_count - 1) : 0
        public float StandardDeviation => _count > 0 ? (float)Math.Sqrt(_m2 / _count) : 0f;

        public long Count => _count;

        /// <summary>
        /// Turbowydajne przetwarzanie wsadowe (SIMD).
        /// Używa algorytmu Chana do bezpiecznego scalania batchy ze stanem globalnym.
        /// </summary>
        public void FitBatch(ReadOnlySpan<float> data)
        {
            if (data.Length == 0)
            {
                return;
            }

            long n2 = data.Length;

            // 1. Liczymy lokalne parametry batcha (używamy SIMD dla float, ale promujemy do double)
            var localMeanFloat = TensorPrimitives.Sum(data) / n2;
            double localMean = localMeanFloat;
            double localM2 = 0;

            var buffer = ArrayPool<float>.Shared.Rent(data.Length);
            var diffs = buffer.AsSpan(0, data.Length);

            try
            {
                // diffs = data - localMeanFloat
                TensorPrimitives.Subtract(data, localMeanFloat, diffs);
                // Iloczyn skalarny (suma kwadratów różnic)
                localM2 = TensorPrimitives.Dot(diffs, diffs);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(buffer);
            }

            // 2. Scalanie stanów (Chan's Update Algorithm - Welford dla paczek)
            if (_count == 0)
            {
                // Pierwszy batch - po prostu inicjalizujemy stan
                _count = n2;
                _mean = localMean;
                _m2 = localM2;
            }
            else
            {
                // Kolejne batche - matematyczne scalanie
                var newCount = _count + n2;
                var delta = localMean - _mean;

                // Nowa średnia
                _mean += delta * n2 / newCount;

                // Nowe m2 (suma kwadratów odchyleń)
                _m2 += localM2 + (delta * delta * _count * n2) / newCount;

                _count = newCount;
            }
        }

        /// <summary>
        /// Algorytm sekwencyjny (Welford's Algorithm). 
        /// </summary>
        public void FitIncremental(float value)
        {
            _count++;

            var delta = value - _mean;

            _mean += delta / _count;

            var delta2 = value - _mean;

            _m2 += delta * delta2;
        }

        public void TransformInPlace(Span<float> data)
        {
            if (data.Length == 0)
            {
                return;
            }

            var stdDev = StandardDeviation;

            if (stdDev < 1e-8f)
            {
                stdDev = 1e-8f;
            }

            TensorPrimitives.Subtract(data, Mean, data);

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