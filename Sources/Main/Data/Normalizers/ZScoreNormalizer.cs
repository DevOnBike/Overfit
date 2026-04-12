using System.Buffers;
using System.Numerics.Tensors;

namespace DevOnBike.Overfit.Data.Normalizers
{
    public sealed class ZScoreNormalizer
    {
        private long _count;
        private double _mean;
        private double _m2;

        public float Mean => (float)_mean;

        // Wariancja populacyjna. Dla próbkowej zmień na: _count > 1 ? _m2 / (_count - 1) : 0
        public float StandardDeviation => _count > 0 ? (float)Math.Sqrt(_m2 / _count) : 0f;

        public long Count => _count;

        /// <summary>
        /// Turbowydajne przetwarzanie wsadowe z algorytmem Chana.
        /// Mean liczona w double aby uniknąć utraty precyzji dla dużych wartości
        /// (np. MemoryWorkingSetBytes ~10⁹).
        /// M2 liczone przez SIMD SumOfSquares na float diffs — wystarczająca precyzja
        /// gdy dane są wstępnie znormalizowane. Dla surowych metryk użyj FitIncremental.
        /// </summary>
        public void FitBatch(ReadOnlySpan<float> data)
        {
            if (data.Length == 0)
            {
                return;
            }

            long n2 = data.Length;

            // Mean w double — chroni przed utratą precyzji dla dużych wartości
            double localMean = 0.0;
            for (var i = 0; i < data.Length; i++) localMean += data[i];
            localMean /= n2;

            var localMeanFloat = (float)localMean;

            // M2 przez SIMD — subtract + SumOfSquares na float diffs
            var buffer = ArrayPool<float>.Shared.Rent(data.Length);
            double localM2;

            try
            {
                var diffs = buffer.AsSpan(0, data.Length);
                TensorPrimitives.Subtract(data, localMeanFloat, diffs);
                localM2 = TensorPrimitives.SumOfSquares(diffs);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(buffer);
            }

            // Chan's merge
            if (_count == 0)
            {
                _count = n2;
                _mean = localMean;
                _m2 = localM2;
            }
            else
            {
                var newCount = _count + n2;
                var delta = localMean - _mean;

                _mean += delta * n2 / newCount;

                // (double) cast chroni przed overflow _count * n2 przy bardzo dużych zbiorach
                _m2 += localM2 + delta * delta * _count * n2 / newCount;

                _count = newCount;
            }
        }

        /// <summary>
        /// Algorytm sekwencyjny Welforda — precyzyjny dla dowolnych wartości,
        /// w tym surowych metryk o dużym zakresie.
        /// </summary>
        public void FitIncremental(float value)
        {
            _count++;

            var delta = value - _mean;
            _mean += delta / _count;
            var delta2 = value - _mean;
            _m2 += delta * delta2;
        }

        /// <summary>
        /// Aplikuje Z-Score in-place: z = (x - mean) / stdDev.
        /// </summary>
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
            TensorPrimitives.Multiply(data, 1.0f / stdDev, data);
        }

        public void Reset()
        {
            _count = 0;
            _mean = 0;
            _m2 = 0;
        }
    }
}