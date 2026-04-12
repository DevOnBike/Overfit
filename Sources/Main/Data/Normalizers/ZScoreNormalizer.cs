using System.Buffers;
using System.Numerics.Tensors;
using DevOnBike.Overfit.Data.Abstractions;

namespace DevOnBike.Overfit.Data.Normalizers
{
    public sealed class ZScoreNormalizer : IFeatureNormalizer
    {
        private long _count;
        private double _mean;
        private double _m2;

        private bool _isFrozen;
        private float _frozenMean;
        private float _frozenInvStdDev;

        public float Mean => _isFrozen ? _frozenMean : (float)_mean;

        // Wariancja populacyjna. Dla próbkowej zmień na: _count > 1 ? _m2 / (_count - 1) : 0
        public float StandardDeviation => _count > 0 ? (float)Math.Sqrt(_m2 / _count) : 0f;

        public long Count => _count;

        public bool IsFrozen => _isFrozen;

        public void FitBatch(ReadOnlySpan<float> data)
        {
            if (_isFrozen)
            {
                throw new InvalidOperationException("Normalizer is frozen.");
            }

            if (data.Length == 0)
            {
                return;
            }

            long n2 = data.Length;

            var localMean = 0.0;

            for (var i = 0; i < data.Length; i++)
            {
                localMean += data[i];
            }

            localMean /= n2;

            var localMeanFloat = (float)localMean;

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
                _m2 += localM2 + delta * delta * (double)_count * n2 / newCount;
                _count = newCount;
            }
        }

        public void FitIncremental(float value)
        {
            if (_isFrozen)
            {
                throw new InvalidOperationException("Normalizer is frozen.");
            }

            _count++;
            var delta = value - _mean;
            _mean += delta / _count;
            var delta2 = value - _mean;
            _m2 += delta * delta2;
        }

        public void Freeze()
        {
            if (_isFrozen)
            {
                return;
            }

            if (_count == 0)
            {
                throw new InvalidOperationException("Cannot freeze without data.");
            }

            var stdDev = StandardDeviation;

            if (stdDev < 1e-8f)
            {
                stdDev = 1e-8f;
            }

            _frozenMean = (float)_mean;
            _frozenInvStdDev = 1.0f / stdDev;
            _isFrozen = true;
        }

        public void TransformInPlace(Span<float> data)
        {
            if (!_isFrozen)
            {
                throw new InvalidOperationException("Normalizer is not frozen.");
            }

            if (data.Length == 0)
            {
                return;
            }

            TensorPrimitives.Subtract(data, _frozenMean, data);
            TensorPrimitives.Multiply(data, _frozenInvStdDev, data);
        }

        public void Reset()
        {
            _count = 0;
            _mean = 0;
            _m2 = 0;
            _isFrozen = false;
            _frozenMean = 0f;
            _frozenInvStdDev = 0f;
        }
    }
}