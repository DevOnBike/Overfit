// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.

using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Statistical
{
    public sealed class ShapKernel : IDisposable
    {
        private readonly Func<ReadOnlySpan<float>, float> _modelFunc;
        private readonly float[] _background;
        private readonly int _m;
        private readonly int _numSamples;

        public ShapKernel(Func<ReadOnlySpan<float>, float> modelFunc, float[] background, int numSamples = 2048)
        {
            _modelFunc = modelFunc;
            _background = background;
            _m = background.Length;
            _numSamples = numSamples;
        }

        public void Explain(ReadOnlySpan<float> x, Span<float> phi)
        {
            phi.Clear();
            
            using var zWithBuf = new PooledBuffer<float>(_m, clearMemory: false);
            using var zWithoutBuf = new PooledBuffer<float>(_m, clearMemory: false);

            var zWith = zWithBuf.Span;
            var zWithout = zWithoutBuf.Span;

            var fX = _modelFunc(x);
            var fNull = _modelFunc(_background);
            var samplesPerFeature = Math.Max(1, _numSamples / _m);

            for (var i = 0; i < _m; i++)
            {
                float marginalSum = 0;
                
                for (var s = 0; s < samplesPerFeature; s++)
                {
                    PrepareCoalition(zWithout, i, x);
                    zWithout.CopyTo(zWith);
                    zWith[i] = x[i];

                    marginalSum += (_modelFunc(zWith) - _modelFunc(zWithout));
                }
                
                phi[i] = marginalSum / samplesPerFeature;
            }

            // Korekta efektywności (Aksjomat SHAP)
            float sum = 0;
            
            for (var i = 0; i < _m; i++)
            {
                sum += phi[i];
            }
            
            var diff = (fX - fNull - sum) / _m;
            
            for (var i = 0; i < _m; i++)
            {
                phi[i] += diff;
            }
        }

        private void PrepareCoalition(Span<float> dest, int excludedIdx, ReadOnlySpan<float> x)
        {
            _background.AsSpan().CopyTo(dest);
            
            var rnd = Random.Shared;
            
            for (var i = 0; i < _m; i++)
            {
                if (i != excludedIdx && rnd.NextDouble() > 0.5)
                {
                    dest[i] = x[i];
                }
            }
        }

        public void Dispose()
        {
            
        }
    }
}