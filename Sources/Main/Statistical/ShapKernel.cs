// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Core;

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

        public void Explain(ReadOnlySpan<float> instance, Span<float> shapValues)
        {
            var baseValue = _modelFunc(_background);
            var instanceValue = _modelFunc(instance);
            var totalDiff = instanceValue - baseValue;

            using var A = new FastMatrix<float>(_m, _m);
            using var b = new FastBuffer<float>(_m);
            using var mask = new FastBuffer<float>(_m);
            using var synthetic = new FastBuffer<float>(_m);

            // GWARANTOWANE PRÓBKOWANIE (1-cecha i M-1 cech)
            // To eliminuje błąd 10^-8, bo model dostaje "czyste" sygnały o pojedynczych cechach.
            for (var i = 0; i < _m; i++)
            {
                // Koalicja: tylko jedna cecha aktywna
                mask.Clear();
                
                mask[i] = 1f;
                
                ProcessSample(1, instance, mask.AsSpan(), synthetic.AsSpan(), baseValue, A, b.AsSpan());

                // Koalicja: wszystkie cechy poza jedną
                for (var j = 0; j < _m; j++)
                {
                    mask[j] = 1f;
                }
                
                mask[i] = 0f;
                
                ProcessSample(_m - 1, instance, mask.AsSpan(), synthetic.AsSpan(), baseValue, A, b.AsSpan());
            }

            // Pozostałe losowe próbki
            var remaining = _numSamples - 2 * _m;
            
            for (var s = 0; s < remaining; s++)
            {
                var k = GenerateRandomCoalition(mask.AsSpan());
                
                ProcessSample(k, instance, mask.AsSpan(), synthetic.AsSpan(), baseValue, A, b.AsSpan());
            }

            SolveWithEfficiencyConstraint(A, b.AsSpan(), totalDiff, shapValues);
        }

        private void ProcessSample(int k, ReadOnlySpan<float> instance, Span<float> mask, Span<float> syn, float baseVal, FastMatrix<float> A, Span<float> b)
        {
            var weight = CalculateShapWeight(k, _m);

            for (var i = 0; i < _m; i++)
            {
                syn[i] = mask[i] > 0.5f ? instance[i] : _background[i];
            }

            var y = _modelFunc(syn) - baseVal;

            // Update Z'WZ i Z'Wy
            TensorPrimitives.MultiplyAdd(mask, weight * y, b, b);
            
            for (var i = 0; i < _m; i++)
            {
                if (mask[i] == 0)
                {
                    continue;
                }
                
                var rowA = A.Row(i);
                
                TensorPrimitives.MultiplyAdd(mask, weight * mask[i], rowA, rowA);
            }
        }

        private void SolveWithEfficiencyConstraint(FastMatrix<float> A, Span<float> b, float totalDiff, Span<float> phi)
        {
            // Regularyzacja przekątnej dla stabilności
            for (var i = 0; i < _m; i++)
            {
                A[i, i] += 1e-4f;
            }

            using var L = CholeskyMultivariateGaussianLogic.DecomposeCholesky(A);

            using var zBuf = new FastBuffer<float>(_m);
            var z = zBuf.AsSpan();
            
            for (var i = 0; i < _m; i++)
            {
                var dot = TensorPrimitives.Dot(L.ReadOnlyRow(i)[..i], z[..i]);
                z[i] = (b[i] - dot) / L[i, i];
            }

            for (var i = _m - 1; i >= 0; i--)
            {
                float sum = 0;
                
                for (var j = i + 1; j < _m; j++)
                {
                    sum += L[j, i] * phi[j];
                }
                
                phi[i] = (z[i] - sum) / L[i, i];
            }

            // WYMUSZENIE SUMY (Efficiency Axiom): Korygujemy błędy numeryczne regresji
            float currentSum = 0;
            
            for (var i = 0; i < _m; i++)
            {
                currentSum += phi[i];
            }
            
            var correction = (totalDiff - currentSum) / _m;
            
            for (var i = 0; i < _m; i++)
            {
                phi[i] += correction;
            }
        }

        private float CalculateShapWeight(int k, int M)
        {
            var comb = Binomial(M, k);
            var w = (double)(M - 1) / (comb * k * (M - k));
            return (float)Math.Max(w, 1e-12); // Clamping zapobiega underflow do zera
        }

        private int GenerateRandomCoalition(Span<float> mask)
        {
            var k = 0;
            
            for (var i = 0; i < _m; i++)
            {
                var bit = Random.Shared.NextDouble() > 0.5;
                
                mask[i] = bit ? 1f : 0f;
                
                if (bit)
                {
                    k++;
                }
            }
            
            return Math.Clamp(k, 1, _m - 1);
        }

        private static double Binomial(int n, int k)
        {
            if (k < 0 || k > n)
            {
                return 0;
            }
            
            if (k == 0 || k == n)
            {
                return 1;
            }
            
            if (k > n / 2)
            {
                k = n - k;
            }
            
            double res = 1;
            
            for (var i = 1; i <= k; i++)
            {
                res = res * (n - i + 1) / i;
            }
            
            return res;
        }

        public void Dispose() { }
    }
}