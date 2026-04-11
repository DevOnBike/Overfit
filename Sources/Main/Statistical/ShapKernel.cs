// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Concurrent;
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
        private readonly float[] _precomputedWeights; // Cache dla wag jądra

        public ShapKernel(Func<ReadOnlySpan<float>, float> modelFunc, float[] background, int numSamples = 2048)
        {
            _modelFunc = modelFunc;
            _background = background;
            _m = background.Length;
            _numSamples = numSamples;

            // OPTYMALIZACJA 1: Prekalkulacja wag
            _precomputedWeights = new float[_m];
            for (var k = 1; k < _m; k++)
            {
                var comb = Binomial(_m, k);
                _precomputedWeights[k] = (float)Math.Max((_m - 1) / (comb * k * (_m - k)), 1e-12);
            }
        }

        public void Explain(ReadOnlySpan<float> instance, Span<float> shapValues)
        {
            var baseValue = _modelFunc(_background);
            var instanceValue = _modelFunc(instance);
            var totalDiff = instanceValue - baseValue;

            using var A = new FastMatrix<float>(_m, _m);
            using var b = new FastBuffer<float>(_m);

            using var deltaBuf = new FastBuffer<float>(_m);

            // OPTYMALIZACJA 2: Różnica (instance - background) raz na całą analizę.
            // Zapisujemy bezpośrednio do deltaBuf.AsSpan()
            TensorPrimitives.Subtract(instance, _background, deltaBuf.AsSpan());

            using var mask = new FastBuffer<float>(_m);
            using var synthetic = new FastBuffer<float>(_m);

            // GWARANTOWANE PRÓBKOWANIE (1-cecha i M-1 cech) wykonywane sekwencyjnie
            for (var i = 0; i < _m; i++)
            {
                mask.Clear();
                mask[i] = 1f;
                // Wywołujemy .AsSpan() na bieżąco, aby uniknąć problemów ze scopingiem
                ProcessSampleSIMD(1, mask.AsSpan(), synthetic.AsSpan(), deltaBuf.AsSpan(), baseValue, A, b.AsSpan());

                mask.AsSpan().Fill(1f);
                mask[i] = 0f;
                ProcessSampleSIMD(_m - 1, mask.AsSpan(), synthetic.AsSpan(), deltaBuf.AsSpan(), baseValue, A, b.AsSpan());
            }

            // Pozostałe losowe próbki - OPTYMALIZACJA 3: Zrównoleglenie
            var remaining = _numSamples - 2 * _m;

            if (remaining > 0)
            {
                // Przechwytujemy referencję do 'deltaBuf' (dozwolone), a NIE 'Span<float>' (niedozwolone)
                Parallel.ForEach(Partitioner.Create(0, remaining), range =>
                {
                    // POPRAWKA CS8175: Tworzymy Span na nowo wewnątrz lambdy (na stosie konkretnego wątku)
                    var threadLocalDelta = deltaBuf.AsSpan();

                    // Thread-local struktury zapobiegające wyścigom
                    using var localA = new FastMatrix<float>(_m, _m);
                    using var localB = new FastBuffer<float>(_m);
                    using var localMask = new FastBuffer<float>(_m);
                    using var localSynthetic = new FastBuffer<float>(_m);

                    for (var s = range.Item1; s < range.Item2; s++)
                    {
                        var k = GenerateFastMask(localMask.AsSpan(), _m);
                        ProcessSampleSIMD(k, localMask.AsSpan(), localSynthetic.AsSpan(), threadLocalDelta, baseValue, localA, localB.AsSpan());
                    }

                    // Bezpieczne scalanie wyników lokalnych do macierzy globalnych
                    lock (A)
                    {
                        MergeSystems(A, b.AsSpan(), localA, localB.AsSpan(), _m);
                    }
                });
            }

            SolveWithEfficiencyConstraint(A, b.AsSpan(), totalDiff, shapValues);
        }

        private void ProcessSampleSIMD(int k, Span<float> mask, Span<float> syn, ReadOnlySpan<float> delta, float baseVal, FastMatrix<float> A, Span<float> b)
        {
            var weight = _precomputedWeights[k];

            // OPTYMALIZACJA 4: SIMD dla próbki syntetycznej bez branchingu:
            // synthetic = (delta * mask) + background
            TensorPrimitives.Multiply(delta, mask, syn);
            TensorPrimitives.Add(syn, _background, syn);

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

        private void MergeSystems(FastMatrix<float> destA, Span<float> destB, FastMatrix<float> srcA, ReadOnlySpan<float> srcB, int m)
        {
            TensorPrimitives.Add(destB, srcB, destB);

            for (var i = 0; i < m; i++)
            {
                TensorPrimitives.Add(destA.Row(i), srcA.Row(i), destA.Row(i));
            }
        }

        private void SolveWithEfficiencyConstraint(FastMatrix<float> A, Span<float> b, float totalDiff, Span<float> phi)
        {
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

        private static int GenerateFastMask(Span<float> mask, int m)
        {
            var k = 0;
            var chunks = (m + 63) / 64; // Obsługa maski dla dowolnej liczby cech

            for (var chunk = 0; chunk < chunks; chunk++)
            {
                var bits = (ulong)Random.Shared.NextInt64();
                var limit = Math.Min(64, m - chunk * 64);

                for (var i = 0; i < limit; i++)
                {
                    // OPTYMALIZACJA 5: Generowanie bitowe
                    var bit = (float)((bits >> i) & 1);
                    mask[chunk * 64 + i] = bit;

                    if (bit > 0)
                    {
                        k++;
                    }
                }
            }

            return Math.Clamp(k, 1, m - 1);
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