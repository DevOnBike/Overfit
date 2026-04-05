// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Optimizers
{
    public sealed class Adam : IOptimizer, IDisposable
    {
        private readonly struct ParamState
        {
            public readonly AutogradNode Node;
            public readonly FastTensor<float> M;
            public readonly FastTensor<float> V;
            public readonly int Size;

            public ParamState(AutogradNode node)
            {
                Node = node;
                Size = node.Data.Size;

                // Używamy true, by FastTensor od razu wyczyścił pamięć z ArrayPool
                M = new FastTensor<float>(true, node.Data.Shape);
                V = new FastTensor<float>(true, node.Data.Shape);
            }
        }

        private readonly ParamState[] _states;

        public float LearningRate { get; set; }
        public float Beta1 { get; set; } = 0.9f;
        public float Beta2 { get; set; } = 0.999f;
        public float Epsilon { get; set; } = 1e-8f;
        public float WeightDecay { get; set; } = 0.0001f;

        private int _t = 0;

        public Adam(IEnumerable<AutogradNode> parameters, float learningRate = 0.001f)
        {
            LearningRate = learningRate;

            var statesList = new List<ParamState>();

            foreach (var p in parameters)
            {
                if (p.RequiresGrad)
                {
                    statesList.Add(new ParamState(p));
                }
            }

            _states = [.. statesList];
        }

        public void Step()
        {
            _t++;

            var bc1 = 1f - MathF.Pow(Beta1, _t);
            var bc2 = 1f - MathF.Pow(Beta2, _t);
            var invBc1 = 1f / bc1;
            var invBc2 = 1f / bc2;

            var wd = WeightDecay;
            var b1 = Beta1;
            var b2 = Beta2;
            var b1Inv = 1f - Beta1;
            var b2Inv = 1f - Beta2;
            var eps = Epsilon;
            var lr = LearningRate;

            foreach (var state in _states)
            {
                var p = state.Node;
                var n = state.Size;

                if (p.Grad == null) continue;

                var g = p.Grad.AsSpan();
                var m = state.M.AsSpan();
                var v = state.V.AsSpan();
                var w = p.Data.AsSpan();

                var i = 0;

                // Fuzja SIMD: 1 przejście przez pamięć zamiast 10!
                if (Vector.IsHardwareAccelerated)
                {
                    var vecSize = Vector<float>.Count;

                    var vWd = new Vector<float>(wd);
                    var vB1 = new Vector<float>(b1);
                    var vB2 = new Vector<float>(b2);
                    var vB1Inv = new Vector<float>(b1Inv);
                    var vB2Inv = new Vector<float>(b2Inv);
                    var vInvBc1 = new Vector<float>(invBc1);
                    var vInvBc2 = new Vector<float>(invBc2);
                    var vEps = new Vector<float>(eps);
                    var vLr = new Vector<float>(lr);

                    for (; i <= n - vecSize; i += vecSize)
                    {
                        var vG = new Vector<float>(g.Slice(i));
                        var vM = new Vector<float>(m.Slice(i));
                        var vV = new Vector<float>(v.Slice(i));
                        var vW = new Vector<float>(w.Slice(i));

                        // 1. gWithL2 = g + wd * w
                        var vGl2 = vG + vW * vWd;

                        // 2. m = b1 * m + b1Inv * gWithL2
                        vM = vM * vB1 + vGl2 * vB1Inv;

                        // 3. v = b2 * v + b2Inv * (gWithL2 * gWithL2)
                        vV = vV * vB2 + (vGl2 * vGl2) * vB2Inv;

                        // 4 & 5. mHat = m * invBc1, vHat = sqrt(v * invBc2) + eps
                        var vMHat = vM * vInvBc1;
                        var vVHat = Vector.SquareRoot(vV * vInvBc2) + vEps;

                        // 6. w -= lr * (mHat / vHat)
                        vW -= (vMHat / vVHat) * vLr;

                        // Bezpośredni zrzut do pamięci
                        vM.CopyTo(m.Slice(i));
                        vV.CopyTo(v.Slice(i));
                        vW.CopyTo(w.Slice(i));
                    }
                }

                // Resztki (jeśli tablica nie jest wielokrotnością szerokości wektora)
                for (; i < n; i++)
                {
                    var gl2 = g[i] + wd * w[i];
                    m[i] = b1 * m[i] + b1Inv * gl2;
                    v[i] = b2 * v[i] + b2Inv * (gl2 * gl2);

                    var mHat = m[i] * invBc1;
                    var vHat = MathF.Sqrt(v[i] * invBc2) + eps;

                    w[i] -= lr * (mHat / vHat);
                }
            }
        }

        public void ZeroGrad()
        {
            foreach (var state in _states)
            {
                if (state.Node.Grad != null)
                {
                    state.Node.Grad.AsSpan().Clear();
                }
            }
        }

        public void ResetTime()
        {
            _t = 0;
        }

        public void Dispose()
        {
            foreach (var state in _states)
            {
                state.M?.Dispose();
                state.V?.Dispose();
            }
        }
    }
}