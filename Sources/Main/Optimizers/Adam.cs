// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;
using System.Runtime.InteropServices;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Optimizers
{
    /// <summary>
    ///     Implements the Adam & AdamW (Adaptive Moment Estimation with Decoupled Weight Decay).
    ///     Features extreme-performance MemoryMarshal SIMD paths.
    /// </summary>
    public sealed class Adam : IOptimizer, IDisposable
    {
        private readonly ParamState[] _states;
        private int _t; // Timestep counter for bias correction

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

        public float Beta1 { get; set; } = 0.9f;
        public float Beta2 { get; set; } = 0.999f;
        public float Epsilon { get; set; } = 1e-8f;
        public float WeightDecay { get; set; } = 0.0001f;

        // ZŁOTY STANDARD DEEP LEARNINGU: AdamW
        public bool UseAdamW { get; set; } = true;

        public float LearningRate { get; set; }

        public void Dispose()
        {
            foreach (var state in _states)
            {
                state.M?.Dispose();
                state.V?.Dispose();
            }
        }

        /// <summary>
        ///     Performs a single optimization step using highly optimized SIMD operations.
        /// </summary>
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
            var useAdamW = UseAdamW;

            foreach (var state in _states)
            {
                var p = state.Node;
                var n = state.Size;

                if (p.Grad == null)
                {
                    continue;
                }

                var gSpan = p.Grad.AsSpan();
                var mSpan = state.M.AsSpan();
                var vSpan = state.V.AsSpan();
                var wSpan = p.Data.AsSpan();

                var elementsProcessed = 0;

                if (Vector.IsHardwareAccelerated)
                {
                    // MemoryMarshal magią pozbywa się bounds-checkingu i tworzenia tymczasowych obiektów.
                    var gVec = MemoryMarshal.Cast<float, Vector<float>>(gSpan);
                    var mVec = MemoryMarshal.Cast<float, Vector<float>>(mSpan);
                    var vVec = MemoryMarshal.Cast<float, Vector<float>>(vSpan);
                    var wVec = MemoryMarshal.Cast<float, Vector<float>>(wSpan);

                    var vWd = new Vector<float>(wd);
                    var vB1 = new Vector<float>(b1);
                    var vB2 = new Vector<float>(b2);
                    var vB1Inv = new Vector<float>(b1Inv);
                    var vB2Inv = new Vector<float>(b2Inv);
                    var vInvBc1 = new Vector<float>(invBc1);
                    var vInvBc2 = new Vector<float>(invBc2);
                    var vEps = new Vector<float>(eps);
                    var vLr = new Vector<float>(lr);

                    for (var i = 0; i < gVec.Length; i++)
                    {
                        var vG = gVec[i];
                        var vW = wVec[i];
                        var vM = mVec[i];
                        var vV = vVec[i];

                        if (useAdamW)
                        {
                            // AdamW: Weight decay zaaplikowane osobno na wagach, NIE na gradientach.
                            vM = vM * vB1 + vG * vB1Inv;
                            vV = vV * vB2 + vG * vG * vB2Inv;

                            var vMHat = vM * vInvBc1;
                            var vVHat = Vector.SquareRoot(vV * vInvBc2) + vEps;

                            vW -= vMHat / vVHat * vLr;
                            if (wd > 0f)
                            {
                                vW -= vW * vWd * vLr; // AdamW penalty
                            }
                        }
                        else
                        {
                            // Klasyczny Adam z L2 Regularization (Przestarzałe)
                            var vGl2 = vG + vW * vWd;
                            vM = vM * vB1 + vGl2 * vB1Inv;
                            vV = vV * vB2 + vGl2 * vGl2 * vB2Inv;

                            var vMHat = vM * vInvBc1;
                            var vVHat = Vector.SquareRoot(vV * vInvBc2) + vEps;

                            vW -= vMHat / vVHat * vLr;
                        }

                        // Zapis wyników bez CopyTo()
                        mVec[i] = vM;
                        vVec[i] = vV;
                        wVec[i] = vW;
                    }
                    elementsProcessed = gVec.Length * Vector<float>.Count;
                }

                // Skalarny fallback dla resztki parametrów (np. 3 ostatnie z warstwy)
                for (var i = elementsProcessed; i < n; i++)
                {
                    var gw = gSpan[i];
                    var ww = wSpan[i];
                    var mw = mSpan[i];
                    var vw = vSpan[i];

                    if (useAdamW)
                    {
                        mw = b1 * mw + b1Inv * gw;
                        vw = b2 * vw + b2Inv * (gw * gw);

                        var mHat = mw * invBc1;
                        var vHat = MathF.Sqrt(vw * invBc2) + eps;

                        ww -= lr * (mHat / vHat);
                        if (wd > 0f)
                        {
                            ww -= ww * wd * lr;
                        }
                    }
                    else
                    {
                        var gl2 = gw + wd * ww;
                        mw = b1 * mw + b1Inv * gl2;
                        vw = b2 * vw + b2Inv * (gl2 * gl2);

                        var mHat = mw * invBc1;
                        var vHat = MathF.Sqrt(vw * invBc2) + eps;

                        ww -= lr * (mHat / vHat);
                    }

                    mSpan[i] = mw;
                    vSpan[i] = vw;
                    wSpan[i] = ww;
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

        private readonly struct ParamState
        {
            public readonly AutogradNode Node;
            public readonly FastTensor<float> M; // First moment vector
            public readonly FastTensor<float> V; // Second moment vector
            public readonly int Size;

            public ParamState(AutogradNode node)
            {
                Node = node;
                Size = node.Data.Size;

                M = new FastTensor<float>(true, node.Data.Shape);
                V = new FastTensor<float>(true, node.Data.Shape);
            }
        }
    }
}