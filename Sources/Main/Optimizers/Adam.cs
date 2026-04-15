using System.Numerics;
using System.Runtime.InteropServices;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Diagnostics;
using DevOnBike.Overfit.Optimizers.Abstractions;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Optimizers
{
    /// <summary>
    /// Implements the Adam + AdamW optimizer with optional diagnostics.
    /// </summary>
    public sealed class Adam : IOptimizer, IDisposable
    {
        private readonly ParamState[] _states;
        private int _t;

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

        public void Step()
        {
            var ctx = ModuleDiagnostics.Begin(
                moduleType: nameof(Adam),
                phase: "step",
                isTraining: true,
                batchSize: 0,
                inputRows: _states.Length,
                inputCols: 0,
                outputRows: _states.Length,
                outputCols: 0);

            try
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

                    if (!p.RequiresGrad)
                    {
                        continue;
                    }

                    var gSpan = p.GradView.AsSpan();
                    var mSpan = state.M.GetView().AsSpan();
                    var vSpan = state.V.GetView().AsSpan();
                    var wSpan = p.DataView.AsSpan();

                    var elementsProcessed = 0;

                    if (Vector.IsHardwareAccelerated)
                    {
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
                                vM = vM * vB1 + vG * vB1Inv;
                                vV = vV * vB2 + vG * vG * vB2Inv;

                                var vMHat = vM * vInvBc1;
                                var vVHat = Vector.SquareRoot(vV * vInvBc2) + vEps;
                                vW -= vMHat / vVHat * vLr;

                                if (wd > 0f)
                                {
                                    vW -= vW * vWd * vLr;
                                }
                            }
                            else
                            {
                                var vGl2 = vG + vW * vWd;
                                vM = vM * vB1 + vGl2 * vB1Inv;
                                vV = vV * vB2 + vGl2 * vGl2 * vB2Inv;

                                var vMHat = vM * vInvBc1;
                                var vVHat = Vector.SquareRoot(vV * vInvBc2) + vEps;
                                vW -= vMHat / vVHat * vLr;
                            }

                            mVec[i] = vM;
                            vVec[i] = vV;
                            wVec[i] = vW;
                        }

                        elementsProcessed = gVec.Length * Vector<float>.Count;
                    }

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
            finally
            {
                ModuleDiagnostics.End(ctx);
            }
        }

        public void ZeroGrad()
        {
            var ctx = ModuleDiagnostics.Begin(
                moduleType: nameof(Adam),
                phase: "zero_grad",
                isTraining: true,
                batchSize: 0,
                inputRows: _states.Length,
                inputCols: 0,
                outputRows: _states.Length,
                outputCols: 0);

            try
            {
                foreach (var state in _states)
                {
                    state.Node.ZeroGrad();
                }
            }
            finally
            {
                ModuleDiagnostics.End(ctx);
            }
        }

        public void ResetTime()
        {
            _t = 0;
        }

        private readonly struct ParamState
        {
            public readonly AutogradNode Node;
            public readonly FastTensor<float> M;
            public readonly FastTensor<float> V;
            public readonly int Size;

            public ParamState(AutogradNode node)
            {
                Node = node;
                Size = node.DataView.Size;
                
                M = new FastTensor<float>(Size, clearMemory: true);
                V = new FastTensor<float>(Size, clearMemory: true);
            }
        }
    }
}
