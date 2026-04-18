// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Diagnostics;
using DevOnBike.Overfit.Optimizers.Abstractions;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Optimizers
{
    /// <summary>
    /// Implements the Adam + AdamW optimizer with AVX-512 acceleration.
    /// </summary>
    public sealed class Adam : IOptimizer, IDisposable
    {
        private const int Avx512Threshold = 512;

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

                // Parallel processing for multiple large parameters
                if (_states.Length >= 4)
                {
                    if (UseAdamW)
                    {
                        Parallel.ForEach(_states, state =>
                        {
                            if (state.Node.RequiresGrad)
                            {
                                StepAdamW(state, b1, b2, b1Inv, b2Inv, invBc1, invBc2, eps, lr, wd);
                            }
                        });
                    }
                    else
                    {
                        Parallel.ForEach(_states, state =>
                        {
                            if (state.Node.RequiresGrad)
                            {
                                StepAdam(state, b1, b2, b1Inv, b2Inv, invBc1, invBc2, eps, lr, wd);
                            }
                        });
                    }
                }
                else
                {
                    // Sequential for few parameters (less overhead)
                    if (UseAdamW)
                    {
                        foreach (var state in _states)
                        {
                            if (state.Node.RequiresGrad)
                            {
                                StepAdamW(state, b1, b2, b1Inv, b2Inv, invBc1, invBc2, eps, lr, wd);
                            }
                        }
                    }
                    else
                    {
                        foreach (var state in _states)
                        {
                            if (state.Node.RequiresGrad)
                            {
                                StepAdam(state, b1, b2, b1Inv, b2Inv, invBc1, invBc2, eps, lr, wd);
                            }
                        }
                    }
                }
            }
            finally
            {
                ModuleDiagnostics.End(ctx);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void StepAdamW(
            in ParamState state,
            float b1, float b2, float b1Inv, float b2Inv,
            float invBc1, float invBc2, float eps, float lr, float wd)
        {
            var n = state.Size;
            var gSpan = state.Node.GradView.AsSpan();
            var mSpan = state.M.GetView().AsSpan();
            var vSpan = state.V.GetView().AsSpan();
            var wSpan = state.Node.DataView.AsSpan();

            ref var gRef = ref MemoryMarshal.GetReference(gSpan);
            ref var mRef = ref MemoryMarshal.GetReference(mSpan);
            ref var vRef = ref MemoryMarshal.GetReference(vSpan);
            ref var wRef = ref MemoryMarshal.GetReference(wSpan);

            var i = 0;

            // AVX-512 path for large parameters
            if (Avx512F.IsSupported && n >= Avx512Threshold)
            {
                var simd512 = Vector512<float>.Count; // 16
                var vB1 = Vector512.Create(b1);
                var vB2 = Vector512.Create(b2);
                var vB1Inv = Vector512.Create(b1Inv);
                var vB2Inv = Vector512.Create(b2Inv);
                var vInvBc1 = Vector512.Create(invBc1);
                var vInvBc2 = Vector512.Create(invBc2);
                var vEps = Vector512.Create(eps);
                var vLr = Vector512.Create(lr);
                var vWdLr = Vector512.Create(wd * lr);

                for (; i <= n - simd512; i += simd512)
                {
                    var vG = Vector512.LoadUnsafe(ref gRef, (nuint)i);
                    var vM = Vector512.LoadUnsafe(ref mRef, (nuint)i);
                    var vV = Vector512.LoadUnsafe(ref vRef, (nuint)i);
                    var vW = Vector512.LoadUnsafe(ref wRef, (nuint)i);

                    // m = b1 * m + b1Inv * g
                    vM = Avx512F.FusedMultiplyAdd(vM, vB1, vG * vB1Inv);

                    // v = b2 * v + b2Inv * g * g
                    vV = Avx512F.FusedMultiplyAdd(vV, vB2, vG * vG * vB2Inv);

                    // mHat = m * invBc1
                    var vMHat = vM * vInvBc1;

                    // vHat = sqrt(v * invBc2) + eps
                    var vVHat = Avx512F.Sqrt(vV * vInvBc2) + vEps;

                    // w -= lr * mHat / vHat
                    vW = Avx512F.FusedMultiplyAddNegated(vMHat / vVHat, vLr, vW);

                    // w -= w * wd * lr (weight decay)
                    vW = Avx512F.FusedMultiplyAddNegated(vW, vWdLr, vW);

                    vM.StoreUnsafe(ref mRef, (nuint)i);
                    vV.StoreUnsafe(ref vRef, (nuint)i);
                    vW.StoreUnsafe(ref wRef, (nuint)i);
                }
            }
            // AVX2 path
            else if (Vector.IsHardwareAccelerated && n >= Vector<float>.Count * 4)
            {
                var gVec = MemoryMarshal.Cast<float, Vector<float>>(gSpan);
                var mVec = MemoryMarshal.Cast<float, Vector<float>>(mSpan);
                var vVec = MemoryMarshal.Cast<float, Vector<float>>(vSpan);
                var wVec = MemoryMarshal.Cast<float, Vector<float>>(wSpan);

                var vecB1 = new Vector<float>(b1);
                var vecB2 = new Vector<float>(b2);
                var vecB1Inv = new Vector<float>(b1Inv);
                var vecB2Inv = new Vector<float>(b2Inv);
                var vecInvBc1 = new Vector<float>(invBc1);
                var vecInvBc2 = new Vector<float>(invBc2);
                var vecEps = new Vector<float>(eps);
                var vecLr = new Vector<float>(lr);
                var vecWdLr = new Vector<float>(wd * lr);

                for (var j = 0; j < gVec.Length; j++)
                {
                    var vG = gVec[j];
                    var vM = mVec[j];
                    var vV = vVec[j];
                    var vW = wVec[j];

                    vM = vM * vecB1 + vG * vecB1Inv;
                    vV = vV * vecB2 + vG * vG * vecB2Inv;

                    var vMHat = vM * vecInvBc1;
                    var vVHat = Vector.SquareRoot(vV * vecInvBc2) + vecEps;
                    vW -= vMHat / vVHat * vecLr;
                    vW -= vW * vecWdLr;

                    mVec[j] = vM;
                    vVec[j] = vV;
                    wVec[j] = vW;
                }

                i = gVec.Length * Vector<float>.Count;
            }

            // Scalar remainder
            for (; i < n; i++)
            {
                var gw = Unsafe.Add(ref gRef, i);
                var mw = Unsafe.Add(ref mRef, i);
                var vw = Unsafe.Add(ref vRef, i);
                var ww = Unsafe.Add(ref wRef, i);

                mw = b1 * mw + b1Inv * gw;
                vw = b2 * vw + b2Inv * (gw * gw);

                var mHat = mw * invBc1;
                var vHat = MathF.Sqrt(vw * invBc2) + eps;
                ww -= lr * (mHat / vHat);
                ww -= ww * wd * lr;

                Unsafe.Add(ref mRef, i) = mw;
                Unsafe.Add(ref vRef, i) = vw;
                Unsafe.Add(ref wRef, i) = ww;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void StepAdam(
            in ParamState state,
            float b1, float b2, float b1Inv, float b2Inv,
            float invBc1, float invBc2, float eps, float lr, float wd)
        {
            var n = state.Size;
            var gSpan = state.Node.GradView.AsSpan();
            var mSpan = state.M.GetView().AsSpan();
            var vSpan = state.V.GetView().AsSpan();
            var wSpan = state.Node.DataView.AsSpan();

            ref var gRef = ref MemoryMarshal.GetReference(gSpan);
            ref var mRef = ref MemoryMarshal.GetReference(mSpan);
            ref var vRef = ref MemoryMarshal.GetReference(vSpan);
            ref var wRef = ref MemoryMarshal.GetReference(wSpan);

            var i = 0;

            // AVX-512 path
            if (Avx512F.IsSupported && n >= Avx512Threshold)
            {
                var simd512 = Vector512<float>.Count;
                var vB1 = Vector512.Create(b1);
                var vB2 = Vector512.Create(b2);
                var vB1Inv = Vector512.Create(b1Inv);
                var vB2Inv = Vector512.Create(b2Inv);
                var vInvBc1 = Vector512.Create(invBc1);
                var vInvBc2 = Vector512.Create(invBc2);
                var vEps = Vector512.Create(eps);
                var vLr = Vector512.Create(lr);
                var vWd = Vector512.Create(wd);

                for (; i <= n - simd512; i += simd512)
                {
                    var vG = Vector512.LoadUnsafe(ref gRef, (nuint)i);
                    var vM = Vector512.LoadUnsafe(ref mRef, (nuint)i);
                    var vV = Vector512.LoadUnsafe(ref vRef, (nuint)i);
                    var vW = Vector512.LoadUnsafe(ref wRef, (nuint)i);

                    // L2 regularization: g += wd * w
                    var vGl2 = Avx512F.FusedMultiplyAdd(vW, vWd, vG);

                    // m = b1 * m + b1Inv * gl2
                    vM = Avx512F.FusedMultiplyAdd(vM, vB1, vGl2 * vB1Inv);

                    // v = b2 * v + b2Inv * gl2 * gl2
                    vV = Avx512F.FusedMultiplyAdd(vV, vB2, vGl2 * vGl2 * vB2Inv);

                    var vMHat = vM * vInvBc1;
                    var vVHat = Avx512F.Sqrt(vV * vInvBc2) + vEps;
                    vW = Avx512F.FusedMultiplyAddNegated(vMHat / vVHat, vLr, vW);

                    vM.StoreUnsafe(ref mRef, (nuint)i);
                    vV.StoreUnsafe(ref vRef, (nuint)i);
                    vW.StoreUnsafe(ref wRef, (nuint)i);
                }
            }
            // AVX2 path
            else if (Vector.IsHardwareAccelerated && n >= Vector<float>.Count * 4)
            {
                var gVec = MemoryMarshal.Cast<float, Vector<float>>(gSpan);
                var mVec = MemoryMarshal.Cast<float, Vector<float>>(mSpan);
                var vVec = MemoryMarshal.Cast<float, Vector<float>>(vSpan);
                var wVec = MemoryMarshal.Cast<float, Vector<float>>(wSpan);

                var vecB1 = new Vector<float>(b1);
                var vecB2 = new Vector<float>(b2);
                var vecB1Inv = new Vector<float>(b1Inv);
                var vecB2Inv = new Vector<float>(b2Inv);
                var vecInvBc1 = new Vector<float>(invBc1);
                var vecInvBc2 = new Vector<float>(invBc2);
                var vecEps = new Vector<float>(eps);
                var vecLr = new Vector<float>(lr);
                var vecWd = new Vector<float>(wd);

                for (var j = 0; j < gVec.Length; j++)
                {
                    var vG = gVec[j];
                    var vM = mVec[j];
                    var vV = vVec[j];
                    var vW = wVec[j];

                    var vGl2 = vG + vW * vecWd;
                    vM = vM * vecB1 + vGl2 * vecB1Inv;
                    vV = vV * vecB2 + vGl2 * vGl2 * vecB2Inv;

                    var vMHat = vM * vecInvBc1;
                    var vVHat = Vector.SquareRoot(vV * vecInvBc2) + vecEps;
                    vW -= vMHat / vVHat * vecLr;

                    mVec[j] = vM;
                    vVec[j] = vV;
                    wVec[j] = vW;
                }

                i = gVec.Length * Vector<float>.Count;
            }

            // Scalar remainder
            for (; i < n; i++)
            {
                var gw = Unsafe.Add(ref gRef, i);
                var ww = Unsafe.Add(ref wRef, i);
                var mw = Unsafe.Add(ref mRef, i);
                var vw = Unsafe.Add(ref vRef, i);

                var gl2 = gw + wd * ww;
                mw = b1 * mw + b1Inv * gl2;
                vw = b2 * vw + b2Inv * (gl2 * gl2);

                var mHat = mw * invBc1;
                var vHat = MathF.Sqrt(vw * invBc2) + eps;
                ww -= lr * (mHat / vHat);

                Unsafe.Add(ref mRef, i) = mw;
                Unsafe.Add(ref vRef, i) = vw;
                Unsafe.Add(ref wRef, i) = ww;
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