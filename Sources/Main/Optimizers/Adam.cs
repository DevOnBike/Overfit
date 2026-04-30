// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Optimizers.Abstractions;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Optimizers
{
    /// <summary>
    /// Adam / AdamW optimizer.
    ///
    /// BeastMode:
    /// - small parameters: sequential SIMD
    /// - large parameters: Parallel.For over element chunks
    /// - global parallelism: OverfitParallel.Options
    /// - ZeroGrad intentionally stays sequential because parallel ZeroGrad was slower
    ///   and allocated more in MNIST BeastMode.
    /// </summary>
    public sealed class Adam : IOptimizer, IDisposable
    {
        private const int VectorThreshold = 512;
        private const int ParallelElementThreshold = 65_536;
        private const int MinChunkElements = 16_384;
        private const bool ParallelZeroGrad = false;

        private readonly ParamState[] _states;
        private int _t;

        /// <summary>
        /// Preferred constructor — accepts <see cref="Parameter"/> directly.
        /// Grad storage is shared with Parameter: backward writes into Parameter.Grad,
        /// optimizer reads from the same buffer. No sync copy needed.
        /// </summary>
        public Adam(IEnumerable<Parameter> parameters, float learningRate = 0.001f)
        {
            _ = OverfitParallel.Options;

            LearningRate = learningRate;

            var statesList = new List<ParamState>();

            foreach (var p in parameters)
            {
                if (p.RequiresGrad)
                {
                    statesList.Add(new ParamState(p));
                }
            }

            _states = statesList.ToArray();
        }

        /// <summary>
        /// Compatibility shim — accepts legacy <see cref="AutogradNode"/> collections.
        /// Prefer <see cref="Adam(IEnumerable{Parameter}, float)"/> for new code.
        /// </summary>
        [Obsolete("Pass IEnumerable<Parameter> via module.TrainableParameters() instead.")]
        public Adam(IEnumerable<AutogradNode> parameters, float learningRate = 0.001f)
        {
            _ = OverfitParallel.Options;

            LearningRate = learningRate;

            var statesList = new List<ParamState>();

            foreach (var p in parameters)
            {
                if (p.RequiresGrad)
                {
                    statesList.Add(new ParamState(p));
                }
            }

            _states = statesList.ToArray();
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
                state.M.Dispose();
                state.V.Dispose();
            }
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

            if (UseAdamW)
            {
                foreach (var state in _states)
                {
                    if (!state.RequiresGrad)
                    {
                        continue;
                    }

                    StepAdamWState(
                        state,
                        b1,
                        b2,
                        b1Inv,
                        b2Inv,
                        invBc1,
                        invBc2,
                        eps,
                        lr,
                        wd);
                }
            }
            else
            {
                foreach (var state in _states)
                {
                    if (!state.RequiresGrad)
                    {
                        continue;
                    }

                    StepAdamState(
                        state,
                        b1,
                        b2,
                        b1Inv,
                        b2Inv,
                        invBc1,
                        invBc2,
                        eps,
                        lr,
                        wd);
                }
            }
        }

        public void ZeroGrad()
        {
            foreach (var state in _states)
            {
                if (ParallelZeroGrad && state.Size >= ParallelElementThreshold)
                {
                    ClearGradParallel(state, state.Size);
                }
                else
                {
                    state.ZeroGrad();
                }
            }
        }

        public void ResetTime()
        {
            _t = 0;
        }

        private static void ClearGradParallel(ParamState state, int size)
        {
            var chunks = GetChunkCount(size);

            if (chunks <= 1)
            {
                state.ZeroGrad();
                return;
            }

            Parallel.For(
                0,
                chunks,
                OverfitParallel.Options,
                chunk =>
                {
                    GetChunkRange(size, chunks, chunk, out var start, out var end);

                    state.GradSpan
                        .Slice(start, end - start)
                        .Clear();
                });
        }

        private static void StepAdamWState(
            ParamState state,
            float b1,
            float b2,
            float b1Inv,
            float b2Inv,
            float invBc1,
            float invBc2,
            float eps,
            float lr,
            float wd)
        {
            var size = state.Size;

            if (size < ParallelElementThreshold)
            {
                StepAdamWRange(
                    state,
                    0,
                    size,
                    b1,
                    b2,
                    b1Inv,
                    b2Inv,
                    invBc1,
                    invBc2,
                    eps,
                    lr,
                    wd);

                return;
            }

            var chunks = GetChunkCount(size);

            Parallel.For(
                0,
                chunks,
                OverfitParallel.Options,
                chunk =>
                {
                    GetChunkRange(size, chunks, chunk, out var start, out var end);

                    StepAdamWRange(
                        state,
                        start,
                        end,
                        b1,
                        b2,
                        b1Inv,
                        b2Inv,
                        invBc1,
                        invBc2,
                        eps,
                        lr,
                        wd);
                });
        }

        private static void StepAdamState(
            ParamState state,
            float b1,
            float b2,
            float b1Inv,
            float b2Inv,
            float invBc1,
            float invBc2,
            float eps,
            float lr,
            float wd)
        {
            var size = state.Size;

            if (size < ParallelElementThreshold)
            {
                StepAdamRange(
                    state,
                    0,
                    size,
                    b1,
                    b2,
                    b1Inv,
                    b2Inv,
                    invBc1,
                    invBc2,
                    eps,
                    lr,
                    wd);

                return;
            }

            var chunks = GetChunkCount(size);

            Parallel.For(
                0,
                chunks,
                OverfitParallel.Options,
                chunk =>
                {
                    GetChunkRange(size, chunks, chunk, out var start, out var end);

                    StepAdamRange(
                        state,
                        start,
                        end,
                        b1,
                        b2,
                        b1Inv,
                        b2Inv,
                        invBc1,
                        invBc2,
                        eps,
                        lr,
                        wd);
                });
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void StepAdamWRange(
            ParamState state,
            int start,
            int end,
            float b1,
            float b2,
            float b1Inv,
            float b2Inv,
            float invBc1,
            float invBc2,
            float eps,
            float lr,
            float wd)
        {
            var length = end - start;

            if (length <= 0)
            {
                return;
            }

            var gSpan = state.GradSpan.Slice(start, length);
            var mSpan = state.M.GetView().AsSpan().Slice(start, length);
            var vSpan = state.V.GetView().AsSpan().Slice(start, length);
            var wSpan = state.DataSpan.Slice(start, length);

            var i = 0;

            if (Vector.IsHardwareAccelerated && length >= VectorThreshold)
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

                    // Existing local AdamW behavior: decay after Adam update.
                    vW -= vW * vecWdLr;

                    mVec[j] = vM;
                    vVec[j] = vV;
                    wVec[j] = vW;
                }

                i = gVec.Length * Vector<float>.Count;
            }

            for (; i < length; i++)
            {
                var gw = gSpan[i];
                var mw = mSpan[i];
                var vw = vSpan[i];
                var ww = wSpan[i];

                mw = b1 * mw + b1Inv * gw;
                vw = b2 * vw + b2Inv * (gw * gw);

                var mHat = mw * invBc1;
                var vHat = MathF.Sqrt(vw * invBc2) + eps;

                ww -= lr * (mHat / vHat);
                ww -= ww * wd * lr;

                mSpan[i] = mw;
                vSpan[i] = vw;
                wSpan[i] = ww;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void StepAdamRange(
            ParamState state,
            int start,
            int end,
            float b1,
            float b2,
            float b1Inv,
            float b2Inv,
            float invBc1,
            float invBc2,
            float eps,
            float lr,
            float wd)
        {
            var length = end - start;

            if (length <= 0)
            {
                return;
            }

            var gSpan = state.GradSpan.Slice(start, length);
            var mSpan = state.M.GetView().AsSpan().Slice(start, length);
            var vSpan = state.V.GetView().AsSpan().Slice(start, length);
            var wSpan = state.DataSpan.Slice(start, length);

            var i = 0;

            if (Vector.IsHardwareAccelerated && length >= VectorThreshold)
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

            for (; i < length; i++)
            {
                var gw = gSpan[i];
                var ww = wSpan[i];
                var mw = mSpan[i];
                var vw = vSpan[i];

                var gl2 = gw + wd * ww;

                mw = b1 * mw + b1Inv * gl2;
                vw = b2 * vw + b2Inv * (gl2 * gl2);

                var mHat = mw * invBc1;
                var vHat = MathF.Sqrt(vw * invBc2) + eps;

                ww -= lr * (mHat / vHat);

                mSpan[i] = mw;
                vSpan[i] = vw;
                wSpan[i] = ww;
            }
        }

        private static int GetChunkCount(int size)
        {
            var bySize = Math.Max(1, (size + MinChunkElements - 1) / MinChunkElements);
            return Math.Min(OverfitParallel.MaxDegreeOfParallelism, bySize);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void GetChunkRange(
            int size,
            int chunks,
            int chunk,
            out int start,
            out int end)
        {
            start = (int)((long)size * chunk / chunks);
            end = (int)((long)size * (chunk + 1) / chunks);
        }

        private readonly struct ParamState
        {
            private readonly Parameter? _param;
            private readonly AutogradNode? _node;

            public readonly FastTensor<float> M;
            public readonly FastTensor<float> V;
            public readonly int Size;

            // Preferred path: Parameter-backed state.
            // Data and Grad are accessed directly — no AutogradNode overhead.
            public ParamState(Parameter param)
            {
                _param = param;
                _node  = null;
                Size   = param.Shape.Size;
                M      = new FastTensor<float>(Size, clearMemory: true);
                V      = new FastTensor<float>(Size, clearMemory: true);
            }

            // Legacy path: AutogradNode-backed state.
            public ParamState(AutogradNode node)
            {
                _param = null;
                _node  = node;
                Size   = node.DataView.Size;
                M      = new FastTensor<float>(Size, clearMemory: true);
                V      = new FastTensor<float>(Size, clearMemory: true);
            }

            public Span<float> DataSpan =>
                _param != null
                    ? _param.Data.AsSpan()
                    : _node!.DataView.AsSpan();

            public Span<float> GradSpan =>
                _param != null
                    ? _param.Grad!.AsSpan()
                    : _node!.GradView.AsSpan();

            public void ZeroGrad()
            {
                if (_param != null) { _param.ZeroGrad(); }
                else { _node!.ZeroGrad(); }
            }

            public bool RequiresGrad =>
                _param != null ? _param.RequiresGrad : _node!.RequiresGrad;
        }
    }
}