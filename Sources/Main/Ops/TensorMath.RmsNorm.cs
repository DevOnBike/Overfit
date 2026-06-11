// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        /// <summary>
        /// RMS normalization (Llama / Qwen style):
        ///   <c>y[i] = x[i] / sqrt(mean(x²) + eps) · gamma[i]</c>
        ///
        /// Unlike <see cref="LayerNorm"/> there is NO mean subtraction and NO beta —
        /// the row is scaled by the reciprocal root-mean-square, then by a learned
        /// per-feature gain <paramref name="gamma"/>. This is the normalization in
        /// every Llama/Qwen transformer block (the trainable-base path).
        ///
        /// Normalizes over the last dimension (features), independently per token.
        /// Input shape: [N, C] or [N, T, C].
        ///
        /// Auxiliary output stored on tape:
        ///   invRms [numRows] — per-row 1/sqrt(mean(x²)+eps), needed for backward.
        ///
        /// Parallelized via <see cref="OverfitParallel"/> above
        /// <see cref="ParallelThreshold"/> total elements; each row is independent.
        /// </summary>
        public static AutogradNode RmsNorm(
            ComputationGraph graph,
            AutogradNode input,
            AutogradNode gamma,
            float eps = 1e-6f)
        {
            var C = input.Shape[input.Shape.Rank - 1];
            var numRows = input.Shape.Size / C;

            var requiresGrad = input.RequiresGrad || gamma.RequiresGrad;
            var output = AllocateNode(graph, input.Shape, requiresGrad, clearMemory: false);

            // Per-row 1/RMS — GraphAuxiliary, disposed by graph.Reset().
            var invRms = AllocateNode(graph, new TensorShape(numRows), requiresGrad: false, clearMemory: false);

            var inS = input.DataView.AsReadOnlySpan();
            var outS = output.DataView.AsSpan();
            var gammaS = gamma.DataView.AsReadOnlySpan();
            var invRmsS = invRms.DataView.AsSpan();

            if ((long)numRows * C < ParallelThreshold)
            {
                RmsNormForwardSeq(inS, outS, gammaS, invRmsS, numRows, C, eps);
            }
            else
            {
                unsafe
                {
                    fixed (float* inPtr = inS, outPtr = outS, gPtr = gammaS, invPtr = invRmsS)
                    {
                        var ctx = new RmsNormForwardCtx
                        {
                            Input = inPtr,
                            Output = outPtr,
                            Gamma = gPtr,
                            InvRms = invPtr,
                            C = C,
                            Eps = eps,
                        };
                        OverfitParallel.For(0, numRows, &RmsNormForwardChunk, &ctx);
                    }
                }
            }

            if (requiresGrad)
            {
                graph?.Record(OpCode.RmsNorm, output, input, c0: gamma, c1: invRms, contextCount: 2);
            }
            else
            {
                invRms.Dispose();
            }

            return output;
        }

        /// <summary>
        /// RMSNorm backward.
        /// With <c>r = invRms</c>, <c>S = Σ_j dY[j]·γ[j]·x[j]</c> (per row):
        ///   <c>dX[i] = r·γ[i]·dY[i] − x[i]·r³·S / C</c>
        ///   <c>dγ[i] = Σ_rows dY[i]·x[i]·r</c>   (accumulated across the batch)
        /// </summary>
        public static void RmsNormBackward(
            AutogradNode input,
            AutogradNode output,
            AutogradNode gamma,
            AutogradNode invRms)
        {
            var C = input.Shape[input.Shape.Rank - 1];
            var numRows = input.Shape.Size / C;

            var inS = input.DataView.AsReadOnlySpan();
            var dOutS = output.GradView.AsReadOnlySpan();
            var gammaS = gamma.DataView.AsReadOnlySpan();
            var invRmsS = invRms.DataView.AsReadOnlySpan();

            var needsDInput = input.RequiresGrad;
            var needsDGamma = gamma.RequiresGrad;

            const int MaxStackallocC = 8192;

            if ((long)numRows * C < ParallelThreshold || C > MaxStackallocC)
            {
                RmsNormBackwardSeq(
                    inS, dOutS, gammaS, invRmsS,
                    needsDInput ? input.GradView.AsSpan() : default,
                    needsDGamma ? gamma.GradView.AsSpan() : default,
                    numRows, C, needsDInput, needsDGamma);
                return;
            }

            var workerCount = OverfitParallel.WorkerCount;
            var chunkCount = Math.Min(workerCount, numRows);
            var perChunk = (numRows + chunkCount - 1) / chunkCount;

            unsafe
            {
                // Per-worker partial dGamma avoids the cross-row shared-write race.
                var dGammaPartial = needsDGamma ? stackalloc float[chunkCount * C] : default;
                if (needsDGamma) { dGammaPartial.Clear(); }

                fixed (float* inPtr = inS, dOutPtr = dOutS, gPtr = gammaS,
                              invPtr = invRmsS, dGammaPartialPtr = dGammaPartial)
                fixed (float* dInPtr = needsDInput ? input.GradView.AsSpan() : default)
                {
                    var ctx = new RmsNormBackwardCtx
                    {
                        Input = inPtr,
                        GradOutput = dOutPtr,
                        Gamma = gPtr,
                        InvRms = invPtr,
                        GradInput = dInPtr,
                        DGammaPartial = dGammaPartialPtr,
                        C = C,
                        PerChunk = perChunk,
                        NeedsDInput = needsDInput,
                        NeedsDGamma = needsDGamma,
                    };

                    OverfitParallel.For(0, numRows, &RmsNormBackwardChunk, &ctx);
                }

                if (needsDGamma)
                {
                    var dGammaFinal = gamma.GradView.AsSpan();
                    for (var w = 0; w < chunkCount; w++)
                    {
                        var slot = dGammaPartial.Slice(w * C, C);
                        TensorPrimitives.Add(dGammaFinal, slot, dGammaFinal);
                    }
                }
            }
        }

        // ── Sequential implementations (small tensors / fallback) ─────────────

        private static void RmsNormForwardSeq(
            ReadOnlySpan<float> inS,
            Span<float> outS,
            ReadOnlySpan<float> gammaS,
            Span<float> invRmsS,
            int numRows, int C, float eps)
        {
            for (var r = 0; r < numRows; r++)
            {
                var row = inS.Slice(r * C, C);
                var outRow = outS.Slice(r * C, C);

                var ms = TensorPrimitives.Dot(row, row) / C;
                var inv = 1f / MathF.Sqrt(ms + eps);
                invRmsS[r] = inv;

                for (var i = 0; i < C; i++)
                {
                    outRow[i] = row[i] * inv * gammaS[i];
                }
            }
        }

        private static void RmsNormBackwardSeq(
            ReadOnlySpan<float> inS,
            ReadOnlySpan<float> dOutS,
            ReadOnlySpan<float> gammaS,
            ReadOnlySpan<float> invRmsS,
            Span<float> dInS,
            Span<float> dGammaS,
            int numRows, int C,
            bool needsDInput, bool needsDGamma)
        {
            for (var r = 0; r < numRows; r++)
            {
                RmsNormBackwardRow(
                    inS.Slice(r * C, C), dOutS.Slice(r * C, C), gammaS, invRmsS[r],
                    dInS, dGammaS, r, C, needsDInput, needsDGamma);
            }
        }

        /// <summary>One row of RMSNorm backward. <paramref name="dGammaS"/> may be a
        /// per-worker partial slot (length C, written with <c>dGammaS[i] +=</c>).</summary>
        private static void RmsNormBackwardRow(
            ReadOnlySpan<float> row,
            ReadOnlySpan<float> dOut,
            ReadOnlySpan<float> gammaS,
            float inv,
            Span<float> dInS,
            Span<float> dGammaS,
            int rowIndex, int C,
            bool needsDInput, bool needsDGamma)
        {
            // S = Σ dY·γ·x
            var s = 0f;
            for (var i = 0; i < C; i++)
            {
                s += dOut[i] * gammaS[i] * row[i];
            }

            var coef = inv * inv * inv * s / C;

            if (needsDInput)
            {
                var dIn = dInS.Slice(rowIndex * C, C);
                for (var i = 0; i < C; i++)
                {
                    dIn[i] += inv * gammaS[i] * dOut[i] - row[i] * coef;
                }
            }

            if (needsDGamma)
            {
                for (var i = 0; i < C; i++)
                {
                    dGammaS[i] += dOut[i] * row[i] * inv;
                }
            }
        }

        // ── OverfitParallel chunk bodies + contexts ────────────────────────

        private unsafe struct RmsNormForwardCtx
        {
            public float* Input;
            public float* Output;
            public float* Gamma;
            public float* InvRms;
            public int C;
            public float Eps;
        }

        private static unsafe void RmsNormForwardChunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<RmsNormForwardCtx>(contextPtr);
            var C = ctx.C;
            for (var r = chunkStart; r < chunkEnd; r++)
            {
                var row = new ReadOnlySpan<float>(ctx.Input + r * C, C);
                var outRow = new Span<float>(ctx.Output + r * C, C);
                var gammaS = new ReadOnlySpan<float>(ctx.Gamma, C);

                var ms = TensorPrimitives.Dot(row, row) / C;
                var inv = 1f / MathF.Sqrt(ms + ctx.Eps);
                ctx.InvRms[r] = inv;

                for (var i = 0; i < C; i++)
                {
                    outRow[i] = row[i] * inv * gammaS[i];
                }
            }
        }

        private unsafe struct RmsNormBackwardCtx
        {
            public float* Input;
            public float* GradOutput;
            public float* Gamma;
            public float* InvRms;
            public float* GradInput;
            public float* DGammaPartial;
            public int C;
            public int PerChunk;
            public bool NeedsDInput;
            public bool NeedsDGamma;
        }

        private static unsafe void RmsNormBackwardChunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<RmsNormBackwardCtx>(contextPtr);
            var C = ctx.C;

            // Slot for this worker's partial dGamma (indexed by chunk, like LayerNorm).
            var chunkIdx = ctx.PerChunk > 0 ? chunkStart / ctx.PerChunk : 0;
            var dGammaSlot = ctx.NeedsDGamma
                ? new Span<float>(ctx.DGammaPartial + chunkIdx * C, C)
                : default;
            var dInAll = ctx.NeedsDInput ? new Span<float>(ctx.GradInput, (chunkEnd) * C) : default;
            var gammaS = new ReadOnlySpan<float>(ctx.Gamma, C);

            for (var r = chunkStart; r < chunkEnd; r++)
            {
                var row = new ReadOnlySpan<float>(ctx.Input + r * C, C);
                var dOut = new ReadOnlySpan<float>(ctx.GradOutput + r * C, C);

                RmsNormBackwardRow(
                    row, dOut, gammaS, ctx.InvRms[r],
                    dInAll, dGammaSlot, r, C,
                    ctx.NeedsDInput, ctx.NeedsDGamma);
            }
        }
    }
}
