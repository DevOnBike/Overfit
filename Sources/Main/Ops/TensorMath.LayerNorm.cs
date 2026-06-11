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
        /// Layer Normalization forward pass.
        ///
        /// Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
        ///
        /// Normalizes over the last dimension (features), independently per token.
        /// Input shape: [N, T, C] or [N, C] — normalization applied over the last dim C.
        ///
        /// Unlike BatchNorm:
        ///   - No running statistics (no train/eval mode difference)
        ///   - Normalizes over features, not batch
        ///   - Standard for Transformers (used in every GPT layer)
        ///
        /// Auxiliary outputs stored on tape:
        ///   mean    [numRows]   — per-token mean, needed for backward
        ///   invStd  [numRows]   — per-token 1/sqrt(var+eps), needed for backward
        ///
        /// Parallelized via <see cref="OverfitParallel"/> above
        /// <see cref="ParallelThreshold"/> rows. Each row is independent —
        /// no shared writes — so the forward parallelizes trivially.
        /// </summary>
        public static AutogradNode LayerNorm(
            ComputationGraph graph,
            AutogradNode input,
            AutogradNode gamma,
            AutogradNode beta,
            float eps = 1e-5f)
        {
            // Flatten to [numRows, C] for uniform handling of [N,C] and [N,T,C].
            // Normalise over last dimension — works for [N,C] and [N,T,C].
            var C = input.Shape[input.Shape.Rank - 1];
            var numRows = input.Shape.Size / C;

            var requiresGrad = input.RequiresGrad || gamma.RequiresGrad || beta.RequiresGrad;
            var output = AllocateNode(graph, input.Shape, requiresGrad, clearMemory: false);

            // Per-row statistics — GraphAuxiliary, disposed by graph.Reset().
            var mean = AllocateNode(graph, new TensorShape(numRows), requiresGrad: false, clearMemory: false);
            var invStd = AllocateNode(graph, new TensorShape(numRows), requiresGrad: false, clearMemory: false);

            var inS = input.DataView.AsReadOnlySpan();
            var outS = output.DataView.AsSpan();
            var gammaS = gamma.DataView.AsReadOnlySpan();
            var betaS = beta.DataView.AsReadOnlySpan();
            var meanS = mean.DataView.AsSpan();
            var invStdS = invStd.DataView.AsSpan();

            // Parallelize when total work (numRows * C) is large enough to amortize dispatch.
            if ((long)numRows * C < ParallelThreshold)
            {
                LayerNormForwardSeq(inS, outS, gammaS, betaS, meanS, invStdS, numRows, C, eps);
            }
            else
            {
                unsafe
                {
                    fixed (float* inPtr = inS, outPtr = outS,
                                  gPtr = gammaS, bPtr = betaS,
                                  meanPtr = meanS, invStdPtr = invStdS)
                    {
                        var ctx = new LayerNormForwardCtx
                        {
                            Input = inPtr,
                            Output = outPtr,
                            Gamma = gPtr,
                            Beta = bPtr,
                            Mean = meanPtr,
                            InvStd = invStdPtr,
                            C = C,
                            Eps = eps,
                        };
                        OverfitParallel.For(0, numRows, &LayerNormForwardChunk, &ctx);
                    }
                }
            }

            if (requiresGrad)
            {
                graph?.Record(
                    OpCode.LayerNorm, output, input,
                    c0: gamma, c1: beta, c2: mean, c3: invStd,
                    contextCount: 4);
            }
            else
            {
                mean.Dispose();
                invStd.Dispose();
            }

            return output;
        }

        /// <summary>
        /// Layer Normalization backward pass.
        ///
        /// Gradients for input, gamma, beta — computed analytically.
        /// Derivation follows the standard LN backward:
        ///   dL/dx = (1/C) * invStd * (C * dL/dy_hat - sum(dL/dy_hat) - y_hat * sum(dL/dy_hat * y_hat))
        ///   where y_hat = (x - mean) * invStd  (normalised, before gamma/beta)
        ///
        /// <para>
        /// Parallelization (over rows): the per-row dInput computation is
        /// independent so it parallelizes trivially. dGamma[i] and dBeta[i]
        /// are accumulated across ALL rows for each feature i — that's a
        /// shared-write race if rows are split across workers. Fix: each
        /// worker accumulates into its OWN partial buffer
        /// (<c>dGammaPartial</c> / <c>dBetaPartial</c>, sized <c>C × N_workers</c>
        /// and stackalloc'd by the caller), then a single sequential merge
        /// sums partials into the final gradient.
        /// </para>
        /// </summary>
        public static void LayerNormBackward(
            AutogradNode input,
            AutogradNode output,
            AutogradNode gamma,
            AutogradNode beta,
            AutogradNode mean,
            AutogradNode invStd)
        {
            // Normalise over last dimension — works for [N,C] and [N,T,C].
            var C = input.Shape[input.Shape.Rank - 1];
            var numRows = input.Shape.Size / C;

            var inS = input.DataView.AsReadOnlySpan();
            var dOutS = output.GradView.AsReadOnlySpan();
            var gammaS = gamma.DataView.AsReadOnlySpan();
            var meanS = mean.DataView.AsReadOnlySpan();
            var invStdS = invStd.DataView.AsReadOnlySpan();

            var needsDInput = input.RequiresGrad;
            var needsDGamma = gamma.RequiresGrad;
            var needsDBeta = beta.RequiresGrad;

            if ((long)numRows * C < ParallelThreshold)
            {
                LayerNormBackwardSeq(
                    inS, dOutS, gammaS, meanS, invStdS,
                    needsDInput ? input.GradView.AsSpan() : default,
                    needsDGamma ? gamma.GradView.AsSpan() : default,
                    needsDBeta ? beta.GradView.AsSpan() : default,
                    numRows, C,
                    needsDInput, needsDGamma, needsDBeta);
                return;
            }

            // Parallel path. Per-worker partial buffers for dGamma/dBeta avoid
            // shared-write races. Size: WorkerCount × C floats each.
            //
            // stackalloc cap: caller-thread stack ~1 MB. For typical models
            // (C ≤ 4096), partial size = 32 × 4096 × 4 B × 2 = 1 MB, near the
            // edge. We bound C to keep the stack safe; over the bound, fall
            // back to sequential. (Could promote to heap workspace later.)
            const int MaxStackallocC = 4096;

            if (C > MaxStackallocC)
            {
                LayerNormBackwardSeq(
                    inS, dOutS, gammaS, meanS, invStdS,
                    needsDInput ? input.GradView.AsSpan() : default,
                    needsDGamma ? gamma.GradView.AsSpan() : default,
                    needsDBeta ? beta.GradView.AsSpan() : default,
                    numRows, C,
                    needsDInput, needsDGamma, needsDBeta);
                return;
            }

            var workerCount = OverfitParallel.WorkerCount;
            var partialSlots = workerCount * C;

            unsafe
            {
                // Stackalloc per call — zero managed alloc. The buffer lives on
                // this caller's stack; workers index into it by their chunkIdx.
                var dGammaPartial = needsDGamma ? stackalloc float[partialSlots] : default;
                var dBetaPartial = needsDBeta ? stackalloc float[partialSlots] : default;

                if (needsDGamma) { dGammaPartial.Clear(); }
                if (needsDBeta) { dBetaPartial.Clear(); }

                // Chunking math: For(0, numRows, ...) hands each worker a range
                // [chunkStart, chunkEnd). chunkIdx = chunkStart / perChunk
                // gives the slot index into the partial buffers.
                var chunkCount = Math.Min(workerCount, numRows);
                var perChunk = (numRows + chunkCount - 1) / chunkCount;

                fixed (float* inPtr = inS, dOutPtr = dOutS, gPtr = gammaS,
                              meanPtr = meanS, invStdPtr = invStdS,
                              dGammaPartialPtr = dGammaPartial,
                              dBetaPartialPtr = dBetaPartial)
                fixed (float* dInPtr = needsDInput ? input.GradView.AsSpan() : default)
                {
                    var ctx = new LayerNormBackwardCtx
                    {
                        Input = inPtr,
                        GradOutput = dOutPtr,
                        Gamma = gPtr,
                        Mean = meanPtr,
                        InvStd = invStdPtr,
                        GradInput = dInPtr,
                        DGammaPartial = dGammaPartialPtr,
                        DBetaPartial = dBetaPartialPtr,
                        C = C,
                        PerChunk = perChunk,
                        NeedsDInput = needsDInput,
                        NeedsDGamma = needsDGamma,
                        NeedsDBeta = needsDBeta,
                    };

                    OverfitParallel.For(0, numRows, &LayerNormBackwardChunk, &ctx);
                }

                // Merge per-worker partials into final dGamma / dBeta.
                // SIMD via TensorPrimitives.Add — Span<float> implicitly converts
                // to ReadOnlySpan<float> for the input args, no allocation.
                if (needsDGamma)
                {
                    var dGammaFinal = gamma.GradView.AsSpan();
                    for (var w = 0; w < chunkCount; w++)
                    {
                        var slot = dGammaPartial.Slice(w * C, C);
                        TensorPrimitives.Add(dGammaFinal, slot, dGammaFinal);
                    }
                }
                if (needsDBeta)
                {
                    var dBetaFinal = beta.GradView.AsSpan();
                    for (var w = 0; w < chunkCount; w++)
                    {
                        var slot = dBetaPartial.Slice(w * C, C);
                        TensorPrimitives.Add(dBetaFinal, slot, dBetaFinal);
                    }
                }
            }
        }

        // ── Sequential implementations (small tensors / fallback) ─────────────

        private static void LayerNormForwardSeq(
            ReadOnlySpan<float> inS,
            Span<float> outS,
            ReadOnlySpan<float> gammaS,
            ReadOnlySpan<float> betaS,
            Span<float> meanS,
            Span<float> invStdS,
            int numRows, int C, float eps)
        {
            for (var r = 0; r < numRows; r++)
            {
                var row = inS.Slice(r * C, C);
                var outRow = outS.Slice(r * C, C);

                var mu = TensorPrimitives.Sum(row) / C;
                meanS[r] = mu;

                var variance = 0f;
                for (var i = 0; i < C; i++)
                {
                    var diff = row[i] - mu;
                    variance += diff * diff;
                }
                variance /= C;

                var inv = 1f / MathF.Sqrt(variance + eps);
                invStdS[r] = inv;

                for (var i = 0; i < C; i++)
                {
                    outRow[i] = gammaS[i] * ((row[i] - mu) * inv) + betaS[i];
                }
            }
        }

        private static void LayerNormBackwardSeq(
            ReadOnlySpan<float> inS,
            ReadOnlySpan<float> dOutS,
            ReadOnlySpan<float> gammaS,
            ReadOnlySpan<float> meanS,
            ReadOnlySpan<float> invStdS,
            Span<float> dInS,
            Span<float> dGammaS,
            Span<float> dBetaS,
            int numRows, int C,
            bool needsDInput, bool needsDGamma, bool needsDBeta)
        {
            for (var r = 0; r < numRows; r++)
            {
                var row = inS.Slice(r * C, C);
                var dOut = dOutS.Slice(r * C, C);

                var mu = meanS[r];
                var inv = invStdS[r];

                var sumDOut = 0f;
                var sumDOutYh = 0f;

                for (var i = 0; i < C; i++)
                {
                    var yHat = (row[i] - mu) * inv;
                    sumDOut += dOut[i] * gammaS[i];
                    sumDOutYh += dOut[i] * gammaS[i] * yHat;

                    if (needsDBeta) { dBetaS[i] += dOut[i]; }
                    if (needsDGamma) { dGammaS[i] += dOut[i] * yHat; }
                }

                if (needsDInput)
                {
                    var dIn = dInS.Slice(r * C, C);
                    var scale = inv / C;

                    for (var i = 0; i < C; i++)
                    {
                        var yHat = (row[i] - mu) * inv;
                        dIn[i] += scale * (C * dOut[i] * gammaS[i] - sumDOut - yHat * sumDOutYh);
                    }
                }
            }
        }

        // ── Parallel chunk bodies + contexts ──────────────────────────────────

        private unsafe struct LayerNormForwardCtx
        {
            public float* Input;
            public float* Output;
            public float* Gamma;
            public float* Beta;
            public float* Mean;
            public float* InvStd;
            public int C;
            public float Eps;
        }

        private static unsafe void LayerNormForwardChunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<LayerNormForwardCtx>(contextPtr);
            var C = ctx.C;
            var eps = ctx.Eps;
            var gamma = new ReadOnlySpan<float>(ctx.Gamma, C);
            var beta = new ReadOnlySpan<float>(ctx.Beta, C);

            for (var r = chunkStart; r < chunkEnd; r++)
            {
                var row = new ReadOnlySpan<float>(ctx.Input + r * C, C);
                var outRow = new Span<float>(ctx.Output + r * C, C);

                var mu = TensorPrimitives.Sum(row) / C;
                ctx.Mean[r] = mu;

                var variance = 0f;
                for (var i = 0; i < C; i++)
                {
                    var diff = row[i] - mu;
                    variance += diff * diff;
                }
                variance /= C;

                var inv = 1f / MathF.Sqrt(variance + eps);
                ctx.InvStd[r] = inv;

                for (var i = 0; i < C; i++)
                {
                    outRow[i] = gamma[i] * ((row[i] - mu) * inv) + beta[i];
                }
            }
        }

        private unsafe struct LayerNormBackwardCtx
        {
            public float* Input;
            public float* GradOutput;
            public float* Gamma;
            public float* Mean;
            public float* InvStd;
            public float* GradInput;       // null if !needsDInput
            public float* DGammaPartial;   // null if !needsDGamma, size workerCount*C
            public float* DBetaPartial;    // null if !needsDBeta, size workerCount*C
            public int C;
            public int PerChunk;
            public bool NeedsDInput;
            public bool NeedsDGamma;
            public bool NeedsDBeta;
        }

        private static unsafe void LayerNormBackwardChunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<LayerNormBackwardCtx>(contextPtr);
            var C = ctx.C;
            var chunkIdx = ctx.PerChunk > 0 ? chunkStart / ctx.PerChunk : 0;

            // Pointers to this worker's slot in the per-worker partial buffers.
            var dGammaSlot = ctx.NeedsDGamma ? ctx.DGammaPartial + chunkIdx * C : null;
            var dBetaSlot = ctx.NeedsDBeta ? ctx.DBetaPartial + chunkIdx * C : null;

            var gammaSpan = new ReadOnlySpan<float>(ctx.Gamma, C);

            for (var r = chunkStart; r < chunkEnd; r++)
            {
                var row = new ReadOnlySpan<float>(ctx.Input + r * C, C);
                var dOut = new ReadOnlySpan<float>(ctx.GradOutput + r * C, C);

                var mu = ctx.Mean[r];
                var inv = ctx.InvStd[r];

                var sumDOut = 0f;
                var sumDOutYh = 0f;

                for (var i = 0; i < C; i++)
                {
                    var yHat = (row[i] - mu) * inv;
                    sumDOut += dOut[i] * gammaSpan[i];
                    sumDOutYh += dOut[i] * gammaSpan[i] * yHat;

                    if (ctx.NeedsDBeta) { dBetaSlot[i] += dOut[i]; }
                    if (ctx.NeedsDGamma) { dGammaSlot[i] += dOut[i] * yHat; }
                }

                if (ctx.NeedsDInput)
                {
                    var dIn = new Span<float>(ctx.GradInput + r * C, C);
                    var scale = inv / C;

                    for (var i = 0; i < C; i++)
                    {
                        var yHat = (row[i] - mu) * inv;
                        dIn[i] += scale * (C * dOut[i] * gammaSpan[i] - sumDOut - yHat * sumDOutYh);
                    }
                }
            }
        }
    }
}
