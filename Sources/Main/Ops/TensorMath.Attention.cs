// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.LanguageModels.Experimental;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        private const long AttentionParallelWorkThreshold = 250_000;

        /// <summary>
        /// Scaled Dot-Product Attention.
        ///
        /// Formula:
        /// S = Q @ K^T / sqrt(d_k) [B, T, T]
        /// A = softmax(S, axis=-1) [B, T, T] (with optional causal mask)
        /// O = A @ V [B, T, d_v]
        /// </summary>
        public static AutogradNode ScaledDotProductAttention(
            ComputationGraph graph,
            AutogradNode q,
            AutogradNode k,
            AutogradNode v,
            bool causalMask = true)
        {
            var batchSize = q.Shape.D0;
            var seqLen = q.Shape.D1;
            var dk = q.Shape.D2;
            var dv = v.Shape.D2;

            if (k.Shape.D0 != batchSize || k.Shape.D1 != seqLen || k.Shape.D2 != dk)
            {
                throw new ArgumentException($"K shape {k.Shape} must match Q shape {q.Shape}.");
            }

            if (v.Shape.D0 != batchSize || v.Shape.D1 != seqLen)
            {
                throw new ArgumentException($"V shape {v.Shape} must have batch={batchSize}, seq={seqLen}.");
            }

            return ScaledDotProductAttentionCore(
                graph,
                q,
                k,
                v,
                batchSize,
                seqLen,
                dk,
                dv,
                causalMask,
                outputShape: new TensorShape(batchSize, seqLen, dv),
                attnShape: new TensorShape(batchSize, seqLen, seqLen));
        }

        /// <summary>
        /// Flattened (<c>[B*T, d]</c>) overload of Scaled Dot-Product Attention.
        ///
        /// Q/K/V arrive as 2-D <c>[batchSize*seqLen, d]</c> nodes — the layout
        /// produced directly by <c>graph.Linear</c> projections — so callers can
        /// skip the <c>[B*T,d] &lt;-&gt; [B,T,d]</c> reshape round-trip around
        /// attention. The output is returned 2-D <c>[batchSize*seqLen, dv]</c>,
        /// ready to feed straight into the next <c>graph.Linear</c>.
        ///
        /// The data layout is byte-identical to the 3-D form (row-major,
        /// batch-major); this is purely a shape-bookkeeping difference. The
        /// forward/backward kernels operate on flat spans and are rank-agnostic.
        /// </summary>
        public static AutogradNode ScaledDotProductAttention(
            ComputationGraph graph,
            AutogradNode q,
            AutogradNode k,
            AutogradNode v,
            int batchSize,
            int seqLen,
            bool causalMask = true)
        {
            var rows = batchSize * seqLen;
            var dk = q.Shape.D1;
            var dv = v.Shape.D1;

            if (q.Shape.D0 != rows)
            {
                throw new ArgumentException(
                    $"Q shape {q.Shape} must have D0=batchSize*seqLen={rows}.");
            }

            if (k.Shape.D0 != rows || k.Shape.D1 != dk)
            {
                throw new ArgumentException($"K shape {k.Shape} must match Q shape {q.Shape}.");
            }

            if (v.Shape.D0 != rows)
            {
                throw new ArgumentException(
                    $"V shape {v.Shape} must have D0=batchSize*seqLen={rows}.");
            }

            return ScaledDotProductAttentionCore(
                graph,
                q,
                k,
                v,
                batchSize,
                seqLen,
                dk,
                dv,
                causalMask,
                outputShape: new TensorShape(rows, dv),
                attnShape: new TensorShape(rows, seqLen));
        }

        private static AutogradNode ScaledDotProductAttentionCore(
            ComputationGraph graph,
            AutogradNode q,
            AutogradNode k,
            AutogradNode v,
            int batchSize,
            int seqLen,
            int dk,
            int dv,
            bool causalMask,
            TensorShape outputShape,
            TensorShape attnShape)
        {
            var scale = 1f / MathF.Sqrt(dk);

            var requiresGrad = q.RequiresGrad || k.RequiresGrad || v.RequiresGrad;

            var attnWeights = AllocateNode(
                graph,
                attnShape,
                requiresGrad: false,
                clearMemory: false);

            var output = AllocateNode(
                graph,
                outputShape,
                requiresGrad,
                clearMemory: false);

            var qS = q.DataView.AsReadOnlySpan();
            var kS = k.DataView.AsReadOnlySpan();
            var vS = v.DataView.AsReadOnlySpan();
            var aS = attnWeights.DataView.AsSpan();
            var oS = output.DataView.AsSpan();

            // Each batch is independent (only writes to per-batch slices of
            // attnWeights/output). Parallel-over-batch when there's enough
            // work to amortize the OverfitParallelFor dispatch (~5 µs warm).
            //
            // Symmetric guard to ScaledDotProductAttentionBackwardParallel:
            // batchSize > 1 AND total work above threshold.
            var work = (long)batchSize * seqLen * seqLen * Math.Max(dk, dv);

            if (batchSize > 1 && work >= AttentionParallelWorkThreshold)
            {
                unsafe
                {
                    fixed (float* qPtr = qS, kPtr = kS, vPtr = vS,
                                  aPtr = aS, oPtr = oS)
                    {
                        var ctx = new SDPAForwardCtx
                        {
                            Q = qPtr,
                            K = kPtr,
                            V = vPtr,
                            A = aPtr,
                            O = oPtr,
                            SeqLen = seqLen,
                            Dk = dk,
                            Dv = dv,
                            Scale = scale,
                            CausalMask = causalMask,
                        };
                        OverfitParallelFor.For(0, batchSize, &SDPAForwardChunk, &ctx);
                    }
                }
            }
            else
            {
                for (var b = 0; b < batchSize; b++)
                {
                    SDPAForwardBatch(b, qS, kS, vS, aS, oS, seqLen, dk, dv, scale, causalMask);
                }
            }

            if (requiresGrad)
            {
                graph?.Record(
                    OpCode.ScaledDotProductAttention,
                    output,
                    q,
                    k,
                    c0: v,
                    c1: attnWeights,
                    i0: seqLen,
                    i1: dk,
                    i2: causalMask ? 1 : 0);
            }
            else
            {
                attnWeights.Dispose();
            }

            return output;
        }

        /// <summary>
        /// Scaled Dot-Product Attention backward pass.
        ///
        /// Stable default: sequential backward path.
        /// Experimental path: parallel backward is used only when
        /// ExperimentalLanguageModelOptions.EnableParallelAttentionBackward is true.
        /// </summary>
        public static void ScaledDotProductAttentionBackward(
            AutogradNode q,
            AutogradNode k,
            AutogradNode v,
            AutogradNode attnWeights,
            AutogradNode output,
            int seqLen,
            int dk,
            bool causalMask)
        {
            // Rank-agnostic dim recovery. q/k/v may be 3-D [B,T,d] (the [B,T,d]
            // overload) or 2-D [B*T,d] (the flattened overload) — the backward
            // kernels slice flat either way. Total element count plus seqLen and
            // dk fully determine batchSize and dv regardless of node rank.
            var batchSize = q.Shape.Size / (seqLen * dk);
            var dv = v.Shape.Size / (batchSize * seqLen);
            var work = (long)batchSize * seqLen * seqLen * Math.Max(dk, dv);

            if (ExperimentalLanguageModelOptions.EnableParallelAttentionBackward &&
                batchSize > 1 &&
                work >= AttentionParallelWorkThreshold)
            {
                ScaledDotProductAttentionBackwardParallel(
                    q,
                    k,
                    v,
                    attnWeights,
                    output,
                    batchSize,
                    dv,
                    seqLen,
                    dk,
                    causalMask);

                return;
            }

            ScaledDotProductAttentionBackwardSequential(
                q,
                k,
                v,
                attnWeights,
                output,
                batchSize,
                dv,
                seqLen,
                dk,
                causalMask);
        }

        private static void ScaledDotProductAttentionBackwardParallel(
            AutogradNode q,
            AutogradNode k,
            AutogradNode v,
            AutogradNode attnWeights,
            AutogradNode output,
            int batchSize,
            int dv,
            int seqLen,
            int dk,
            bool causalMask)
        {
            var bufferLength = seqLen * seqLen;

            Parallel.For(0, batchSize, OverfitParallel.Options, b =>
            {
                var dAArray = ArrayPool<float>.Shared.Rent(bufferLength);
                var dSArray = ArrayPool<float>.Shared.Rent(bufferLength);

                try
                {
                    var dA = dAArray.AsSpan(0, bufferLength);
                    var dS = dSArray.AsSpan(0, bufferLength);

                    ScaledDotProductAttentionBackwardBatch(
                        b,
                        q,
                        k,
                        v,
                        attnWeights,
                        output,
                        dA,
                        dS,
                        seqLen,
                        dk,
                        dv,
                        causalMask);
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(dSArray);
                    ArrayPool<float>.Shared.Return(dAArray);
                }
            });
        }

        private static void ScaledDotProductAttentionBackwardSequential(
            AutogradNode q,
            AutogradNode k,
            AutogradNode v,
            AutogradNode attnWeights,
            AutogradNode output,
            int batchSize,
            int dv,
            int seqLen,
            int dk,
            bool causalMask)
        {
            var bufferLength = seqLen * seqLen;

            var dAArray = ArrayPool<float>.Shared.Rent(bufferLength);
            var dSArray = ArrayPool<float>.Shared.Rent(bufferLength);

            try
            {
                var dA = dAArray.AsSpan(0, bufferLength);
                var dS = dSArray.AsSpan(0, bufferLength);

                for (var b = 0; b < batchSize; b++)
                {
                    ScaledDotProductAttentionBackwardBatch(
                        b,
                        q,
                        k,
                        v,
                        attnWeights,
                        output,
                        dA,
                        dS,
                        seqLen,
                        dk,
                        dv,
                        causalMask);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(dSArray);
                ArrayPool<float>.Shared.Return(dAArray);
            }
        }

        private static void ScaledDotProductAttentionBackwardBatch(
            int batchIndex,
            AutogradNode q,
            AutogradNode k,
            AutogradNode v,
            AutogradNode attnWeights,
            AutogradNode output,
            Span<float> dA,
            Span<float> dS,
            int seqLen,
            int dk,
            int dv,
            bool causalMask)
        {
            var scale = 1f / MathF.Sqrt(dk);

            var dO = output.GradView.AsReadOnlySpan();
            var aS = attnWeights.DataView.AsReadOnlySpan();
            var vS = v.DataView.AsReadOnlySpan();
            var qS = q.DataView.AsReadOnlySpan();
            var kS = k.DataView.AsReadOnlySpan();

            var dOBatch = dO.Slice(batchIndex * seqLen * dv, seqLen * dv);
            var aBatch = aS.Slice(batchIndex * seqLen * seqLen, seqLen * seqLen);
            var vBatch = vS.Slice(batchIndex * seqLen * dv, seqLen * dv);
            var qBatch = qS.Slice(batchIndex * seqLen * dk, seqLen * dk);
            var kBatch = kS.Slice(batchIndex * seqLen * dk, seqLen * dk);

            // dV = A^T @ dO
            if (v.RequiresGrad)
            {
                var dVBatch = v.GradView.AsSpan()
                    .Slice(batchIndex * seqLen * dv, seqLen * dv);

                for (var j = 0; j < seqLen; j++)
                {
                    var dVRow = dVBatch.Slice(j * dv, dv);

                    for (var i = 0; i < seqLen; i++)
                    {
                        var aij = aBatch[i * seqLen + j];

                        if (aij == 0f)
                        {
                            continue;
                        }

                        TensorPrimitives.MultiplyAdd(
                            dOBatch.Slice(i * dv, dv),
                            aij,
                            dVRow,
                            dVRow);
                    }
                }
            }

            // dA = dO @ V^T
            for (var i = 0; i < seqLen; i++)
            {
                var dORow = dOBatch.Slice(i * dv, dv);
                var dARow = dA.Slice(i * seqLen, seqLen);

                for (var j = 0; j < seqLen; j++)
                {
                    if (causalMask && j > i)
                    {
                        dARow[j] = 0f;
                    }
                    else
                    {
                        dARow[j] = TensorPrimitives.Dot(
                            dORow,
                            vBatch.Slice(j * dv, dv));
                    }
                }
            }

            // dS = softmax_backward(dA, A)
            for (var i = 0; i < seqLen; i++)
            {
                var aRow = aBatch.Slice(i * seqLen, seqLen);
                var dARow = dA.Slice(i * seqLen, seqLen);
                var dSRow = dS.Slice(i * seqLen, seqLen);

                var dot = TensorPrimitives.Dot(
                    dARow,
                    aRow);

                for (var j = 0; j < seqLen; j++)
                {
                    dSRow[j] = aRow[j] * (dARow[j] - dot);
                }
            }

            // dQ = dS @ K / sqrt(dk)
            if (q.RequiresGrad)
            {
                var dQBatch = q.GradView.AsSpan()
                    .Slice(batchIndex * seqLen * dk, seqLen * dk);

                for (var i = 0; i < seqLen; i++)
                {
                    var dSRow = dS.Slice(i * seqLen, seqLen);
                    var dQRow = dQBatch.Slice(i * dk, dk);

                    for (var j = 0; j < seqLen; j++)
                    {
                        var dsij = dSRow[j];

                        if (dsij == 0f)
                        {
                            continue;
                        }

                        TensorPrimitives.MultiplyAdd(
                            kBatch.Slice(j * dk, dk),
                            dsij * scale,
                            dQRow,
                            dQRow);
                    }
                }
            }

            // dK = dS^T @ Q / sqrt(dk)
            if (k.RequiresGrad)
            {
                var dKBatch = k.GradView.AsSpan()
                    .Slice(batchIndex * seqLen * dk, seqLen * dk);

                for (var j = 0; j < seqLen; j++)
                {
                    var dKRow = dKBatch.Slice(j * dk, dk);

                    for (var i = 0; i < seqLen; i++)
                    {
                        var dsij = dS[i * seqLen + j];

                        if (dsij == 0f)
                        {
                            continue;
                        }

                        TensorPrimitives.MultiplyAdd(
                            qBatch.Slice(i * dk, dk),
                            dsij * scale,
                            dKRow,
                            dKRow);
                    }
                }
            }
        }

        // ── SDPA forward — per-batch + parallel chunk dispatch ────────────────

        /// <summary>
        /// Forward pass for one batch slice of SDPA. Independent of other batches —
        /// only writes to its own [seqLen × seqLen] slice of <paramref name="aS"/>
        /// and [seqLen × dv] slice of <paramref name="oS"/>.
        /// </summary>
        private static void SDPAForwardBatch(
            int b,
            ReadOnlySpan<float> qS,
            ReadOnlySpan<float> kS,
            ReadOnlySpan<float> vS,
            Span<float> aS,
            Span<float> oS,
            int seqLen, int dk, int dv,
            float scale, bool causalMask)
        {
            var qBatch = qS.Slice(b * seqLen * dk, seqLen * dk);
            var kBatch = kS.Slice(b * seqLen * dk, seqLen * dk);
            var vBatch = vS.Slice(b * seqLen * dv, seqLen * dv);
            var aBatch = aS.Slice(b * seqLen * seqLen, seqLen * seqLen);
            var oBatch = oS.Slice(b * seqLen * dv, seqLen * dv);

            // Causal: query row i only attends to keys [0, i]. The masked
            // suffix [i+1, seqLen) has a fixed post-softmax value of 0
            // (softmax of -inf). Computing softmax + A·V over the full row
            // wastes ~half the inner work averaged across rows — instead we
            // run softmax + accumulation over the valid prefix only and zero
            // the masked suffix directly. Bit-identical to the −inf round
            // trip (the masked entries are 0 either way), and backward is
            // consistent (`SDPABackwardBatch` skips `aij == 0`).
            for (var i = 0; i < seqLen; i++)
            {
                var qRow = qBatch.Slice(i * dk, dk);
                var aRow = aBatch.Slice(i * seqLen, seqLen);
                var validLen = causalMask ? i + 1 : seqLen;

                for (var j = 0; j < validLen; j++)
                {
                    aRow[j] = TensorPrimitives.Dot(qRow, kBatch.Slice(j * dk, dk)) * scale;
                }

                if (validLen < seqLen)
                {
                    aRow.Slice(validLen).Clear();
                }

                var aValid = aRow.Slice(0, validLen);
                StableSoftmax(aValid, aValid);
            }

            for (var i = 0; i < seqLen; i++)
            {
                var aRow = aBatch.Slice(i * seqLen, seqLen);
                var oRow = oBatch.Slice(i * dv, dv);
                var validLen = causalMask ? i + 1 : seqLen;
                oRow.Clear();

                for (var j = 0; j < validLen; j++)
                {
                    TensorPrimitives.MultiplyAdd(vBatch.Slice(j * dv, dv), aRow[j], oRow, oRow);
                }
            }
        }

        private unsafe struct SDPAForwardCtx
        {
            public float* Q;
            public float* K;
            public float* V;
            public float* A;
            public float* O;
            public int SeqLen;
            public int Dk;
            public int Dv;
            public float Scale;
            public bool CausalMask;
        }

        private static unsafe void SDPAForwardChunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<SDPAForwardCtx>(contextPtr);

            // Re-create spans from raw pointers + the OUTER batch×... lengths.
            // (Caller pinned the underlying arrays with `fixed`, so these
            // pointers stay valid for the duration of the OverfitParallelFor call.)
            //
            // We need to know the OUTER buffer length so the per-batch slicing
            // inside SDPAForwardBatch doesn't trip span bounds — chunkEnd is
            // already the exclusive end of the batch range, so the largest
            // batch we'll touch is `chunkEnd - 1`. Buffer must cover up to
            // `chunkEnd * seqLen * d`.
            var seqLen = ctx.SeqLen;
            var dk = ctx.Dk;
            var dv = ctx.Dv;

            var qLen = chunkEnd * seqLen * dk;
            var kLen = chunkEnd * seqLen * dk;
            var vLen = chunkEnd * seqLen * dv;
            var aLen = chunkEnd * seqLen * seqLen;
            var oLen = chunkEnd * seqLen * dv;

            var qS = new ReadOnlySpan<float>(ctx.Q, qLen);
            var kS = new ReadOnlySpan<float>(ctx.K, kLen);
            var vS = new ReadOnlySpan<float>(ctx.V, vLen);
            var aS = new Span<float>(ctx.A, aLen);
            var oS = new Span<float>(ctx.O, oLen);

            for (var b = chunkStart; b < chunkEnd; b++)
            {
                SDPAForwardBatch(b, qS, kS, vS, aS, oS, seqLen, dk, dv, ctx.Scale, ctx.CausalMask);
            }
        }
    }
}
