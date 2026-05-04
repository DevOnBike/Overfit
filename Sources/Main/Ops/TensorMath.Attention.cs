// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using System.Numerics.Tensors;
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
            var scale = 1f / MathF.Sqrt(dk);

            if (k.Shape.D0 != batchSize || k.Shape.D1 != seqLen || k.Shape.D2 != dk)
            {
                throw new ArgumentException($"K shape {k.Shape} must match Q shape {q.Shape}.");
            }

            if (v.Shape.D0 != batchSize || v.Shape.D1 != seqLen)
            {
                throw new ArgumentException($"V shape {v.Shape} must have batch={batchSize}, seq={seqLen}.");
            }

            var requiresGrad = q.RequiresGrad || k.RequiresGrad || v.RequiresGrad;

            var attnShape = new TensorShape(batchSize, seqLen, seqLen);
            var attnWeights = AllocateNode(
                graph,
                attnShape,
                requiresGrad: false,
                clearMemory: false);

            var output = AllocateNode(
                graph,
                new TensorShape(batchSize, seqLen, dv),
                requiresGrad,
                clearMemory: false);

            var qS = q.DataView.AsReadOnlySpan();
            var kS = k.DataView.AsReadOnlySpan();
            var vS = v.DataView.AsReadOnlySpan();
            var aS = attnWeights.DataView.AsSpan();
            var oS = output.DataView.AsSpan();

            for (var b = 0; b < batchSize; b++)
            {
                var qBatch = qS.Slice(b * seqLen * dk, seqLen * dk);
                var kBatch = kS.Slice(b * seqLen * dk, seqLen * dk);
                var vBatch = vS.Slice(b * seqLen * dv, seqLen * dv);
                var aBatch = aS.Slice(b * seqLen * seqLen, seqLen * seqLen);
                var oBatch = oS.Slice(b * seqLen * dv, seqLen * dv);

                for (var i = 0; i < seqLen; i++)
                {
                    var qRow = qBatch.Slice(i * dk, dk);
                    var aRow = aBatch.Slice(i * seqLen, seqLen);

                    for (var j = 0; j < seqLen; j++)
                    {
                        if (causalMask && j > i)
                        {
                            aRow[j] = float.NegativeInfinity;
                        }
                        else
                        {
                            aRow[j] =
                                TensorPrimitives.Dot(
                                    qRow,
                                    kBatch.Slice(j * dk, dk)) *
                                scale;
                        }
                    }

                    StableSoftmax(
                        aRow,
                        aRow);
                }

                for (var i = 0; i < seqLen; i++)
                {
                    var aRow = aBatch.Slice(i * seqLen, seqLen);
                    var oRow = oBatch.Slice(i * dv, dv);

                    oRow.Clear();

                    for (var j = 0; j < seqLen; j++)
                    {
                        TensorPrimitives.MultiplyAdd(
                            vBatch.Slice(j * dv, dv),
                            aRow[j],
                            oRow,
                            oRow);
                    }
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
            var batchSize = q.Shape.D0;
            var dv = v.Shape.D2;
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
            int seqLen,
            int dk,
            bool causalMask)
        {
            var batchSize = q.Shape.D0;
            var dv = v.Shape.D2;
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
            int seqLen,
            int dk,
            bool causalMask)
        {
            var batchSize = q.Shape.D0;
            var dv = v.Shape.D2;
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
    }
}
