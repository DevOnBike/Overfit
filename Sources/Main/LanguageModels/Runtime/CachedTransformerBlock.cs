// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Rope;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Cached single-token transformer block for autoregressive decode.
    ///
    /// This composes the runtime building blocks built so far:
    ///
    /// Pre-LN transformer block:
    ///
    ///   ln1 = LayerNorm(input)
    ///   attn = CachedMultiHeadAttention(ln1)
    ///   x = input + attn
    ///   ln2 = LayerNorm(x)
    ///   ffn = FeedForward(ln2)
    ///   output = x + ffn
    ///
    /// Scope:
    /// - batch = 1,
    /// - one token,
    /// - one transformer layer,
    /// - FP32,
    /// - Pre-LN only,
    /// - caller-owned output,
    /// - cache lifetime and position are controlled by caller.
    ///
    /// Important:
    /// The caller must call cache.Advance() before decoding the new position.
    /// The cache length must already include the position being decoded.
    /// </summary>
    public class CachedTransformerBlock
    {
        private readonly CachedMultiHeadAttention _attention;
        private readonly CachedFeedForwardBlock _feedForward;
        private readonly Qwen2MoeFeedForwardBlock? _moe;   // non-null only for MoE blocks

        private readonly float[] _ln1Output;
        private readonly float[] _attentionOutput;
        private readonly float[] _afterAttentionResidual;
        private readonly float[] _ln2Output;
        private readonly float[] _feedForwardOutput;

        public CachedTransformerBlock(
            int dModel,
            int headCount,
            int dFF,
            int maxSequenceLength,
            float layerNormEpsilon = 1e-5f,
            FeedForwardActivation feedForwardActivation = FeedForwardActivation.GeLU,
            int kvHeadCount = 0,
            int expertCount = 0,
            int expertUsedCount = 0,
            int expertFeedForwardLength = 0,
            bool normalizeExpertWeights = true,
            bool hasSharedExpert = true)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dModel);

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headCount);

            if (dModel % headCount != 0)
            {
                throw new ArgumentException(
                    $"dModel ({dModel}) must be divisible by headCount ({headCount}).",
                    nameof(dModel));
            }

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dFF);

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSequenceLength);

            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(layerNormEpsilon, 0f);

            DModel = dModel;
            HeadCount = headCount;
            HeadDimension = dModel / headCount;
            DFF = dFF;
            MaxSequenceLength = maxSequenceLength;
            LayerNormEpsilon = layerNormEpsilon;
            FeedForwardActivation = feedForwardActivation;

            _attention = new CachedMultiHeadAttention(
                dModel,
                headCount,
                maxSequenceLength,
                kvHeadCount: kvHeadCount > 0 ? kvHeadCount : headCount);

            _feedForward = new CachedFeedForwardBlock(
                dModel,
                dFF,
                feedForwardActivation);

            // MoE: routed experts (expertFeedForwardLength) + (Qwen-MoE) a sigmoid-gated shared expert
            // of length dFF. Mixtral has no shared expert — pass shared length 0. Replaces the dense
            // FFN at decode when the block's weights are MoE.
            _moe = expertCount > 0
                ? new Qwen2MoeFeedForwardBlock(
                    dModel,
                    expertFeedForwardLength > 0 ? expertFeedForwardLength : dFF,
                    hasSharedExpert ? dFF : 0,
                    expertCount,
                    expertUsedCount,
                    normalizeExpertWeights)
                : null;

            _ln1Output = new float[dModel];
            _attentionOutput = new float[dModel];
            _afterAttentionResidual = new float[dModel];
            _ln2Output = new float[dModel];
            _feedForwardOutput = new float[dModel];
        }

        public int DModel { get; }

        public int HeadCount { get; }

        public int HeadDimension { get; }

        public int DFF { get; }

        public int MaxSequenceLength { get; }

        public float LayerNormEpsilon { get; }

        public FeedForwardActivation FeedForwardActivation { get; }

        internal void Decode(
            ReadOnlySpan<float> input,
            in BlockWeights weights,
            KeyValueCache cache,
            int layerIndex,
            int position,
            Span<float> output,
            RopeTable? rope = null)
        {
            // Per-token shape contract — fail loudly on caller-side bugs so the
            // downstream kernels don't surface them as IndexOutOfRangeException
            // half-way through the block. Cost: four int compares per token.
            if (input.Length != DModel)
            {
                throw new ArgumentException(
                    $"input.Length must equal DModel ({DModel}), got {input.Length}.",
                    nameof(input));
            }
            if (output.Length != DModel)
            {
                throw new ArgumentException(
                    $"output.Length must equal DModel ({DModel}), got {output.Length}.",
                    nameof(output));
            }
            // Dense-FFN shape guards — skipped for MoE blocks (their FFN weights live in the
            // expert arrays + shared expert, not FfnW1/FfnW2, which are intentionally empty).
            if (!weights.IsMoe)
            {
                if (weights.FfnW1.ElementCount != (long)DModel * DFF)
                {
                    throw new ArgumentException(
                        $"weights.FfnW1 must hold DModel*DFF ({DModel * DFF}) elements, got {weights.FfnW1.ElementCount}.",
                        nameof(weights));
                }
                if (weights.FfnW2.ElementCount != (long)DFF * DModel)
                {
                    throw new ArgumentException(
                        $"weights.FfnW2 must hold DFF*DModel ({DFF * DModel}) elements, got {weights.FfnW2.ElementCount}.",
                        nameof(weights));
                }
            }

            // Llama/Qwen: RMSNorm when beta is empty; GPT-2: standard LayerNorm
            if (weights.Ln1Beta.IsEmpty)
            {
                RmsNormalize(input, weights.Ln1Gamma, _ln1Output, DModel, LayerNormEpsilon);
            }
            else
            {
                SingleTokenLayerNormKernel.Normalize(
                input, weights.Ln1Gamma, weights.Ln1Beta, _ln1Output, DModel, LayerNormEpsilon);
            }

            _attention.Decode(
                _ln1Output,
                weights,
                cache,
                layerIndex,
                position,
                _attentionOutput,
                rope);

            SingleTokenLayerNormKernel.AddResidual(
                input,
                _attentionOutput,
                _afterAttentionResidual,
                DModel);

            if (weights.Ln2Beta.IsEmpty)
            {
                RmsNormalize(_afterAttentionResidual, weights.Ln2Gamma, _ln2Output, DModel, LayerNormEpsilon);
            }
            else
            {
                SingleTokenLayerNormKernel.Normalize(
                _afterAttentionResidual, weights.Ln2Gamma, weights.Ln2Beta, _ln2Output, DModel, LayerNormEpsilon);
            }

            // MoE (qwen2moe): routed experts + sigmoid-gated shared expert.
            if (weights.IsMoe)
            {
                _moe!.Decode(
                    _ln2Output,
                    weights.MoeRouter,
                    weights.MoeGate,
                    weights.MoeUp,
                    weights.MoeDown,
                    weights.MoeSharedGateInp,
                    weights.MoeSharedGate,
                    weights.MoeSharedUp,
                    weights.MoeSharedDown,
                    _feedForwardOutput);
            }
            // SwiGLU (Llama/Mistral/Qwen): FfnGate is present.
            // GeLU/ReLU (GPT-1/GPT-2): FfnGate is empty.
            else if (!weights.FfnGate.IsEmpty)
            {
                // SwiGLU (Llama/Mistral/Qwen): per-weight dispatch — each of
                // gate/up/down picks its kernel from its resident format
                // (F32 / Q8_0 / Q4_K), handling heterogeneous K-quant files.
                _feedForward.DecodeSwiGluDispatched(
                    _ln2Output,
                    weights.FfnGate,
                    weights.FfnW1,
                    weights.FfnW2,
                    _feedForwardOutput);
            }
            else
            {
                _feedForward.Decode(
                    _ln2Output,
                    weights.FfnW1.F32,
                    weights.FfnB1,
                    weights.FfnW2.F32,
                    weights.FfnB2,
                    _feedForwardOutput);
            }

            SingleTokenLayerNormKernel.AddResidual(
                _afterAttentionResidual,
                _feedForwardOutput,
                output,
                DModel);
        }

        /// <summary>
        /// Batched (prefill) Pre-LN block forward over <paramref name="rows"/> tokens —
        /// the multi-row counterpart of <see cref="Decode"/>. Per-row LayerNorm
        /// (<see cref="SingleTokenLayerNormKernel"/>) → batched MHA
        /// (<see cref="CachedMultiHeadAttention.DecodeBatched"/>) → residual → LayerNorm
        /// → batched FFN (<see cref="CachedFeedForwardBlock.DecodeBatched"/>) → residual.
        /// Composes the per-row-independent ops with the two verified batched blocks,
        /// so the result is <b>bit-identical</b> to N× <see cref="Decode"/>. Scoped to
        /// the F32 / GPT-2 path: standard LayerNorm (not RMSNorm), GeLU/ReLU FFN (not
        /// SwiGLU), MHA (not GQA), no RoPE — throws otherwise. The cache must already
        /// be advanced to length <c>basePosition + rows</c>.
        /// </summary>
        internal void DecodeBatched(
            ReadOnlySpan<float> input,
            int rows,
            in BlockWeights weights,
            KeyValueCache cache,
            int layerIndex,
            int basePosition,
            Span<float> output,
            RopeTable? rope = null)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);

            if (weights.Ln1Beta.IsEmpty || weights.Ln2Beta.IsEmpty)
            {
                throw new NotSupportedException("Batched prefill supports standard LayerNorm only (RMSNorm is a follow-on).");
            }
            if (!weights.FfnGate.IsEmpty)
            {
                throw new NotSupportedException("Batched prefill supports GeLU/ReLU FFN only (SwiGLU is a follow-on).");
            }

            var dModel = DModel;

            // Per-prefill scratch (one-time pass, not the 0-alloc decode hot path).
            var ln1 = new float[rows * dModel];
            var attnOut = new float[rows * dModel];
            var afterAttn = new float[rows * dModel];
            var ln2 = new float[rows * dModel];
            var ffnOut = new float[rows * dModel];

            for (var n = 0; n < rows; n++)
            {
                SingleTokenLayerNormKernel.Normalize(
                    input.Slice(n * dModel, dModel), weights.Ln1Gamma, weights.Ln1Beta,
                    ln1.AsSpan(n * dModel, dModel), dModel, LayerNormEpsilon);
            }

            _attention.DecodeBatched(ln1, rows, in weights, cache, layerIndex, basePosition, attnOut, rope);

            for (var n = 0; n < rows; n++)
            {
                SingleTokenLayerNormKernel.AddResidual(
                    input.Slice(n * dModel, dModel), attnOut.AsSpan(n * dModel, dModel),
                    afterAttn.AsSpan(n * dModel, dModel), dModel);
            }

            for (var n = 0; n < rows; n++)
            {
                SingleTokenLayerNormKernel.Normalize(
                    afterAttn.AsSpan(n * dModel, dModel), weights.Ln2Gamma, weights.Ln2Beta,
                    ln2.AsSpan(n * dModel, dModel), dModel, LayerNormEpsilon);
            }

            _feedForward.DecodeBatched(
                ln2, rows, weights.FfnW1.F32, weights.FfnB1, weights.FfnW2.F32, weights.FfnB2, ffnOut);

            for (var n = 0; n < rows; n++)
            {
                SingleTokenLayerNormKernel.AddResidual(
                    afterAttn.AsSpan(n * dModel, dModel), ffnOut.AsSpan(n * dModel, dModel),
                    output.Slice(n * dModel, dModel), dModel);
            }
        }




        public void GetLastLayerNorm1Output(Span<float> destination)
        {
            CopyDModelBuffer(_ln1Output, destination, nameof(destination));
        }

        public void GetLastAttentionOutput(Span<float> destination)
        {
            CopyDModelBuffer(_attentionOutput, destination, nameof(destination));
        }

        public void GetLastAfterAttentionResidual(Span<float> destination)
        {
            CopyDModelBuffer(_afterAttentionResidual, destination, nameof(destination));
        }

        public void GetLastLayerNorm2Output(Span<float> destination)
        {
            CopyDModelBuffer(_ln2Output, destination, nameof(destination));
        }

        public void GetLastFeedForwardOutput(Span<float> destination)
        {
            CopyDModelBuffer(_feedForwardOutput, destination, nameof(destination));
        }



        private void CopyDModelBuffer(
            float[] source,
            Span<float> destination,
            string parameterName)
        {
            if (destination.Length < DModel)
            {
                throw new ArgumentException("Destination span is smaller than dModel.", parameterName);
            }

            source.AsSpan().CopyTo(destination);
        }


        // RMSNorm: x / sqrt(mean(x²) + eps) * gamma  (no mean subtraction)
        private static void RmsNormalize(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> gamma,
            Span<float> output,
            int dModel,
            float eps)
        {
            var sumSq = 0f;
            for (var i = 0; i < dModel; i++)
            {
                sumSq += input[i] * input[i];
            }
            var scale = 1f / MathF.Sqrt(sumSq / dModel + eps);
            if (gamma.IsEmpty)
            {
                for (var i = 0; i < dModel; i++)
                {
                    output[i] = input[i] * scale;
                }
            }
            else
            {
                for (var i = 0; i < dModel; i++)
                {
                    output[i] = input[i] * scale * gamma[i];
                }
            }
        }


    }
}
