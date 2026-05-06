// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

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
            FeedForwardActivation feedForwardActivation = FeedForwardActivation.GeLU)
        {
            if (dModel <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(dModel));
            }

            if (headCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(headCount));
            }

            if (dModel % headCount != 0)
            {
                throw new ArgumentException(
                    $"dModel ({dModel}) must be divisible by headCount ({headCount}).",
                    nameof(dModel));
            }

            if (dFF <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(dFF));
            }

            if (maxSequenceLength <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxSequenceLength));
            }

            if (layerNormEpsilon <= 0f)
            {
                throw new ArgumentOutOfRangeException(nameof(layerNormEpsilon));
            }

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
                maxSequenceLength);

            _feedForward = new CachedFeedForwardBlock(
                dModel,
                dFF,
                feedForwardActivation);

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
            Span<float> output)
        {
            SingleTokenLayerNormKernel.Normalize(
                input,
                weights.Ln1Gamma,
                weights.Ln1Beta,
                _ln1Output,
                DModel,
                LayerNormEpsilon);

            _attention.Decode(
                _ln1Output,
                weights,
                cache,
                layerIndex,
                position,
                _attentionOutput);

            SingleTokenLayerNormKernel.AddResidual(
                input,
                _attentionOutput,
                _afterAttentionResidual,
                DModel);

            SingleTokenLayerNormKernel.Normalize(
                _afterAttentionResidual,
                weights.Ln2Gamma,
                weights.Ln2Beta,
                _ln2Output,
                DModel,
                LayerNormEpsilon);

            _feedForward.Decode(
                _ln2Output,
                weights.FfnW1,
                weights.FfnB1,
                weights.FfnW2,
                weights.FfnB2,
                _feedForwardOutput);

            SingleTokenLayerNormKernel.AddResidual(
                _afterAttentionResidual,
                _feedForwardOutput,
                output,
                DModel);
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


    }
}
