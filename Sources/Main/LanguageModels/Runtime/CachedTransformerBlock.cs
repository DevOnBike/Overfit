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

        public void Decode(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> ln1Gamma,
            ReadOnlySpan<float> ln1Beta,
            IReadOnlyList<float[]> wqHeads,
            IReadOnlyList<float[]> wkHeads,
            IReadOnlyList<float[]> wvHeads,
            IReadOnlyList<float[]> bqHeads,
            IReadOnlyList<float[]> bkHeads,
            IReadOnlyList<float[]> bvHeads,
            IReadOnlyList<float[]> woHeads,
            ReadOnlySpan<float> attentionOutputBias,
            ReadOnlySpan<float> ln2Gamma,
            ReadOnlySpan<float> ln2Beta,
            ReadOnlySpan<float> ffnW1,
            ReadOnlySpan<float> ffnB1,
            ReadOnlySpan<float> ffnW2,
            ReadOnlySpan<float> ffnB2,
            KeyValueCache cache,
            int layerIndex,
            int position,
            Span<float> output)
        {
            ValidateDecodeArguments(
                input,
                ln1Gamma,
                ln1Beta,
                attentionOutputBias,
                ln2Gamma,
                ln2Beta,
                ffnW1,
                ffnB1,
                ffnW2,
                ffnB2,
                cache,
                position,
                output);

            SingleTokenLayerNormKernel.Normalize(
                input,
                ln1Gamma,
                ln1Beta,
                _ln1Output,
                DModel,
                LayerNormEpsilon);

            _attention.Decode(
                _ln1Output,
                wqHeads,
                wkHeads,
                wvHeads,
                bqHeads,
                bkHeads,
                bvHeads,
                woHeads,
                attentionOutputBias,
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
                ln2Gamma,
                ln2Beta,
                _ln2Output,
                DModel,
                LayerNormEpsilon);

            _feedForward.Decode(
                _ln2Output,
                ffnW1,
                ffnB1,
                ffnW2,
                ffnB2,
                _feedForwardOutput);

            SingleTokenLayerNormKernel.AddResidual(
                _afterAttentionResidual,
                _feedForwardOutput,
                output,
                DModel);
        }

        public void DecodeWithoutLayerNormAffine(
            ReadOnlySpan<float> input,
            IReadOnlyList<float[]> wqHeads,
            IReadOnlyList<float[]> wkHeads,
            IReadOnlyList<float[]> wvHeads,
            IReadOnlyList<float[]> bqHeads,
            IReadOnlyList<float[]> bkHeads,
            IReadOnlyList<float[]> bvHeads,
            IReadOnlyList<float[]> woHeads,
            ReadOnlySpan<float> attentionOutputBias,
            ReadOnlySpan<float> ffnW1,
            ReadOnlySpan<float> ffnB1,
            ReadOnlySpan<float> ffnW2,
            ReadOnlySpan<float> ffnB2,
            KeyValueCache cache,
            int layerIndex,
            int position,
            Span<float> output)
        {
            Decode(
                input,
                ReadOnlySpan<float>.Empty,
                ReadOnlySpan<float>.Empty,
                wqHeads,
                wkHeads,
                wvHeads,
                bqHeads,
                bkHeads,
                bvHeads,
                woHeads,
                attentionOutputBias,
                ReadOnlySpan<float>.Empty,
                ReadOnlySpan<float>.Empty,
                ffnW1,
                ffnB1,
                ffnW2,
                ffnB2,
                cache,
                layerIndex,
                position,
                output);
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

        private void ValidateDecodeArguments(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> ln1Gamma,
            ReadOnlySpan<float> ln1Beta,
            ReadOnlySpan<float> attentionOutputBias,
            ReadOnlySpan<float> ln2Gamma,
            ReadOnlySpan<float> ln2Beta,
            ReadOnlySpan<float> ffnW1,
            ReadOnlySpan<float> ffnB1,
            ReadOnlySpan<float> ffnW2,
            ReadOnlySpan<float> ffnB2,
            KeyValueCache cache,
            int position,
            Span<float> output)
        {
            if (input.Length < DModel)
            {
                throw new ArgumentException("Input span is smaller than dModel.", nameof(input));
            }

            if (!ln1Gamma.IsEmpty && ln1Gamma.Length < DModel)
            {
                throw new ArgumentException("LN1 gamma span is smaller than dModel.", nameof(ln1Gamma));
            }

            if (!ln1Beta.IsEmpty && ln1Beta.Length < DModel)
            {
                throw new ArgumentException("LN1 beta span is smaller than dModel.", nameof(ln1Beta));
            }

            if (!attentionOutputBias.IsEmpty && attentionOutputBias.Length < DModel)
            {
                throw new ArgumentException("Attention output bias span is smaller than dModel.", nameof(attentionOutputBias));
            }

            if (!ln2Gamma.IsEmpty && ln2Gamma.Length < DModel)
            {
                throw new ArgumentException("LN2 gamma span is smaller than dModel.", nameof(ln2Gamma));
            }

            if (!ln2Beta.IsEmpty && ln2Beta.Length < DModel)
            {
                throw new ArgumentException("LN2 beta span is smaller than dModel.", nameof(ln2Beta));
            }

            if (ffnW1.Length < DModel * DFF)
            {
                throw new ArgumentException("FFN W1 span is smaller than dModel * dFF.", nameof(ffnW1));
            }

            if (!ffnB1.IsEmpty && ffnB1.Length < DFF)
            {
                throw new ArgumentException("FFN B1 span is smaller than dFF.", nameof(ffnB1));
            }

            if (ffnW2.Length < DFF * DModel)
            {
                throw new ArgumentException("FFN W2 span is smaller than dFF * dModel.", nameof(ffnW2));
            }

            if (!ffnB2.IsEmpty && ffnB2.Length < DModel)
            {
                throw new ArgumentException("FFN B2 span is smaller than dModel.", nameof(ffnB2));
            }

            if (cache is null)
            {
                throw new ArgumentNullException(nameof(cache));
            }

            if (cache.Shape.HeadCount < HeadCount)
            {
                throw new ArgumentException("Cache head count is smaller than block head count.", nameof(cache));
            }

            if (cache.Shape.HeadDimension != HeadDimension)
            {
                throw new ArgumentException("Cache head dimension does not match block head dimension.", nameof(cache));
            }

            if (position < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(position));
            }

            if (position >= MaxSequenceLength)
            {
                throw new ArgumentOutOfRangeException(nameof(position));
            }

            if (position >= cache.CurrentLength)
            {
                throw new InvalidOperationException(
                    $"Position {position} is not visible in the cache. CurrentLength={cache.CurrentLength}. Advance the cache before Decode.");
            }

            if (output.Length < DModel)
            {
                throw new ArgumentException("Output span is smaller than dModel.", nameof(output));
            }
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
