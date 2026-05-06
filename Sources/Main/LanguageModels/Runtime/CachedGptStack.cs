// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Cached GPT-style transformer stack for single-token autoregressive decode.
    /// </summary>
    public class CachedGptStack
    {
        private readonly CachedTransformerBlock[] _blocks;
        private readonly float[] _currentHidden;
        private readonly float[] _nextHidden;
        private readonly float[] _finalHidden;
        private readonly float[] _lastLogits;

        public CachedGptStack(
            int layerCount,
            int dModel,
            int headCount,
            int dFF,
            int vocabSize,
            int maxSequenceLength,
            float layerNormEpsilon = 1e-5f,
            FeedForwardActivation feedForwardActivation = FeedForwardActivation.GeLU)
        {
            if (layerCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(layerCount));
            }

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

            if (vocabSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(vocabSize));
            }

            if (maxSequenceLength <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxSequenceLength));
            }

            if (layerNormEpsilon <= 0f)
            {
                throw new ArgumentOutOfRangeException(nameof(layerNormEpsilon));
            }

            LayerCount = layerCount;
            DModel = dModel;
            HeadCount = headCount;
            HeadDimension = dModel / headCount;
            DFF = dFF;
            VocabSize = vocabSize;
            MaxSequenceLength = maxSequenceLength;
            LayerNormEpsilon = layerNormEpsilon;
            FeedForwardActivation = feedForwardActivation;

            _blocks = new CachedTransformerBlock[layerCount];

            for (var layer = 0; layer < layerCount; layer++)
            {
                _blocks[layer] = new CachedTransformerBlock(
                    dModel,
                    headCount,
                    dFF,
                    maxSequenceLength,
                    layerNormEpsilon,
                    feedForwardActivation);
            }

            _currentHidden = new float[dModel];
            _nextHidden = new float[dModel];
            _finalHidden = new float[dModel];
            _lastLogits = new float[vocabSize];
        }

        public int LayerCount { get; }

        public int DModel { get; }

        public int HeadCount { get; }

        public int HeadDimension { get; }

        public int DFF { get; }

        public int VocabSize { get; }

        public int MaxSequenceLength { get; }

        public float LayerNormEpsilon { get; }

        public FeedForwardActivation FeedForwardActivation { get; }

        /// <summary>
        /// Decodes one token through all transformer layers using KV-cache.
        /// Zero allocations — all weights accessed via StackWeights references.
        /// </summary>
        internal void Decode(
            ReadOnlySpan<float> inputHidden,
            StackWeights weights,
            KeyValueCache cache,
            int position,
            Span<float> logits)
        {
            if (inputHidden.Length < DModel)
            {
                throw new ArgumentException($"inputHidden length {inputHidden.Length} < DModel {DModel}.");
            }
            if (logits.Length < VocabSize)
            {
                throw new ArgumentException($"logits length {logits.Length} < VocabSize {VocabSize}.");
            }

            inputHidden.Slice(0, DModel).CopyTo(_currentHidden);

            var current = _currentHidden;
            var next = _nextHidden;

            for (var layer = 0; layer < LayerCount; layer++)
            {
                _blocks[layer].Decode(
                    current,
                    in weights.Block(layer),
                    cache,
                    layerIndex: layer,
                    position,
                    next);

                (current, next) = (next, current);
            }

            SingleTokenLayerNormKernel.Normalize(
                current,
                weights.FinalNormGamma,
                weights.FinalNormBeta,
                _finalHidden,
                DModel,
                LayerNormEpsilon);

            SingleTokenProjectionKernel.Project(
                _finalHidden,
                weights.LmHeadWeights,
                ReadOnlySpan<float>.Empty,
                logits,
                DModel,
                VocabSize);

            logits.Slice(0, VocabSize).CopyTo(_lastLogits);
        }

        // Validation helpers removed — StackWeights guarantees correct dimensions
        // by construction (bound directly to GPT1Model parameters).



        public void GetLastFinalHidden(Span<float> destination)
            => _finalHidden.AsSpan(0, DModel).CopyTo(destination);

        public void GetLastLogits(Span<float> destination)
            => _lastLogits.AsSpan(0, VocabSize).CopyTo(destination);

        /// <summary>Exposes internal blocks for testing.</summary>
        internal CachedTransformerBlock[] Blocks => _blocks;
    }
}


