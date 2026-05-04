// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Cached GPT-style transformer stack for single-token autoregressive decode.
    ///
    /// This composes:
    ///
    /// - N cached transformer blocks,
    /// - final LayerNorm,
    /// - LM head projection.
    ///
    /// Scope:
    /// - batch = 1,
    /// - one token,
    /// - FP32,
    /// - Pre-LN transformer blocks,
    /// - caller-owned logits buffer,
    /// - caller controls KeyValueCache length and position.
    ///
    /// This class still does not do token embedding or positional embedding.
    /// The caller provides the already embedded hidden vector for the current token.
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

        public void Decode(
            ReadOnlySpan<float> inputHidden,
            IReadOnlyList<float[]> ln1Gammas,
            IReadOnlyList<float[]> ln1Betas,
            IReadOnlyList<IReadOnlyList<float[]>> wqHeadsByLayer,
            IReadOnlyList<IReadOnlyList<float[]>> wkHeadsByLayer,
            IReadOnlyList<IReadOnlyList<float[]>> wvHeadsByLayer,
            IReadOnlyList<IReadOnlyList<float[]>> woHeadsByLayer,
            IReadOnlyList<float[]> attentionOutputBiases,
            IReadOnlyList<float[]> ln2Gammas,
            IReadOnlyList<float[]> ln2Betas,
            IReadOnlyList<float[]> ffnW1ByLayer,
            IReadOnlyList<float[]> ffnB1ByLayer,
            IReadOnlyList<float[]> ffnW2ByLayer,
            IReadOnlyList<float[]> ffnB2ByLayer,
            ReadOnlySpan<float> finalLayerNormGamma,
            ReadOnlySpan<float> finalLayerNormBeta,
            ReadOnlySpan<float> lmHeadWeights,
            ReadOnlySpan<float> lmHeadBias,
            KeyValueCache cache,
            int position,
            Span<float> logits)
        {
            ValidateDecodeArguments(
                inputHidden,
                ln1Gammas,
                ln1Betas,
                wqHeadsByLayer,
                wkHeadsByLayer,
                wvHeadsByLayer,
                woHeadsByLayer,
                attentionOutputBiases,
                ln2Gammas,
                ln2Betas,
                ffnW1ByLayer,
                ffnB1ByLayer,
                ffnW2ByLayer,
                ffnB2ByLayer,
                finalLayerNormGamma,
                finalLayerNormBeta,
                lmHeadWeights,
                lmHeadBias,
                cache,
                position,
                logits);

            inputHidden
                .Slice(0, DModel)
                .CopyTo(_currentHidden);

            var current = _currentHidden;
            var next = _nextHidden;

            for (var layer = 0; layer < LayerCount; layer++)
            {
                _blocks[layer].Decode(
                    current,
                    ln1Gammas[layer],
                    ln1Betas[layer],
                    wqHeadsByLayer[layer],
                    wkHeadsByLayer[layer],
                    wvHeadsByLayer[layer],
                    woHeadsByLayer[layer],
                    attentionOutputBiases[layer],
                    ln2Gammas[layer],
                    ln2Betas[layer],
                    ffnW1ByLayer[layer],
                    ffnB1ByLayer[layer],
                    ffnW2ByLayer[layer],
                    ffnB2ByLayer[layer],
                    cache,
                    layerIndex: layer,
                    position,
                    next);

                (current, next) = (next, current);
            }

            SingleTokenLayerNormKernel.Normalize(
                current,
                finalLayerNormGamma,
                finalLayerNormBeta,
                _finalHidden,
                DModel,
                LayerNormEpsilon);

            SingleTokenProjectionKernel.Project(
                _finalHidden,
                lmHeadWeights,
                lmHeadBias,
                logits,
                DModel,
                VocabSize);

            logits
                .Slice(0, VocabSize)
                .CopyTo(_lastLogits);
        }

        public void DecodeWithoutLayerNormAffine(
            ReadOnlySpan<float> inputHidden,
            IReadOnlyList<IReadOnlyList<float[]>> wqHeadsByLayer,
            IReadOnlyList<IReadOnlyList<float[]>> wkHeadsByLayer,
            IReadOnlyList<IReadOnlyList<float[]>> wvHeadsByLayer,
            IReadOnlyList<IReadOnlyList<float[]>> woHeadsByLayer,
            IReadOnlyList<float[]> attentionOutputBiases,
            IReadOnlyList<float[]> ffnW1ByLayer,
            IReadOnlyList<float[]> ffnB1ByLayer,
            IReadOnlyList<float[]> ffnW2ByLayer,
            IReadOnlyList<float[]> ffnB2ByLayer,
            ReadOnlySpan<float> lmHeadWeights,
            ReadOnlySpan<float> lmHeadBias,
            KeyValueCache cache,
            int position,
            Span<float> logits)
        {
            var emptyDModelByLayer = CreateEmptyLayerArray();
            var emptyDffByLayer = CreateEmptyLayerArray();

            Decode(
                inputHidden,
                emptyDModelByLayer,
                emptyDModelByLayer,
                wqHeadsByLayer,
                wkHeadsByLayer,
                wvHeadsByLayer,
                woHeadsByLayer,
                attentionOutputBiases,
                emptyDModelByLayer,
                emptyDModelByLayer,
                ffnW1ByLayer,
                ffnB1ByLayer,
                ffnW2ByLayer,
                ffnB2ByLayer,
                finalLayerNormGamma: ReadOnlySpan<float>.Empty,
                finalLayerNormBeta: ReadOnlySpan<float>.Empty,
                lmHeadWeights,
                lmHeadBias,
                cache,
                position,
                logits);
        }

        public void GetLastFinalHidden(Span<float> destination)
        {
            if (destination.Length < DModel)
            {
                throw new ArgumentException("Destination span is smaller than dModel.", nameof(destination));
            }

            _finalHidden.AsSpan().CopyTo(destination);
        }

        public void GetLastLogits(Span<float> destination)
        {
            if (destination.Length < VocabSize)
            {
                throw new ArgumentException("Destination span is smaller than vocabSize.", nameof(destination));
            }

            _lastLogits.AsSpan().CopyTo(destination);
        }

        public CachedTransformerBlock GetBlock(int layerIndex)
        {
            if ((uint)layerIndex >= (uint)LayerCount)
            {
                throw new ArgumentOutOfRangeException(nameof(layerIndex));
            }

            return _blocks[layerIndex];
        }

        private float[][] CreateEmptyLayerArray()
        {
            var values = new float[LayerCount][];

            for (var i = 0; i < values.Length; i++)
            {
                values[i] = [];
            }

            return values;
        }

        private void ValidateDecodeArguments(
            ReadOnlySpan<float> inputHidden,
            IReadOnlyList<float[]> ln1Gammas,
            IReadOnlyList<float[]> ln1Betas,
            IReadOnlyList<IReadOnlyList<float[]>> wqHeadsByLayer,
            IReadOnlyList<IReadOnlyList<float[]>> wkHeadsByLayer,
            IReadOnlyList<IReadOnlyList<float[]>> wvHeadsByLayer,
            IReadOnlyList<IReadOnlyList<float[]>> woHeadsByLayer,
            IReadOnlyList<float[]> attentionOutputBiases,
            IReadOnlyList<float[]> ln2Gammas,
            IReadOnlyList<float[]> ln2Betas,
            IReadOnlyList<float[]> ffnW1ByLayer,
            IReadOnlyList<float[]> ffnB1ByLayer,
            IReadOnlyList<float[]> ffnW2ByLayer,
            IReadOnlyList<float[]> ffnB2ByLayer,
            ReadOnlySpan<float> finalLayerNormGamma,
            ReadOnlySpan<float> finalLayerNormBeta,
            ReadOnlySpan<float> lmHeadWeights,
            ReadOnlySpan<float> lmHeadBias,
            KeyValueCache cache,
            int position,
            Span<float> logits)
        {
            if (inputHidden.Length < DModel)
            {
                throw new ArgumentException("Input hidden span is smaller than dModel.", nameof(inputHidden));
            }

            ValidateLayerCollection(ln1Gammas, nameof(ln1Gammas));
            ValidateLayerCollection(ln1Betas, nameof(ln1Betas));
            ValidateLayerCollection(attentionOutputBiases, nameof(attentionOutputBiases));
            ValidateLayerCollection(ln2Gammas, nameof(ln2Gammas));
            ValidateLayerCollection(ln2Betas, nameof(ln2Betas));
            ValidateLayerCollection(ffnW1ByLayer, nameof(ffnW1ByLayer));
            ValidateLayerCollection(ffnB1ByLayer, nameof(ffnB1ByLayer));
            ValidateLayerCollection(ffnW2ByLayer, nameof(ffnW2ByLayer));
            ValidateLayerCollection(ffnB2ByLayer, nameof(ffnB2ByLayer));

            ValidateHeadCollection(wqHeadsByLayer, nameof(wqHeadsByLayer));
            ValidateHeadCollection(wkHeadsByLayer, nameof(wkHeadsByLayer));
            ValidateHeadCollection(wvHeadsByLayer, nameof(wvHeadsByLayer));
            ValidateHeadCollection(woHeadsByLayer, nameof(woHeadsByLayer));

            for (var layer = 0; layer < LayerCount; layer++)
            {
                if (!ln1Gammas[layer].AsSpan().IsEmpty && ln1Gammas[layer].Length < DModel)
                {
                    throw new ArgumentException($"LN1 gamma for layer {layer} is smaller than dModel.", nameof(ln1Gammas));
                }

                if (!ln1Betas[layer].AsSpan().IsEmpty && ln1Betas[layer].Length < DModel)
                {
                    throw new ArgumentException($"LN1 beta for layer {layer} is smaller than dModel.", nameof(ln1Betas));
                }

                if (!attentionOutputBiases[layer].AsSpan().IsEmpty && attentionOutputBiases[layer].Length < DModel)
                {
                    throw new ArgumentException($"Attention output bias for layer {layer} is smaller than dModel.", nameof(attentionOutputBiases));
                }

                if (!ln2Gammas[layer].AsSpan().IsEmpty && ln2Gammas[layer].Length < DModel)
                {
                    throw new ArgumentException($"LN2 gamma for layer {layer} is smaller than dModel.", nameof(ln2Gammas));
                }

                if (!ln2Betas[layer].AsSpan().IsEmpty && ln2Betas[layer].Length < DModel)
                {
                    throw new ArgumentException($"LN2 beta for layer {layer} is smaller than dModel.", nameof(ln2Betas));
                }

                if (ffnW1ByLayer[layer].Length < DModel * DFF)
                {
                    throw new ArgumentException($"FFN W1 for layer {layer} is smaller than dModel * dFF.", nameof(ffnW1ByLayer));
                }

                if (!ffnB1ByLayer[layer].AsSpan().IsEmpty && ffnB1ByLayer[layer].Length < DFF)
                {
                    throw new ArgumentException($"FFN B1 for layer {layer} is smaller than dFF.", nameof(ffnB1ByLayer));
                }

                if (ffnW2ByLayer[layer].Length < DFF * DModel)
                {
                    throw new ArgumentException($"FFN W2 for layer {layer} is smaller than dFF * dModel.", nameof(ffnW2ByLayer));
                }

                if (!ffnB2ByLayer[layer].AsSpan().IsEmpty && ffnB2ByLayer[layer].Length < DModel)
                {
                    throw new ArgumentException($"FFN B2 for layer {layer} is smaller than dModel.", nameof(ffnB2ByLayer));
                }
            }

            if (!finalLayerNormGamma.IsEmpty && finalLayerNormGamma.Length < DModel)
            {
                throw new ArgumentException("Final LayerNorm gamma span is smaller than dModel.", nameof(finalLayerNormGamma));
            }

            if (!finalLayerNormBeta.IsEmpty && finalLayerNormBeta.Length < DModel)
            {
                throw new ArgumentException("Final LayerNorm beta span is smaller than dModel.", nameof(finalLayerNormBeta));
            }

            if (lmHeadWeights.Length < DModel * VocabSize)
            {
                throw new ArgumentException("LM head weights span is smaller than dModel * vocabSize.", nameof(lmHeadWeights));
            }

            if (!lmHeadBias.IsEmpty && lmHeadBias.Length < VocabSize)
            {
                throw new ArgumentException("LM head bias span is smaller than vocabSize.", nameof(lmHeadBias));
            }

            if (cache is null)
            {
                throw new ArgumentNullException(nameof(cache));
            }

            if (cache.Shape.LayerCount < LayerCount)
            {
                throw new ArgumentException("Cache layer count is smaller than stack layer count.", nameof(cache));
            }

            if (cache.Shape.HeadCount < HeadCount)
            {
                throw new ArgumentException("Cache head count is smaller than stack head count.", nameof(cache));
            }

            if (cache.Shape.HeadDimension != HeadDimension)
            {
                throw new ArgumentException("Cache head dimension does not match stack head dimension.", nameof(cache));
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

            if (logits.Length < VocabSize)
            {
                throw new ArgumentException("Logits span is smaller than vocabSize.", nameof(logits));
            }
        }

        private void ValidateLayerCollection<T>(
            IReadOnlyList<T> values,
            string name)
        {
            if (values is null)
            {
                throw new ArgumentNullException(name);
            }

            if (values.Count < LayerCount)
            {
                throw new ArgumentException($"{name} collection is smaller than layerCount.", name);
            }
        }

        private void ValidateHeadCollection(
            IReadOnlyList<IReadOnlyList<float[]>> values,
            string name)
        {
            if (values is null)
            {
                throw new ArgumentNullException(name);
            }

            if (values.Count < LayerCount)
            {
                throw new ArgumentException($"{name} collection is smaller than layerCount.", name);
            }

            for (var layer = 0; layer < LayerCount; layer++)
            {
                if (values[layer] is null)
                {
                    throw new ArgumentException($"{name}[{layer}] is null.", name);
                }

                if (values[layer].Count < HeadCount)
                {
                    throw new ArgumentException($"{name}[{layer}] has fewer heads than headCount.", name);
                }
            }
        }
    }
}
