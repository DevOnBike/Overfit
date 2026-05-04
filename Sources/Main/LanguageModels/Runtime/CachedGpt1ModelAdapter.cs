// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Adapter from the current GPT1Model object graph to the cached runtime stack.
    ///
    /// This is the first bridge between:
    ///
    /// - GPT1Model / TransformerBlock / Parameter objects,
    /// - CachedGptStack / KeyValueCache runtime primitives.
    ///
    /// Scope:
    /// - batch = 1,
    /// - one token per call,
    /// - FP32,
    /// - Pre-LN GPT configuration only,
    /// - no token sampling,
    /// - no SlmSession integration yet.
    ///
    /// The adapter copies model weights into runtime-owned float arrays once.
    /// Call RefreshWeightsFromModel() after loading a checkpoint or mutating model
    /// parameters.
    ///
    /// For tied LM head models, the runtime LM head is transposed directly from
    /// TokenEmbedding.Weight. This avoids relying on GPT1Model.LMHead being synced
    /// by a previous graph forward pass.
    /// </summary>
    public class CachedGpt1ModelAdapter : IDisposable
    {
        private readonly GPT1Model _model;
        private readonly CachedGptStack _stack;
        private readonly KeyValueCache _cache;

        private readonly float[] _tokenEmbeddingBuffer;
        private readonly float[] _positionEmbeddingBuffer;
        private readonly float[] _inputHidden;
        private readonly float[] _lastLogits;

        private readonly float[][] _ln1Gammas;
        private readonly float[][] _ln1Betas;
        private readonly IReadOnlyList<float[]>[] _wqHeadsByLayer;
        private readonly IReadOnlyList<float[]>[] _wkHeadsByLayer;
        private readonly IReadOnlyList<float[]>[] _wvHeadsByLayer;
        private readonly IReadOnlyList<float[]>[] _woHeadsByLayer;
        private readonly float[][] _attentionOutputBiases;
        private readonly float[][] _ln2Gammas;
        private readonly float[][] _ln2Betas;
        private readonly float[][] _ffnW1ByLayer;
        private readonly float[][] _ffnB1ByLayer;
        private readonly float[][] _ffnW2ByLayer;
        private readonly float[][] _ffnB2ByLayer;
        private readonly float[] _finalLayerNormGamma;
        private readonly float[] _finalLayerNormBeta;
        private readonly float[] _lmHeadWeights;

        private bool _disposed;

        public CachedGpt1ModelAdapter(GPT1Model model)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));

            var config = model.Config;

            if (!config.PreLayerNorm)
            {
                throw new NotSupportedException(
                    "CachedGpt1ModelAdapter currently supports Pre-LN GPT blocks only.");
            }

            LayerCount = config.NLayers;
            DModel = config.DModel;
            HeadCount = config.NHeads;
            HeadDimension = config.DModel / config.NHeads;
            DFF = config.DFF;
            VocabSize = config.VocabSize;
            MaxContextLength = config.ContextLength;

            _stack = new CachedGptStack(
                LayerCount,
                DModel,
                HeadCount,
                DFF,
                VocabSize,
                MaxContextLength,
                config.LNEps,
                FeedForwardActivation.GeLU);

            _cache = KeyValueCache.Create(
                LayerCount,
                HeadCount,
                MaxContextLength,
                HeadDimension);

            _tokenEmbeddingBuffer = new float[DModel];
            _positionEmbeddingBuffer = new float[DModel];
            _inputHidden = new float[DModel];
            _lastLogits = new float[VocabSize];

            _ln1Gammas = CreateLayerArray(LayerCount, DModel);
            _ln1Betas = CreateLayerArray(LayerCount, DModel);
            _attentionOutputBiases = CreateLayerArray(LayerCount, DModel);
            _ln2Gammas = CreateLayerArray(LayerCount, DModel);
            _ln2Betas = CreateLayerArray(LayerCount, DModel);
            _ffnW1ByLayer = CreateLayerArray(LayerCount, DModel * DFF);
            _ffnB1ByLayer = CreateLayerArray(LayerCount, DFF);
            _ffnW2ByLayer = CreateLayerArray(LayerCount, DFF * DModel);
            _ffnB2ByLayer = CreateLayerArray(LayerCount, DModel);

            _wqHeadsByLayer = CreateHeadsByLayer(
                LayerCount,
                HeadCount,
                DModel * HeadDimension);

            _wkHeadsByLayer = CreateHeadsByLayer(
                LayerCount,
                HeadCount,
                DModel * HeadDimension);

            _wvHeadsByLayer = CreateHeadsByLayer(
                LayerCount,
                HeadCount,
                DModel * HeadDimension);

            _woHeadsByLayer = CreateHeadsByLayer(
                LayerCount,
                HeadCount,
                HeadDimension * DModel);

            _finalLayerNormGamma = new float[DModel];
            _finalLayerNormBeta = new float[DModel];
            _lmHeadWeights = new float[DModel * VocabSize];

            RefreshWeightsFromModel();
        }

        public int LayerCount { get; }

        public int DModel { get; }

        public int HeadCount { get; }

        public int HeadDimension { get; }

        public int DFF { get; }

        public int VocabSize { get; }

        public int MaxContextLength { get; }

        public int CurrentPosition => _cache.CurrentLength;

        public bool IsFull => _cache.IsFull;

        public void Reset()
        {
            ThrowIfDisposed();

            _cache.Reset();
            Array.Clear(_lastLogits);
            Array.Clear(_inputHidden);
            Array.Clear(_tokenEmbeddingBuffer);
            Array.Clear(_positionEmbeddingBuffer);
        }

        public void RefreshWeightsFromModel()
        {
            ThrowIfDisposed();

            for (var layer = 0; layer < LayerCount; layer++)
            {
                var block = _model.Blocks[layer];

                Copy(block.Norm1.Gamma.DataReadOnlySpan, _ln1Gammas[layer]);
                Copy(block.Norm1.Beta.DataReadOnlySpan, _ln1Betas[layer]);

                for (var head = 0; head < HeadCount; head++)
                {
                    Copy(block.Attention.WqHeads[head].DataReadOnlySpan, ((float[][])_wqHeadsByLayer[layer])[head]);
                    Copy(block.Attention.WkHeads[head].DataReadOnlySpan, ((float[][])_wkHeadsByLayer[layer])[head]);
                    Copy(block.Attention.WvHeads[head].DataReadOnlySpan, ((float[][])_wvHeadsByLayer[layer])[head]);
                    Copy(block.Attention.WoHeads[head].DataReadOnlySpan, ((float[][])_woHeadsByLayer[layer])[head]);
                }

                Copy(block.Attention.Bo.DataReadOnlySpan, _attentionOutputBiases[layer]);

                Copy(block.Norm2.Gamma.DataReadOnlySpan, _ln2Gammas[layer]);
                Copy(block.Norm2.Beta.DataReadOnlySpan, _ln2Betas[layer]);

                Copy(block.FFN.W1.DataReadOnlySpan, _ffnW1ByLayer[layer]);
                Copy(block.FFN.B1.DataReadOnlySpan, _ffnB1ByLayer[layer]);
                Copy(block.FFN.W2.DataReadOnlySpan, _ffnW2ByLayer[layer]);
                Copy(block.FFN.B2.DataReadOnlySpan, _ffnB2ByLayer[layer]);
            }

            Copy(_model.FinalNorm.Gamma.DataReadOnlySpan, _finalLayerNormGamma);
            Copy(_model.FinalNorm.Beta.DataReadOnlySpan, _finalLayerNormBeta);

            if (_model.Config.TieWeights)
            {
                TransposeTokenEmbeddingToLmHead();
            }
            else
            {
                Copy(_model.LMHead.DataReadOnlySpan, _lmHeadWeights);
            }
        }

        public void DecodeNextToken(
            int tokenId,
            Span<float> logits)
        {
            ThrowIfDisposed();

            if (logits.Length < VocabSize)
            {
                throw new ArgumentException("Logits span is smaller than vocab size.", nameof(logits));
            }

            if (_cache.IsFull)
            {
                throw new InvalidOperationException(
                    $"Cannot decode next token because KV cache is full. MaxContextLength={MaxContextLength}.");
            }

            var position = _cache.CurrentLength;

            _model.TokenEmbedding.LookupInference(
                tokenId,
                _tokenEmbeddingBuffer);

            _model.PositionEmbedding.LookupInference(
                position,
                _positionEmbeddingBuffer);

            for (var i = 0; i < DModel; i++)
            {
                _inputHidden[i] = _tokenEmbeddingBuffer[i] + _positionEmbeddingBuffer[i];
            }

            _cache.Advance();

            _stack.Decode(
                _inputHidden,
                _ln1Gammas,
                _ln1Betas,
                _wqHeadsByLayer,
                _wkHeadsByLayer,
                _wvHeadsByLayer,
                _woHeadsByLayer,
                _attentionOutputBiases,
                _ln2Gammas,
                _ln2Betas,
                _ffnW1ByLayer,
                _ffnB1ByLayer,
                _ffnW2ByLayer,
                _ffnB2ByLayer,
                _finalLayerNormGamma,
                _finalLayerNormBeta,
                _lmHeadWeights,
                lmHeadBias: ReadOnlySpan<float>.Empty,
                _cache,
                position,
                logits);

            logits
                .Slice(0, VocabSize)
                .CopyTo(_lastLogits);
        }

        public void GetLastLogits(Span<float> destination)
        {
            ThrowIfDisposed();

            if (destination.Length < VocabSize)
            {
                throw new ArgumentException("Destination span is smaller than vocab size.", nameof(destination));
            }

            _lastLogits.AsSpan().CopyTo(destination);
        }

        public KeyValueCache Cache => _cache;

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _cache.Dispose();
        }

        private void TransposeTokenEmbeddingToLmHead()
        {
            var source = _model.TokenEmbedding.Weight.DataReadOnlySpan;

            for (var token = 0; token < VocabSize; token++)
            {
                for (var dim = 0; dim < DModel; dim++)
                {
                    _lmHeadWeights[dim * VocabSize + token] =
                        source[token * DModel + dim];
                }
            }
        }

        private static float[][] CreateLayerArray(
            int layerCount,
            int length)
        {
            var values = new float[layerCount][];

            for (var layer = 0; layer < layerCount; layer++)
            {
                values[layer] = new float[length];
            }

            return values;
        }

        private static IReadOnlyList<float[]>[] CreateHeadsByLayer(
            int layerCount,
            int headCount,
            int length)
        {
            var layers = new IReadOnlyList<float[]>[layerCount];

            for (var layer = 0; layer < layerCount; layer++)
            {
                var heads = new float[headCount][];

                for (var head = 0; head < headCount; head++)
                {
                    heads[head] = new float[length];
                }

                layers[layer] = heads;
            }

            return layers;
        }

        private static void Copy(
            ReadOnlySpan<float> source,
            float[] destination)
        {
            if (source.Length != destination.Length)
            {
                throw new InvalidOperationException(
                    $"Source length {source.Length} does not match destination length {destination.Length}.");
            }

            source.CopyTo(destination);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(CachedGpt1ModelAdapter));
            }
        }
    }
}
