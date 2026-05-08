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
        private readonly GPT1Model    _model;
        private readonly CachedGptStack _stack;
        private readonly KeyValueCache   _cache;
        private readonly StackWeights    _weights;

        private readonly float[] _tokenEmbeddingBuffer;
        private readonly float[] _positionEmbeddingBuffer;
        private readonly float[] _inputHidden;
        private readonly float[] _lastLogits;

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
                kvHeadCount: HeadCount,  // MHA — KV heads == Q heads
                MaxContextLength,
                HeadDimension);

            _tokenEmbeddingBuffer    = new float[DModel];
            _positionEmbeddingBuffer = new float[DModel];
            _inputHidden             = new float[DModel];
            _lastLogits              = new float[VocabSize];

            // Zero-copy weight binding — no data is copied.
            // TensorStorage refs point directly to model parameters.
            _weights = new StackWeights(model);
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

        /// <summary>
        /// No-op in the zero-copy StackWeights architecture.
        /// TensorStorage references in StackWeights point directly to model
        /// parameters — updates to model weights are immediately visible
        /// through the spans without any copying.
        /// </summary>
        public void RefreshWeightsFromModel()
        {
            ThrowIfDisposed();
            // Zero-copy: nothing to refresh. StackWeights spans are live views
            // into TensorStorage, which points to the current model parameters.
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
                _weights,
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
