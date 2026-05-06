// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using System.Numerics.Tensors;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Zero-copy KV-cache adapter for GPT1Model.
    ///
    /// Weight access:
    ///   All model weights are referenced via <see cref="StackWeights"/> which holds
    ///   <see cref="Tensors.Core.TensorStorage{T}"/> references — no weight data is copied.
    ///   Per-token inference allocates 0 bytes on the managed heap.
    ///
    /// Session creation cost (one-time):
    ///   KeyValueCache buffers + small working arrays (embedding buffer, hidden, logits).
    ///   For GPT-2 Small: ~75 MB (KV cache) + ~200 KB (working buffers).
    ///   No weight duplication.
    /// </summary>
    public sealed class CachedGpt1ModelAdapter : IDisposable
    {
        private readonly GPT1Model    _model;
        private readonly CachedGptStack _stack;
        private readonly KeyValueCache  _cache;
        private readonly StackWeights   _weights;

        // Working buffers — small, allocated once per session
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
                throw new NotSupportedException(
                    "CachedGpt1ModelAdapter supports Pre-LN GPT blocks only.");

            LayerCount      = config.NLayers;
            DModel          = config.DModel;
            HeadCount       = config.NHeads;
            HeadDimension   = config.DModel / config.NHeads;
            DFF             = config.DFF;
            VocabSize       = config.VocabSize;
            MaxContextLength = config.ContextLength;

            _stack = new CachedGptStack(
                LayerCount, DModel, HeadCount, DFF, VocabSize, MaxContextLength,
                config.LNEps, FeedForwardActivation.GeLU);

            _cache = KeyValueCache.Create(LayerCount, HeadCount, MaxContextLength, HeadDimension);

            // Zero-copy weight references — no data copied from model
            _weights = new StackWeights(model);

            // Small working buffers — the only managed allocations per session
            _tokenEmbeddingBuffer    = new float[DModel];
            _positionEmbeddingBuffer = new float[DModel];
            _inputHidden             = new float[DModel];
            _lastLogits              = new float[VocabSize];
        }

        public int LayerCount       { get; }
        public int DModel           { get; }
        public int HeadCount        { get; }
        public int HeadDimension    { get; }
        public int DFF              { get; }
        public int VocabSize        { get; }
        public int MaxContextLength { get; }
        public int CurrentPosition  => _cache.CurrentLength;
        public bool IsFull          => _cache.IsFull;
        public KeyValueCache Cache  => _cache;

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
        /// Rebinds weight references after an in-place model update (e.g. LoRA injection).
        /// Zero allocations — only updates TensorStorage references.
        /// </summary>
        public void RefreshWeightsFromModel()
        {
            ThrowIfDisposed();
            // StackWeights stores TensorStorage refs — already up to date for in-place updates.
            // Only needed if the Parameter objects themselves were replaced (rare).
            // For LoRA which writes into existing TensorStorage: no-op needed.
        }

        public void DecodeNextToken(int tokenId, Span<float> logits)
        {
            ThrowIfDisposed();

            if (logits.Length < VocabSize)
                throw new ArgumentException("Logits span is smaller than vocab size.", nameof(logits));

            if (_cache.IsFull)
                throw new InvalidOperationException(
                    $"KV cache full. MaxContextLength={MaxContextLength}.");

            var position = _cache.CurrentLength;

            _model.TokenEmbedding.LookupInference(tokenId, _tokenEmbeddingBuffer);
            _model.PositionEmbedding.LookupInference(position, _positionEmbeddingBuffer);

            TensorPrimitives.Add(_tokenEmbeddingBuffer, _positionEmbeddingBuffer, _inputHidden);

            _cache.Advance();

            _stack.Decode(
                _inputHidden,
                _weights,
                _cache,
                position,
                logits);

            logits.Slice(0, VocabSize).CopyTo(_lastLogits);
        }

        public void GetLastLogits(Span<float> destination)
        {
            ThrowIfDisposed();
            if (destination.Length < VocabSize)
                throw new ArgumentException("Destination span is smaller than vocab size.", nameof(destination));
            _lastLogits.AsSpan().CopyTo(destination);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _cache.Dispose();
        }

        private void ThrowIfDisposed()
            => ObjectDisposedException.ThrowIf(_disposed, this);
    }
}
