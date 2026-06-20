// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors;

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
    public sealed class CachedGpt1ModelAdapter : IDisposable
    {
        private readonly GPT1Model _model;
        private readonly CachedGptStack _stack;
        private readonly KeyValueCache _cache;
        private readonly StackWeights _weights;

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
                throw new OverfitRuntimeException(
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
                config.LNEps);

            _cache = KeyValueCache.Create(
                LayerCount,
                kvHeadCount: HeadCount,  // MHA — KV heads == Q heads
                MaxContextLength,
                HeadDimension);

            _tokenEmbeddingBuffer = new float[DModel];
            _positionEmbeddingBuffer = new float[DModel];
            _inputHidden = new float[DModel];
            _lastLogits = new float[VocabSize];

            // Zero-copy weight binding — no data is copied.
            // TensorStorage refs point directly to model parameters.
            _weights = new StackWeights(model);
        }

        public int LayerCount
        {
            get;
        }

        public int DModel
        {
            get;
        }

        public int HeadCount
        {
            get;
        }

        public int HeadDimension
        {
            get;
        }

        public int DFF
        {
            get;
        }

        public int VocabSize
        {
            get;
        }

        public int MaxContextLength
        {
            get;
        }

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

            var position = AdvanceAndEmbed(tokenId);

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

        /// <summary>
        /// Prefill variant: same KV-cache and hidden-state update as
        /// <see cref="DecodeNextToken"/>, but skips the LM-head projection
        /// and does NOT update last-logits. Use this for every prompt token
        /// except the one whose logits the caller actually consumes
        /// (typically the final prompt token).
        ///
        /// Skipping the LM head saves the largest single per-token cost
        /// (~27 % on GPT-2 Small) for tokens whose logits would be
        /// immediately overwritten by the next decode anyway.
        /// </summary>
        public void PrefillToken(int tokenId)
        {
            ThrowIfDisposed();

            var position = AdvanceAndEmbed(tokenId);

            _stack.DecodeWithoutLogits(
                _inputHidden,
                _weights,
                _cache,
                position);
        }

        /// <summary>
        /// True when the wrapped model's stack is on the batched-prefill fast path
        /// (standard LayerNorm, GeLU/ReLU FFN, MHA — not GQA, no RoPE). Always true
        /// for the GPT-1 / GPT-2 architecture this adapter wraps.
        /// </summary>
        public bool SupportsBatchedPrefill
        {
            get
            {
                ref readonly var b0 = ref _weights.Block(0);
                return !b0.Ln1Beta.IsEmpty && !b0.Ln2Beta.IsEmpty && b0.FfnGate.IsEmpty && !b0.HasGqa;
            }
        }

        /// <summary>
        /// Batched prefill (Phase 3): embeds all <paramref name="tokens"/>, advances
        /// the cache, runs the whole prompt through the stack in one batched pass
        /// (<see cref="CachedGptStack.PrefillBatched"/>), and projects the LAST
        /// token's logits into <paramref name="lastLogits"/> — leaving the session
        /// in exactly the state the single-token prefill loop would, but reading the
        /// weight set once per layer instead of once per token. Caller must check
        /// <see cref="SupportsBatchedPrefill"/> first.
        /// </summary>
        public void PrefillBatched(ReadOnlySpan<int> tokens, Span<float> lastLogits)
        {
            ThrowIfDisposed();

            var n = tokens.Length;
            if (n == 0)
            {
                return;
            }
            if (_cache.CurrentLength + n > MaxContextLength)
            {
                throw new OverfitRuntimeException(
                    $"Batched prefill of {n} tokens would exceed MaxContextLength {MaxContextLength}.");
            }

            var basePos = _cache.CurrentLength;
            var batch = PooledBuffer<float>.RentArray(n * DModel);
            try
            {
                for (var i = 0; i < n; i++)
                {
                    _model.TokenEmbedding.LookupInference(tokens[i], _tokenEmbeddingBuffer);
                    _model.PositionEmbedding.LookupInference(basePos + i, _positionEmbeddingBuffer);

                    var dst = batch.AsSpan(i * DModel, DModel);
                    for (var d = 0; d < DModel; d++)
                    {
                        dst[d] = _tokenEmbeddingBuffer[d] + _positionEmbeddingBuffer[d];
                    }

                    _cache.Advance();
                }

                _stack.PrefillBatched(batch, n, _weights, _cache, basePos, rope: null);
                _stack.ProjectLogits(_weights, lastLogits);
            }
            finally
            {
                PooledBuffer<float>.ReturnArray(batch);
            }
        }

        private int AdvanceAndEmbed(int tokenId)
        {
            if (_cache.IsFull)
            {
                throw new OverfitRuntimeException(
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
            return position;
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
            Span<float> destination)
        {
            if (source.Length != destination.Length)
            {
                throw new OverfitRuntimeException(
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
