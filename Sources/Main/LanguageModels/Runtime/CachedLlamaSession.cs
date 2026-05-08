// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Rope;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Single-session stateful inference for Llama / Qwen / Mistral / Phi models.
    ///
    /// Differences from <see cref="CachedSlmSession"/> (GPT-1/2):
    ///   - No additive positional embedding — position is encoded via RoPE.
    ///   - Token embedding lookup is a direct row-read from embed_weights.
    ///   - RoPE table is passed into stack.Decode per step.
    ///   - GQA cache uses kvHeadCount &lt; nHeads slots.
    ///
    /// Thread-safety: one session per thread.
    /// </summary>
    public sealed class CachedLlamaSession : IDisposable
    {
        private readonly GPT1Config            _config;
        private readonly CachedGptStack        _stack;
        private readonly StackWeights          _weights;
        private readonly KeyValueCache         _cache;
        private readonly RopeTable?            _rope;
        private readonly ReadOnlyMemory<float> _embedWeights;

        // Per-token working buffers (allocated once)
        private readonly float[]  _hidden;
        private readonly float[]  _logits;
        private readonly int[]    _indexScratch;
        private readonly float[]  _scoreScratch;
        private readonly Random   _random;

        private bool _disposed;

        internal CachedLlamaSession(
            GPT1Config            config,
            CachedGptStack        stack,
            StackWeights          weights,
            KeyValueCache         cache,
            RopeTable?            rope,
            ReadOnlySpan<float>   embedWeights)
        {
            _config  = config;
            _stack   = stack;
            _weights = weights;
            _cache   = cache;
            _rope    = rope;

            // Copy embed weights reference — this points to engine-owned TensorStorage.
            _embedWeights = embedWeights.ToArray().AsMemory();

            _hidden       = new float[config.DModel];
            _logits       = new float[config.VocabSize];
            _indexScratch = new int[config.VocabSize];
            _scoreScratch = new float[config.VocabSize];
            _random       = new Random();
        }

        public int Position => _cache.CurrentLength;
        public bool IsFull  => _cache.IsFull;

        // ── Session lifecycle ─────────────────────────────────────────────────

        /// <summary>
        /// Resets the session and prefills the KV cache with prompt tokens.
        /// After Reset the session is positioned at the last prompt token.
        /// Call GenerateNextToken to generate the first new token.
        /// </summary>
        public void Reset(ReadOnlySpan<int> promptTokens)
        {
            ThrowIfDisposed();
            _cache.Reset();

            foreach (var token in promptTokens)
            {
                if (_cache.IsFull)
                    throw new InvalidOperationException(
                        $"Prompt length {promptTokens.Length} exceeds ContextLength {_config.ContextLength}.");

                DecodeToken(token);
            }
        }

        /// <summary>
        /// Generates the next token using the current cache state.
        /// The generated token is automatically fed back as context.
        /// Returns the token ID.
        /// </summary>
        public int GenerateNextToken(in SamplingOptions sampling)
        {
            ThrowIfDisposed();

            if (_cache.IsFull)
                throw new InvalidOperationException(
                    $"KV cache is full (ContextLength={_config.ContextLength}). Start a new session.");

            if (Position == 0)
                throw new InvalidOperationException(
                    "Session is empty. Call Reset with at least one prompt token first.");

            var token = TokenSampler.Sample(
                _logits, in sampling, _random, _indexScratch, _scoreScratch);
            DecodeToken(token);
            return token;
        }

        /// <summary>Exposes last logits for custom sampling.</summary>
        public ReadOnlySpan<float> LastLogits => _logits;

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _cache.Dispose();
        }

        // ── Private decode ────────────────────────────────────────────────────

        private void DecodeToken(int tokenId)
        {
            // Token embedding lookup: row tokenId of embed_weights [vocab × dModel]
            var embedSpan = _embedWeights.Span;
            var row = embedSpan.Slice(tokenId * _config.DModel, _config.DModel);
            row.CopyTo(_hidden);

            // No additive positional embedding — RoPE handles positions inside attention.

            var position = _cache.CurrentLength;
            _cache.Advance();

            _stack.Decode(
                _hidden,
                _weights,
                _cache,
                position,
                _logits,
                _rope);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(CachedLlamaSession));
        }
    }
}
