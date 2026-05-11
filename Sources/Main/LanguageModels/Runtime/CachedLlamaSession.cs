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
        private readonly GPT1Config _config;
        private readonly CachedGptStack _stack;
        private readonly StackWeights _weights;
        private readonly KeyValueCache _cache;
        private readonly RopeTable? _rope;
        private readonly ReadOnlyMemory<float> _embedWeights;

        // Per-token working buffers (allocated once)
        private readonly float[] _hidden;
        private readonly float[] _logits;
        private readonly int[] _indexScratch;
        private readonly float[] _scoreScratch;
        private readonly Random _random;

        // Repetition penalty: track token history + scratch for penalized logits.
        // Allocated lazily on first use to keep zero-cost when penalty disabled.
        private readonly int[] _tokenHistory;
        private int _historyCount;
        private float[]? _logitsForSampling;

        private bool _disposed;

        internal CachedLlamaSession(
            GPT1Config config,
            CachedGptStack stack,
            StackWeights weights,
            KeyValueCache cache,
            RopeTable? rope,
            ReadOnlySpan<float> embedWeights)
        {
            _config = config;
            _stack = stack;
            _weights = weights;
            _cache = cache;
            _rope = rope;

            // Copy embed weights reference — this points to engine-owned TensorStorage.
            _embedWeights = embedWeights.ToArray().AsMemory();

            _hidden = new float[config.DModel];
            _logits = new float[config.VocabSize];
            _indexScratch = new int[config.VocabSize];
            _scoreScratch = new float[config.VocabSize];
            _random = new Random();

            _tokenHistory = new int[config.ContextLength];
            _historyCount = 0;
        }

        public int Position => _cache.CurrentLength;
        public bool IsFull => _cache.IsFull;

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
            _historyCount = 0;

            foreach (var token in promptTokens)
            {
                if (_cache.IsFull)
                {
                    throw new InvalidOperationException(
                        $"Prompt length {promptTokens.Length} exceeds ContextLength {_config.ContextLength}.");
                }

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
            {
                throw new InvalidOperationException(
                    $"KV cache is full (ContextLength={_config.ContextLength}). Start a new session.");
            }

            if (Position == 0)
            {
                throw new InvalidOperationException(
                    "Session is empty. Call Reset with at least one prompt token first.");
            }

            int token;
            if (sampling.RepetitionPenalty > 1.0f && _historyCount > 0)
            {
                // Lazy-allocate penalty scratch (vocab_size floats)
                _logitsForSampling ??= new float[_config.VocabSize];

                _logits.AsSpan().CopyTo(_logitsForSampling);

                var ctxSize = sampling.RepetitionPenaltyContextSize;
                var windowSize = ctxSize > 0
                    ? Math.Min(ctxSize, _historyCount)
                    : _historyCount;
                var historyStart = _historyCount - windowSize;

                TokenSampler.ApplyRepetitionPenalty(
                    _logitsForSampling.AsSpan(),
                    _tokenHistory.AsSpan(historyStart, windowSize),
                    sampling.RepetitionPenalty);

                token = TokenSampler.Sample(
                    _logitsForSampling, in sampling, _random, _indexScratch, _scoreScratch);
            }
            else
            {
                token = TokenSampler.Sample(
                    _logits, in sampling, _random, _indexScratch, _scoreScratch);
            }

            DecodeToken(token);
            return token;
        }

        /// <summary>Exposes last logits for custom sampling.</summary>
        public ReadOnlySpan<float> LastLogits => _logits;

        /// <summary>
        /// Hidden state AFTER all transformer layers, BEFORE final RMSNorm.
        /// Matches Python: x before rms_norm(x, fg2, eps).
        /// Previously incorrectly returned _hidden (token embedding input).
        /// </summary>
        public ReadOnlySpan<float> LastHiddenState => _stack.LastFinalHidden;

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _cache.Dispose();
        }

        // ── Private decode ────────────────────────────────────────────────────

        internal CachedGptStack Stack => _stack;

        private void DecodeToken(int tokenId)
        {
            // Track for repetition penalty (circular if needed)
            if (_historyCount < _tokenHistory.Length)
            {
                _tokenHistory[_historyCount++] = tokenId;
            }
            else
            {
                // Shift left, keep newest at end (rare path, only if ctx exceeded)
                Array.Copy(_tokenHistory, 1, _tokenHistory, 0, _tokenHistory.Length - 1);
                _tokenHistory[^1] = tokenId;
            }

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
            {
                throw new ObjectDisposedException(nameof(CachedLlamaSession));
            }
        }
    }
}
