// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Rope;
using DevOnBike.Overfit.Tensors.Core;

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
    public sealed class CachedLlamaSession : ISlmSession
    {
        private readonly GPT1Config _config;
        private readonly CachedGptStack _stack;
        private readonly StackWeights _weights;
        private readonly KeyValueCache _cache;
        private readonly RopeTable? _rope;

        // Engine-owned embedding matrix [vocab × dModel], referenced (NOT copied):
        // the engine outlives every session it creates and disposes this storage.
        // The row for the current token is sliced on demand at lookup. Previously
        // this was a per-session ToArray() copy — 1.24 GB duplicated for a 3B model.
        private readonly TensorStorage<float> _embedWeights;

        // Per-token working buffers (allocated once)
        private readonly float[] _hidden;
        private readonly float[] _logits;
        private readonly int[] _indexScratch;
        private readonly float[] _scoreScratch;
        private readonly Random _random;

        private bool _disposed;
        private bool _slidingWindow;
        private int _evictBlock;

        internal CachedLlamaSession(
            GPT1Config config,
            CachedGptStack stack,
            StackWeights weights,
            KeyValueCache cache,
            RopeTable? rope,
            TensorStorage<float> embedWeights)
        {
            _config = config;
            _stack = stack;
            _weights = weights;
            _cache = cache;
            _rope = rope;

            // Reference the engine-owned embedding storage — no copy. The engine
            // owns its lifetime (disposes it); sessions never outlive their engine.
            _embedWeights = embedWeights;

            _hidden = new float[config.DModel];
            _logits = new float[config.VocabSize];
            _indexScratch = new int[config.VocabSize];
            _scoreScratch = new float[config.VocabSize];
            _random = new Random();
        }

        public int Position => _cache.CurrentLength;
        public bool IsFull => _cache.IsFull;

        // ── ISlmSession surface (wires this engine into ChatSession + the SLM contract) ──

        /// <summary>Live token count in the KV cache (alias of <see cref="Position"/>).</summary>
        public int CurrentPosition => _cache.CurrentLength;

        /// <summary>Maximum context the cache can hold this session.</summary>
        public int MaxContextLength => _cache.MaxLength;

        /// <summary>Vocabulary size (logits width).</summary>
        public int VocabularySize => _config.VocabSize;

        /// <summary>This runtime always decodes through a KV cache.</summary>
        public bool HasKeyValueCache => true;

        /// <summary>Tokens evicted so far by the sliding window (0 until the cache first fills).</summary>
        public int BasePosition => _cache.BasePosition;

        /// <summary>True when sliding-window eviction is enabled (see <see cref="EnableSlidingWindow"/>).</summary>
        public bool SlidingWindowEnabled => _slidingWindow;

        /// <summary>
        /// Enables sliding-window KV eviction (RoPE models only): once the cache fills,
        /// the oldest <paramref name="evictBlock"/> tokens are dropped instead of throwing,
        /// so generation/prefill can continue indefinitely over a rolling context. Retained
        /// K/V are not re-rotated — <see cref="KeyValueCache.BasePosition"/> keeps RoPE's
        /// relative offsets correct. Default block = ¼ of the cache. Requires a RoPE config
        /// (learned absolute-position models cannot slide). No-op effect until the cache fills.
        /// </summary>
        public void EnableSlidingWindow(int evictBlock = 0)
        {
            ThrowIfDisposed();
            if (_rope is null)
            {
                throw new NotSupportedException(
                    "Sliding-window eviction requires a RoPE model; learned absolute-position models cannot slide without re-embedding.");
            }
            _slidingWindow = true;
            _evictBlock = evictBlock > 0 ? evictBlock : Math.Max(1, _cache.MaxLength / 4);
        }

        private void MakeRoomIfSliding()
        {
            if (!_slidingWindow || !_cache.IsFull)
            {
                return;
            }
            var count = Math.Min(_evictBlock, _cache.CurrentLength - 1);
            if (count > 0)
            {
                _cache.Evict(count);
            }
        }

        // ── Session lifecycle ─────────────────────────────────────────────────

        /// <summary>
        /// Clears the KV cache without feeding any prompt. After <c>Reset()</c>
        /// the session is empty (<see cref="Position"/> == 0); follow with
        /// <see cref="Prefill"/> or call the convenience overload
        /// <see cref="Reset(System.ReadOnlySpan{int})"/> which does both in one
        /// step.
        /// </summary>
        public void Reset()
        {
            ThrowIfDisposed();
            _cache.Reset();
        }

        /// <summary>
        /// Feeds prompt tokens into the KV cache one at a time. Each token goes
        /// through embedding → transformer stack → cache write, leaving the
        /// session ready for <see cref="GenerateNextToken"/>.
        ///
        /// Can be called multiple times to append context incrementally
        /// (e.g. chat history: system → user → assistant → user → …) without
        /// dropping cache state, provided total tokens stay within
        /// <c>ContextLength</c>.
        ///
        /// **Performance note:** today this is a single-token decode loop —
        /// O(N) calls through the transformer stack. A multi-token batched
        /// prefill path (one GEMM per layer over the whole prompt) is the
        /// upcoming optimization tracked in ROADMAP under
        /// "Prefill: multi-token batched matmul".
        /// </summary>
        public void Prefill(ReadOnlySpan<int> promptTokens)
        {
            ThrowIfDisposed();

            if (promptTokens.IsEmpty) { return; }

            // Skip the LM-head projection for every prompt token except the last:
            // their logits would be overwritten anyway. The final token runs the
            // full decode so _logits reflects the end-of-prompt prediction.
            var lastIndex = promptTokens.Length - 1;

            for (var i = 0; i < promptTokens.Length; i++)
            {
                if (_cache.IsFull && !_slidingWindow)
                {
                    throw new InvalidOperationException(
                        $"Prefill of {promptTokens.Length} tokens would exceed ContextLength {_config.ContextLength} " +
                        $"(current position {Position}).");
                }

                if (i < lastIndex)
                {
                    DecodeTokenWithoutLogits(promptTokens[i]);
                }
                else
                {
                    DecodeToken(promptTokens[i]);
                }
            }
        }

        /// <summary>
        /// Convenience overload: clears the cache and prefills it with the
        /// supplied prompt. Equivalent to <see cref="Reset()"/> followed by
        /// <see cref="Prefill"/>.
        /// </summary>
        public void Reset(ReadOnlySpan<int> promptTokens)
        {
            Reset();
            Prefill(promptTokens);
        }

        /// <summary>
        /// Generates the next token using the current cache state.
        /// The generated token is automatically fed back as context.
        /// Returns the token ID.
        /// </summary>
        public int GenerateNextToken(in SamplingOptions sampling)
        {
            ThrowIfDisposed();

            if (_cache.IsFull && !_slidingWindow)
            {
                throw new InvalidOperationException(
                    $"KV cache is full (ContextLength={_config.ContextLength}). Start a new session.");
            }

            if (Position == 0)
            {
                throw new InvalidOperationException(
                    "Session is empty. Call Reset with at least one prompt token first.");
            }

            var token = TokenSampler.Sample(
                _logits, in sampling, _random, _indexScratch, _scoreScratch);
            DecodeToken(token);
            return token;
        }

        /// <summary>
        /// Prefills <paramref name="promptTokens"/> then greedily fills
        /// <paramref name="outputTokens"/> (bounded by <see cref="GenerationOptions.MaxNewTokens"/>),
        /// stopping early on the configured end-of-text token. Returns the number of
        /// tokens written. Mirrors <see cref="CachedSlmSession.Generate"/>.
        /// </summary>
        public int Generate(
            ReadOnlySpan<int> promptTokens,
            Span<int> outputTokens,
            in GenerationOptions options)
        {
            ThrowIfDisposed();

            Reset(promptTokens);

            var sampling = options.Sampling;
            var generated = 0;
            while (generated < outputTokens.Length && generated < options.MaxNewTokens)
            {
                var token = GenerateNextToken(in sampling);
                outputTokens[generated] = token;
                generated++;

                if (options.StopOnEndOfTextToken &&
                    options.EndOfTextTokenId >= 0 &&
                    token == options.EndOfTextTokenId)
                {
                    break;
                }
            }
            return generated;
        }

        /// <summary>Copies the end-of-prompt / last-decode logits into <paramref name="destination"/>.</summary>
        public void GetLastLogits(Span<float> destination)
        {
            ThrowIfDisposed();
            if (destination.Length < VocabularySize)
            {
                throw new ArgumentException("Destination span is too small for logits.", nameof(destination));
            }
            _logits.AsSpan().CopyTo(destination);
        }

        /// <summary>
        /// Streams generated tokens one-by-one as an async sequence.
        /// Yields each newly-generated token ID; the stream terminates when
        /// any of these conditions is met:
        ///   - <paramref name="options"/>.MaxTokens reached
        ///   - A token from <paramref name="options"/>.StopTokens is sampled
        ///     (the stop token IS yielded before termination)
        ///   - The KV cache fills (ContextLength reached)
        ///   - <paramref name="cancellationToken"/> is signaled
        ///
        /// Each iteration yields control via <see cref="Task.Yield"/> so UI
        /// threads can render token-by-token without blocking.
        /// </summary>
        /// <example>
        /// <code>
        /// var opts = StreamingOptions.WithStopTokens(
        ///     maxTokens: 256, QwenTokenizer.ImEnd, QwenTokenizer.EndOfText);
        ///
        /// await foreach (var token in session.StreamGenerate(opts, ct))
        /// {
        ///     Console.Write(tokenizer.DecodeToken(token));
        /// }
        /// </code>
        /// </example>
        public async IAsyncEnumerable<int> StreamGenerate(
            StreamingOptions options,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            if (Position == 0)
            {
                throw new InvalidOperationException(
                    "Session is empty. Call Reset with at least one prompt token first.");
            }

            var sampling = options.Sampling;

            for (var i = 0; i < options.MaxTokens; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                if (_cache.IsFull && !_slidingWindow)
                {
                    // KV cache exhausted — graceful stop
                    yield break;
                }

                var token = TokenSampler.Sample(
                    _logits, in sampling, _random, _indexScratch, _scoreScratch);

                DecodeToken(token);
                yield return token;

                // Check stop tokens AFTER yielding so consumer sees the terminator.
                if (ContainsStopToken(options.StopTokens, token))
                {
                    yield break;
                }

                // Hand control back to the scheduler so the consumer (and any
                // UI thread) can process the yielded token before we compute
                // the next one. For server scenarios this is essentially free;
                // for UI scenarios it's what makes streaming "feel real".
                await Task.Yield();
            }
        }

        private static bool ContainsStopToken(IReadOnlyList<int> stops, int token)
        {
            // Avoid LINQ in main code path; small list, linear scan is fine.
            for (var i = 0; i < stops.Count; i++)
            {
                if (stops[i] == token) { return true; }
            }
            return false;
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
            var position = EmbedAndAdvance(tokenId);

            _stack.Decode(
                _hidden,
                _weights,
                _cache,
                position,
                _logits,
                _rope);
        }

        /// <summary>
        /// Prefill variant — same KV-cache + hidden-state update as
        /// <see cref="DecodeToken"/>, but skips the LM-head projection
        /// and does NOT touch <c>_logits</c>. Used by <see cref="Prefill"/>
        /// for every prompt token except the last.
        /// </summary>
        private void DecodeTokenWithoutLogits(int tokenId)
        {
            var position = EmbedAndAdvance(tokenId);

            _stack.DecodeWithoutLogits(
                _hidden,
                _weights,
                _cache,
                position,
                _rope);
        }

        private int EmbedAndAdvance(int tokenId)
        {
            // Sliding window: free a block of oldest tokens before writing a new one
            // when the cache is full (RoPE-only; no-op otherwise).
            MakeRoomIfSliding();

            // Token embedding lookup: row tokenId of embed_weights [vocab × dModel],
            // sliced directly from the engine-owned storage (no per-session copy).
            var row = _embedWeights.AsReadOnlySpan().Slice(tokenId * _config.DModel, _config.DModel);
            row.CopyTo(_hidden);

            // No additive positional embedding — RoPE handles positions inside attention.

            var position = _cache.CurrentLength;
            _cache.Advance();
            return position;
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
