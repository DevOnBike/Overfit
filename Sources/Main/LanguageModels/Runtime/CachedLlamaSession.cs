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
        // the engine outlives every session it creates and disposes its backing. The row for
        // the current token is read on demand at lookup (F32 slice, or per-row dequant when the
        // table is K-quant-resident). Previously this was a per-session ToArray() copy — 1.24 GB
        // duplicated for a 3B model.
        private readonly DecodeWeight _embedWeights;

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
            DecodeWeight embedWeights)
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

        /// <summary>This RoPE-capable session supports sliding-window eviction (<see cref="ISlmSession"/>).</summary>
        public bool SupportsSlidingWindow => _rope is not null;

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
        /// session ready for <see cref="GenerateNextToken(in SamplingOptions)"/>.
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

            // Batched (multi-token) prefill: one set of batched GEMMs per layer over the whole prompt
            // instead of N single-token passes — amortises the (weight-bandwidth-bound) weight reads
            // ~N×. Eligible for the dense SwiGLU Llama/Qwen path, non-sliding, when the prompt fits the
            // remaining context. MoE / sliding-window / tiny prompts fall back to the single-token loop.
            if (!DisableBatchedPrefillForParity
                && promptTokens.Length >= BatchedPrefillThreshold
                && !_slidingWindow
                && _config.FfnActivation == FeedForwardActivation.SwiGLU
                && _cache.CurrentLength + promptTokens.Length <= _cache.MaxLength)
            {
                PrefillBatchedQuant(promptTokens);
                return;
            }

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
        /// Snapshots the current KV-cache state as a reusable <b>prefix</b> (e.g. after prefilling a fixed
        /// system prompt). Restore it into this or another same-model session with <see cref="RestorePrefix"/>
        /// to skip re-encoding the prefix on every request — the agentic / multi-turn TTFT win.
        /// </summary>
        public KvCacheSnapshot SavePrefix()
        {
            ThrowIfDisposed();
            return _cache.Snapshot();
        }

        /// <summary>
        /// Restores a <see cref="SavePrefix"/> snapshot: the cache becomes exactly as if the prefix had
        /// just been prefilled (a memcpy, not a forward pass). Append the request's turn with
        /// <see cref="Prefill"/> afterwards — it attends over the restored prefix and refreshes the logits.
        /// </summary>
        public void RestorePrefix(KvCacheSnapshot prefix)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(prefix);
            _cache.RestoreFrom(prefix);
        }

        /// <summary>
        /// Generates the next token using the current cache state.
        /// The generated token is automatically fed back as context.
        /// Returns the token ID.
        /// </summary>
        public int GenerateNextToken(in SamplingOptions sampling)
            => GenerateNextToken(in sampling, constraint: null);

        /// <summary>
        /// Generates the next token, optionally under a decode-time <paramref name="constraint"/>
        /// (e.g. <c>JsonGrammarConstraint</c> for guaranteed well-formed JSON). The constraint masks
        /// the logits in place before sampling — disallowed tokens become <c>-inf</c>, so they cannot
        /// be drawn — and is then advanced by the chosen token. The masked buffer is the per-decode
        /// logits, overwritten by the next decode, so masking in place is safe.
        /// </summary>
        public int GenerateNextToken(in SamplingOptions sampling, ITokenConstraint? constraint)
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

            DecodeProfiler.BeginToken();
            constraint?.ApplyMask(_logits.AsSpan(0, VocabularySize));

            var profSample = DecodeProfiler.Start();
            var token = TokenSampler.Sample(
                _logits, in sampling, _random, _indexScratch, _scoreScratch);
            DecodeProfiler.Stop(DecodeProfiler.Component.Sampler, profSample);

            constraint?.Accept(token);
            DecodeToken(token);
            DecodeProfiler.EndToken();
            return token;
        }

        /// <summary>
        /// Forced decode of a known token (no sampling): advances the KV cache by <paramref name="token"/>
        /// and refreshes <c>_logits</c> to predict the token after it. Used by draft-model speculative
        /// decoding to condition a draft session on tokens the target chose (and to re-apply a correction).
        /// </summary>
        internal void Feed(int token)
        {
            ThrowIfDisposed();
            if (_cache.IsFull && !_slidingWindow)
            {
                throw new InvalidOperationException(
                    $"KV cache is full (ContextLength={_config.ContextLength}). Start a new session.");
            }
            if (Position == 0)
            {
                throw new InvalidOperationException("Session is empty. Call Reset with a prompt first.");
            }
            DecodeToken(token);
        }

        /// <summary>
        /// Rolls the KV cache back to <paramref name="length"/> positions (drops later K/V) — used by
        /// draft-model speculative decoding to discard a draft session's rejected proposal tokens.
        /// </summary>
        internal void RollbackTo(int length) => _cache.TruncateTo(length);

        // ── Adaptive speculative gating state (see GenerateSpeculative) ──
        private const double SpecGateThreshold = 3.0;  // committed-per-verify break-even ≈ 3.5; gate below it
        private const double SpecEmaAlpha = 0.5;       // EMA responsiveness — fast so it gates after a few rejects
        private const int SpecProbeInterval = 64;      // while gated, draft once every N steps to re-detect echo
        private double _specAcceptEma;                 // start pessimistic (0 → gated): single-token until a probe
                                                       // proves drafting pays. Novel text (chat) stays ≈ 1× — one
                                                       // probe per SpecProbeInterval; repetitive text ramps up fast.
        private int _specProbeCountdown;               // 0 → the first step probes immediately

        /// <summary>True when this session can run speculative decoding (SwiGLU FFN, non-sliding) — lets a
        /// generate loop pick the speculative path. <c>GenerateSpeculative</c> also falls back to a
        /// single-token step internally when this is false, so calling it is always safe.</summary>
        public bool CanSpeculate => !_slidingWindow && _config.FfnActivation == FeedForwardActivation.SwiGLU;

        /// <summary>Greedy speculative-decode step (overload of <see cref="GenerateSpeculative(ReadOnlySpan{int}, Span{int}, in SamplingOptions, int, int, int)"/>).</summary>
        public int GenerateSpeculative(
            ReadOnlySpan<int> history,
            Span<int> committed,
            int maxDraft = 4,
            int ngramMin = 1,
            int ngramMax = 3)
            => GenerateSpeculative(history, committed, SamplingOptions.Greedy, maxDraft, ngramMin, ngramMax);

        /// <summary>
        /// One <b>sampling-correct speculative-decode</b> step (prompt-lookup, no draft model): drafts the
        /// next tokens from <paramref name="history"/> via <see cref="PromptLookupDrafter"/> and verifies
        /// them in ONE batched forward. Each draft is accepted by speculative rejection sampling — accept
        /// with probability <c>p(draft)</c> under the sampler's target distribution, else resample from the
        /// renormalised residual <c>norm(max(0, p − e_draft))</c> — so the committed tokens are
        /// <b>distributed exactly as sampling from the target model directly</b> (greedy is the T→0 case,
        /// then it is bit-identical to single-token greedy). Commits the accepted prefix plus the
        /// correction/bonus token (forwarded so the cache + <c>_logits</c> stay consistent) into
        /// <paramref name="committed"/>; returns the count (≥1, ≤ maxDraft+2). The win is throughput on
        /// repetitive / structured output (the agentic moat); ~1× on novel text. Requires the batched path
        /// (RoPE/SwiGLU, non-sliding) — otherwise a plain single-token step.
        /// </summary>
        public int GenerateSpeculative(
            ReadOnlySpan<int> history,
            Span<int> committed,
            in SamplingOptions sampling,
            int maxDraft = 4,
            int ngramMin = 1,
            int ngramMax = 3)
            => GenerateSpeculativeCore(history, committed, in sampling, maxDraft, ngramMin, ngramMax, drafter: null);

        /// <summary>
        /// Draft-MODEL speculative overload: proposals come from <paramref name="drafter"/> (a small draft
        /// model) instead of prompt-lookup, so speculation wins on NOVEL text too. Same verify /
        /// accept-or-resample machinery; the drafter keeps its own KV in lockstep via its Sync callback.
        /// </summary>
        internal int GenerateSpeculative(
            ReadOnlySpan<int> history,
            Span<int> committed,
            in SamplingOptions sampling,
            int maxDraft,
            ISpeculativeDrafter drafter)
            => GenerateSpeculativeCore(history, committed, in sampling, maxDraft, ngramMin: 1, ngramMax: 3, drafter: drafter);

        private int GenerateSpeculativeCore(
            ReadOnlySpan<int> history,
            Span<int> committed,
            in SamplingOptions sampling,
            int maxDraft,
            int ngramMin,
            int ngramMax,
            ISpeculativeDrafter? drafter)
        {
            ThrowIfDisposed();
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxDraft);
            if (committed.Length < maxDraft + 2)
            {
                throw new ArgumentException("committed buffer must hold at least maxDraft+2 tokens.", nameof(committed));
            }
            if (Position == 0)
            {
                throw new InvalidOperationException("Session is empty. Call Reset with a prompt first.");
            }

            var dModel = _config.DModel;
            var vocab = VocabularySize;

            // Next token from the current (target) distribution — the same draw a normal step would make.
            var t0 = TokenSampler.Sample(_logits, in sampling, _random, _indexScratch, _scoreScratch);

            var canSpeculate = !_slidingWindow
                && _config.FfnActivation == FeedForwardActivation.SwiGLU
                && maxDraft > 0;

            // Adaptive gating: a verify forward only pays off if it commits enough tokens to beat its
            // (batch = 1 + dn) cost — break-even is ≈ 3.5 committed/verify on this path. We track an EMA of
            // committed tokens per VERIFY step; once it drops below the threshold (novel text → drafts get
            // rejected), suppress drafting and fall back to single-token steps so speculative never
            // underperforms plain decode. A periodic probe re-enables drafting if the output turns
            // repetitive/structured again (RAG, summarization, code — where it wins).
            var gated = _specAcceptEma < SpecGateThreshold;
            var probe = false;
            if (gated)
            {
                if (_specProbeCountdown > 0) { _specProbeCountdown--; }
                else { probe = true; _specProbeCountdown = SpecProbeInterval; }
            }

            var dn = 0;
            Span<int> draft = stackalloc int[maxDraft];
            if (drafter is not null)
            {
                // Draft-MODEL path: always propose (a model predicts, so the echo-detection gate doesn't
                // apply); the drafter keeps its own KV in lockstep via Sync at the commit points below.
                if (canSpeculate)
                {
                    dn = drafter.Draft(t0, draft);
                }
            }
            else if (canSpeculate && (!gated || probe))
            {
                var anchor = new int[history.Length + 1];
                history.CopyTo(anchor);
                anchor[^1] = t0;
                dn = PromptLookupDrafter.Draft(anchor, draft, ngramMin, ngramMax);
            }

            var batch = 1 + dn;
            var basePosition = _cache.CurrentLength;
            if (dn == 0 || basePosition + batch + 1 > _cache.MaxLength)
            {
                // No draft (or no room for verify + the bonus forward): a plain single-token step.
                committed[0] = t0;
                DecodeToken(t0);
                drafter?.Sync(committed.Slice(0, 1));
                return 1;
            }

            // Embed [t0, draft…] and run ONE batched verify forward → per-row target logits.
            var hidden = new float[batch * dModel];
            _embedWeights.DequantizeRow(t0, hidden.AsSpan(0, dModel));
            for (var j = 0; j < dn; j++)
            {
                _embedWeights.DequantizeRow(draft[j], hidden.AsSpan((1 + j) * dModel, dModel));
            }
            _cache.Advance(batch);

            var finalNorm = new float[batch * dModel];
            _stack.PrefillBatchedQuantAllRows(hidden, batch, _weights, _cache, basePosition, finalNorm, _rope);

            // Batched LM head — read the (large) head weights ONCE for all draft rows, else the per-row
            // re-read cancels the batched stack's saving (measured: 1.01× before this).
            var verifyLogits = new float[batch * vocab];
            _stack.ProjectLogitsBatched(finalNorm, batch, _weights, verifyLogits);

            committed[0] = t0;
            var accepted = 0;
            var probs = new float[vocab];
            var residual = new float[vocab];
            var correction = -1;
            for (var j = 0; j < dn; j++)
            {
                TokenSampler.ComputeProbabilities(
                    verifyLogits.AsSpan(j * vocab, vocab), in sampling, _indexScratch, _scoreScratch, probs);

                // Speculative rejection sampling: accept draft d w.p. p(d), else resample the residual.
                var token = SpeculativeSampler.AcceptOrResample(probs, draft[j], _random, residual);
                if (token == draft[j])
                {
                    committed[1 + j] = draft[j];
                    accepted++;
                }
                else
                {
                    correction = token;
                    break;
                }
            }

            if (correction < 0)
            {
                // All drafts accepted — the bonus token is sampled from the row after the last draft.
                TokenSampler.ComputeProbabilities(
                    verifyLogits.AsSpan(dn * vocab, vocab), in sampling, _indexScratch, _scoreScratch, probs);
                correction = SpeculativeSampler.Sample(probs, _random);
            }

            // Keep t0 + accepted drafts' K/V (drop rejected drafts), then forward the correction/bonus so
            // it is cached and _logits reflects the prediction after it (the session invariant).
            _cache.TruncateTo(basePosition + 1 + accepted);
            committed[1 + accepted] = correction;
            DecodeToken(correction);

            // Feed the verify outcome back into the adaptive gate: committed tokens this step (= 2 + accepted).
            var committedThisStep = 2 + accepted;
            drafter?.Sync(committed.Slice(0, committedThisStep));
            _specAcceptEma = SpecEmaAlpha * committedThisStep + (1 - SpecEmaAlpha) * _specAcceptEma;
            return committedThisStep;
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

        /// <summary>Embedding vector length (model dimension).</summary>
        public int EmbeddingDimension => _config.DModel;

        /// <summary>
        /// Encodes <paramref name="tokens"/> into a single embedding vector by pooling the
        /// per-token final hidden states — the in-process embeddings primitive for RAG /
        /// vector-store use. RESETS the session (it's a fresh encode pass, not generation;
        /// the KV cache ends filled with these tokens). <paramref name="destination"/> must be
        /// at least <see cref="EmbeddingDimension"/> long. L2-normalised by default (cosine-ready).
        /// </summary>
        public void Embed(
            ReadOnlySpan<int> tokens,
            Span<float> destination,
            EmbeddingPooling pooling = EmbeddingPooling.Mean,
            bool normalize = true)
        {
            ThrowIfDisposed();
            if (tokens.IsEmpty) { throw new ArgumentException("Cannot embed an empty token sequence.", nameof(tokens)); }
            var d = _config.DModel;
            if (destination.Length < d)
            {
                throw new ArgumentException($"Destination ({destination.Length}) is smaller than embedding dimension ({d}).", nameof(destination));
            }
            if (tokens.Length > _cache.MaxLength)
            {
                throw new ArgumentException(
                    $"Embedding input ({tokens.Length} tokens) exceeds context length {_cache.MaxLength}.", nameof(tokens));
            }

            Reset();
            var dst = destination[..d];
            dst.Clear();

            for (var i = 0; i < tokens.Length; i++)
            {
                DecodeTokenWithoutLogits(tokens[i]);   // updates _stack.LastFinalHidden
                var h = _stack.LastFinalHidden;
                if (pooling == EmbeddingPooling.Mean)
                {
                    for (var j = 0; j < d; j++) { dst[j] += h[j]; }
                }
                else if (i == tokens.Length - 1)
                {
                    h[..d].CopyTo(dst);
                }
            }

            if (pooling == EmbeddingPooling.Mean)
            {
                var inv = 1f / tokens.Length;
                for (var j = 0; j < d; j++) { dst[j] *= inv; }
            }

            if (normalize)
            {
                var norm = 0f;
                for (var j = 0; j < d; j++) { norm += dst[j] * dst[j]; }
                norm = MathF.Sqrt(norm);
                if (norm > 1e-12f)
                {
                    var inv = 1f / norm;
                    for (var j = 0; j < d; j++) { dst[j] *= inv; }
                }
            }
        }

        /// <summary>Convenience overload: allocates and returns the embedding vector.</summary>
        public float[] Embed(ReadOnlySpan<int> tokens, EmbeddingPooling pooling = EmbeddingPooling.Mean, bool normalize = true)
        {
            var result = new float[_config.DModel];
            Embed(tokens, result, pooling, normalize);
            return result;
        }

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

        /// <summary>Prompt length at/above which <see cref="Prefill"/> uses the batched path.</summary>
        private const int BatchedPrefillThreshold = 16;

        /// <summary>Test hook: force the single-token prefill loop (for batched-vs-single parity).</summary>
        internal bool DisableBatchedPrefillForParity { get; set; }

        /// <summary>
        /// Batched prefill: embed all prompt tokens, advance the cache, run one batched pass per layer
        /// (<see cref="CachedGptStack.PrefillBatchedQuant"/>), then project the last token's logits —
        /// leaving the session in exactly the state the single-token loop would (bit-identical).
        /// </summary>
        private void PrefillBatchedQuant(ReadOnlySpan<int> promptTokens)
        {
            var rows = promptTokens.Length;
            var dModel = _config.DModel;
            var basePosition = _cache.CurrentLength;

            var hidden = new float[rows * dModel];
            for (var i = 0; i < rows; i++)
            {
                _embedWeights.DequantizeRow(promptTokens[i], hidden.AsSpan(i * dModel, dModel));
                _cache.Advance();
            }

            _stack.PrefillBatchedQuant(hidden, rows, _weights, _cache, basePosition, _rope);
            _stack.ProjectLogits(_weights, _logits);
        }

        private int EmbedAndAdvance(int tokenId)
        {
            // Sliding window: free a block of oldest tokens before writing a new one
            // when the cache is full (RoPE-only; no-op otherwise).
            MakeRoomIfSliding();

            // Token embedding lookup: row tokenId of embed_weights [vocab × dModel], written
            // straight into _hidden. F32 backing → a plain slice-copy; K-quant backing →
            // dequantize just this one row (the per-token cost of the quantized embedding table).
            _embedWeights.DequantizeRow(tokenId, _hidden.AsSpan(0, _config.DModel));

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
