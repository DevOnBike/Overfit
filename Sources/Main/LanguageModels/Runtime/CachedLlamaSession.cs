// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Rope;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;

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

        // DRY (Don't-Repeat-Yourself) anti-loop: a rolling history of the tokens THIS session generated, plus
        // reusable Z-algorithm scratch. ApplyDry penalises would-be verbatim repetitions on the logits before
        // sampling — bounded to the last DryHistoryCap tokens so the scratch stays fixed-size (zero per-token alloc).
        private const int DryHistoryCap = 512;
        private readonly List<int> _generatedTokens = new(DryHistoryCap);
        private readonly int[] _dryRev = new int[DryHistoryCap];
        private readonly int[] _dryZ = new int[DryHistoryCap];

        // Prompt cache: the token ids currently represented in the KV cache, indexed BY CACHE POSITION.
        // The live region is always [0, _cache.CurrentLength) — which is what makes truncation free
        // bookkeeping-wise: dropping KV state past N automatically drops these too, and the stale tail is
        // overwritten when the cache refills. Sized to the context length once, so recording a token is a
        // single array store and the zero-allocation decode invariant is preserved.
        private readonly int[] _cacheTokens;

        // Cleared whenever cache positions stop corresponding to recorded ids — sliding-window eviction
        // (every id shifts down) and prefix restore (ids belong to whoever took the snapshot). Reuse then
        // falls back to a full prefill rather than attending over K/V that does not match the prompt.
        private bool _cacheTokensValid = true;

        // The logits left by the most recent prefill, plus the cache length they belong to (-1 = none).
        //
        // K/V reuse alone still costs one forward pass, because logits are a by-product of the stack rather
        // than cache state: even a prompt the cache holds in full has to re-run its last token to learn what
        // comes next. Keeping the end-of-prompt logits removes that last pass — an exact match restores them
        // with a copy and forwards nothing at all. Valid for as long as tokens [0, position) are untouched,
        // which a whole turn of generation is, since decoding only ever appends.
        //
        // Costs one float[vocab] per session (~608 KB for Qwen-3B's 151936-wide vocabulary) against a KV
        // cache measured in tens of megabytes, and one memcpy per prefill.
        private readonly float[] _promptLogits;
        private int _promptLogitsPosition = -1;

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
            _cacheTokens = new int[cache.MaxLength];
            _promptLogits = new float[config.VocabSize];
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
                throw new OverfitRuntimeException(
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
                // Eviction shifts every surviving token down by `count`, so the recorded ids no longer sit
                // at their own positions. Rebuilding the map would be cheap, but a slid session's prompt no
                // longer starts at position 0 either — prefix reuse is meaningless once the head is gone.
                _cacheTokensValid = false;
                _promptLogitsPosition = -1;
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
            _generatedTokens.Clear();
            _cacheTokensValid = true;
            _promptLogitsPosition = -1;
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

            if (promptTokens.IsEmpty)
            {
                return;
            }

            // Batched (multi-token) prefill: one set of batched GEMMs per layer over the whole prompt
            // instead of N single-token passes — amortises the (weight-bandwidth-bound) weight reads
            // ~N×. Eligible for the dense SwiGLU Llama/Qwen path, non-sliding, when the prompt fits the
            // remaining context. MoE / sliding-window / tiny prompts fall back to the single-token loop.
            if (!DisableBatchedPrefillForParity
                && promptTokens.Length >= BatchedPrefillThreshold
                && !_slidingWindow
                && _config.FfnActivation is FeedForwardActivation.SwiGLU or FeedForwardActivation.GeGLU
                && _cache.CurrentLength + promptTokens.Length <= _cache.MaxLength)
            {
                PrefillProfiler.BeginRequest(promptTokens.Length);
                PrefillBatchedQuant(promptTokens);
                PrefillProfiler.EndRequest();
                SnapshotPromptLogits();
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
                    throw new OverfitRuntimeException(
                        $"Prefill of {promptTokens.Length} tokens would exceed ContextLength {_config.ContextLength} " +
                        $"(current position {Position}).");
                }

                if (i < lastIndex)
                {
                    DecodeTokenWithoutLogits(promptTokens[i]);
                }

                if (!(i < lastIndex))
                {
                    DecodeToken(promptTokens[i]);
                }
            }

            SnapshotPromptLogits();
        }

        /// <summary>
        /// Records the logits this prefill just produced against the cache length they describe, so a later
        /// prompt that the cache already holds in full can skip the forward pass entirely.
        /// </summary>
        private void SnapshotPromptLogits()
        {
            _logits.AsSpan(0, VocabularySize).CopyTo(_promptLogits.AsSpan(0, VocabularySize));
            _promptLogitsPosition = _cache.CurrentLength;
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

            // The restored K/V belongs to whoever took the snapshot; this session never saw those token ids,
            // so it cannot claim any prefix matches them.
            _cacheTokensValid = false;
            _promptLogitsPosition = -1;
            _cache.RestoreFrom(prefix);
        }

        /// <summary>
        /// Prefills <paramref name="promptTokens"/> <b>reusing the longest prefix already present in the KV
        /// cache</b>, and returns how many tokens that reuse saved. The remainder is prefilled normally, so
        /// the resulting state is the same one <see cref="Reset(System.ReadOnlySpan{int})"/> would leave —
        /// this trades no accuracy for the saving, because K/V for a given position depends only on the
        /// tokens at and before it, which are by construction identical across the matched prefix.
        ///
        /// <para><b>What it is for.</b> In a chat server every turn re-sends the whole conversation, so turn
        /// N re-encodes everything turns 1..N-1 already encoded. Measured against a competing pure-.NET
        /// engine through the same load driver, its prompt cache answered a repeated prompt in 47 ms where
        /// its own cold prefill of the same prompt took 865 ms — an 18x difference that has nothing to do
        /// with kernel quality and everything to do with not doing the work twice.</para>
        ///
        /// <para><b>One token is always re-forwarded.</b> Even on an exact match the last token is dropped
        /// and re-run, because <c>_logits</c> must predict the token that follows the prompt, and those
        /// logits are a by-product of the forward pass rather than cache state. So a fully cached prompt
        /// still costs one decode step, not zero.</para>
        ///
        /// <para>Falls back to a full reset+prefill when the recorded ids cannot be trusted (after
        /// sliding-window eviction or <see cref="RestorePrefix"/>) or when nothing matches. The DRY history
        /// is deliberately <i>not</i> rewound — it is a bounded anti-repetition heuristic over what this
        /// session emitted, not part of the cache contract.</para>
        /// </summary>
        public int PrefillReusingCache(ReadOnlySpan<int> promptTokens)
        {
            ThrowIfDisposed();

            var match = MatchingPrefixLength(promptTokens);

            if (DisableLogitsCache)
            {
                return PrefillReusingKeyValuesOnly(promptTokens, match);
            }

            // Everything matches AND we kept the logits this exact prompt produced: restore them and forward
            // nothing. This is the re-sent-prompt case (a retry, a regenerate, a load test), where even the
            // one-token fallback below would be re-deriving something already computed.
            if (match == promptTokens.Length && _promptLogitsPosition == promptTokens.Length)
            {
                _cache.TruncateTo(promptTokens.Length);
                _promptLogits.AsSpan(0, VocabularySize).CopyTo(_logits.AsSpan(0, VocabularySize));
                return promptTokens.Length;
            }

            return PrefillReusingKeyValuesOnly(promptTokens, match);
        }

        /// <summary>Test hook: skip the kept-logits fast path so the K/V-only behaviour can be A/B'd against
        /// it. Defaults from <see cref="OverfitEnvironment.DisableLogitsCache"/> so a server process can be
        /// started in either configuration and both measured in one interleaved run.</summary>
        internal static bool DisableLogitsCache =
            Environment.GetEnvironmentVariable(OverfitEnvironment.DisableLogitsCache) == "1";

        /// <summary>
        /// Reuse K/V only: hold one token back — logits are a by-product of the stack, so the last token has
        /// to go through it for the session to learn what follows the prompt.
        /// </summary>
        private int PrefillReusingKeyValuesOnly(ReadOnlySpan<int> promptTokens, int match)
        {
            var reusable = Math.Min(match, promptTokens.Length - 1);
            if (reusable <= 0)
            {
                Reset(promptTokens);
                return 0;
            }

            // Any truncation below the snapshot leaves it describing K/V the cache no longer holds. The
            // Prefill below re-establishes it; dropping it first means no window where it could be believed.
            _promptLogitsPosition = -1;
            _cache.TruncateTo(reusable);
            Prefill(promptTokens[reusable..]);
            return reusable;
        }

        /// <summary>
        /// How many leading tokens of <paramref name="promptTokens"/> are already in the cache at the very
        /// positions they would occupy — the raw match, before any decision about holding a token back.
        /// </summary>
        private int MatchingPrefixLength(ReadOnlySpan<int> promptTokens)
        {
            if (!_cacheTokensValid || _slidingWindow || _cache.BasePosition != 0)
            {
                return 0;
            }

            var limit = Math.Min(_cache.CurrentLength, promptTokens.Length);
            var match = 0;
            while (match < limit && _cacheTokens[match] == promptTokens[match])
            {
                match++;
            }

            return match;
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
            => GenerateNextToken(in sampling, constraint, onSampled: null);

        /// <summary>
        /// As <see cref="GenerateNextToken(in SamplingOptions, ITokenConstraint?)"/>, but hands the sampled
        /// token to <paramref name="onSampled"/> <b>before</b> the forward pass that follows it.
        ///
        /// <para><b>Why the ordering is worth an API.</b> A decode step samples token N from the logits it
        /// already holds, then runs a full pass over the weights so that logits predict token N+1. Emitting
        /// after that pass makes every token — including the first — arrive one whole pass late. Measured
        /// through the server on Qwen-3B with a fully cached prompt: time to first token was 74.8 ms against
        /// an inter-token latency of 37.2 ms, i.e. exactly two passes, where one is all the answer needs.</para>
        ///
        /// <para>Returning <c>true</c> from the hook means the caller is finished with this token (a stop
        /// sequence, end-of-text, a closed constraint), so the trailing pass is skipped entirely — it would
        /// only have prepared logits nobody reads. The token is then <b>not</b> fed back into the cache, which
        /// is the honest state: the cache holds what was forwarded. Do not continue generating on the same
        /// session after returning <c>true</c> without resetting or prefilling — the logits still predict the
        /// token just sampled, so the next step would draw it again.</para>
        /// </summary>
        public int GenerateNextToken(
            in SamplingOptions sampling,
            ITokenConstraint? constraint,
            Func<int, bool>? onSampled)
        {
            ThrowIfDisposed();

            if (_cache.IsFull && !_slidingWindow)
            {
                throw new OverfitRuntimeException(
                    $"KV cache is full (ContextLength={_config.ContextLength}). Start a new session.");
            }

            if (Position == 0)
            {
                throw new OverfitRuntimeException(
                    "Session is empty. Call Reset with at least one prompt token first.");
            }

            DecodeProfiler.BeginToken();
            constraint?.ApplyMask(_logits.AsSpan(0, VocabularySize));
            ApplyDry(in sampling);

            var profSample = DecodeProfiler.Start();
            var token = TokenSampler.Sample(
                _logits, in sampling, _random, _indexScratch, _scoreScratch);
            DecodeProfiler.Stop(DecodeProfiler.Component.Sampler, profSample);

            constraint?.Accept(token);
            TrackGenerated(token);

            // Hand the token over before the pass that prepares the NEXT logits, so a streaming caller can
            // put it on the wire a full weight-pass earlier — and can tell us the answer is finished, in
            // which case that pass is pure waste and is skipped.
            if (onSampled is not null && onSampled(token))
            {
                DecodeProfiler.EndToken();
                return token;
            }

            DecodeToken(token);
            DecodeProfiler.EndToken();
            return token;
        }

        /// <summary>
        /// Applies the DRY (Don't-Repeat-Yourself) penalty to the current logits before sampling: subtracts a
        /// length-scaled penalty from any token that would extend a verbatim repetition of the recent output.
        /// Acts on the logits (not the sampler), so it constrains greedy and stochastic decode alike, and reuses
        /// fixed scratch for the Z-algorithm — zero per-token allocation. No-op when DryMultiplier ≤ 0.
        /// </summary>
        private void ApplyDry(in SamplingOptions sampling)
        {
            if (sampling.DryMultiplier <= 0f || _generatedTokens.Count < 2)
            {
                return;
            }

            var count = _generatedTokens.Count;
            var lastN = sampling.DryPenaltyLastN;
            var start = lastN > 0 && count > lastN ? count - lastN : 0;
            var m = count - start;
            if (m < 2 || m > DryHistoryCap)
            {
                return;
            }

            // Zero-alloc window over the list's backing store; the DRY core reuses _dryRev/_dryZ for the Z-algorithm.
            var window = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_generatedTokens).Slice(start, m);
            Sampling.SamplingPipeline.Dry.Apply(
                _logits.AsSpan(0, VocabularySize),
                window,
                sampling.DryMultiplier,
                sampling.DryBase,
                sampling.DryAllowedLength,
                _dryRev.AsSpan(0, m),
                _dryZ.AsSpan(0, m));
        }

        /// <summary>Appends a generated token to the DRY history, dropping the oldest once the cap is hit so the
        /// window (and thus the reusable scratch) stays bounded. No-op cost when DRY is unused beyond the append.</summary>
        private void TrackGenerated(int token)
        {
            if (_generatedTokens.Count >= DryHistoryCap)
            {
                _generatedTokens.RemoveAt(0);
            }
            _generatedTokens.Add(token);
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
                throw new OverfitRuntimeException(
                    $"KV cache is full (ContextLength={_config.ContextLength}). Start a new session.");
            }
            if (Position == 0)
            {
                throw new OverfitRuntimeException("Session is empty. Call Reset with a prompt first.");
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
        private const int SpecProbeInterval = 64;

        /// <summary>
        /// Hard ceiling on the speculative-decode draft length. It exists to bound a <c>stackalloc</c>: without
        /// it a caller-supplied draft length reaches the stack unchecked, which turns a tuning knob into a
        /// StackOverflowException. 64 ints is 256 B, and no useful draft is anywhere near that long.
        /// </summary>
        private const int MaxSpeculativeDraft = 64;      // while gated, draft once every N steps to re-detect echo
        private double _specAcceptEma;                 // start pessimistic (0 → gated): single-token until a probe
                                                       // proves drafting pays. Novel text (chat) stays ≈ 1× — one
                                                       // probe per SpecProbeInterval; repetitive text ramps up fast.
        private int _specProbeCountdown;               // 0 → the first step probes immediately

        /// <summary>True when this session can run speculative decoding (SwiGLU FFN, non-sliding) — lets a
        /// generate loop pick the speculative path. <c>GenerateSpeculative</c> also falls back to a
        /// single-token step internally when this is false, so calling it is always safe.</summary>
        public bool CanSpeculate => !_slidingWindow && _config.FfnActivation is FeedForwardActivation.SwiGLU or FeedForwardActivation.GeGLU;

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
            => GenerateSpeculativeCore(
                history, committed, in sampling, maxDraft, ngramMin, ngramMax, drafter: null, onSampled: null);

        /// <summary>
        /// Speculative step with the same early-emit hook as
        /// <see cref="GenerateNextToken(in SamplingOptions, ITokenConstraint?, Func{int, bool})"/>: the first
        /// token of the step is drawn from the logits already held, so it can go out before the verify
        /// forward runs. Returning <c>true</c> ends the step immediately — the whole verify is skipped, not
        /// just a single pass.
        /// </summary>
        public int GenerateSpeculative(
            ReadOnlySpan<int> history,
            Span<int> committed,
            in SamplingOptions sampling,
            int maxDraft,
            Func<int, bool>? onSampled)
            => GenerateSpeculativeCore(
                history, committed, in sampling, maxDraft, ngramMin: 1, ngramMax: 3, drafter: null,
                onSampled: onSampled);

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
            => GenerateSpeculativeCore(
                history, committed, in sampling, maxDraft, ngramMin: 1, ngramMax: 3, drafter: drafter,
                onSampled: null);

        private int GenerateSpeculativeCore(
            ReadOnlySpan<int> history,
            Span<int> committed,
            in SamplingOptions sampling,
            int maxDraft,
            int ngramMin,
            int ngramMax,
            ISpeculativeDrafter? drafter,
            Func<int, bool>? onSampled)
        {
            ThrowIfDisposed();
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxDraft);
            if (committed.Length < maxDraft + 2)
            {
                throw new ArgumentException("committed buffer must hold at least maxDraft+2 tokens.", nameof(committed));
            }
            if (Position == 0)
            {
                throw new OverfitRuntimeException("Session is empty. Call Reset with a prompt first.");
            }

            var dModel = _config.DModel;
            var vocab = VocabularySize;

            // Next token from the current (target) distribution — the same draw a normal step would make.
            var t0 = TokenSampler.Sample(_logits, in sampling, _random, _indexScratch, _scoreScratch);

            // Early emit: t0 comes from logits we already hold, so it can reach the client before the verify
            // forward. If the caller says the answer ends here, the entire verify is wasted work — skip it.
            if (onSampled is not null && onSampled(t0))
            {
                committed[0] = t0;
                return 1;
            }

            var canSpeculate = !_slidingWindow
                && _config.FfnActivation is FeedForwardActivation.SwiGLU or FeedForwardActivation.GeGLU
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
                // Capture BEFORE decrementing: a second `_specProbeCountdown > 0` test would read the
                // already-decremented value, so countdown == 1 would both decrement AND probe.
                var countingDown = _specProbeCountdown > 0;

                if (countingDown)
                {
                    _specProbeCountdown--;
                }

                if (!countingDown)
                {
                    probe = true;
                    _specProbeCountdown = SpecProbeInterval;
                }
            }

            var dn = 0;

            // maxDraft reached the stack unvalidated: a caller passing a large value would have turned a
            // speculative-decode knob into a stack overflow. Bounded explicitly, which also makes the
            // allocation below provably small (64 ints = 256 B, inside the OVERFIT025 budget).
            if (maxDraft is < 1 or > MaxSpeculativeDraft)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(maxDraft), maxDraft, $"maxDraft must be in [1, {MaxSpeculativeDraft}].");
            }

#pragma warning disable OVERFIT026 // BOUND: maxDraft is validated to [1, MaxSpeculativeDraft] directly above.
            Span<int> draft = stackalloc int[maxDraft];
#pragma warning restore OVERFIT026
            if (drafter is not null)
            {
                // Draft-MODEL path: always propose (a model predicts, so the echo-detection gate doesn't
                // apply); the drafter keeps its own KV in lockstep via Sync at the commit points below.
                if (canSpeculate)
                {
                    dn = drafter.Draft(t0, draft);
                }
            }
            if (drafter is null && canSpeculate && (!gated || probe))
            {
#pragma warning disable OVERFIT001 // exact-length contract: PromptLookupDrafter.Draft reads anchor.Length; tiny per-step array
                var anchor = new int[history.Length + 1];
#pragma warning restore OVERFIT001
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
            // Pooled per-step scratch; probs/residual are EXACT slices (the samplers read .Length).
            using var hiddenArr = new PooledBuffer<float>(batch * dModel, clearMemory: false);
            using var finalNormArr = new PooledBuffer<float>(batch * dModel, clearMemory: false);
            using var verifyLogitsArr = new PooledBuffer<float>(batch * vocab, clearMemory: false);
            using var probsArr = new PooledBuffer<float>(vocab, clearMemory: false);
            using var residualArr = new PooledBuffer<float>(vocab, clearMemory: false);

            var hidden = hiddenArr.Span;
            _embedWeights.DequantizeRow(t0, hidden.Slice(0, dModel));
            ApplyEmbeddingScale(hidden.Slice(0, dModel));
            _cacheTokens[basePosition] = t0;
            for (var j = 0; j < dn; j++)
            {
                _embedWeights.DequantizeRow(draft[j], hidden.Slice((1 + j) * dModel, dModel));
                ApplyEmbeddingScale(hidden.Slice((1 + j) * dModel, dModel));

                // Record the drafts too: this batch bypasses EmbedAndAdvance, and the truncation below keeps
                // exactly the accepted prefix — so recording all of them and letting TruncateTo cut the
                // rejected tail leaves the map correct without a second pass.
                _cacheTokens[basePosition + 1 + j] = draft[j];
            }
            _cache.Advance(batch);

            var finalNorm = finalNormArr.Span;
            _stack.PrefillBatchedQuantAllRows(hidden, batch, _weights, _cache, basePosition, finalNorm, _rope);

            // Batched LM head — read the (large) head weights ONCE for all draft rows, else the per-row
            // re-read cancels the batched stack's saving (measured: 1.01× before this).
            var verifyLogits = verifyLogitsArr.Span;
            _stack.ProjectLogitsBatched(finalNorm, batch, _weights, verifyLogits);

            committed[0] = t0;
            var accepted = 0;
            var probs = probsArr.Span;
            var residual = residualArr.Span;
            var correction = -1;
            for (var j = 0; j < dn; j++)
            {
                TokenSampler.ComputeProbabilities(
                    verifyLogits.Slice(j * vocab, vocab), in sampling, _indexScratch, _scoreScratch, probs);

                // Speculative rejection sampling: accept draft d w.p. p(d), else resample the residual.
                var token = SpeculativeSampler.AcceptOrResample(probs, draft[j], _random, residual);
                if (token == draft[j])
                {
                    committed[1 + j] = draft[j];
                    accepted++;
                }

                if (!(token == draft[j]))
                {
                    correction = token;
                    break;
                }
            }

            if (correction < 0)
            {
                // All drafts accepted — the bonus token is sampled from the row after the last draft.
                TokenSampler.ComputeProbabilities(
                    verifyLogits.Slice(dn * vocab, vocab), in sampling, _indexScratch, _scoreScratch, probs);
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
                throw new OverfitRuntimeException(
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
                if (stops[i] == token)
                {
                    return true;
                }
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
            if (tokens.IsEmpty)
            {
                throw new ArgumentException("Cannot embed an empty token sequence.", nameof(tokens));
            }
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
                    for (var j = 0; j < d; j++)
                    {
                        dst[j] += h[j];
                    }
                }
                if (pooling != EmbeddingPooling.Mean && i == tokens.Length - 1)
                {
                    h[..d].CopyTo(dst);
                }
            }

            if (pooling == EmbeddingPooling.Mean)
            {
                var inv = 1f / tokens.Length;
                for (var j = 0; j < d; j++)
                {
                    dst[j] *= inv;
                }
            }

            if (normalize)
            {
                var norm = 0f;
                for (var j = 0; j < d; j++)
                {
                    norm += dst[j] * dst[j];
                }
                norm = MathF.Sqrt(norm);
                if (norm > 1e-12f)
                {
                    var inv = 1f / norm;
                    for (var j = 0; j < d; j++)
                    {
                        dst[j] *= inv;
                    }
                }
            }
        }

        /// <summary>Convenience overload: allocates and returns the embedding vector.</summary>
        public float[] Embed(ReadOnlySpan<int> tokens, EmbeddingPooling pooling = EmbeddingPooling.Mean, bool normalize = true)
        {
#pragma warning disable OVERFIT001 // public contract: returns a caller-owned fresh array (dModel-sized, per Embed call)
            var result = new float[_config.DModel];
#pragma warning restore OVERFIT001
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
        internal bool DisableBatchedPrefillForParity
        {
            get; set;
        }

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

            using var hidden = new PooledBuffer<float>(rows * dModel, clearMemory: false);

            for (var i = 0; i < rows; i++)
            {
                _embedWeights.DequantizeRow(promptTokens[i], hidden.Span.Slice(i * dModel, dModel));
                ApplyEmbeddingScale(hidden.Span.Slice(i * dModel, dModel));
                _cacheTokens[basePosition + i] = promptTokens[i];
                _cache.Advance();
            }

            _stack.PrefillBatchedQuant(hidden.Span, rows, _weights, _cache, basePosition, _rope);
            _stack.ProjectLogits(_weights, _logits);
        }

        // Gemma scales the token embedding by sqrt(d_model) after lookup; no-op (scale 1) for every other arch.
        private void ApplyEmbeddingScale(Span<float> hidden)
        {
            var s = _config.EmbeddingScale;
            if (s == 1f)
            {
                return;
            }
            for (var i = 0; i < hidden.Length; i++)
            {
                hidden[i] *= s;
            }
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
            ApplyEmbeddingScale(_hidden.AsSpan(0, _config.DModel));

            // No additive positional embedding — RoPE handles positions inside attention.

            var position = _cache.CurrentLength;
            _cacheTokens[position] = tokenId;
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
