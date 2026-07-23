// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Contracts
{
    public interface ISlmSession : IDisposable
    {
        int CurrentPosition
        {
            get;
        }

        int MaxContextLength
        {
            get;
        }

        int VocabularySize
        {
            get;
        }

        bool HasKeyValueCache
        {
            get;
        }

        void Reset();

        void Reset(ReadOnlySpan<int> promptTokens);

        /// <summary>
        /// Prefills <paramref name="promptTokens"/>, reusing whatever leading portion is already in this
        /// session's KV cache, and returns how many tokens that saved. The end state matches
        /// <see cref="Reset(System.ReadOnlySpan{int})"/> exactly — reuse is an optimisation, never a
        /// behaviour change. The default implementation reuses nothing, so sessions that do not track
        /// their cached tokens keep working unchanged.
        ///
        /// <para>This is the multi-turn chat lever: every turn re-sends the whole conversation, so without
        /// reuse turn N re-encodes everything turns 1..N-1 already encoded.</para>
        /// </summary>
        int PrefillReusingCache(ReadOnlySpan<int> promptTokens)
        {
            Reset(promptTokens);
            return 0;
        }

        int GenerateNextToken(in SamplingOptions sampling);

        /// <summary>
        /// True when this session supports sliding-window KV eviction — a rolling context that
        /// keeps generating past the cache length by dropping the oldest tokens (RoPE models).
        /// </summary>
        bool SupportsSlidingWindow => false;

        /// <summary>
        /// Enables sliding-window KV eviction: once the cache fills, the oldest tokens are dropped
        /// instead of throwing, so generation and prefill continue over a rolling context.
        /// <paramref name="evictBlock"/> = how many tokens to drop per eviction (0 ⇒ a sensible
        /// default). Throws <see cref="OverfitRuntimeException"/> on sessions that don't support it.
        /// </summary>
#pragma warning disable RS0030 // Type.Name = compile-time-safe type name for a diagnostic, not runtime reflection (AOT-safe)
        void EnableSlidingWindow(int evictBlock = 0)
            => throw new OverfitRuntimeException(
                $"{GetType().Name} does not support sliding-window eviction.");
#pragma warning restore RS0030

        /// <summary>
        /// Generates the next token under a decode-time <paramref name="constraint"/> (e.g. JSON-mode):
        /// the constraint masks the logits before sampling and is advanced by the chosen token.
        /// Sessions that don't support constrained generation throw <see cref="OverfitRuntimeException"/>
        /// when a non-null constraint is supplied (a null constraint always defers to the plain path).
        /// </summary>
        int GenerateNextToken(in SamplingOptions sampling, ITokenConstraint? constraint)
#pragma warning disable RS0030 // Type.Name = compile-time-safe type name for a diagnostic, not runtime reflection (AOT-safe)
            => constraint is null
                ? GenerateNextToken(in sampling)
                : throw new OverfitRuntimeException(
                    $"{GetType().Name} does not support constrained generation.");
#pragma warning restore RS0030

        /// <summary>
        /// Generates the next token and hands it to <paramref name="onSampled"/> as early as the
        /// implementation can — ideally before the forward pass that prepares the following logits, which is
        /// what lets a streaming caller put the token on the wire a whole weight-pass sooner. Returning
        /// <c>true</c> from the hook says the caller is finished with generation, letting the implementation
        /// skip that pass entirely.
        ///
        /// <para>The default implementation invokes the hook <i>after</i> the step instead, so a caller can
        /// rely on it firing exactly once per token whatever the session: it is a latency optimisation where
        /// supported, never a difference in what gets emitted.</para>
        /// </summary>
        int GenerateNextToken(in SamplingOptions sampling, ITokenConstraint? constraint, Func<int, bool>? onSampled)
        {
            var token = GenerateNextToken(in sampling, constraint);
            onSampled?.Invoke(token);
            return token;
        }

        int Generate(
            ReadOnlySpan<int> promptTokens,
            Span<int> outputTokens,
            in GenerationOptions options);

        void GetLastLogits(Span<float> destination);
    }
}
