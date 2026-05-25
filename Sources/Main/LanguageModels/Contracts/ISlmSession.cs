// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Contracts
{
    public interface ISlmSession : IDisposable
    {
        int CurrentPosition { get; }

        int MaxContextLength { get; }

        int VocabularySize { get; }

        bool HasKeyValueCache { get; }

        void Reset();

        void Reset(ReadOnlySpan<int> promptTokens);

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
        /// default). Throws <see cref="NotSupportedException"/> on sessions that don't support it.
        /// </summary>
        void EnableSlidingWindow(int evictBlock = 0)
            => throw new NotSupportedException(
                $"{GetType().Name} does not support sliding-window eviction.");

        /// <summary>
        /// Generates the next token under a decode-time <paramref name="constraint"/> (e.g. JSON-mode):
        /// the constraint masks the logits before sampling and is advanced by the chosen token.
        /// Sessions that don't support constrained generation throw <see cref="NotSupportedException"/>
        /// when a non-null constraint is supplied (a null constraint always defers to the plain path).
        /// </summary>
        int GenerateNextToken(in SamplingOptions sampling, ITokenConstraint? constraint)
            => constraint is null
                ? GenerateNextToken(in sampling)
                : throw new NotSupportedException(
                    $"{GetType().Name} does not support constrained generation.");

        int Generate(
            ReadOnlySpan<int> promptTokens,
            Span<int> outputTokens,
            in GenerationOptions options);

        void GetLastLogits(Span<float> destination);
    }
}
