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
