// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Evolutionary
{
    public interface ILanguageModelMutator<TCandidate>
    {
        ISlmInferenceEngine Engine { get; }

        bool TryBuildPrompt(
            in TCandidate candidate,
            Span<char> promptDestination,
            out int charsWritten);

        bool TryParseMutation(
            ReadOnlySpan<char> generatedText,
            in TCandidate original,
            out TCandidate mutated);

        bool TryMutate(
            in TCandidate candidate,
            in PromptMutationOptions options,
            out TCandidate mutated);
    }
}
