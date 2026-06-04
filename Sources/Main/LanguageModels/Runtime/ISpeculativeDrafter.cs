// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Source of speculative draft tokens for <c>CachedLlamaSession.GenerateSpeculative</c>. The default
    /// path uses an inline prompt-lookup (n-gram) drafter; supplying a custom drafter — e.g. a small draft
    /// MODEL via <see cref="DraftModelSpeculativeDrafter"/> — lets speculation win on NOVEL text, not just
    /// repetition. The verify / accept-or-resample machinery is unchanged: a drafter only proposes tokens.
    /// </summary>
    internal interface ISpeculativeDrafter
    {
        /// <summary>
        /// Proposes up to <paramref name="draftOut"/>.Length continuation tokens that follow
        /// <paramref name="seed"/> (the token the target just chose, which itself follows the already-synced
        /// committed context). Returns the count written (0 = no proposal). May advance internal state
        /// speculatively; <see cref="Sync"/> reconciles it afterwards.
        /// </summary>
        int Draft(int seed, Span<int> draftOut);

        /// <summary>
        /// Reconciles internal state with the tokens actually committed this step
        /// (<paramref name="committedThisStep"/> = seed + accepted drafts + correction/bonus), discarding any
        /// speculative state from the matching <see cref="Draft"/> call that was not committed.
        /// </summary>
        void Sync(ReadOnlySpan<int> committedThisStep);
    }
}
