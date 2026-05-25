// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Contracts
{
    /// <summary>
    /// A stateful decode-time constraint: before each token is sampled it masks the logits of any
    /// token that would violate the constraint (sets them to <see cref="float.NegativeInfinity"/>),
    /// and after a token is chosen it advances its internal state. This is how structured output
    /// (e.g. guaranteed well-formed JSON) is enforced — the model physically cannot emit an invalid
    /// token, rather than merely being asked to behave in the prompt.
    ///
    /// Stateful and single-generation: create one per generation, never share across sessions.
    /// </summary>
    public interface ITokenConstraint
    {
        /// <summary>
        /// Masks <paramref name="logits"/> in place for the current state — every token that cannot
        /// legally come next is set to <see cref="float.NegativeInfinity"/>. Indexed by token id
        /// (so <paramref name="logits"/>.Length must equal the vocabulary size).
        /// </summary>
        void ApplyMask(Span<float> logits);

        /// <summary>Advances the constraint state by the token that was actually sampled.</summary>
        void Accept(int token);

        /// <summary>
        /// True when generation may validly stop now (the produced output already satisfies the
        /// constraint). The end-of-text token is only left unmasked by <see cref="ApplyMask"/> when
        /// this is true.
        /// </summary>
        bool IsComplete { get; }
    }
}
