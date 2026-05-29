// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Ops
{
    /// <summary>
    /// A label-level language model for CTC beam-search rescoring: the log-probability of the next
    /// label given the labels decoded so far. <see cref="CtcDecoder.BeamSearchDecode(ReadOnlySpan{float}, int, int, int, int, ICtcLanguageModel, double, double)"/>
    /// adds <c>weight · LogProbability(prefix, nextLabel)</c> each time a beam's labeling grows by
    /// <c>nextLabel</c>, steering decoding toward sequences the model considers plausible (e.g. a char
    /// n-gram over a lexicon). Return <see cref="double.NegativeInfinity"/> to forbid a continuation.
    /// </summary>
    public interface ICtcLanguageModel
    {
        /// <param name="prefix">The labels decoded so far (blank-collapsed), oldest first.</param>
        /// <param name="nextLabel">The candidate label about to extend the prefix.</param>
        /// <returns>Natural-log probability of <paramref name="nextLabel"/> following <paramref name="prefix"/>.</returns>
        double LogProbability(ReadOnlySpan<int> prefix, int nextLabel);
    }
}
