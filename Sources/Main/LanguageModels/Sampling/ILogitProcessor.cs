// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Sampling
{
    /// <summary>
    /// An in-place logit transform that depends on the generation HISTORY (unlike a stateless
    /// <see cref="ISamplerStep"/>) — e.g. a repetition / frequency / presence penalty, or a logit bias. Run
    /// by a <see cref="SamplingPipeline"/> before the stateless steps. Custom implementations plug new
    /// history-aware behaviour into the sampling pipeline.
    /// </summary>
    public interface ILogitProcessor
    {
        /// <summary>Transforms <paramref name="logits"/> in place using the tokens generated so far
        /// (<paramref name="history"/>, oldest→newest).</summary>
        void Process(Span<float> logits, ReadOnlySpan<int> history);
    }
}
