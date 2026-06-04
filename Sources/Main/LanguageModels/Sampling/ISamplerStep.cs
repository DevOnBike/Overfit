// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Sampling
{
    /// <summary>
    /// A stateless, in-place transform of a logit vector — one stage of a <see cref="SamplingPipeline"/>
    /// (temperature, top-k, top-p, min-p, or a custom transform). Filtering steps mask removed tokens by
    /// setting their logit to <see cref="float.NegativeInfinity"/>; scaling steps (temperature) rescale the
    /// finite logits. Steps compose in order; the pipeline's terminal softmax+sample sees the result.
    ///
    /// This is the additive, composable counterpart to the monolithic <c>TokenSampler</c>/<c>SamplingOptions</c>
    /// path — it lets a caller assemble (and extend) a custom sampling strategy without touching the engine's
    /// default hot path.
    /// </summary>
    public interface ISamplerStep
    {
        /// <summary>Transforms <paramref name="logits"/> in place.</summary>
        void Apply(System.Span<float> logits);
    }
}
