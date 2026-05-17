// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Factory for selecting SLM runtime implementation.
    ///
    /// This deliberately does not remove SlmInferenceEngine.
    ///
    /// Current recommendation:
    ///
    /// - use Legacy for compatibility, parity tests and baseline benchmarks,
    /// - use Cached for continuation-heavy autoregressive generation.
    /// </summary>
    public static class SlmRuntimeFactory
    {
        public static SlmRuntimeHandle CreateGpt1(
            GPT1Model model,
            SlmRuntimeMode mode = SlmRuntimeMode.Cached)
        {
            if (model is null)
            {
                throw new ArgumentNullException(nameof(model));
            }

            return mode switch
            {
                SlmRuntimeMode.Legacy => CreateLegacyGpt1(model),
                SlmRuntimeMode.Cached => CreateCachedGpt1(model),
                _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, "Unsupported SLM runtime mode.")
            };
        }

        public static SlmRuntimeHandle CreateLegacyGpt1(GPT1Model model)
        {
            if (model is null)
            {
                throw new ArgumentNullException(nameof(model));
            }

            var engine = SlmInferenceEngine.FromGpt1(model);
            var session = engine.CreateSession(model.Config.ContextLength);

            return new SlmRuntimeHandle(
                SlmRuntimeMode.Legacy,
                session,
                engine);
        }

        public static SlmRuntimeHandle CreateCachedGpt1(GPT1Model model)
        {
            if (model is null)
            {
                throw new ArgumentNullException(nameof(model));
            }

            var engine = CachedSlmInferenceEngine.FromGpt1(model);
            var session = engine.CreateSession();

            return new SlmRuntimeHandle(
                SlmRuntimeMode.Cached,
                session,
                engine);
        }
    }
}
