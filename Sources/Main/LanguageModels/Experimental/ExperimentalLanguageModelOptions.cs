// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Experimental
{
    /// <summary>
    /// Experimental language-model switches.
    ///
    /// Defaults must keep the stable/reference path unchanged.
    ///
    /// These options are intentionally global and explicit. They are for manual
    /// experiments, demos, and benchmarks while the GPT runtime/training surface
    /// is being stabilized.
    /// </summary>
    public static class ExperimentalLanguageModelOptions
    {
        private static volatile bool _enableParallelAttentionBackward;

        /// <summary>
        /// Enables the experimental parallel implementation of
        /// ScaledDotProductAttentionBackward.
        ///
        /// Default: false.
        ///
        /// Keep false for normal correctness tests and stable training paths.
        /// Set true only in explicit performance experiments, e.g. the
        /// TinyShakespeare data-parallel training demo.
        /// </summary>
        public static bool EnableParallelAttentionBackward
        {
            get => _enableParallelAttentionBackward;
            set => _enableParallelAttentionBackward = value;
        }
    }
}
