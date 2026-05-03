// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Selects the runtime implementation used for GPT-style SLM inference.
    /// </summary>
    public enum SlmRuntimeMode
    {
        /// <summary>
        /// Existing graph/context based session path.
        ///
        /// Keep this as a compatibility path, a parity reference and a benchmark
        /// baseline while the cached runtime matures.
        /// </summary>
        Legacy = 0,

        /// <summary>
        /// KV-cache backed session path.
        ///
        /// This is the new continuation-optimized runtime.
        /// </summary>
        Cached = 1
    }
}
