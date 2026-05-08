// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Activation used by the cached single-token feed-forward decode block.
    /// </summary>
    public enum FeedForwardActivation
    {
        None   = 0,
        ReLU   = 1,
        GeLU   = 2,

        /// <summary>
        /// SwiGLU (Shazeer, 2020) — used by Llama, Mistral, Qwen, Phi.
        ///
        /// FFN(x) = (SiLU(x @ Wgate) * (x @ Wup)) @ Wdown
        ///
        /// Three weight matrices (Wgate, Wup, Wdown), no biases.
        /// Wgate and Wup have shape [dModel, dFF].
        /// Wdown has shape [dFF, dModel].
        /// dFF is typically 2/3 × 4 × dModel (rounded up to multiple of 256).
        /// </summary>
        SwiGLU = 3,
    }
}
