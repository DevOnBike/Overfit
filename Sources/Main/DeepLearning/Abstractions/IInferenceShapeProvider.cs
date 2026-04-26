// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.DeepLearning.Abstractions
{
    /// <summary>
    /// Optional inference metadata for modules that change feature width.
    ///
    /// This is intentionally separate from IModule so older/simple modules can
    /// still compile. Sequential uses it to slice reusable inference workspaces
    /// without allocating temporary buffers per call.
    /// </summary>
    public interface IInferenceShapeProvider
    {
        int InferenceInputSize { get; }

        int InferenceOutputSize { get; }

        /// <summary>
        /// Prepares reusable inference caches outside the hot path.
        /// Must not allocate during ForwardInference after this has been called.
        /// </summary>
        void PrepareInference();
    }
}