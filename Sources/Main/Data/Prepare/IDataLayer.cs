// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Data.Contracts;

namespace DevOnBike.Overfit.Data.Prepare
{
    /// <summary>
    /// Defines the fundamental contract for a data processing layer within the Overfit pipeline.
    /// </summary>
    /// <remarks>
    /// Each layer represents a single step in data preparation (e.g., filtering, selection, or scaling).
    /// Layers are executed sequentially by the <see cref="DataPipeline"/>.
    /// </remarks>
    public interface IDataLayer
    {
        /// <summary>
        /// Processes the data state within the provided context and returns the transformed result.
        /// </summary>
        /// <param name="context">The current pipeline context holding feature and target tensors.</param>
        /// <returns>The processed <see cref="PipelineContext"/>.</returns>
        PipelineContext Process(PipelineContext context);
    }
}