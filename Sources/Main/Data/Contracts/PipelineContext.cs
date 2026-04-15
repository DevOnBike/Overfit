// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Data.Contracts
{
    /// <summary>
    ///     Represents the state of data within the processing pipeline.
    /// </summary>
    public sealed class PipelineContext : IDisposable
    {
        private bool _disposed;

        public PipelineContext(FastTensor<float> features, FastTensor<float> targets)
        {
            Features = features ?? throw new ArgumentNullException(nameof(features));
            Targets = targets ?? throw new ArgumentNullException(nameof(targets));
        }

        /// <summary>
        ///     Input features tensor.
        /// </summary>
        public FastTensor<float> Features { get; set; }

        /// <summary>
        ///     Target values tensor (labels).
        /// </summary>
        public FastTensor<float> Targets { get; set; }

        /// <summary>
        ///     Diagnostic metadata populated by the DataPipeline after each processing step.
        /// </summary>
        public List<LayerDiagnostic> Diagnostics { get; } = [];

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;

            Features?.Dispose();
            Targets?.Dispose();

            Features = null;
            Targets = null;
        }

        /// <summary>
        ///     Replaces the current Features tensor and disposes of the old one.
        /// </summary>
        public void ReplaceFeatures(FastTensor<float> newFeatures)
        {
            ArgumentNullException.ThrowIfNull(newFeatures);

            var old = Features;
            Features = newFeatures;

            if (!ReferenceEquals(old, newFeatures))
            {
                old?.Dispose();
            }
        }

        /// <summary>
        ///     Replaces both Features and Targets tensors and disposes of the old ones.
        ///     Typically used by row-filtering layers (e.g., TechnicalSanityLayer, DuplicateRowFilter).
        /// </summary>
        public void ReplaceAll(FastTensor<float> newFeatures, FastTensor<float> newTargets)
        {
            ArgumentNullException.ThrowIfNull(newFeatures);
            ArgumentNullException.ThrowIfNull(newTargets);

            var oldF = Features;
            var oldT = Targets;

            Features = newFeatures;
            Targets = newTargets;

            if (!ReferenceEquals(oldF, newFeatures))
            {
                oldF?.Dispose();
            }

            if (!ReferenceEquals(oldT, newTargets))
            {
                oldT?.Dispose();
            }
        }
    }
}