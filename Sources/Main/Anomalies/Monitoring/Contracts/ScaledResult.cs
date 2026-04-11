// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    /// <summary>
    /// Scaled tensors ready for the LSTM autoencoder.
    /// All values normalised to ~[-3, 3] via RobustScaler.
    /// </summary>
    public sealed class ScaledResult
    {
        /// <summary>Shape [dcCount, WindowSize, MetricCount]. One row per DC per timestep.</summary>
        public required FastTensor<float> FleetBaseline { get; init; }

        /// <summary>Shape [podCount, WindowSize, MetricCount]. One row per pod per timestep.</summary>
        public required FastTensor<float> PodDeviations { get; init; }

        /// <summary>Pod identity — PodIndex[i] corresponds to PodDeviations[i].</summary>
        public required List<PodKey> PodIndex { get; init; }
    }
}