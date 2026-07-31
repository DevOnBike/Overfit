// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>One absolute rule: which metric, and the threshold it is judged against.</summary>
    /// <param name="Metric">The channel to evaluate.</param>
    /// <param name="Options">Threshold and persistence — see <see cref="SustainedThresholdOptions"/>.</param>
    public readonly record struct RuleProfile(MetricIndex Metric, SustainedThresholdOptions Options);
}
