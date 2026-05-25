// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring.Abstractions
{
    /// <summary>
    /// A multi-pod raw-metric scrape source — one <see cref="ReadAsync"/> returns the current
    /// window of <see cref="RawMetricSeries"/> across all matched pods. Implemented by
    /// <see cref="PrometheusMetricSource"/>; the seam lets <c>LiveMonitoringPipeline</c> run
    /// against a scripted/in-memory source in tests without a live Prometheus instance.
    /// (Distinct from <see cref="IMetricSource"/>, which is a single-pod single-snapshot read.)
    /// </summary>
    public interface IRawMetricSource : IDisposable
    {
        Task<List<RawMetricSeries>> ReadAsync(CancellationToken ct = default);
    }
}
