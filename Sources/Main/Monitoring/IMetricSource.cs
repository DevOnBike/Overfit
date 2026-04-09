// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Monitoring
{
    /// <summary>
    /// Metric source contract.
    /// ValueTask — synchronous paths (tests, buffered reads) do not allocate a Task.
    /// </summary>
    public interface IMetricSource : IDisposable
    {
        string PodName { get; }

        /// <summary>
        /// Reads the current metric sample.
        /// Implementation may await the end of the scraping window before returning.
        /// </summary>
        ValueTask<MetricSnapshot> ReadAsync(CancellationToken ct = default);
    }
}