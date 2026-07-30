// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    /// <summary>One of this deployment's metrics, and which known feature it supplies.</summary>
    /// <param name="Target">The known feature it fills. See <see cref="MetricIndex"/>.</param>
    /// <param name="SourceMetric">
    /// The metric name as <b>this cluster's exporter</b> emits it. For a histogram, the family name without
    /// the <c>_bucket</c> suffix.
    ///
    /// <para><b>Metric names are a property of whoever wrote the exporter, never a contract.</b> The built-in
    /// templates use OpenTelemetry naming; this project's own server emits <c>overfit_chat_*</c> and
    /// <c>dotnet_*</c>. A name that matches nothing returns an empty result which Prometheus reports as
    /// <c>success</c> — indistinguishable from a quiet system until something counts coverage.</para>
    /// </param>
    /// <param name="Kind">The shape, which decides the PromQL wrapped around the name.</param>
    /// <param name="Quantile">
    /// For <see cref="MetricSourceKind.HistogramSeconds"/>, which quantile to take. Ignored otherwise, and
    /// defaulted from the target feature when left at zero — <c>LatencyP95Ms</c> means 0.95.
    /// </param>
    public readonly record struct MetricBinding(
        MetricIndex Target,
        string SourceMetric,
        MetricSourceKind Kind,
        double Quantile = 0.0)
    {
        /// <summary>Whether this binding can produce a query at all.</summary>
        public bool IsUsable => !string.IsNullOrWhiteSpace(SourceMetric);
    }
}
