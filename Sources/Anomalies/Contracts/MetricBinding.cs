// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
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
    /// <param name="Query">
    /// Verbatim PromQL replacing the name-and-kind template; empty for the ordinary case. See the property
    /// of the same name for why it exists and what it must contain.
    /// </param>
    public readonly record struct MetricBinding(
        MetricIndex Target,
        string SourceMetric,
        MetricSourceKind Kind,
        double Quantile = 0.0,
        string Query = "")
    {
        /// <summary>
        /// Verbatim PromQL, used in place of everything <see cref="Kind"/> would have wrapped around
        /// <see cref="SourceMetric"/>. Empty for the ordinary case, which is nearly all of them.
        ///
        /// <para><b>It exists because a name and a shape cannot express every correct query, and the gap is
        /// not academic.</b> <c>OomEventsRate</c> has to read the restart counter joined against the
        /// last-terminated reason — kube-state-metrics carries the OOM fact in one series and the event in
        /// another, and no combination of name-plus-kind produces a join. Measured 2026-08-08: the channel
        /// had been bound to <c>container_oom_events_total</c>, which is zero on every series this runtime
        /// produces, so the guard's OOM channel could never fire.</para>
        ///
        /// <para><b>The reason this had to be added rather than worked around</b>: the built-in template was
        /// fixed first, and it had no effect, because a configured binding overrides it. A deployment with
        /// an explicit <c>metrics</c> block — which is every real one — would have kept reading the dead
        /// series. Deleting the entry instead is worse still: an unbound channel is issued no query at all
        /// and reports blind.</para>
        ///
        /// <para>Must contain the selector token <c>%selector%</c> (<c>PromqlCatalog.SelectorToken</c>), or
        /// the query would ignore the namespace and pod matchers and silently report on the whole
        /// cluster.</para>
        /// </summary>
        public string Query
        {
            get;
        } = Query;

        /// <summary>Whether this binding can produce a query at all.</summary>
        public bool IsUsable => !string.IsNullOrWhiteSpace(SourceMetric)
                                || !string.IsNullOrWhiteSpace(Query);
    }
}
