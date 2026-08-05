// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// One metric series that could feed a channel, and the evidence for it.
    /// </summary>
    /// <param name="Source">The Prometheus series name.</param>
    /// <param name="Kind">How it must be read — a counter and a gauge of the same name mean different things.</param>
    /// <param name="Stack">
    /// The runtime family it belongs to, for a human reading the report. <c>"container"</c> for the
    /// cAdvisor and kube-state-metrics series, which are the same on every stack.
    /// </param>
    /// <param name="PodsReporting">
    /// How many of the pods under inspection actually export it.
    ///
    /// <para><b>This is the field that makes the report evidence rather than a guess.</b> A metric name
    /// existing somewhere in the cluster says nothing about whether <i>these</i> pods emit it — a cluster
    /// running one Java service and twelve .NET ones has <c>jvm_*</c> in its name list either way. Zero here
    /// means the name matched and the workload does not export it, which is precisely the distinction
    /// between a binding and a channel that will report blind for ever.</para>
    /// </param>
    /// <param name="Quantile">Quantile to take, for histogram sources; NaN otherwise.</param>
    /// <param name="Inferred">
    /// Whether this came from the shape of the name rather than from the list of known conventions.
    ///
    /// <para><b>Worth showing a human, because the two carry different confidence.</b>
    /// <c>dotnet_gc_heap_size_bytes</c> is a name this project has seen and knows the meaning of;
    /// <c>orders_api_requests_total</c> is a name that merely <i>ends</i> like a request counter, and the
    /// convention it follows is a convention, not a guarantee. Both are worth proposing; only one is worth
    /// accepting without looking.</para>
    /// </param>
    public readonly record struct MetricCandidate(
        string Source,
        MetricSourceKind Kind,
        string Stack,
        int PodsReporting,
        double Quantile,
        bool Inferred = false)
    {
        /// <summary>Whether the pods under inspection actually emit this series.</summary>
        public bool IsEvidenced => PodsReporting > 0;
    }
}
