// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// Everything needed to turn a <see cref="MetricIndex"/> into PromQL, shared by the instant and the range
    /// source.
    ///
    /// <para><b>This interface exists because the duplication already cost something.</b> Both sources
    /// carried their own copy of the same twelve query templates and the same label-matcher assembly, so a
    /// census that found three defects in one of them — a hard-coded <c>dc</c> label that reduced every query
    /// to the empty set, metric names from a different exporter, and no way to override either — found the
    /// identical three, untouched, in the other. Fixing one is not fixing the pair unless there is only one
    /// copy to fix.</para>
    /// </summary>
    public interface IPrometheusQuerySelector
    {
        /// <summary>PromQL regex matching the pods to monitor, e.g. <c>overfit-server-.*</c>.</summary>
        string PodRegex
        {
            get;
        }

        /// <summary>Kubernetes namespace to restrict every query to; empty for no namespace matcher.</summary>
        string Namespace
        {
            get;
        }

        /// <summary>
        /// Label separating data centres. Empty means a single-data-centre deployment: no matcher is emitted
        /// and one pass is issued rather than two.
        /// </summary>
        string DataCenterLabel
        {
            get;
        }

        /// <summary>Value of <see cref="DataCenterLabel"/> for <c>DataCenter.West</c>.</summary>
        string DcWestLabel
        {
            get;
        }

        /// <summary>Value of <see cref="DataCenterLabel"/> for <c>DataCenter.East</c>.</summary>
        string DcEastLabel
        {
            get;
        }

        /// <summary>
        /// Per-metric PromQL overriding the built-in templates; an empty value declares that the metric has
        /// no source in this deployment. See <c>PromqlCatalog</c>.
        /// </summary>
        IReadOnlyDictionary<MetricIndex, string>? QueryOverrides
        {
            get;
        }
    }
}
