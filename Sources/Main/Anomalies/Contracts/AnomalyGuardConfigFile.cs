// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// The shape of the guard's configuration file, as a client fills it in.
    ///
    /// <para><b>Keyed by name, never by position.</b> The absolute floors used to be an
    /// <c>IReadOnlyList&lt;double&gt;</c> indexed by <see cref="MetricIndex"/> — configuration that reads
    /// <c>[0, 0, 100000000, 0, …]</c>, which nobody can write and nobody can check. Here every entry names
    /// the feature it is about, and a missing key means something: absent from <see cref="Metrics"/> is "this
    /// cluster does not have it", absent from <see cref="Thresholds"/> is "that gate is off".</para>
    ///
    /// <para>Plain properties with parameterless construction, so the standard configuration binder fills it
    /// from JSON in a ConfigMap without this project taking a dependency on the binder.</para>
    /// </summary>
    public sealed class AnomalyGuardConfigFile
    {
        /// <summary>Prometheus HTTP API base URL.</summary>
        public string Prometheus { get; set; } = string.Empty;

        /// <summary>Kubernetes namespace to watch.</summary>
        public string Namespace { get; set; } = string.Empty;

        /// <summary>Pod-name regex selecting the group to watch.</summary>
        public string PodRegex { get; set; } = string.Empty;

        /// <summary>
        /// Which known feature comes from which of this cluster's metrics, keyed by
        /// <see cref="MetricIndex"/> name.
        /// </summary>
        public Dictionary<string, MetricEntry> Metrics { get; set; } = new(StringComparer.OrdinalIgnoreCase);

        /// <summary>
        /// Metrics this project does not model, keyed by the name findings will carry. Evaluated by the
        /// rules, peer and trend families and not by the learned one.
        /// </summary>
        public Dictionary<string, CustomEntry> CustomMetrics { get; set; } =
            new(StringComparer.OrdinalIgnoreCase);

        /// <summary>
        /// Per-feature absolute floors, keyed by <see cref="MetricIndex"/> name. Values accept a unit —
        /// <c>100MB</c>, <c>50ms</c> — because a bare number is where an order-of-magnitude slip hides.
        /// </summary>
        public Dictionary<string, ThresholdEntry> Thresholds { get; set; } =
            new(StringComparer.OrdinalIgnoreCase);

        /// <summary>One known feature's source.</summary>
        public class MetricEntry
        {
            /// <summary>The metric name as this cluster's exporter emits it; for a histogram, without
            /// <c>_bucket</c>.</summary>
            public string Source { get; set; } = string.Empty;

            /// <summary>Its shape — see <see cref="MetricSourceKind"/>.</summary>
            public string Kind { get; set; } = nameof(MetricSourceKind.Gauge);

            /// <summary>Histogram quantile; the feature's own default when left at zero.</summary>
            public double Quantile
            {
                get; set;
            }
        }

        /// <summary>One metric outside the modelled set.</summary>
        public sealed class CustomEntry : MetricEntry
        {
            /// <summary>
            /// Whether uneven load can explain its magnitude. <b>Only the operator knows</b>, and a wrong
            /// answer is expensive: classifying memory this way made it the largest single source of false
            /// peer findings, because dividing a fixed cost by a varying one manufactures the traffic
            /// imbalance as a difference.
            /// </summary>
            public bool LoadSensitive
            {
                get; set;
            }

            /// <summary>Where it sits between cause and consequence: Infrastructure, Resource or Symptom.</summary>
            public string Class { get; set; } = "Resource";

            /// <summary>Smallest peer difference worth reporting, with a unit.</summary>
            public string MinGap { get; set; } = string.Empty;

            /// <summary>Smallest trend change worth reporting, with a unit.</summary>
            public string MinTrendChange { get; set; } = string.Empty;

            /// <summary>Optional absolute rule: the level, with a unit.</summary>
            public string RuleThreshold { get; set; } = string.Empty;

            /// <summary>Share of the window that must be at or above it, 0…1.</summary>
            public double RuleMinBreachFraction { get; set; } = 0.25;
        }

        /// <summary>One feature's absolute floors.</summary>
        public sealed class ThresholdEntry
        {
            /// <summary>Smallest peer difference worth reporting, with a unit.</summary>
            public string MinGap { get; set; } = string.Empty;

            /// <summary>Smallest trend change worth reporting, with a unit.</summary>
            public string MinTrendChange { get; set; } = string.Empty;
        }
    }
}
