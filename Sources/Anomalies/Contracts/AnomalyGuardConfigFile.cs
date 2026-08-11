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

        /// <summary>
        /// Pod label naming which replicas may be compared against each other — <c>role</c> for most database
        /// and queue operators. Empty means none is declared and every pod compares against every other.
        ///
        /// <para><b>One string, not a list of pods.</b> Nobody maintains it after a scale-up and nobody
        /// updates it after a failover, because the operator that runs the workload already publishes the
        /// fact as a label and changes it when leadership moves. This names which label to read.</para>
        ///
        /// <para>It has to be declared because it cannot be inferred: a rollout, a canary and an elected
        /// leader all look like a minority of replicas behaving unlike the majority, and they call for
        /// opposite answers. An attempt to key this on the ReplicaSet was written and reverted within the
        /// hour, when a test showed it made canaries invisible.</para>
        /// </summary>
        public string PeerGroupLabel { get; set; } = string.Empty;

        /// <summary>Pod-name regex selecting the group to watch.</summary>
        public string PodRegex { get; set; } = string.Empty;

        /// <summary>
        /// Several populations watched by one process, instead of the single <see cref="Namespace"/> /
        /// <see cref="PodRegex"/> pair above.
        ///
        /// <para><b>Empty means single-scope, and that is the migration.</b> A file written before this
        /// existed resolves to a one-element list built from the top-level fields, so nothing deployed needs
        /// editing on the day multi-scope ships. Setting both forms is refused rather than merged — see
        /// <see cref="Monitoring.GuardScopeResolver"/> for why a rule that guessed would be worse than the
        /// rejection.</para>
        ///
        /// <para>Bindings in <see cref="Metrics"/> and gates in <see cref="Thresholds"/> stay at file level
        /// and are shared: fifty scopes repeating thirteen metric names is a configuration file nobody
        /// reads.</para>
        /// </summary>
        public List<GuardScopeEntry> Scopes { get; set; } = [];

        /// <summary>
        /// The deployment being watched. Used to key the seasonal baseline, to match maintenance windows, and
        /// as the subject of findings that are about the workload rather than any one replica.
        ///
        /// <para><b>Leaving it blank is not neutral, which is why it is here.</b> Without it the guard ran
        /// with an empty workload name, and two things broke quietly. A maintenance window naming a workload
        /// could never match, so an operator who declared one for their rollout was paged during it anyway.
        /// And the incident tracker keys a pod-less subject on the workload, so every deployment-level
        /// finding in a namespace collapsed to <c>"namespace/"</c> — one identity shared by unrelated
        /// problems, reported as a single continuing incident. Visible in the lab's own logs as
        /// <c>Anomaly incident in lab/:</c> with nothing after the slash.</para>
        /// </summary>
        public string Workload { get; set; } = string.Empty;

        /// <summary>
        /// Which known feature comes from which of this cluster's metrics, keyed by
        /// <see cref="MetricIndex"/> name.
        /// </summary>
        public Dictionary<string, MetricEntry> Metrics { get; set; } = new(StringComparer.OrdinalIgnoreCase);

        /// <summary>
        /// Metrics this project does not model, keyed by the name findings will carry. Evaluated by the
        /// rules, peer and trend families and not by the learned one.
        /// </summary>
        public Dictionary<string, CustomEntry> CustomMetrics
        {
            get; set;
        } =
            new(StringComparer.OrdinalIgnoreCase);

        /// <summary>
        /// Per-feature absolute floors, keyed by <see cref="MetricIndex"/> name. Values accept a unit —
        /// <c>100MB</c>, <c>50ms</c> — because a bare number is where an order-of-magnitude slip hides.
        /// </summary>
        public Dictionary<string, ThresholdEntry> Thresholds
        {
            get; set;
        } =
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

            /// <summary>
            /// Verbatim PromQL, replacing whatever <see cref="Kind"/> would have built around
            /// <see cref="Source"/>. Leave empty unless the correct query cannot be expressed as a name and
            /// a shape — see <see cref="MetricBinding.Query"/> for the channel that forced this to exist.
            /// Must contain <c>%selector%</c>, and the reader rejects the file if it does not.
            ///
            /// <para><b>The rate range is yours to write and is not substituted.</b> A templated query takes
            /// the window the guard is configured with; a verbatim one carries whatever you typed, so a
            /// change to that setting will not reach it. Keep the two in step by hand, or do not use this
            /// field.</para>
            /// </summary>
            public string Query { get; set; } = string.Empty;
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

            /// <summary>
            /// Smallest change in this channel's PEER GAP worth treating as news, with a unit. Required by
            /// the peer-novelty gate and by nothing else.
            ///
            /// <para><b>Added 2026-08-10 because the property existed and the file could not express it.</b>
            /// <see cref="CustomMetricBinding.MinAbsoluteGapChange"/> was there, the reader never set it, and
            /// nothing noticed — the novelty gate was proven in-process with options assigned in code, never
            /// through the file a deployment actually loads. It is harmless while the gate is off and fatal
            /// the day it is switched on, because <c>AnomalyGuard.RestoreNovelty</c> refuses a binding whose
            /// value is zero. A feature that cannot be configured is not shipped, however well it is
            /// tested.</para>
            ///
            /// <para>Distinct from <see cref="MinGap"/> on purpose: that one asks how large a difference
            /// between replicas matters, this one asks how much that difference must MOVE before it is news
            /// again. A pod 19 MB heavier than its peers since it started is not an event; the same pod
            /// growing another 19 MB is.</para>
            /// </summary>
            public string MinGapChange { get; set; } = string.Empty;

            /// <summary>
            /// Smallest STEP in this channel's level worth reporting. Absent falls back to
            /// <see cref="MinTrendChange"/>, which is what the step gate used before this field existed —
            /// so adding it moves no configured channel on its own. See the built-in equivalent on
            /// <c>ThresholdEntry.MinStepChange</c> for why the two distributions differ.
            /// </summary>
            public string MinStepChange { get; set; } = string.Empty;

            /// <summary>Optional absolute rule: the level, with a unit.</summary>
            public string RuleThreshold { get; set; } = string.Empty;

            /// <summary>Share of the window that must be at or above it, 0…1.</summary>
            public double RuleMinBreachFraction { get; set; } = 0.25;

            /// <summary>
            /// Whether a peer finding on this channel must recur before it is reported. False — the default —
            /// is what every custom channel did before this existed.
            ///
            /// <para>How many cycles is not set here: it is <c>silentPodCycles</c>, so the two cannot drift
            /// apart. See <see cref="CustomMetricBinding.RequirePersistence"/>.</para>
            /// </summary>
            public bool RequirePersistence
            {
                get; set;
            }

            /// <summary>
            /// Whether a healthy period's observations of this channel may become a floor, and whether the
            /// inert-channel check may judge it. True — the default — is what every custom channel did before
            /// this existed. See <see cref="CustomMetricBinding.Calibrated"/>.
            /// </summary>
            public bool Calibrated { get; set; } = true;
        }

        /// <summary>One feature's absolute floors.</summary>
        /// <summary>
        /// Periods declared abnormal on purpose — a deployment, a node pool upgrade, a load test.
        ///
        /// <para><b>Declarative, in the same file as everything else, and that is the whole point.</b> The
        /// alternative is an endpoint the operator calls to silence the guard, which means the guard needs a
        /// write API, authentication for it, and a way to survive its own restart with that state intact. A
        /// list in a ConfigMap needs none of those, reviews like code, and keeps the promise that the only
        /// dependency is an HTTP route to Prometheus.</para>
        ///
        /// <para>Timestamps are ISO-8601. A window that cannot be parsed is <b>reported and dropped</b>, never
        /// silently widened to cover everything — a suppression that quietly applies for ever is the one
        /// mistake here that produces total, invisible deafness.</para>
        /// </summary>
        public List<MaintenanceEntry> Maintenance
        {
            get;
            set;
        } = [];

        /// <summary>One declared window.</summary>
        public sealed class MaintenanceEntry
        {
            /// <summary>Start, ISO-8601, inclusive.</summary>
            public string From { get; set; } = string.Empty;

            /// <summary>End, ISO-8601, exclusive.</summary>
            public string To { get; set; } = string.Empty;

            /// <summary>Workload it covers; empty means the whole scope.</summary>
            public string Workload { get; set; } = string.Empty;

            /// <summary>Why, in the operator's words. Carried into every suppressed report.</summary>
            public string Reason { get; set; } = string.Empty;
        }

        public sealed class ThresholdEntry
        {
            /// <summary>
            /// Smallest STEP in the workload's own level worth reporting, in the signal's units.
            ///
            /// <para><b>Absent falls back to <see cref="MinTrendChange"/>, which is what this gate used
            /// before this field existed.</b> That fallback is the whole compatibility story: adding the
            /// field must not move a single deployed threshold on its own, because dropping every existing
            /// config onto the calibrator overnight is a silent change to what the guard reports.</para>
            ///
            /// <para><b>Why it needed its own entry.</b> A trend floor is fitted to how far ONE pod's series
            /// moves across a window; a step floor governs how far the median across pods moves between the
            /// halves of one. Measured 2026-08-11 (`AN-D4b`) over 30.8 h of lab data: reusing the trend
            /// floor made the step gate demand 40% of the level on `MemoryWorkingSetBytes` — where the 25%
            /// relative gate then never binds at all, 100% of the time — and at the low decile of
            /// `GcGen2HeapBytes` it demanded 123%, i.e. the heap had to more than double before a step was
            /// reportable. <c>FloorCalibrator.ProposedMinAbsoluteLevelShift</c> already computes the right
            /// number from the step distribution; this is where an operator writes it down.</para>
            /// </summary>
            public string MinStepChange { get; set; } = string.Empty;

            /// <summary>Smallest peer difference worth reporting, with a unit.</summary>
            public string MinGap { get; set; } = string.Empty;

            /// <summary>Smallest trend change worth reporting, with a unit.</summary>
            public string MinTrendChange { get; set; } = string.Empty;
        }
    }
}
