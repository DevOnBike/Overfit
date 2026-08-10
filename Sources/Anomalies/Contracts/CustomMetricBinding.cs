// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// A metric this project does not model, that a deployment wants watched anyway.
    ///
    /// <para><b>Why it is a separate type rather than a new <see cref="MetricIndex"/> member.</b>
    /// <c>MetricSnapshot.FeatureCount</c> is the trained model's input contract — the file says so, and it is
    /// not a formality: adding a thirteenth channel once moved the tokeniser vocabulary from 768 to 832 and
    /// broke two committed checkpoints, <c>ContextLength</c>, a 201 000-row CSV fixture and the Python
    /// generator behind it. The enum cannot grow per customer.</para>
    ///
    /// <para><b>And it does not need to.</b> <c>MetricIndex.Count</c> is already larger than
    /// <c>FeatureCount</c> for exactly this reason: the rules, peer and trend families read raw series, while
    /// the model reads a fixed feature set. Below them <c>SignalFinding.Signal</c> is a string, so the
    /// grouper, the tracker and the reporter never see the enum at all.</para>
    ///
    /// <para><b>The consequence, which the operator must be told rather than discover:</b> a custom metric is
    /// evaluated by the rules, the peer comparison and the trend detector, and is <b>not</b> fed to the
    /// learned family.</para>
    /// </summary>
    /// <param name="Name">
    /// What the finding will be called. Stable across pods, because the grouper counts distinct signals and
    /// the tracker matches on them.
    /// </param>
    /// <param name="Source">The metric name as this cluster's exporter emits it.</param>
    /// <param name="Kind">Its shape, which decides the PromQL wrapped around the name.</param>
    /// <param name="SignalKind">
    /// Whether uneven load can explain its magnitude.
    ///
    /// <para><b>Only the operator knows, and a wrong answer is expensive.</b> Measured here: memory was
    /// classified load-sensitive and divided by request rate, which made it the single largest source of
    /// false peer findings — a working set is a fixed cost, and dividing a fixed quantity by a varying one
    /// manufactures a difference the size of the traffic imbalance.</para>
    /// </param>
    /// <param name="Class">
    /// Where it sits between cause and consequence. The grouper orders an incident cause-first, so without
    /// this a new metric lands wherever the default puts it regardless of what it means.
    /// </param>
    /// <param name="MinAbsoluteGap">
    /// Smallest peer difference worth reporting, in this metric's units. Zero disables the gate — and leaving
    /// it off is how a difference of three tenths of a millisecond became the top false-positive source on a
    /// healthy population.
    /// </param>
    /// <param name="MinAbsoluteTrendChange">Smallest fitted trend change worth reporting, in its units.</param>
    /// <param name="Quantile">For a histogram, which quantile to take; 0.95 when left at zero.</param>
    /// <param name="Rule">
    /// Optional absolute rule — "above this, for this share of the window". The case the relative families
    /// cannot reach: a queue depth or a consumer lag that is simply too high, whatever the siblings are doing.
    /// </param>
    /// <param name="SaturationLimit">
    /// The ceiling this signal is heading towards, in its own units — a disk capacity, a queue bound, a
    /// connection-pool size. <see cref="double.NaN"/> skips the projection.
    ///
    /// <para>Its own field rather than an entry in the per-metric table, because that table is indexed by
    /// <see cref="MetricIndex"/> and cannot hold a name the enum does not have — the same reason custom
    /// channels carry their own floors.</para>
    /// </param>
    /// <param name="MinAbsoluteGapChange">
    /// Smallest <b>movement in the peer gap</b>, across the novelty window and in this metric's units, that
    /// counts as the deviation changing rather than standing. Read only when
    /// <c>AnomalyGuardOptions.PeerNovelty</c> is configured, and <b>required</b> then — a guard with the
    /// novelty gate on and this left at zero refuses to start.
    ///
    /// <para><b>Nobody has measured it, which is exactly why it has no default.</b> Calibrating it needs the
    /// distribution of fitted gap-change across healthy pods, which no accumulator here collects yet. A
    /// silent value would be a guess with a threshold's authority, on a gate whose failure mode is
    /// silence.</para>
    /// </param>
    /// <param name="Query">
    /// Verbatim PromQL replacing the name-and-kind template; empty for the ordinary case. See the property of
    /// the same name for what forced it to exist on this record too.
    /// </param>
    /// <param name="RequirePersistence">
    /// Whether a peer finding on this channel must recur before it is forwarded. False — the default — is the
    /// behaviour every custom channel had before this existed: forward on the first cycle.
    ///
    /// <para><b>Opt-in per channel, because persistence is not free.</b> A gate that waits costs latency on
    /// every real fault it delays, so it belongs only on a channel whose transient dips are known to be
    /// uninformative. Scrape coverage is one: a pod being replaced stops answering for a moment on every
    /// rollout, and <c>AN-A1</c> already fails on incident rate, so a same-cycle gate there would add a false
    /// positive per replaced pod per rollout.</para>
    ///
    /// <para><b>A flag rather than a count, deliberately.</b> The number of cycles is
    /// <c>AnomalyGuardOptions.SilentPodCycles</c> — one number answers "how many cycles before I believe a
    /// coverage-related degradation is real" for both total silence and partial degradation, and two knobs
    /// with the same meaning drift apart. A <c>MinConsecutiveCycles</c> integer was the shape first proposed
    /// and was rejected on writing it: a field whose value is ignored above 1 is a configuration key that
    /// accepts 5 and silently does 2.</para>
    /// </param>
    /// <param name="Calibrated">
    /// Whether a healthy period's observations of this channel may be turned into a floor, and whether the
    /// inert-channel check may judge it. True — the default — is what every custom channel did before this
    /// existed.
    ///
    /// <para><b>This is the custom-channel equivalent of <c>PeerSignalCatalog.IsCountedEvent</c></b>, which
    /// can only speak for the built-in enum: a quantity whose <i>scale</i> is arbitrary can be calibrated from
    /// data, and a quantity whose <i>unit</i> is already the thing you care about cannot. Set it false for a
    /// channel that is constant while healthy — a counter of a rare event, or a coverage fraction pinned at
    /// 1.0 — where fitting a floor either does nothing or sets the bar above the only event the channel
    /// exists to report.</para>
    ///
    /// <para>It reaches the calibrator as a name on <c>AnomalyGuardOptions.NonCalibratedCustomChannels</c>,
    /// because <c>FloorCalibrator</c> never receives a binding.</para>
    /// </param>
    public readonly record struct CustomMetricBinding(
        string Name,
        string Source,
        MetricSourceKind Kind,
        PeerSignalKind SignalKind,
        SignalClass Class,
        double MinAbsoluteGap = 0.0,
        double MinAbsoluteTrendChange = 0.0,
        double Quantile = 0.0,
        SustainedThresholdOptions? Rule = null,
        double SaturationLimit = double.NaN,
        double MinAbsoluteGapChange = 0.0,
        string Query = "",
        bool RequirePersistence = false,
        bool Calibrated = true)
    {
        /// <summary>
        /// Verbatim PromQL, used in place of everything <see cref="Kind"/> would have wrapped around
        /// <see cref="Source"/>. Empty for the ordinary case.
        ///
        /// <para><b>The built-in path got this and the custom path did not, and the omission was load-bearing
        /// rather than cosmetic.</b> <see cref="MetricBinding.Query"/> exists because a name and a shape
        /// cannot express a join (<c>OomEventsRate</c>) or a division (<c>CpuThrottleRatio</c>). Scrape
        /// coverage needs a third thing none of the five kinds produce — a range vector inside an aggregation,
        /// <c>avg_over_time(up{%selector%}[15m])</c>. <see cref="MetricSourceKind.Ratio"/> renders
        /// <c>name{selector}</c>, which puts the selector and the range in the wrong places; there is no kind
        /// that renders the right thing, and adding one for a single channel would be a worse trade than the
        /// escape hatch the built-in path already has.</para>
        ///
        /// <para>Must contain <c>%selector%</c> (<c>PromqlCatalog.SelectorToken</c>), or the query ignores the
        /// namespace and pod matchers and silently reports on the whole cluster. The reader enforces it —
        /// <c>AnomalyGuardConfigReader</c> applies the same check the built-in path has always applied.</para>
        ///
        /// <para><b>A verbatim query carries its own range and nothing substitutes it.</b> A templated one
        /// takes the window the guard is configured with; this one carries whatever was typed, so a change to
        /// that setting will not reach it.</para>
        /// </summary>
        public string Query
        {
            get;
        } = Query;

        /// <summary>Whether this binding can produce a query at all.</summary>
        public bool IsUsable => !string.IsNullOrWhiteSpace(Name)
                                && (!string.IsNullOrWhiteSpace(Source)
                                    || !string.IsNullOrWhiteSpace(Query));
    }
}
