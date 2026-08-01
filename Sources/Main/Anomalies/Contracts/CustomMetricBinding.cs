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
        double SaturationLimit = double.NaN)
    {
        /// <summary>Whether this binding can produce a query at all.</summary>
        public bool IsUsable => !string.IsNullOrWhiteSpace(Name) && !string.IsNullOrWhiteSpace(Source);
    }
}
