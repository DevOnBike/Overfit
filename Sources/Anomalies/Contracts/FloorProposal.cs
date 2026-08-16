// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// What a healthy deployment actually did on one signal, and the absolute floors that follow from it.
    ///
    /// <para><b>This exists because the relative gates cannot defend themselves and the absolute ones have
    /// nobody to set them.</b> Measured on a twelve-replica lab: the gen2 heap sat at three to five
    /// megabytes, so an ordinary swing between collections was a <b>66% relative change</b> and cleared every
    /// percentage gate there is — while being two megabytes, which is nothing. The same run had CPU at about
    /// 2% of a core, with the same shape. Both produced incidents every single cycle.</para>
    ///
    /// <para><b>And the floors cannot be shipped as constants, which is the part that took a day to learn.</b>
    /// A 256 MiB working-set floor was measured and set; the heap was deliberately left without one because
    /// on the synthetic population it produced no false positives at all. That population's heap was around
    /// 480 MB. The lab's is 3 MB — a hundred and sixty times smaller, and the same absence of a floor that
    /// cost nothing there costs an incident per cycle here.</para>
    /// </summary>
    /// <param name="Samples">Observations behind the figures; a proposal from a handful means little.</param>
    /// <param name="Windows">
    /// Evaluation windows the observations span. <b>Not derivable from <paramref name="Samples"/></b>: one
    /// cycle on twelve replicas produces twelve observations covering a single moment, and a maximum needs
    /// time. See <see cref="MinimumWindows"/> for the false positive that made the distinction necessary.
    /// </param>
    /// <param name="TypicalMagnitude">
    /// The signal's own scale — the median across pods and cycles. Present so a reader can sanity-check the
    /// proposal against something they recognise rather than take a number on trust.
    /// </param>
    /// <param name="PeerGapP99">
    /// The 99th percentile of the absolute distance between a pod's median and its peers', over everything
    /// observed. Anything at or below this is something a <b>healthy</b> deployment did.
    /// </param>
    /// <param name="PeerGapMax">The largest such distance seen.</param>
    /// <param name="TrendChangeP99">
    /// The 99th percentile of the absolute fitted change across a window, per pod — the quantity
    /// <c>MinAbsoluteTrendChange</c> gates, in the same units.
    /// </param>
    /// <param name="TrendChangeMax">The largest such change seen.</param>
    /// <param name="LevelShiftP99">
    /// The 99th percentile of how far the <b>workload's common level</b> moved across a window — one
    /// observation per cycle, measured on the median across replicas.
    ///
    /// <para><b>A separate figure from <see cref="TrendChangeP99"/>, and expect it to be much smaller.</b> A
    /// trend change is fitted to one pod's noisy series; a step is measured on a median over all of them,
    /// which is roughly <c>√N</c> less scattered. Reading one as the other put the CPU step floor at about
    /// 1.5× the signal's own level and made a real cluster-wide rise unreportable.</para>
    /// </param>
    /// <param name="LevelShiftMax">The largest such step seen.</param>
    /// <param name="ProposedMinAbsoluteGap">
    /// <see cref="PeerGapMax"/> with a margin. The maximum rather than the 99th percentile on purpose: a
    /// floor's job here is to suppress what a healthy cluster does, and at a five-minute cadence over a week
    /// the top percentile is still hundreds of findings.
    /// </param>
    /// <param name="ProposedMinAbsoluteTrendChange">The same, for the trend family.</param>
    /// <param name="ProposedMinAbsoluteLevelShift">The same, for the step family.</param>
    /// <param name="CappedByOperator">
    /// Whether an operator's <c>--real</c> label held the proposal below what the data alone suggested.
    /// <b>Worth surfacing rather than hiding</b>: it is the one direction in which the feedback loop pulls
    /// against silence, so a proposal that was capped is evidence the loop is working, and a fleet where it
    /// is never true is a fleet converging on a detector that reports nothing.
    /// </param>
    public readonly record struct FloorProposal(
        int Samples,
        double TypicalMagnitude,
        double PeerGapP99,
        double PeerGapMax,
        double TrendChangeP99,
        double TrendChangeMax,
        double LevelShiftP99,
        double LevelShiftMax,
        double ProposedMinAbsoluteGap,
        double ProposedMinAbsoluteTrendChange,
        double ProposedMinAbsoluteLevelShift,
        bool CappedByOperator = false,
        int Windows = 0)
    {
        /// <summary>
        /// Whether an operator's <c>--real</c> label held this proposal below what the data alone suggested.
        ///
        /// <para>Reported rather than applied silently, because a capped proposal is a disagreement between
        /// two sources of truth and the operator is entitled to see it: the observed period says a gap this
        /// large is normal, and somebody who looked at one said it was not. Suppressing that would make the
        /// proposal look like a measurement when it is a negotiated number.</para>
        /// </summary>
        public bool WasCapped => CappedByOperator;

        /// <summary>Whether enough was observed for the proposal to be worth reading.</summary>
        /// <summary>
        /// Windows the proposal must span before it is offered, whatever the replica count.
        ///
        /// <para><b>Samples alone were the wrong unit, and it cost a false positive.</b> The old gate was
        /// thirty observations, and one observation is one pod in one window — so on a twelve-replica lab it
        /// was satisfied after <b>three cycles, fifteen minutes</b>. Twelve replicas measured at the same
        /// instant are twelve views of one moment, not twelve moments; a maximum needs time coverage and
        /// they supply none.</para>
        ///
        /// <para><b>Measured 2026-08-09.</b> A `LockContentions` floor calibrated from a three-minute window
        /// gave a healthy peak of 0.0514/s. The same population over twenty minutes gave <b>0.0952/s</b> —
        /// nearly double, and still rising — so healthy replicas routinely exceeded the floor set from the
        /// short window, and it reported a false positive on an unfaulted pod within the hour.</para>
        ///
        /// <para>Twenty-four windows is two hours at the default five-minute cadence. It is chosen as the
        /// smallest round figure comfortably past the point where the maximum was still visibly climbing —
        /// <b>not</b> as a value shown to be sufficient, because nothing here has measured where the maximum
        /// actually settles. A floor is a suggestion for a human either way; this stops it being offered
        /// from a window that describes a rollout rather than a workload.</para>
        /// </summary>
        public const int MinimumWindows = 24;

        /// <summary>
        /// Whether this is worth offering. Both conditions are required and they answer different questions:
        /// <see cref="Samples"/> asks whether there is enough data, <see cref="Windows"/> asks whether it
        /// covers enough time.
        /// </summary>
        public bool IsUsable => Samples >= 30 && Windows >= MinimumWindows;
    }
}
