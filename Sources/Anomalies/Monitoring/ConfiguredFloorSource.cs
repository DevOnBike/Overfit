// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// The floors the operator configured, falling back to what a healthy period turned out to look like.
    ///
    /// <para><b>An explicit value always wins, even a lower one.</b> A configured floor is a decision somebody
    /// made and may encode something no measurement can see. Silently overriding it would make the
    /// configuration a suggestion; where it is too low the guard says so in its proposal and the fix stays
    /// with the operator.</para>
    ///
    /// <para><b>The fallback is what makes day one survivable.</b> An absent floor means the gate is off, and
    /// a guard with every absolute gate off is the configuration measured at <b>209 false incidents a day</b>
    /// on a lab where nothing was wrong.</para>
    ///
    /// <para><b>It inherits the calibrator's one hazard.</b> A fault inside the observed period raises the
    /// bar above that fault and blinds the guard to it at that size, quietly. Construct with
    /// <c>applyCalibrated: false</c> where the observation period cannot be trusted.</para>
    /// </summary>
    public sealed class ConfiguredFloorSource : IAbsoluteFloorSource
    {
        private readonly IReadOnlyList<double>? _gap;
        private readonly IReadOnlyList<double>? _trend;
        private readonly IReadOnlyList<double>? _step;
        private readonly FloorCalibrator? _calibrator;

        /// <param name="gap">Configured per-metric peer floors, or null.</param>
        /// <param name="trend">Configured per-metric trend floors, or null.</param>
        /// <param name="calibrator">Learned floors, consulted only where the configured one is absent.</param>
        /// <param name="applyCalibrated">Whether the fallback is used at all.</param>
        public ConfiguredFloorSource(
            IReadOnlyList<double>? gap,
            IReadOnlyList<double>? trend,
            FloorCalibrator? calibrator,
            bool applyCalibrated)
            : this(gap, trend, step: null, calibrator, applyCalibrated)
        {
        }

        /// <param name="gap">Configured per-metric peer floors, or null.</param>
        /// <param name="trend">Configured per-metric trend floors, or null.</param>
        /// <param name="step">
        /// Configured per-metric STEP floors, or null. Zero for a metric means "fall back to that metric's
        /// trend floor" — see <see cref="MinAbsoluteLevelShift(MetricIndex)"/>.
        /// </param>
        /// <param name="calibrator">Learned floors, consulted only where the configured one is absent.</param>
        /// <param name="applyCalibrated">Whether the fallback is used at all.</param>
        public ConfiguredFloorSource(
            IReadOnlyList<double>? gap,
            IReadOnlyList<double>? trend,
            IReadOnlyList<double>? step,
            FloorCalibrator? calibrator,
            bool applyCalibrated)
        {
            _gap = gap;
            _trend = trend;
            _step = step;
            _calibrator = applyCalibrated ? calibrator : null;
        }

        /// <summary>Which of the three floors a lookup is for. A bool stopped being enough at three.</summary>
        private enum Kind
        {
            Gap,
            Trend,
            Step,
        }

        /// <inheritdoc/>
        public double MinAbsoluteGap(MetricIndex metric) => Resolve(_gap, metric, Kind.Gap);

        /// <inheritdoc/>
        public double MinAbsoluteTrendChange(MetricIndex metric) => Resolve(_trend, metric, Kind.Trend);

        /// <inheritdoc/>
        /// <remarks>
        /// <b>There is now a separate step floor, and the trend table is its fallback.</b> Until 2026-08-11
        /// this read the trend table outright, on the reasoning that adding a step floor would drop every
        /// existing deployment onto the calibrator overnight. That reasoning was right about the hazard and
        /// wrong about the remedy: falling back per metric to the trend floor keeps every deployed threshold
        /// exactly where it was while letting an operator write the correct number.
        ///
        /// <para>Why it had to be separable, measured rather than argued (<c>AN-D4b</c>): a trend floor is
        /// fitted to how far ONE pod's series travels across a window, a step floor to how far the median
        /// across pods moves between the halves of one. Sharing them made the step gate demand 40% of the
        /// level on MemoryWorkingSetBytes — so its 25% relative gate never bound, 100% of the time — and
        /// 123% at the low decile of GcGen2HeapBytes.</para>
        /// </remarks>
        public double MinAbsoluteLevelShift(MetricIndex metric)
        {
            // A configured step floor wins outright. This is the only place the two tables differ, and the
            // ORDER is the compatibility contract: falling back to the trend table — not to the calibrator —
            // is what makes adding this field move no deployed threshold on its own.
            var explicitly = AnomalyGuardOptions.FloorFor(_step, metric);

            return explicitly > 0.0 ? explicitly : Resolve(_trend, metric, Kind.Step);
        }

        /// <inheritdoc/>
        /// <remarks>
        /// No configured table to consult: a custom channel's explicit floor lives on its
        /// <c>CustomMetricBinding</c>, which the guard holds and applies ahead of asking here. This answers
        /// only the second half of the question - what a healthy period turned out to look like.
        /// </remarks>
        public double MinAbsoluteGap(string signal) => Learned(signal, Kind.Gap);

        /// <inheritdoc cref="MinAbsoluteGap(string)"/>
        public double MinAbsoluteTrendChange(string signal) => Learned(signal, Kind.Trend);

        /// <inheritdoc cref="MinAbsoluteGap(string)"/>
        public double MinAbsoluteLevelShift(string signal) => Learned(signal, Kind.Step);

        /// <summary>The proposal's answer for one kind of gate. One place, so the three cannot diverge.</summary>
        private static double Proposed(in FloorProposal proposal, Kind kind)
        {
            return kind switch
            {
                Kind.Gap => proposal.ProposedMinAbsoluteGap,
                Kind.Trend => proposal.ProposedMinAbsoluteTrendChange,
                _ => proposal.ProposedMinAbsoluteLevelShift,
            };
        }

        private double Learned(string signal, Kind kind)
        {
            ArgumentNullException.ThrowIfNull(signal);

            if (_calibrator == null)
            {
                return 0.0;
            }

            var proposal = _calibrator.Propose(signal);

            if (!proposal.IsUsable)
            {
                return 0.0;
            }

            return Proposed(proposal, kind);
        }

        private double Resolve(IReadOnlyList<double>? configured, MetricIndex metric, Kind kind)
        {
            var explicitly = AnomalyGuardOptions.FloorFor(configured, metric);

            if (explicitly > 0.0 || _calibrator == null)
            {
                return explicitly;
            }

            var proposal = _calibrator.Propose()[(int)metric];

            if (!proposal.IsUsable)
            {
                return 0.0;
            }

            return Proposed(proposal, kind);
        }
    }
}
