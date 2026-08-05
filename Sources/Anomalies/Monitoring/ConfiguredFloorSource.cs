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
    /// bar above that fault and blinds the guard to it at that size, quietly. Pass
    /// <paramref name="applyCalibrated"/> as false where the observation period cannot be trusted.</para>
    /// </summary>
    public sealed class ConfiguredFloorSource : IAbsoluteFloorSource
    {
        private readonly IReadOnlyList<double>? _gap;
        private readonly IReadOnlyList<double>? _trend;
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
        {
            _gap = gap;
            _trend = trend;
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
        /// <b>The configured table is the trend one, deliberately.</b> There is no separate step floor in the
        /// config file and adding one would make every existing deployment's step gate fall back to the
        /// calibrator overnight. An operator who wrote a number for a signal meant "do not report movements
        /// below this on this signal", and that reading still holds. What changes is the fallback: where
        /// nothing is configured, the learned floor now comes from the step distribution rather than the
        /// slope distribution, which is the defect being fixed.
        /// </remarks>
        public double MinAbsoluteLevelShift(MetricIndex metric) => Resolve(_trend, metric, Kind.Step);

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

            if (_calibrator is null)
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

            if (explicitly > 0.0 || _calibrator is null)
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
