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

        /// <inheritdoc/>
        public double MinAbsoluteGap(MetricIndex metric) => Resolve(_gap, metric, trend: false);

        /// <inheritdoc/>
        public double MinAbsoluteTrendChange(MetricIndex metric) => Resolve(_trend, metric, trend: true);

        private double Resolve(IReadOnlyList<double>? configured, MetricIndex metric, bool trend)
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

            return trend ? proposal.ProposedMinAbsoluteTrendChange : proposal.ProposedMinAbsoluteGap;
        }
    }
}
