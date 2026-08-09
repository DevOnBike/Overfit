// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// Everything the guard has learned about what normal looks like, in one payload.
    ///
    /// <para><b>Two things, one file, and that is deliberate — unlike incidents, which get their own.</b>
    /// The per-hour baseline and the floor calibration are the same concern seen twice: both are "what this
    /// cluster does when nothing is wrong", both are worthless without the other after a restart, and both
    /// are rebuilt over days rather than cycles. Open incidents are a different lifetime and a different
    /// format, and coupling those would mean a version bump in one silently invalidating the other.</para>
    ///
    /// <para><b>A partial read is a cold start for the part that failed, not for both.</b> An unreadable
    /// baseline must not throw away a good calibration: the guard is better off with one of the two than with
    /// neither, and better off with neither than not running.</para>
    /// </summary>
    public static class LearnedState
    {
        private const string HistorySection = "### history";
        private const string CalibrationSection = "### calibration";
        private const string LabelSection = "### labels";
        private const string SuppressionSection = "### suppressions";

        /// <summary>
        /// Added for AN-D1. Written last and read by position, so a payload from before it existed parses
        /// exactly as it always did and the novelty part cold-starts alone — the same graceful degradation
        /// already proven twice, for labels and for suppressions.
        /// </summary>
        private const string PeerNoveltySection = "### peer-novelty";

        /// <summary>Renders both parts into one payload.</summary>
        public static string Write(
            MetricHistory history,
            FloorCalibrator calibrator,
            OperatorLabelStore? labels = null,
            SuppressionStore? suppressions = null,
            PeerNoveltyTracker? novelty = null)
        {
            ArgumentNullException.ThrowIfNull(history);
            ArgumentNullException.ThrowIfNull(calibrator);

            var text = new StringBuilder();

            text.Append(HistorySection).Append('\n').Append(history.Write());

            if (!text.ToString().EndsWith('\n'))
            {
                text.Append('\n');
            }

            text.Append(CalibrationSection).Append('\n').Append(calibrator.Write());

            if (!text.ToString().EndsWith('\n'))
            {
                text.Append('\n');
            }

            // Written last and read by position, so a payload from before labels existed still parses: the
            // section simply is not there and Read hands back an empty store.
            text.Append(LabelSection).Append('\n').Append(labels?.Write() ?? string.Empty);

            if (!text.ToString().EndsWith('\n'))
            {
                text.Append('\n');
            }

            text.Append(SuppressionSection).Append('\n').Append(suppressions?.Write() ?? string.Empty);

            if (!text.ToString().EndsWith('\n'))
            {
                text.Append('\n');
            }

            text.Append(PeerNoveltySection).Append('\n').Append(novelty?.Write() ?? string.Empty);

            return text.ToString();
        }

        /// <summary>
        /// Splits a payload back into its two parts. Either may come back empty; neither throws.
        /// </summary>
        public static LearnedStateSnapshot Read(string? state, TrendOptions? trendOptions = null)
        {
            if (string.IsNullOrWhiteSpace(state))
            {
                return new LearnedStateSnapshot(
                    new MetricHistory(),
                    new FloorCalibrator(trendOptions),
                    new OperatorLabelStore(),
                    new SuppressionStore());
            }

            var historyStart = state.IndexOf(HistorySection, StringComparison.Ordinal);
            var calibrationStart = state.IndexOf(CalibrationSection, StringComparison.Ordinal);
            var labelStart = state.IndexOf(LabelSection, StringComparison.Ordinal);
            var suppressionStart = state.IndexOf(SuppressionSection, StringComparison.Ordinal);
            var noveltyStart = state.IndexOf(PeerNoveltySection, StringComparison.Ordinal);

            // A payload written before the calibration section existed carries the baseline alone. Reading it
            // as "no sections found, therefore nothing" would silently discard a week of learning and look
            // exactly like a cold start — the failure mode this whole subsystem is built to make loud.
            if (historyStart < 0 && calibrationStart < 0)
            {
                return new LearnedStateSnapshot(
                    MetricHistory.Read(state),
                    new FloorCalibrator(trendOptions),
                    new OperatorLabelStore(),
                    new SuppressionStore());
            }

            var history = historyStart >= 0
                ? Section(state, historyStart + HistorySection.Length, calibrationStart)
                : null;

            var calibration = calibrationStart >= 0
                ? Section(state, calibrationStart + CalibrationSection.Length, labelStart)
                : null;

            var labels = labelStart >= 0
                ? Section(state, labelStart + LabelSection.Length, suppressionStart)
                : null;

            var suppressions = suppressionStart >= 0
                ? Section(state, suppressionStart + SuppressionSection.Length, noveltyStart)
                : null;

            var novelty = noveltyStart >= 0
                ? state[(noveltyStart + PeerNoveltySection.Length)..]
                : string.Empty;

            var calibrator = FloorCalibrator.Read(calibration, trendOptions);
            var store = OperatorLabelStore.Read(labels);

            calibrator.UseLabels(store);

            return new LearnedStateSnapshot(
                MetricHistory.Read(history),
                calibrator,
                store,
                SuppressionStore.Read(suppressions))
            {
                PeerNovelty = novelty,
            };
        }

        private static string Section(string state, int from, int until)
        {
            var end = until > from ? until : state.Length;

            return state[from..end];
        }
    }
}
