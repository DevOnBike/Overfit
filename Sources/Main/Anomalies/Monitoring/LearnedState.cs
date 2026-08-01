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

        /// <summary>Renders both parts into one payload.</summary>
        public static string Write(MetricHistory history, FloorCalibrator calibrator)
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

            return text.ToString();
        }

        /// <summary>
        /// Splits a payload back into its two parts. Either may come back empty; neither throws.
        /// </summary>
        public static (MetricHistory History, FloorCalibrator Calibrator) Read(
            string? state, TrendOptions? trendOptions = null)
        {
            if (string.IsNullOrWhiteSpace(state))
            {
                return (new MetricHistory(), new FloorCalibrator(trendOptions));
            }

            var historyStart = state.IndexOf(HistorySection, StringComparison.Ordinal);
            var calibrationStart = state.IndexOf(CalibrationSection, StringComparison.Ordinal);

            // A payload written before the calibration section existed carries the baseline alone. Reading it
            // as "no sections found, therefore nothing" would silently discard a week of learning and look
            // exactly like a cold start — the failure mode this whole subsystem is built to make loud.
            if (historyStart < 0 && calibrationStart < 0)
            {
                return (MetricHistory.Read(state), new FloorCalibrator(trendOptions));
            }

            var history = historyStart >= 0
                ? Section(state, historyStart + HistorySection.Length, calibrationStart)
                : null;

            var calibration = calibrationStart >= 0
                ? state[(calibrationStart + CalibrationSection.Length)..]
                : null;

            return (MetricHistory.Read(history), FloorCalibrator.Read(calibration, trendOptions));
        }

        private static string Section(string state, int from, int until)
        {
            var end = until > from ? until : state.Length;

            return state[from..end];
        }
    }
}
