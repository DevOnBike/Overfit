// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// What each workload normally does, per signal, per hour of the day — across days.
    ///
    /// <para><b>This is the memory the guard has never had, and its absence is the root of three separate
    /// limitations.</b> Every cycle judges one twenty-minute window in isolation, so: a seasonal expectation
    /// is impossible even though <see cref="Statistics.SeasonalBaseline"/> exists and was measured to cut
    /// 2551 false incidents a day to 376; a <b>slow leak is invisible</b>, because two megabytes an hour never
    /// moves far enough inside twenty minutes and still kills a pod in a week; and "this has been wrong for
    /// three days" cannot be told apart from "for ten minutes", which are opposite operational decisions.</para>
    ///
    /// <para><b>Keyed by WORKLOAD, not by pod, and that is a fact about Kubernetes rather than a
    /// simplification.</b> Pod names do not survive a deployment — every rollout replaces them — so a
    /// per-pod history would reset itself exactly when the interesting comparison ("is the new version worse
    /// than the old one?") became possible. Anything genuinely per-pod is short-lived by construction and
    /// belongs in the cycle, not here.</para>
    ///
    /// <para><b>One observation per hour per day.</b> At a five-minute cadence an hour produces twelve
    /// cycles; recording all of them would fill a week's worth of storage with half a day of data and make
    /// "days" a lie. A repeat within the same hour of the same day replaces the previous entry.</para>
    ///
    /// <para><b>Bounded on purpose.</b> A store that grows with every pod name a cluster has ever had is a
    /// leak in the monitoring tool, which is a particularly poor look. Days are capped and entries expire.</para>
    ///
    /// <para>Not thread-safe; the guard drives it from one cycle at a time.</para>
    /// </summary>
    public sealed class MetricHistory
    {
        private const string Header = "overfit-metric-history\tv1";

        /// <summary>Distinct days kept per bucket. A week — the shadow period a deployment starts with.</summary>
        public const int MaxDays = 7;

        private readonly Dictionary<string, double[]> _values = new(StringComparer.Ordinal);
        private readonly Dictionary<string, int[]> _days = new(StringComparer.Ordinal);
        private readonly int _maxBuckets;

        /// <param name="maxBuckets">
        /// Ceiling on distinct (workload, signal, hour) buckets. Reached only by a cluster with hundreds of
        /// workloads; past it, new buckets are dropped rather than the process growing without limit.
        /// </param>
        public MetricHistory(int maxBuckets = 100_000)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(maxBuckets, 1);

            _maxBuckets = maxBuckets;
        }

        /// <summary>Buckets held, for a diagnostic that wants to say how much has been learned.</summary>
        public int Buckets => _values.Count;

        /// <summary>
        /// Records what <paramref name="workload"/> did on <paramref name="metric"/> in the hour containing
        /// <paramref name="at"/>.
        /// </summary>
        /// <param name="workload">The deployment this level belongs to. The bucket key, so two workloads
        /// with the same metric never share a baseline.</param>
        /// <param name="metric">Which channel the value is for.</param>
        /// <param name="at">When it was observed. Only the hour-of-day is kept — the baseline answers "what
        /// does this workload normally do at this time", not "what did it do on Tuesday".</param>
        /// <param name="value">
        /// The workload's own level — normally the median across its replicas, so a single odd pod does not
        /// move the baseline the whole deployment is later judged against.
        /// </param>
        public void Observe(string workload, MetricIndex metric, DateTimeOffset at, double value)
        {
            ArgumentNullException.ThrowIfNull(workload);

            if (!double.IsFinite(value))
            {
                return;
            }

            var key = Key(workload, metric, at.Hour);

            if (!_values.TryGetValue(key, out var values))
            {
                if (_values.Count >= _maxBuckets)
                {
                    return;
                }

                values = new double[MaxDays];
                var stamps = new int[MaxDays];

                for (var i = 0; i < MaxDays; i++)
                {
                    stamps[i] = int.MinValue;
                }

                _values[key] = values;
                _days[key] = stamps;
            }

            var days = _days[key];
            var today = DayNumber(at);

            // Same hour, same day: the FIRST observation stands. Two reasons, and the second is the load-
            // bearing one. Twelve cycles an hour would otherwise crowd out the previous six days and leave
            // "Days" reporting a week of history it does not have — that is why only one is kept. And the one
            // kept is the earliest, so the stored figure describes the level near the START of the hour,
            // which is what TryExpectation anchors its interpolation on. Keeping the last would silently
            // shift every anchor to the hour's end and put a phase error into every expectation.
            for (var i = 0; i < MaxDays; i++)
            {
                if (days[i] == today)
                {
                    return;
                }
            }

            var oldest = 0;

            for (var i = 1; i < MaxDays; i++)
            {
                if (days[i] < days[oldest])
                {
                    oldest = i;
                }
            }

            days[oldest] = today;
            values[oldest] = value;
        }

        /// <summary>
        /// What this workload usually does on this signal at this hour, or <c>false</c> when nothing is known.
        /// </summary>
        public bool TryGet(string workload, MetricIndex metric, DateTimeOffset at, out HistorySummary summary)
        {
            ArgumentNullException.ThrowIfNull(workload);

            summary = default;

            var key = Key(workload, metric, at.Hour);

            if (!_values.TryGetValue(key, out var values))
            {
                return false;
            }

            var days = _days[key];

            Span<double> present = stackalloc double[MaxDays];
            var count = 0;
            var newestDay = int.MinValue;
            var newest = double.NaN;

            for (var i = 0; i < MaxDays; i++)
            {
                if (days[i] == int.MinValue)
                {
                    continue;
                }

                present[count++] = values[i];

                if (days[i] > newestDay)
                {
                    newestDay = days[i];
                    newest = values[i];
                }
            }

            if (count == 0)
            {
                return false;
            }

            present.Slice(0, count).Sort();

            summary = new HistorySummary(
                count,
                present[count / 2],
                count >= 4 ? present[(3 * count) / 4] - present[count / 4] : 0.0,
                newest);

            return true;
        }

        /// <summary>
        /// Fills <paramref name="expectation"/> with what this workload usually does across the window, one
        /// entry per sample, so it can be handed straight to <see cref="Statistics.TrendDetector"/>.
        ///
        /// <para><b>Interpolated between hours, not held flat across one, and the difference is the whole
        /// point.</b> A flat expectation subtracts the <i>level</i> and leaves the <i>slope</i> untouched — so
        /// a twenty-minute window climbing with the daily traffic curve still reads as a trend, which is
        /// exactly the false positive this exists to remove. The lab measured that climb at 3.2% of typical at
        /// the median and 10.1% at p90 against a 10% gate, correlating with traffic at +1.00. Only a reference
        /// with the same slope can cancel it, so each hour's figure anchors the start of that hour and the
        /// expectation runs linearly to the next one.</para>
        ///
        /// <para>Both surrounding hours must be known. Extrapolating past the last one would invent a
        /// continuation of a curve nobody has observed.</para>
        /// </summary>
        /// <returns>
        /// Whether every sample could be given an expectation from at least <paramref name="minimumDays"/>
        /// days of history. <b>False leaves the buffer untouched</b> — a partly-filled expectation is worse
        /// than none, because the detector cannot tell which half to trust.
        /// </returns>
        public bool TryExpectation(
            string workload,
            MetricIndex metric,
            DateTimeOffset windowStart,
            TimeSpan step,
            int minimumDays,
            Span<double> expectation)
        {
            ArgumentNullException.ThrowIfNull(workload);

            if (expectation.Length == 0 || step <= TimeSpan.Zero || minimumDays < 1)
            {
                return false;
            }

            // Checked before anything is written, so a refusal cannot leave a half-filled buffer behind.
            for (var i = 0; i < expectation.Length; i++)
            {
                if (!TryInterpolate(workload, metric, windowStart + (step * i), minimumDays, out _))
                {
                    return false;
                }
            }

            for (var i = 0; i < expectation.Length; i++)
            {
                TryInterpolate(workload, metric, windowStart + (step * i), minimumDays, out expectation[i]);
            }

            return true;
        }

        /// <summary>
        /// The expected level at <paramref name="at"/>, running linearly from the containing hour's figure to
        /// the next hour's.
        /// </summary>
        private bool TryInterpolate(
            string workload, MetricIndex metric, DateTimeOffset at, int minimumDays, out double value)
        {
            value = double.NaN;

            if (!TryGet(workload, metric, at, out var start) || !start.IsUsable(minimumDays))
            {
                return false;
            }

            if (!TryGet(workload, metric, at.AddHours(1), out var next) || !next.IsUsable(minimumDays))
            {
                return false;
            }

            var through = (at.Minute + (at.Second / 60.0)) / 60.0;

            value = start.Median + ((next.Median - start.Median) * through);

            return true;
        }

        /// <summary>Drops buckets untouched for <paramref name="maxAge"/>, so a deleted workload lets go.</summary>
        public int Forget(DateTimeOffset now, TimeSpan maxAge)
        {
            var cutoff = DayNumber(now) - (int)Math.Ceiling(maxAge.TotalDays);
            var stale = new List<string>();

            foreach (var (key, days) in _days)
            {
                var newest = int.MinValue;

                for (var i = 0; i < days.Length; i++)
                {
                    if (days[i] > newest)
                    {
                        newest = days[i];
                    }
                }

                if (newest < cutoff)
                {
                    stale.Add(key);
                }
            }

            for (var i = 0; i < stale.Count; i++)
            {
                _values.Remove(stale[i]);
                _days.Remove(stale[i]);
            }

            return stale.Count;
        }

        /// <summary>Serialises the whole store. Tab-separated, one observation per line.</summary>
        public string Write()
        {
            var text = new StringBuilder();

            text.Append(Header).Append('\n');

            foreach (var (key, values) in _values)
            {
                var days = _days[key];

                for (var i = 0; i < MaxDays; i++)
                {
                    if (days[i] == int.MinValue)
                    {
                        continue;
                    }

                    text.Append(key).Append('\t')
                        .Append(days[i].ToString(CultureInfo.InvariantCulture)).Append('\t')
                        .Append(values[i].ToString("R", CultureInfo.InvariantCulture)).Append('\n');
                }
            }

            return text.ToString();
        }

        /// <summary>
        /// Restores a store. A missing, empty or unrecognised payload yields an empty history rather than an
        /// error: starting without a baseline is correct-but-quiet, and refusing to start is not.
        /// </summary>
        public static MetricHistory Read(string? state, int maxBuckets = 100_000)
        {
            var history = new MetricHistory(maxBuckets);

            if (string.IsNullOrWhiteSpace(state))
            {
                return history;
            }

            var lines = state.Split('\n', StringSplitOptions.RemoveEmptyEntries);

            if (lines.Length == 0 || !lines[0].StartsWith(Header, StringComparison.Ordinal))
            {
                return history;
            }

            for (var i = 1; i < lines.Length; i++)
            {
                var parts = lines[i].Split('\t');

                if (parts.Length != 5
                    || !Enum.TryParse<MetricIndex>(parts[1], out var metric)
                    || !int.TryParse(parts[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out var hour)
                    || !int.TryParse(parts[3], NumberStyles.Integer, CultureInfo.InvariantCulture, out var day)
                    || !double.TryParse(parts[4], NumberStyles.Float, CultureInfo.InvariantCulture, out var value))
                {
                    continue;
                }

                history.Restore(parts[0], metric, hour, day, value);
            }

            return history;
        }

        /// <summary>Replays one stored observation, preserving its original day stamp.</summary>
        private void Restore(string workload, MetricIndex metric, int hour, int day, double value)
        {
            var key = Key(workload, metric, hour);

            if (!_values.TryGetValue(key, out var values))
            {
                if (_values.Count >= _maxBuckets)
                {
                    return;
                }

                values = new double[MaxDays];
                var stamps = new int[MaxDays];

                for (var i = 0; i < MaxDays; i++)
                {
                    stamps[i] = int.MinValue;
                }

                _values[key] = values;
                _days[key] = stamps;
            }

            var days = _days[key];

            for (var i = 0; i < MaxDays; i++)
            {
                if (days[i] == day)
                {
                    values[i] = value;

                    return;
                }
            }

            var oldest = 0;

            for (var i = 1; i < MaxDays; i++)
            {
                if (days[i] < days[oldest])
                {
                    oldest = i;
                }
            }

            days[oldest] = day;
            values[oldest] = value;
        }

        /// <summary>Days since the epoch, in UTC — the calendar day an observation belongs to.</summary>
        private static int DayNumber(DateTimeOffset at) => (int)(at.UtcDateTime.Ticks / TimeSpan.TicksPerDay);

        private static string Key(string workload, MetricIndex metric, int hour)
            => $"{workload}\t{metric}\t{hour.ToString(CultureInfo.InvariantCulture)}";
    }
}
