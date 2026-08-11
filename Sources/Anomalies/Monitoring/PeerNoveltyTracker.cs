// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// Remembers, per pod and per metric, how far that replica has sat from its peers over recent cycles, and
    /// answers whether that separation is <i>changing</i>.
    ///
    /// <para><b>The defect this exists for.</b> A replica with a permanently heavier role clears the peer
    /// gate every cycle, correctly, for as long as it runs — measured on the lab at roughly 97% of a day for
    /// one pod on <c>MemoryWorkingSetBytes</c>. Every one of those findings is arithmetically right and none
    /// of them is news after the first. What an operator needs to hear about is the gap <i>moving</i>.</para>
    ///
    /// <para><b>The change test is <see cref="TrendDetector"/>, reused rather than reinvented.</b> Feeding it
    /// the gap-over-time series turns "is this pod above its peers" into "is this pod pulling away from
    /// them", using the same significance test, the same materiality floor and the same five-status
    /// vocabulary as everything else here. Validated before it was wired in: the recorded 15-sample
    /// <c>pj7r8</c> gap sequence reads <c>Healthy</c> (tau 0.16, p 0.23), and synthetic gaps rising and
    /// falling at 400 kB a cycle read <c>Anomalous</c> with the matching direction.</para>
    ///
    /// <para><b>Absence of history is never stability.</b> Below
    /// <see cref="PeerNoveltyOptions.MinimumCycles"/>, and for a pod this tracker has never seen, the answer
    /// is <see cref="NoveltyKind.New"/> and the finding is forwarded at full severity. A suppression gate
    /// that fails closed on missing information reintroduces the original defect as silence, which is
    /// strictly worse than the noise it replaces.</para>
    ///
    /// <para><b>A shrinking gap counts as standing, not as change.</b> <c>Anomalous</c> with
    /// <see cref="TrendDirection.Falling"/> means the replica is becoming <i>more</i> like its peers; treating
    /// that as a new event would page somebody for an improvement. The lab's own −2.65 MB over three hours is
    /// exactly this case.</para>
    ///
    /// <para><b>Keyed twice, because the guard has two peer call sites.</b> Built-in channels arrive as a
    /// <see cref="MetricIndex"/> and custom ones as a name, matching <c>FloorCalibrator</c>'s own split; the
    /// serialised form marks a custom key with the same <c>~</c> convention, so a channel named like an enum
    /// member cannot be read back as one.</para>
    ///
    /// <para>Not thread-safe, like the rest of this subsystem: one guard, one cycle at a time.</para>
    /// </summary>
    public sealed class PeerNoveltyTracker
    {
        /// <summary>
        /// Marks a custom channel's key in the serialised form, so a channel named like a
        /// <see cref="MetricIndex"/> member — or like an integer, which also parses as one — cannot be read
        /// back as that member. The same character and the same reasoning as <c>FloorCalibrator</c>'s own
        /// marker: two collision schemes for one enum in one assembly is the defect waiting to happen.
        /// </summary>
        private const char CustomMarker = '~';

        /// <summary>
        /// Defence in depth against a malformed roster, mirroring <c>MetricHistory</c>'s own bucket ceiling.
        /// Ordinary pruning is against the live pod list; this is what stops a roster that reports nonsense
        /// from growing the dictionary without bound between prunes.
        /// </summary>
        private const int MaxTrackedPods = 10_000;

        private readonly Dictionary<string, PodState> _pods = new(StringComparer.Ordinal);
        private readonly TrendDetector _trend = new();
        private readonly PeerNoveltyOptions _options;

        /// <param name="options">
        /// Cadence and thresholds. Must be one of the named profiles rather than <c>default</c> — see
        /// <see cref="PeerNoveltyOptions"/> for why there is no balanced one.
        /// </param>
        public PeerNoveltyTracker(PeerNoveltyOptions options)
        {
            if (!options.IsValid)
            {
                throw new ArgumentException(
                    "Peer-novelty options are not usable — use PeerNoveltyOptions.PerShift/Daily/Weekly "
                    + "rather than default. Two of these values cannot be fitted by any measurement, which is "
                    + "why an implicit default is refused rather than guessed.",
                    nameof(options));
            }

            _options = options;
        }

        /// <summary>How many pods carry novelty state right now.</summary>
        public int TrackedPods => _pods.Count;

        /// <summary>
        /// Folds this cycle's gap for a built-in channel into the pod's history and classifies it.
        /// </summary>
        /// <param name="pod">Pod the gap was measured on.</param>
        /// <param name="createdAt">
        /// What the cluster says about when this pod was created. A value that differs from the one recorded
        /// alongside the history <b>resets that history</b>: a StatefulSet pod keeps its name across a
        /// restart, and inheriting the previous incarnation's learning would silence a genuinely new
        /// deviation. <c>default</c> — topology cannot say — is recorded and compared like any other value.
        /// </param>
        /// <param name="metric">Channel.</param>
        /// <param name="gap">The peer detector's <c>AbsoluteGap</c> for this pod this cycle.</param>
        /// <param name="at">Cycle timestamp.</param>
        /// <param name="minAbsoluteGapChange">
        /// Smallest change in the gap, across the retained window and in the metric's own units, worth calling
        /// new. Supplied per call rather than held here, because only the caller owns the per-metric table.
        /// </param>
        public NoveltyDecision Observe(
            string pod,
            DateTimeOffset createdAt,
            MetricIndex metric,
            double gap,
            DateTimeOffset at,
            double minAbsoluteGapChange)
        {
            ArgumentNullException.ThrowIfNull(pod);

            var state = Resolve(pod, createdAt);

            if (state == null)
            {
                return NoveltyDecision.Unknown;
            }

            var index = (int)metric;

            if (index < 0 || index >= (int)MetricIndex.Count)
            {
                return NoveltyDecision.Unknown;
            }

            var series = state.Builtin[index];

            if (series == null)
            {
                series = new GapSeries(_options.RetainedCyclesPerSeries);
                state.Builtin[index] = series;
            }

            return Classify(series, gap, at, minAbsoluteGapChange);
        }

        /// <summary>
        /// Folds this cycle's gap for a <c>CustomMetricBinding</c> channel into the pod's history and
        /// classifies it, exactly as the built-in overload does. Kept separate rather than generalised for
        /// the same reason <c>FloorCalibrator</c> keeps its two: one indexes an enum, the other looks up a
        /// name.
        /// </summary>
        /// <param name="pod">Pod the gap was measured on.</param>
        /// <param name="createdAt">
        /// What the cluster says about when this pod was created; a change discards the history. See the
        /// built-in overload.
        /// </param>
        /// <param name="channel">A <c>CustomMetricBinding</c> name.</param>
        /// <param name="gap">The peer detector's <c>AbsoluteGap</c> for this pod this cycle.</param>
        /// <param name="at">Cycle timestamp.</param>
        /// <param name="minAbsoluteGapChange">
        /// Smallest change in the gap, across the retained window and in the channel's own units, worth
        /// calling new.
        /// </param>
        public NoveltyDecision Observe(
            string pod,
            DateTimeOffset createdAt,
            string channel,
            double gap,
            DateTimeOffset at,
            double minAbsoluteGapChange)
        {
            ArgumentNullException.ThrowIfNull(pod);
            ArgumentNullException.ThrowIfNull(channel);

            var state = Resolve(pod, createdAt);

            if (state == null)
            {
                return NoveltyDecision.Unknown;
            }

            if (!state.Custom.TryGetValue(channel, out var series))
            {
                series = new GapSeries(_options.RetainedCyclesPerSeries);
                state.Custom[channel] = series;
            }

            return Classify(series, gap, at, minAbsoluteGapChange);
        }

        /// <summary>
        /// Drops pods the cluster no longer lists, so a scale-down does not leave state growing for ever —
        /// the same roster-based pruning <c>AnomalyGuard</c> already applies to its silent-pod counters.
        /// </summary>
        public void Prune(IReadOnlyList<string> livePods)
        {
            ArgumentNullException.ThrowIfNull(livePods);

            if (livePods.Count == 0 || _pods.Count <= livePods.Count)
            {
                return;
            }

            var live = new HashSet<string>(StringComparer.Ordinal);

            for (var i = 0; i < livePods.Count; i++)
            {
                live.Add(livePods[i]);
            }

            var stale = new List<string>();

            foreach (var pod in _pods.Keys)
            {
                if (!live.Contains(pod))
                {
                    stale.Add(pod);
                }
            }

            for (var i = 0; i < stale.Count; i++)
            {
                _pods.Remove(stale[i]);
            }
        }

        /// <summary>
        /// Serialises everything learned, one line per (pod, channel).
        ///
        /// <para>Seven tab-separated columns, and the count is the forward-compatibility guard — a reader
        /// that predates a future eighth column skips the line rather than misreading it, the same shape
        /// <c>FloorCalibrator</c>'s <c>#windows</c> marker uses.</para>
        /// </summary>
        public string Write()
        {
            var text = new StringBuilder();

            foreach (var (pod, state) in _pods)
            {
                for (var m = 0; m < state.Builtin.Length; m++)
                {
                    if (state.Builtin[m] is { } series)
                    {
                        WriteRow(text, pod, state, ((MetricIndex)m).ToString(), series);
                    }
                }

                foreach (var (channel, series) in state.Custom)
                {
                    WriteRow(
                        text, pod, state, CustomMarker + LearnedStateText.Escape(channel), series);
                }
            }

            return text.ToString();
        }

        /// <summary>
        /// Restores a tracker. Anything unreadable yields an empty one, for the same reason every other store
        /// here does: a guard that refuses to start because its own scratch file is malformed has turned a
        /// soft degradation into an outage — and this store failing open means <i>more</i> reporting, never
        /// less.
        /// </summary>
        /// <param name="state">What <see cref="Write"/> produced.</param>
        /// <param name="options">Cadence and thresholds for the restored tracker.</param>
        /// <param name="roster">
        /// What the cluster currently says each pod's creation time is. A row whose recorded creation time
        /// disagrees is <b>dropped</b>, not adopted: a reused pod name is a different incarnation, and
        /// inheriting its predecessor's learning across a restart would reintroduce exactly the bug the
        /// in-memory reset exists to prevent, just delayed. A pod the roster does not mention is adopted —
        /// topology is frequently unavailable at construction, and refusing every row then would make
        /// persistence worthless.
        /// </param>
        public static PeerNoveltyTracker Read(
            string? state,
            PeerNoveltyOptions options,
            IReadOnlyDictionary<string, DateTimeOffset>? roster = null)
        {
            var tracker = new PeerNoveltyTracker(options);

            if (string.IsNullOrWhiteSpace(state))
            {
                return tracker;
            }

            var lines = state.Split('\n', StringSplitOptions.RemoveEmptyEntries);

            for (var i = 0; i < lines.Length; i++)
            {
                tracker.ReadRow(lines[i], roster);
            }

            return tracker;
        }

        private void ReadRow(string line, IReadOnlyDictionary<string, DateTimeOffset>? roster)
        {
            var parts = line.Split('\t');

            if (parts.Length != 7)
            {
                return;
            }

            if (!long.TryParse(
                    parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var createdTicks)
                || !long.TryParse(
                    parts[3], NumberStyles.Integer, CultureInfo.InvariantCulture, out var forwardedTicks)
                || !int.TryParse(parts[4], NumberStyles.Integer, CultureInfo.InvariantCulture, out var kind))
            {
                return;
            }

            var pod = LearnedStateText.Unescape(parts[0]);
            var createdAt = new DateTimeOffset(createdTicks, TimeSpan.Zero);

            // The reuse check the ADR requires: a name the cluster still knows, carrying a different creation
            // time, is a different pod wearing an old name.
            if (roster != null
                && roster.TryGetValue(pod, out var current)
                && current != createdAt)
            {
                return;
            }

            if (_pods.Count >= MaxTrackedPods && !_pods.ContainsKey(pod))
            {
                return;
            }

            if (!_pods.TryGetValue(pod, out var state) || state.CreatedAt != createdAt)
            {
                state = new PodState(createdAt);
                _pods[pod] = state;
            }

            var series = new GapSeries(_options.RetainedCyclesPerSeries)
            {
                LastForwarded = new DateTimeOffset(forwardedTicks, TimeSpan.Zero),
                Last = kind == (int)NoveltyKind.Standing ? NoveltyKind.Standing : NoveltyKind.New,
            };

            series.Fill(parts[5], parts[6]);

            var key = parts[2];

            if (key.Length > 1 && key[0] == CustomMarker)
            {
                state.Custom[LearnedStateText.Unescape(key.Substring(1))] = series;

                return;
            }

            if (Enum.TryParse<MetricIndex>(key, out var metric) && metric != MetricIndex.Count)
            {
                state.Builtin[(int)metric] = series;
            }
        }

        private static void WriteRow(
            StringBuilder text, string pod, PodState state, string key, GapSeries series)
        {
            text.Append(LearnedStateText.Escape(pod)).Append('\t')
                .Append(state.CreatedAt.UtcTicks.ToString(CultureInfo.InvariantCulture)).Append('\t')
                .Append(key).Append('\t')
                .Append(series.LastForwarded.UtcTicks.ToString(CultureInfo.InvariantCulture)).Append('\t')
                .Append(((int)series.Last).ToString(CultureInfo.InvariantCulture)).Append('\t');

            series.WriteTimes(text);
            text.Append('\t');
            series.WriteValues(text);
            text.Append('\n');
        }

        /// <summary>
        /// The pod's state, resetting it when the cluster reports a different creation time, or <c>null</c>
        /// when the ceiling refuses a new pod.
        /// </summary>
        private PodState? Resolve(string pod, DateTimeOffset createdAt)
        {
            if (_pods.TryGetValue(pod, out var state))
            {
                if (state.CreatedAt == createdAt)
                {
                    return state;
                }

                // Same name, different incarnation. Everything learned belonged to the previous one.
                state = new PodState(createdAt);
                _pods[pod] = state;

                return state;
            }

            if (_pods.Count >= MaxTrackedPods)
            {
                return null;
            }

            state = new PodState(createdAt);
            _pods[pod] = state;

            return state;
        }

        private NoveltyDecision Classify(
            GapSeries series, double gap, DateTimeOffset at, double minAbsoluteGapChange)
        {
            series.Add(at, gap);

            var options = new TrendOptions(
                TrendOptions.Balanced.MaxPValue,
                TrendOptions.Balanced.MinTau,
                TrendOptions.Balanced.MinRelativeChangeOverWindow,
                _options.MinimumCycles,
                minAbsoluteGapChange);

            var verdict = _trend.Detect(series.Values, series.Times, options);

            // Rising is the only reading that means "this is changing". Healthy is a settled difference, and
            // Falling is the pod converging back on its peers — neither is news.
            var rising = verdict.Status == DetectionStatus.Anomalous
                         && verdict.Direction == TrendDirection.Rising;

            // Undecidable is fail-open, and it is the same branch as rising on purpose: WarmingUp and
            // InsufficientData must never reach the suppression path.
            var decided = verdict.Status == DetectionStatus.Healthy
                          || verdict.Status == DetectionStatus.Anomalous;

            if (rising || !decided)
            {
                series.Last = NoveltyKind.New;
                series.LastForwarded = at;

                return new NoveltyDecision(
                    NoveltyKind.New, true, verdict.Status, verdict.Direction, verdict.SampleCount,
                    verdict.Reason);
            }

            var due = at - series.LastForwarded >= _options.StandingReassertionInterval;

            series.Last = NoveltyKind.Standing;

            if (due)
            {
                series.LastForwarded = at;
            }

            return new NoveltyDecision(
                NoveltyKind.Standing, due, verdict.Status, verdict.Direction, verdict.SampleCount,
                verdict.Reason);
        }

        /// <summary>One pod's novelty state, split the way the guard's two peer call sites key it.</summary>
        private sealed class PodState
        {
            public PodState(DateTimeOffset createdAt)
            {
                CreatedAt = createdAt;
                Builtin = new GapSeries?[(int)MetricIndex.Count];
            }

            /// <summary>What the cluster said when this state was created. See <c>Resolve</c>.</summary>
            public DateTimeOffset CreatedAt
            {
                get;
            }

            /// <summary>Per <see cref="MetricIndex"/>, allocated on first use — most pods deviate on none.</summary>
            public GapSeries?[] Builtin
            {
                get;
            }

            public Dictionary<string, GapSeries> Custom
            {
                get;
            } = new(StringComparer.Ordinal);
        }

        /// <summary>
        /// A bounded, ordered ring of (timestamp, gap) for one pod and one channel.
        ///
        /// <para><b>Not <c>BoundedSamples</c>, and the difference is load-bearing.</b> That type stores values
        /// with no timestamps, and <see cref="TrendDetector.Detect"/> requires an index-aligned time span;
        /// keeping two of them in step would desynchronise the first time a non-finite gap was skipped on one
        /// side and not the other. Its reservoir also spreads samples across the <i>whole</i> stream, which is
        /// the wrong shape for a question about recent movement.</para>
        /// </summary>
        private sealed class GapSeries
        {
            private readonly double[] _times;
            private readonly double[] _values;
            private int _count;

            public GapSeries(int capacity)
            {
                _times = new double[capacity];
                _values = new double[capacity];
            }

            /// <summary>When this series was last allowed through to the incident pipeline.</summary>
            public DateTimeOffset LastForwarded
            {
                get; set;
            }

            /// <summary>The classification the previous cycle reached, carried across a restart.</summary>
            public NoveltyKind Last
            {
                get; set;
            }

            /// <summary>Timestamps in seconds, oldest first — compacted, so the span is contiguous.</summary>
            public ReadOnlySpan<double> Times => _times.AsSpan(0, _count);

            /// <inheritdoc cref="Times"/>
            public ReadOnlySpan<double> Values => _values.AsSpan(0, _count);

            public void Add(DateTimeOffset at, double gap)
            {
                var seconds = at.ToUnixTimeMilliseconds() / 1000.0;

                if (_count < _times.Length)
                {
                    _times[_count] = seconds;
                    _values[_count] = gap;
                    _count++;

                    return;
                }

                // Full: shift down by one. A ring with a moving head would avoid the copy, but the detector
                // needs a contiguous, ordered span and a cycle is five minutes apart — the copy is a hundred
                // doubles once per cycle per deviating pod.
                _times.AsSpan(1, _count - 1).CopyTo(_times.AsSpan(0));
                _values.AsSpan(1, _count - 1).CopyTo(_values.AsSpan(0));

                _times[_count - 1] = seconds;
                _values[_count - 1] = gap;
            }

            public void WriteTimes(StringBuilder text) => Append(text, _times);

            public void WriteValues(StringBuilder text) => Append(text, _values);

            /// <summary>Restores both columns, keeping only as many pairs as both carry.</summary>
            public void Fill(string times, string values)
            {
                var t = times.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                var v = values.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                var pairs = Math.Min(Math.Min(t.Length, v.Length), _times.Length);

                for (var i = 0; i < pairs; i++)
                {
                    if (!double.TryParse(
                            t[i], NumberStyles.Float, CultureInfo.InvariantCulture, out var time)
                        || !double.TryParse(
                            v[i], NumberStyles.Float, CultureInfo.InvariantCulture, out var value))
                    {
                        continue;
                    }

                    _times[_count] = time;
                    _values[_count] = value;
                    _count++;
                }
            }

            private void Append(StringBuilder text, double[] source)
            {
                for (var i = 0; i < _count; i++)
                {
                    if (i > 0)
                    {
                        text.Append(' ');
                    }

                    text.Append(source[i].ToString("R", CultureInfo.InvariantCulture));
                }
            }
        }
    }
}
