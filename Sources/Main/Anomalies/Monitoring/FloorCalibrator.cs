// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// Watches a deployment that is believed to be healthy and works out, per signal, how large a difference
    /// has to be before it is worth reporting.
    ///
    /// <para><b>This is what a shadow week is for.</b> The first question anyone asks about the absolute
    /// gates is "what do I put here", and until now the honest answer was that only they could know — true,
    /// and useless. The floors are not knowable in advance, but they are <i>measurable</i>: run against the
    /// cluster, watch what normal looks like, and set the bar above it.</para>
    ///
    /// <para><b>Both quantities are computed exactly as the gates compute them</b>, or the proposal would be
    /// in the wrong units. The peer gap is a pod's median against its peers' median, which is what
    /// <c>MinAbsoluteGap</c> is compared to; the trend change is the fitted slope multiplied by the window,
    /// which is what <c>MinAbsoluteTrendChange</c> is compared to.</para>
    ///
    /// <para><b>The one thing that can ruin it: the period has to have been healthy.</b> A real fault inside
    /// the observation window raises the maximum, the floor is set above the fault, and the guard is
    /// permanently blind to that fault at that size. This is the same failure as calibrating a generator
    /// against a contaminated recording — fast, repeatable, and wrong — and it is why the proposal is a
    /// suggestion for a human to accept, never something applied on its own.</para>
    ///
    /// <para>Accumulates across cycles; call <see cref="Observe"/> once per window and
    /// <see cref="Propose"/> whenever a report is wanted. Not thread-safe.</para>
    /// </summary>
    public sealed class FloorCalibrator
    {
        /// <summary>Headroom over the largest healthy observation, for what a week did not happen to show.</summary>
        private const double Margin = 1.25;

        /// <summary>Marks a custom channel's line in the serialised form, so a channel named like a
        /// <see cref="MetricIndex"/> member - or like an integer, which also parses as one - cannot be read
        /// back as that member.</summary>
        private const char CustomMarker = '~';

        private BoundedSamples[] _peerGaps;
        private BoundedSamples[] _trendChanges;
        private BoundedSamples[] _magnitudes;

        /// <summary>
        /// The same three accumulators for channels the enum does not have, keyed by name.
        ///
        /// <para><b>Custom channels were observed by every detector and by nothing that proposes a floor.</b>
        /// Their gates read <c>CustomMetricBinding.MinAbsoluteGap</c>, which defaults to zero, and zero means
        /// the gate is off - so the one part of the configuration a customer is most likely to own started in
        /// exactly the state measured at 209 false incidents a day, with no proposal ever offered to get it
        /// out of there.</para>
        /// </summary>
        private readonly Dictionary<string, CustomChannel> _customChannels = new(StringComparer.Ordinal);
        /// <summary>
        /// The last computed proposal, or null when an observation has invalidated it. Not thread-safe, like
        /// the rest of this type: one guard, one cycle at a time.
        /// </summary>
        private FloorProposal[]? _cached;

        private readonly TrendDetector _trend = new();
        private readonly TrendOptions _trendOptions;

        /// <summary>
        /// Accumulates into <see cref="BoundedSamples"/> rather than plain lists, and that is a fix rather
        /// than a style choice: the first version appended one value per pod per metric per cycle for as long
        /// as the process ran — about a million doubles across a shadow week on twelve replicas, and eight
        /// million on a hundred, with no ceiling. The maximum, which is what a floor is actually set from,
        /// stays exact; only the percentiles become sampled.
        /// </summary>
        public FloorCalibrator(TrendOptions? trendOptions = null)
        {
            _trendOptions = trendOptions ?? TrendOptions.Balanced;

            var count = (int)MetricIndex.Count;
            _peerGaps = new BoundedSamples[count];
            _trendChanges = new BoundedSamples[count];
            _magnitudes = new BoundedSamples[count];

            for (var i = 0; i < count; i++)
            {
                _peerGaps[i] = new BoundedSamples();
                _trendChanges[i] = new BoundedSamples();
                _magnitudes[i] = new BoundedSamples();
            }
        }

        /// <summary>Folds one window into the accumulated picture.</summary>
        public void Observe(MetricWindow window)
        {
            ArgumentNullException.ThrowIfNull(window);

            // Invalidated before the early returns as well: a window that contributes nothing still leaves
            // the cache correct, and reasoning about which returns are "safe" is how a stale cache is born.
            _cached = null;

            foreach (var channel in _customChannels.Values)
            {
                channel.Cached = null;
            }

            var pods = window.Pods.Count;

            if (pods == 0 || window.Length == 0)
            {
                return;
            }

            var times = new double[window.Length];
            window.WriteTimestampSeconds(times);

            var windowSeconds = times[^1] - times[0];
            var medians = new double[pods];

            for (var m = 0; m < (int)MetricIndex.Count; m++)
            {
                var metric = (MetricIndex)m;
                var usable = 0;

                for (var pod = 0; pod < pods; pod++)
                {
                    var series = window.Series(pod, metric);
                    var median = Median(series);

                    medians[pod] = median;

                    if (!double.IsFinite(median))
                    {
                        continue;
                    }

                    usable++;
                    _magnitudes[m].Add(Math.Abs(median));

                    // The same fitted change the trend gate is compared against, so the proposal lands in the
                    // units the gate reads. A verdict is not needed — only the slope.
                    var verdict = _trend.Detect(series, times, _trendOptions);

                    if (double.IsFinite(verdict.SlopePerSecond) && windowSeconds > 0.0)
                    {
                        _trendChanges[m].Add(Math.Abs(verdict.SlopePerSecond) * windowSeconds);
                    }
                }

                // A gap needs someone to be apart FROM. Below three there is no "rest of the group", which is
                // the same bound the peer detector itself refuses under.
                if (usable < 3)
                {
                    continue;
                }

                for (var pod = 0; pod < pods; pod++)
                {
                    if (!double.IsFinite(medians[pod]))
                    {
                        continue;
                    }

                    var others = MedianOfOthers(medians, pod);

                    if (double.IsFinite(others))
                    {
                        _peerGaps[m].Add(Math.Abs(medians[pod] - others));
                    }
                }
            }

            var custom = window.CustomChannels;

            for (var c = 0; c < custom.Count; c++)
            {
                var name = custom[c];

                if (!_customChannels.TryGetValue(name, out var channel))
                {
                    channel = new CustomChannel();
                    _customChannels[name] = channel;
                }

                ObserveChannel(window, name, channel, times, windowSeconds, medians);
            }
        }

        /// <summary>
        /// One custom channel, folded exactly as a built-in one is. Kept as its own method rather than
        /// generalising the loop above: the built-in path indexes by <see cref="MetricIndex"/> and this one
        /// looks up by name, and merging them would put a dictionary lookup on the inner loop of the common
        /// case to save a duplicated shape.
        /// </summary>
        private void ObserveChannel(
            MetricWindow window,
            string name,
            CustomChannel channel,
            double[] times,
            double windowSeconds,
            double[] medians)
        {
            var pods = window.Pods.Count;
            var usable = 0;

            for (var pod = 0; pod < pods; pod++)
            {
                var series = window.Series(pod, name);
                var median = Median(series);

                medians[pod] = median;

                if (!double.IsFinite(median))
                {
                    continue;
                }

                usable++;
                channel.Magnitudes.Add(Math.Abs(median));

                var verdict = _trend.Detect(series, times, _trendOptions);

                if (double.IsFinite(verdict.SlopePerSecond) && windowSeconds > 0.0)
                {
                    channel.TrendChanges.Add(Math.Abs(verdict.SlopePerSecond) * windowSeconds);
                }
            }

            if (usable < 3)
            {
                return;
            }

            for (var pod = 0; pod < pods; pod++)
            {
                if (!double.IsFinite(medians[pod]))
                {
                    continue;
                }

                var others = MedianOfOthers(medians, pod);

                if (double.IsFinite(others))
                {
                    channel.PeerGaps.Add(Math.Abs(medians[pod] - others));
                }
            }
        }

        /// <summary>
        /// Serialises everything learned so far, so a restart does not start from nothing.
        ///
        /// <para><b>This matters more than it looks.</b> Without it, every restart of the guard leaves it with
        /// no floors for as long as it takes to relearn them — and the state it is in during that hour is
        /// exactly the one measured at 209 false incidents a day. A rollout of the monitoring tool would
        /// reliably produce a burst of noise from the monitoring tool.</para>
        /// </summary>
        public string Write()
        {
            var text = new StringBuilder();

            for (var m = 0; m < (int)MetricIndex.Count; m++)
            {
                text.Append((MetricIndex)m).Append('\t')
                    .Append(_peerGaps[m].Write()).Append('\t')
                    .Append(_trendChanges[m].Write()).Append('\t')
                    .Append(_magnitudes[m].Write()).Append('\n');
            }

            // Custom lines come after the fixed ones and are marked, so a file written before custom channels
            // existed still reads, and a reader that does not know the marker skips them rather than
            // mistaking a name for an enum member.
            foreach (var (name, channel) in _customChannels)
            {
                text.Append(CustomMarker).Append(name).Append('\t')
                    .Append(channel.PeerGaps.Write()).Append('\t')
                    .Append(channel.TrendChanges.Write()).Append('\t')
                    .Append(channel.Magnitudes.Write()).Append('\n');
            }

            return text.ToString();
        }

        /// <summary>
        /// Restores a calibrator. Anything unreadable yields an empty one: a guard that refuses to start
        /// because its own scratch file is malformed has turned a soft degradation into an outage.
        /// </summary>
        public static FloorCalibrator Read(string? state, TrendOptions? trendOptions = null)
        {
            var calibrator = new FloorCalibrator(trendOptions);

            if (string.IsNullOrWhiteSpace(state))
            {
                return calibrator;
            }

            var lines = state.Split('\n', StringSplitOptions.RemoveEmptyEntries);

            for (var i = 0; i < lines.Length; i++)
            {
                var parts = lines[i].Split('\t');

                if (parts.Length != 4)
                {
                    continue;
                }

                if (parts[0].Length > 1 && parts[0][0] == CustomMarker)
                {
                    calibrator._cached = null;
                    calibrator._customChannels[parts[0][1..]] = new CustomChannel
                    {
                        PeerGaps = BoundedSamples.Read(parts[1]),
                        TrendChanges = BoundedSamples.Read(parts[2]),
                        Magnitudes = BoundedSamples.Read(parts[3])
                    };

                    continue;
                }

                if (!Enum.TryParse<MetricIndex>(parts[0], out var metric) || metric == MetricIndex.Count)
                {
                    continue;
                }

                var m = (int)metric;

                calibrator._cached = null;
                calibrator._peerGaps[m] = BoundedSamples.Read(parts[1]);
                calibrator._trendChanges[m] = BoundedSamples.Read(parts[2]);
                calibrator._magnitudes[m] = BoundedSamples.Read(parts[3]);
            }

            return calibrator;
        }

        /// <summary>
        /// The proposal so far, one entry per metric, indexed by <see cref="MetricIndex"/>.
        ///
        /// <para><b>Cached between observations, and that is a fix rather than an optimisation.</b> Computing
        /// it sorts a copy of every retained sample — thirteen signals times three statistics, each up to a
        /// thousand values — and <c>ConfiguredFloorSource</c> asks for it once per signal per gate, which is
        /// thirty-nine times a cycle for an answer that cannot change within one. Measured with
        /// <c>MemoryDiagnoser</c> on <c>AnomalyGuardScaleBenchmark</c>: <b>7.41 MB allocated per cycle at four
        /// replicas</b>, almost all of it this, and almost none of it varying with pod count — which is what
        /// gave it away, since a cost that ignores the size of the cluster is not doing work about the
        /// cluster.</para>
        ///
        /// <para>A copy is returned rather than the cached array itself. Handing out the internal instance
        /// would make a caller's stray write silently rewrite the guard's floors, and the copy costs a
        /// thirteen-element array against the sort it replaces.</para>
        /// </summary>
        public FloorProposal[] Propose()
        {
            if (_cached is null)
            {
                _cached = Compute();
            }

            var copy = new FloorProposal[_cached.Length];

            Array.Copy(_cached, copy, _cached.Length);

            return copy;
        }

        /// <summary>
        /// The proposal for a custom channel, or a proposal with zero samples when nothing has been observed
        /// under that name - which <see cref="FloorProposal.IsUsable"/> already reports as unusable, so a
        /// caller has one thing to check rather than two.
        ///
        /// <para><b>Every custom channel is treated as fittable</b>, unlike the built-ins, because nothing
        /// here can tell a restart counter from a latency. The equivalent of
        /// <c>PeerSignalCatalog.IsCountedEvent</c> is a statement about a signal's unit, and for a channel the
        /// customer named there is no catalogue to ask. Where that is wrong the proposal sets the bar above a
        /// single event; it is a suggestion for a human either way, and a configured value wins over it.</para>
        /// </summary>
        public FloorProposal Propose(string custom)
        {
            ArgumentNullException.ThrowIfNull(custom);

            if (!_customChannels.TryGetValue(custom, out var channel))
            {
                return default;
            }

            if (channel.Cached is { } cached)
            {
                return cached;
            }

            var gapMax = channel.PeerGaps.Count > 0 ? channel.PeerGaps.Max : 0.0;
            var changeMax = channel.TrendChanges.Count > 0 ? channel.TrendChanges.Max : 0.0;

            var proposal = new FloorProposal(
                channel.Magnitudes.Count,
                channel.Magnitudes.Quantile(0.5),
                channel.PeerGaps.Quantile(0.99),
                gapMax,
                channel.TrendChanges.Quantile(0.99),
                changeMax,
                gapMax * Margin,
                changeMax * Margin);

            channel.Cached = proposal;

            return proposal;
        }

        /// <summary>Custom channel names observed so far, so a report can enumerate them.</summary>
        public IReadOnlyCollection<string> CustomChannels => _customChannels.Keys;

        private FloorProposal[] Compute()
        {
            var proposals = new FloorProposal[(int)MetricIndex.Count];

            for (var m = 0; m < proposals.Length; m++)
            {
                var gaps = _peerGaps[m];
                var changes = _trendChanges[m];
                var magnitudes = _magnitudes[m];

                var gapMax = gaps.Count > 0 ? gaps.Max : 0.0;
                var changeMax = changes.Count > 0 ? changes.Max : 0.0;

                // Counted events are observed and reported like everything else, and proposed for by nobody.
                // The observations are still worth reading — how often peers differ by a restart is a real
                // fact about the cluster — but turning that fact into a floor would set the bar above a single
                // restart, which is the event the signal exists to report. See PeerSignalCatalog.
                var fittable = !PeerSignalCatalog.IsCountedEvent((MetricIndex)m);

                proposals[m] = new FloorProposal(
                    magnitudes.Count,
                    magnitudes.Quantile(0.5),
                    gaps.Quantile(0.99),
                    gapMax,
                    changes.Quantile(0.99),
                    changeMax,
                    fittable ? gapMax * Margin : 0.0,
                    fittable ? changeMax * Margin : 0.0);
            }

            return proposals;
        }

        private static double MedianOfOthers(double[] medians, int skip)
        {
            var others = new List<double>(medians.Length - 1);

            for (var i = 0; i < medians.Length; i++)
            {
                if (i != skip && double.IsFinite(medians[i]))
                {
                    others.Add(medians[i]);
                }
            }

            if (others.Count == 0)
            {
                return double.NaN;
            }

            others.Sort();

            return Quantile(others, 0.5);
        }

        private static double Median(ReadOnlySpan<double> values)
        {
            var finite = new List<double>(values.Length);

            for (var i = 0; i < values.Length; i++)
            {
                if (double.IsFinite(values[i]))
                {
                    finite.Add(values[i]);
                }
            }

            if (finite.Count == 0)
            {
                return double.NaN;
            }

            finite.Sort();

            return Quantile(finite, 0.5);
        }

        /// <summary>Nearest-rank quantile of an already sorted list; zero when there is nothing to rank.</summary>
        private static double Quantile(List<double> sorted, double q)
        {
            if (sorted.Count == 0)
            {
                return 0.0;
            }

            var index = (int)(q * (sorted.Count - 1));

            return sorted[Math.Clamp(index, 0, sorted.Count - 1)];
        }

        /// <summary>
        /// The three accumulators plus a cached proposal, for one named channel. A class rather than a struct
        /// because it is mutated in place through a dictionary lookup, and a struct would be updating a copy.
        /// </summary>
        private sealed class CustomChannel
        {
            public BoundedSamples PeerGaps { get; init; } = new();

            public BoundedSamples TrendChanges { get; init; } = new();

            public BoundedSamples Magnitudes { get; init; } = new();

            public FloorProposal? Cached { get; set; }
        }
    }
}
