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
    /// What operators have said about incidents, persisted beside the baseline and the calibration.
    ///
    /// <para><b>It lives in the learned state because it is learned state.</b> A label is exactly the input
    /// <see cref="FloorCalibrator"/> never had: the calibrator's whole premise is that the period it observed
    /// was healthy, and until now that premise was an assumption nobody could correct. It shares the file's
    /// lifetime, its wipe and its restore, which is what a caller expects — wiping the calibration and
    /// keeping the labels would leave a guard constrained by evidence about numbers it no longer holds.</para>
    ///
    /// <para><b>The floor constraint is the whole point of the class.</b> <see cref="SmallestRealMagnitude"/>
    /// answers "what is the smallest finding on this signal an operator has confirmed", and a proposed floor
    /// at or above that number would have silenced it. Nothing else here is load-bearing.</para>
    ///
    /// <para>Not thread-safe. One instance per guard, like everything else in this directory.</para>
    /// </summary>
    public sealed class OperatorLabelStore
    {
        /// <summary>
        /// Hard ceiling on retained labels. Bounded for the same reason the sample buffers are: a process
        /// meant to run for months cannot hold a list that only grows. The oldest go first, and a
        /// <see cref="OperatorLabelKind.Real"/> label outranks a noise one when both are candidates —
        /// forgetting the constraint is worse than forgetting the dismissal.
        /// </summary>
        public const int MaxLabels = 2000;

        private readonly List<OperatorLabel> _labels = [];

        /// <summary>How many labels are retained.</summary>
        public int Count => _labels.Count;

        /// <summary>The labels, oldest first.</summary>
        public IReadOnlyList<OperatorLabel> Labels => _labels;

        /// <summary>Records a judgement.</summary>
        public void Add(in OperatorLabel label)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(label.Signal);

            _labels.Add(label);

            if (_labels.Count <= MaxLabels)
            {
                return;
            }

            Evict();
        }

        /// <summary>
        /// The smallest confirmed-real magnitude for <paramref name="signal"/>, or
        /// <see cref="double.PositiveInfinity"/> when no operator has confirmed one.
        ///
        /// <para>Infinity rather than NaN or zero deliberately: a caller comparing a proposed floor against it
        /// needs "no constraint" to mean "nothing stops you", and both of the other two would read as a
        /// constraint of zero and silence the calibrator entirely.</para>
        /// </summary>
        public double SmallestRealMagnitude(string signal)
        {
            ArgumentNullException.ThrowIfNull(signal);

            var smallest = double.PositiveInfinity;

            for (var i = 0; i < _labels.Count; i++)
            {
                var label = _labels[i];

                if (!label.ConstrainsFloors
                    || !string.Equals(label.Signal, signal, StringComparison.Ordinal))
                {
                    continue;
                }

                if (label.Magnitude < smallest)
                {
                    smallest = label.Magnitude;
                }
            }

            return smallest;
        }

        /// <summary>Serialises every label, one per line.</summary>
        public string Write()
        {
            var text = new StringBuilder();

            for (var i = 0; i < _labels.Count; i++)
            {
                var label = _labels[i];

                text.Append(label.IncidentId).Append('\t')
                    .Append((int)label.Kind).Append('\t')
                    .Append(label.Magnitude.ToString("R", CultureInfo.InvariantCulture)).Append('\t')
                    .Append(label.At.ToUnixTimeSeconds()).Append('\t')
                    .Append(Escape(label.Signal)).Append('\t')
                    .Append(Escape(label.Reason)).Append('\n');
            }

            return text.ToString();
        }

        /// <summary>
        /// Restores a store. An unreadable line is skipped rather than fatal, on the same reasoning as the
        /// calibrator: a guard that refuses to start because one line of its own scratch file is malformed
        /// has turned a soft degradation into an outage.
        /// </summary>
        public static OperatorLabelStore Read(string? state)
        {
            var store = new OperatorLabelStore();

            if (string.IsNullOrWhiteSpace(state))
            {
                return store;
            }

            var lines = state.Split('\n', StringSplitOptions.RemoveEmptyEntries);

            for (var i = 0; i < lines.Length; i++)
            {
                var parts = lines[i].Split('\t');

                if (parts.Length < 6
                    || !long.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out var id)
                    || !int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var kind)
                    || !double.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out var mag)
                    || !long.TryParse(parts[3], NumberStyles.Integer, CultureInfo.InvariantCulture, out var at)
                    || parts[4].Length == 0)
                {
                    continue;
                }

                store._labels.Add(new OperatorLabel(
                    id,
                    Unescape(parts[4]),
                    kind == (int)OperatorLabelKind.Real ? OperatorLabelKind.Real : OperatorLabelKind.Noise,
                    mag,
                    DateTimeOffset.FromUnixTimeSeconds(at),
                    Unescape(parts[5])));
            }

            return store;
        }

        /// <summary>
        /// Drops the label that is safest to drop: the oldest noise one if any exists, and only then the
        /// <b>largest</b> real one.
        ///
        /// <para>A dismissal that is forgotten costs a repeated alert; a constraint that is forgotten costs
        /// the guard's ability to see something an operator confirmed matters. That much was already
        /// right.</para>
        ///
        /// <para><b>What was wrong is which real label goes.</b> It dropped the oldest, and age is not what
        /// makes a real label useful — every one of them caps future floor proposals through
        /// <c>SmallestRealMagnitude</c>, so <b>only the smallest is doing any work</b>. Dropping by age
        /// removes the binding constraint roughly one time in N, and the symptom is a floor drifting
        /// upward past something an operator has explicitly confirmed is real: the guard going quiet about
        /// exactly the thing it was told to keep reporting. Dropping the largest is free — it was already
        /// dominated by a smaller one.</para>
        /// </summary>
        private void Evict()
        {
            for (var i = 0; i < _labels.Count; i++)
            {
                if (_labels[i].Kind == OperatorLabelKind.Noise)
                {
                    _labels.RemoveAt(i);

                    return;
                }
            }

            // All real. Drop the one that constrains nothing: the largest magnitude, since the cap is a
            // minimum over them. A non-finite magnitude sorts as largest, because it constrains nothing
            // either — SmallestRealMagnitude skips it.
            var victim = 0;

            for (var i = 1; i < _labels.Count; i++)
            {
                var current = _labels[i].Magnitude;
                var best = _labels[victim].Magnitude;

                if (!double.IsFinite(best))
                {
                    continue;
                }

                if (!double.IsFinite(current) || current > best)
                {
                    victim = i;
                }
            }

            _labels.RemoveAt(victim);
        }

        /// <summary>Tabs and newlines are the record separators, so they cannot survive inside a field.</summary>
        private static string Escape(string value)
        {
            return value.Replace("\\", "\\\\", StringComparison.Ordinal)
                        .Replace("\t", "\\t", StringComparison.Ordinal)
                        .Replace("\n", "\\n", StringComparison.Ordinal);
        }

        private static string Unescape(string value)
        {
            return value.Replace("\\n", "\n", StringComparison.Ordinal)
                        .Replace("\\t", "\t", StringComparison.Ordinal)
                        .Replace("\\\\", "\\", StringComparison.Ordinal);
        }
    }
}
