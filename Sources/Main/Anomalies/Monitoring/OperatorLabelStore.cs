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
        /// Drops the oldest label that is safe to drop: a noise one if any exists, and only then the oldest
        /// real one. A dismissal that is forgotten costs a repeated alert; a constraint that is forgotten
        /// costs the guard's ability to see something an operator confirmed matters.
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

            _labels.RemoveAt(0);
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
