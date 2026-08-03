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
    /// The suppressions an operator has opened, persisted with the rest of the learned state.
    ///
    /// <para><b>Expiry is enforced here rather than trusted.</b> <see cref="Prune"/> drops what has lapsed and
    /// is called every cycle, so a store that is never pruned still answers correctly —
    /// <see cref="IsSuppressed"/> checks the clock itself. Two mechanisms for one rule, because the rule is
    /// the entire safety argument for letting an operator silence anything at all.</para>
    ///
    /// <para><b>A maximum, and it fails loud.</b> Beyond <see cref="MaxSuppressions"/> the oldest-expiring
    /// entry is dropped: a store that grows without bound is a guard being switched off one line at a time,
    /// and the entry closest to lapsing is the one whose loss costs least.</para>
    /// </summary>
    public sealed class SuppressionStore : ISignalSuppressor
    {
        /// <summary>How many suppressions may be held at once.</summary>
        public const int MaxSuppressions = 500;

        private readonly List<SignalSuppression> _suppressions = [];

        /// <summary>How many are held, expired or not. Use <see cref="ActiveCount"/> for what is muting.</summary>
        public int Count => _suppressions.Count;

        /// <summary>The suppressions, for a listing an operator can read.</summary>
        public IReadOnlyList<SignalSuppression> Suppressions => _suppressions;

        /// <summary>How many are muting something right now.</summary>
        public int ActiveCount(DateTimeOffset at)
        {
            var active = 0;

            for (var i = 0; i < _suppressions.Count; i++)
            {
                active += _suppressions[i].IsActive(at) ? 1 : 0;
            }

            return active;
        }

        /// <summary>Opens a suppression. An expiry already in the past is refused rather than stored dead.</summary>
        public void Add(in SignalSuppression suppression, DateTimeOffset now)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(suppression.Signal);

            if (!suppression.IsActive(now))
            {
                throw new ArgumentException(
                    $"The suppression expires at {suppression.Until:u}, which has already passed. Storing it "
                    + "would leave an entry in the listing that mutes nothing, and an operator reading that "
                    + "listing would believe they had silenced something.",
                    nameof(suppression));
            }

            _suppressions.Add(suppression);

            if (_suppressions.Count <= MaxSuppressions)
            {
                return;
            }

            var soonest = 0;

            for (var i = 1; i < _suppressions.Count; i++)
            {
                if (_suppressions[i].Until < _suppressions[soonest].Until)
                {
                    soonest = i;
                }
            }

            _suppressions.RemoveAt(soonest);
        }

        /// <inheritdoc/>
        public bool IsSuppressed(
            in IncidentSubject subject, string signal, DateTimeOffset at, double magnitude = double.NaN)
        {
            ArgumentNullException.ThrowIfNull(signal);

            for (var i = 0; i < _suppressions.Count; i++)
            {
                var suppression = _suppressions[i];

                if (suppression.IsActive(at) && suppression.Covers(subject, signal, magnitude))
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>Drops what has lapsed. Returns how many went.</summary>
        public int Prune(DateTimeOffset at)
        {
            var removed = 0;

            for (var i = _suppressions.Count - 1; i >= 0; i--)
            {
                if (_suppressions[i].IsActive(at))
                {
                    continue;
                }

                _suppressions.RemoveAt(i);
                removed++;
            }

            return removed;
        }

        /// <summary>Serialises every suppression, one per line.</summary>
        public string Write()
        {
            var text = new StringBuilder();

            for (var i = 0; i < _suppressions.Count; i++)
            {
                var s = _suppressions[i];

                text.Append(s.Until.ToUnixTimeSeconds()).Append('\t')
                    .Append(s.IncidentId).Append('\t')
                    .Append(s.Magnitude.ToString("R", CultureInfo.InvariantCulture)).Append('\t')
                    .Append(Escape(s.Pod)).Append('\t')
                    .Append(Escape(s.Workload)).Append('\t')
                    .Append(Escape(s.Signal)).Append('\t')
                    .Append(Escape(s.Reason)).Append('\n');
            }

            return text.ToString();
        }

        /// <summary>Restores a store, skipping any line it cannot read.</summary>
        public static SuppressionStore Read(string? state)
        {
            var store = new SuppressionStore();

            if (string.IsNullOrWhiteSpace(state))
            {
                return store;
            }

            var lines = state.Split('\n', StringSplitOptions.RemoveEmptyEntries);

            for (var i = 0; i < lines.Length; i++)
            {
                var parts = lines[i].Split('\t');

                if (parts.Length < 7
                    || !long.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture,
                                      out var until)
                    || !long.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var id)
                    || !double.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture,
                                        out var magnitude)
                    || parts[5].Length == 0)
                {
                    continue;
                }

                store._suppressions.Add(new SignalSuppression(
                    Unescape(parts[3]),
                    Unescape(parts[4]),
                    Unescape(parts[5]),
                    DateTimeOffset.FromUnixTimeSeconds(until),
                    id,
                    Unescape(parts[6]),
                    magnitude));
            }

            return store;
        }

        /// <inheritdoc cref="LearnedStateText.Escape"/>
        private static string Escape(string value) => LearnedStateText.Escape(value);

        /// <inheritdoc cref="LearnedStateText.Unescape"/>
        private static string Unescape(string value) => LearnedStateText.Unescape(value);
    }
}
