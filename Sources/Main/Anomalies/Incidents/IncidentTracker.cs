// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Incidents.Contracts;

namespace DevOnBike.Overfit.Anomalies.Incidents
{
    /// <summary>
    /// Gives an incident an identity that outlives the cycle that found it, so a problem is reported once and
    /// then updated rather than rediscovered every few minutes.
    ///
    /// <para><b>Without this the guard is not deployable, whatever its false-positive rate.</b>
    /// <see cref="IncidentPipeline"/> is stateless by design — it groups one window and forgets. Evaluated
    /// every five minutes, a problem lasting an hour produces twelve unrelated incidents that each look new.
    /// No threshold fixes that, because every one of the twelve is correct.</para>
    ///
    /// <para><b>Matching is by overlap of (subject, signal) pairs, not by equality.</b> A real incident gains
    /// and loses findings constantly: a symptom crosses its threshold, a second pod joins, a marginal signal
    /// drops out. Demanding an identical group would open a new incident on each of those, which is the
    /// behaviour being removed. So a group continues a previous one when it shares enough of its pairs —
    /// intersection over union, against <see cref="IncidentTrackingOptions.MinOverlap"/>.</para>
    ///
    /// <para><b>Closing waits.</b> A finding sitting on its threshold flickers, and resolving on the first
    /// missed cycle converts that flicker into resolve/open/resolve/open — the same storm wearing a different
    /// hat. An incident closes after <see cref="IncidentTrackingOptions.ResolveAfterMissingCycles"/>
    /// consecutive absences, and the wait costs only a late close.</para>
    ///
    /// <para><b>What this deliberately does not model: splits and merges.</b> If one incident becomes two,
    /// the better-overlapping half continues it and the other opens as new; if two become one, it continues
    /// whichever it matches best and the other resolves. Both are defensible, neither is right in general,
    /// and inventing a lattice for a situation nobody has yet watched on real data would be guessing. The
    /// counters on <see cref="TrackedIncident"/> make the choice visible when it happens.</para>
    ///
    /// <para><b>Nothing here persists.</b> Identity is unique within one instance's lifetime; a restart
    /// starts again and every open incident reopens under a new id. Fixing that means durable state, which is
    /// a decision about the deployment rather than about detection.</para>
    ///
    /// <para>Not thread-safe. One instance per monitored scope, driven by one loop.</para>
    /// </summary>
    public sealed class IncidentTracker
    {
        /// <summary>
        /// Ceiling on how many (subject, signal) pairs one incident contributes to matching. A group larger
        /// than this is already beyond what a human can act on, and the comparison is quadratic in pairs.
        /// </summary>
        public const int MaxKeysPerIncident = 256;

        private readonly IncidentTrackingOptions _options;
        private readonly List<Tracked> _open = [];
        private readonly HashSet<string> _left = new(StringComparer.Ordinal);
        private long _nextId = 1;

        public IncidentTracker(IncidentTrackingOptions options)
        {
            if (!options.IsValid)
            {
                throw new ArgumentException(
                    "Tracking options are not usable — use IncidentTrackingOptions.Balanced/Sticky/Strict "
                    + "rather than default.",
                    nameof(options));
            }

            _options = options;
        }

        /// <summary>Incidents currently open, including any inside their grace period.</summary>
        public int OpenCount => _open.Count;

        /// <summary>Forgets everything, so the instance can serve a different scope.</summary>
        public void Clear()
        {
            _open.Clear();
        }

        /// <summary>
        /// Folds one cycle's grouping into the running state and returns what changed.
        ///
        /// <para>The result carries every incident observed this cycle — <see cref="IncidentState.Opened"/>
        /// or <see cref="IncidentState.Ongoing"/> — plus one <see cref="IncidentState.Resolved"/> row for
        /// each incident that has now been absent long enough to close. Incidents merely inside their grace
        /// period are <b>not</b> returned: they have not changed state, and reporting them would put an
        /// incident in the output of a cycle that did not observe it.</para>
        /// </summary>
        /// <param name="incidents">This cycle's output from <see cref="IncidentPipeline.Group"/>.</param>
        /// <param name="observedAt">Cycle timestamp; ages and durations are measured against it.</param>
        public IReadOnlyList<TrackedIncident> Observe(
            IReadOnlyList<Incident> incidents,
            DateTimeOffset observedAt)
        {
            ArgumentNullException.ThrowIfNull(incidents);

            var results = new List<TrackedIncident>(incidents.Count + _open.Count);

            for (var i = 0; i < _open.Count; i++)
            {
                _open[i].MatchedThisCycle = false;
            }

            // Greedy, best-overlap-first. One pass over this cycle's groups, each taking the best unmatched
            // open incident above the threshold. Greedy rather than optimal because a globally optimal
            // assignment would need a matching algorithm to settle cases that the overlap threshold has
            // already declared ambiguous — precision the input does not support.
            for (var i = 0; i < incidents.Count; i++)
            {
                var incident = incidents[i];

                CollectKeys(incident, _left);

                var best = -1;
                var bestOverlap = 0.0;

                for (var o = 0; o < _open.Count; o++)
                {
                    if (_open[o].MatchedThisCycle)
                    {
                        continue;
                    }

                    var overlap = Overlap(_left, _open[o].Keys);

                    if (overlap > bestOverlap)
                    {
                        bestOverlap = overlap;
                        best = o;
                    }
                }

                if (best >= 0 && bestOverlap >= _options.MinOverlap)
                {
                    var tracked = _open[best];

                    tracked.MatchedThisCycle = true;
                    tracked.Incident = incident;
                    tracked.LastSeen = observedAt;
                    tracked.CyclesSeen++;
                    tracked.CyclesMissing = 0;
                    CopyKeys(_left, tracked.Keys);

                    results.Add(Snapshot(tracked, IncidentState.Ongoing));

                    continue;
                }

                results.Add(Open(incident, observedAt));
            }

            CloseAbsent(results);
            Evict();

            return results;
        }

        private TrackedIncident Open(Incident incident, DateTimeOffset observedAt)
        {
            var tracked = new Tracked
            {
                Id = _nextId,
                Incident = incident,
                FirstSeen = observedAt,
                LastSeen = observedAt,
                CyclesSeen = 1,
                CyclesMissing = 0,
                MatchedThisCycle = true,
            };

            _nextId++;
            CopyKeys(_left, tracked.Keys);
            _open.Add(tracked);

            return Snapshot(tracked, IncidentState.Opened);
        }

        /// <summary>
        /// Ages the unmatched, and closes whatever has been gone long enough. Walked backwards so removal
        /// does not disturb the indices still to be visited.
        /// </summary>
        private void CloseAbsent(List<TrackedIncident> results)
        {
            for (var i = _open.Count - 1; i >= 0; i--)
            {
                var tracked = _open[i];

                if (tracked.MatchedThisCycle)
                {
                    continue;
                }

                tracked.CyclesMissing++;

                if (tracked.CyclesMissing < _options.ResolveAfterMissingCycles)
                {
                    // Inside the grace period. Not reported: it has not changed state, and emitting it would
                    // place an incident in the output of a cycle that did not observe it.
                    continue;
                }

                results.Add(Snapshot(tracked, IncidentState.Resolved));

                _open.RemoveAt(i);
            }
        }

        /// <summary>
        /// Drops the least recently seen once the cap is reached. Unbounded state is a leak in a process
        /// meant to run for months, and hitting this at all means a detector upstream has stopped filtering.
        /// </summary>
        private void Evict()
        {
            // #pragma BOUND: at most _open.Count iterations, and each removes one element.
            while (_open.Count > _options.MaxOpenIncidents)
            {
                var oldest = 0;

                for (var i = 1; i < _open.Count; i++)
                {
                    if (_open[i].LastSeen < _open[oldest].LastSeen)
                    {
                        oldest = i;
                    }
                }

                _open.RemoveAt(oldest);
            }
        }

        private static TrackedIncident Snapshot(Tracked tracked, IncidentState state)
        {
            return new TrackedIncident(
                tracked.Id,
                state,
                tracked.Incident,
                tracked.FirstSeen,
                tracked.LastSeen,
                tracked.CyclesSeen,
                tracked.CyclesMissing);
        }

        /// <summary>
        /// The identity of a group, as the set of (subject, signal) pairs it covers.
        ///
        /// <para>Subject is the pod where there is one and the workload otherwise, so a common-mode finding —
        /// which names no pod on purpose — still contributes a key rather than colliding with every other
        /// pod-less finding in the namespace.</para>
        /// </summary>
        private static void CollectKeys(Incident incident, HashSet<string> destination)
        {
            destination.Clear();

            var count = incident.Findings.Count;

            if (count > MaxKeysPerIncident)
            {
                count = MaxKeysPerIncident;
            }

            for (var i = 0; i < count; i++)
            {
                var finding = incident.Findings[i];
                var subject = finding.Subject;
                var who = subject.Pod.Length > 0 ? subject.Pod : subject.Workload;

                destination.Add(string.Concat(subject.Namespace, "/", who, "|", finding.Signal));
            }
        }

        private static void CopyKeys(HashSet<string> source, HashSet<string> destination)
        {
            destination.Clear();

            foreach (var key in source)
            {
                destination.Add(key);
            }
        }

        /// <summary>Jaccard similarity: shared pairs over total distinct pairs.</summary>
        private static double Overlap(HashSet<string> left, HashSet<string> right)
        {
            if (left.Count == 0 || right.Count == 0)
            {
                return 0.0;
            }

            var shared = 0;

            foreach (var key in left)
            {
                if (right.Contains(key))
                {
                    shared++;
                }
            }

            if (shared == 0)
            {
                return 0.0;
            }

            return shared / (double)(left.Count + right.Count - shared);
        }

        /// <summary>Mutable running state for one open incident.</summary>
        private sealed class Tracked
        {
            public long Id
            {
                get; init;
            }

            public Incident Incident
            {
                get; set;
            } = null!;

            public DateTimeOffset FirstSeen
            {
                get; init;
            }

            public DateTimeOffset LastSeen
            {
                get; set;
            }

            public int CyclesSeen
            {
                get; set;
            }

            public int CyclesMissing
            {
                get; set;
            }

            public bool MatchedThisCycle
            {
                get; set;
            }

            public HashSet<string> Keys { get; } = new(StringComparer.Ordinal);
        }
    }
}
