// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;

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
    /// <para><b>Matching is by primary subject.</b> A group continues an open incident when it is centred on
    /// the same pod. Subject overlap still ranks the candidates when more than one qualifies, but it does not
    /// veto: a group that shares a centre with an open incident continues it however much of the periphery
    /// came and went.</para>
    ///
    /// <para><b>The veto was there and it was wrong, and the difference matters more than the threshold.</b>
    /// The rule used to be "same primary <i>and</i> at least a third of the subjects shared", which cost a
    /// shadow run an incident at an overlap of <b>0.33 against a bar of 0.34</b> — the same pod, the same
    /// fault, one hundredth short. The fix is not a lower bar. The bar was measuring the wrong thing: an
    /// incident's periphery is the findings the grouper attached to it this cycle, and those rotate by
    /// design — that is the very property that killed the (subject, signal) key below. Once the centre is
    /// required to be the same pod, a shrinking group is one incident getting better, not a different
    /// incident.</para>
    ///
    /// <para>Removing it costs nothing the primary requirement was not already paying for. The immortality
    /// bug it was introduced alongside — a group about the healthy replicas inheriting the identity of one
    /// about the degraded replica at 0.75 overlap — is blocked by the primary key, which those two groups do
    /// not share. What remains is the cost already stated on <see cref="IncidentTrackingOptions"/>: two
    /// unrelated problems on one pod are one incident, which is the same policy the grouper applies inside a
    /// single cycle.</para>
    ///
    /// <para><b>Subjects rather than (subject, signal) pairs, because the pair version was measured and it
    /// failed.</b> A real incident gains and loses findings constantly, and once a large one clears what
    /// remains is one to three — at which size a single signal rotating out drops pair-overlap below any
    /// usable threshold. On a shadow run of one fault introduced and removed, pairs opened <b>eight</b>
    /// incidents instead of two, alternating opened/ongoing every other cycle, every one of them the same pod
    /// with a different signal. An incident is about who is in trouble; signals are evidence, and evidence
    /// rotates.</para>
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
    /// <para><b>State survives a restart when the caller gives it somewhere to live.</b>
    /// <see cref="Snapshot()"/> and <see cref="Restore"/> carry the open incidents across; without them a
    /// rolling update of the guard reopens everything that was running, at the worst possible moment —
    /// while somebody is already looking at a change.</para>
    ///
    /// <para>Not thread-safe. One instance per monitored scope, driven by one loop.</para>
    /// </summary>
    public sealed class IncidentTracker
    {
        /// <summary>
        /// Ceiling on how many findings one incident contributes subjects from. A group larger than this is
        /// already beyond what a human can act on, and the comparison is quadratic in keys.
        /// </summary>
        public const int MaxKeysPerIncident = 256;

        /// <summary>
        /// Overlap above which a trace calls a non-match <see cref="IncidentMatchOutcome.PrimaryChanged"/>
        /// rather than <see cref="IncidentMatchOutcome.NoResemblance"/>.
        ///
        /// <para><b>A label on a diagnostic, not a threshold in the matcher.</b> Nothing behaves differently
        /// on either side of it; it exists so a reader of a shadow run can tell "the incident's centre moved"
        /// from "this is a different problem". It is deliberately a constant rather than an option — a knob
        /// that changes only how a trace is worded would invite someone to tune it and expect a behaviour
        /// change.</para>
        /// </summary>
        private const double SubstantialOverlap = 0.34;

        private readonly IncidentTrackingOptions _options;
        private readonly List<Tracked> _open = [];
        private readonly HashSet<string> _left = new(StringComparer.Ordinal);
        private string _leftPrimary = string.Empty;
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
        /// The open incidents, flattened for storage. Identity and enough to close honestly — not the
        /// findings, which the consumer has already seen and which would make the state grow with the
        /// noisiest cycle.
        /// </summary>
        public IReadOnlyList<PersistedIncident> Snapshot()
        {
            var state = new List<PersistedIncident>(_open.Count);

            for (var i = 0; i < _open.Count; i++)
            {
                var tracked = _open[i];
                var primary = tracked.Incident.Primary;
                var subject = primary.Subject;
                var keys = new string[tracked.Keys.Count];
                var k = 0;

                foreach (var key in tracked.Keys)
                {
                    keys[k] = key;
                    k++;
                }

                state.Add(new PersistedIncident(
                    tracked.Id,
                    tracked.FirstSeen,
                    tracked.LastSeen,
                    tracked.CyclesSeen,
                    tracked.CyclesMissing,
                    tracked.PrimaryKey,
                    keys,
                    subject.Namespace,
                    subject.Workload,
                    subject.ReplicaSet,
                    subject.Pod,
                    subject.Node,
                    primary.Signal,
                    primary.Class,
                    tracked.Incident.PeakSeverity,
                    tracked.Incident.Start,
                    tracked.Incident.End,
                    tracked.Incident.AffectedSubjects,
                    tracked.Incident.DistinctSignals,
                    tracked.Incident.Summary,
                    primary.Novelty));
            }

            return state;
        }

        /// <summary>The identifier the next new incident will take, so numbering does not restart.</summary>
        public long NextId => _nextId;

        /// <summary>
        /// Saved incidents that were fresh enough to adopt and were refused because
        /// <c>MaxOpenIncidents</c> was already reached. Zero when the whole payload fitted; non-zero means
        /// the caller is running with less identity than it saved.
        ///
        /// <para>Counts only capacity refusals. Records dropped for age are not truncation — they were
        /// deliberately let go, and folding the two together would report a working staleness bound as a
        /// sizing problem.</para>
        /// </summary>
        public int Truncated
        {
            get;
            private set;
        }

        /// <summary>
        /// Replaces the running state with a saved one.
        /// </summary>
        /// <param name="incidents">What <see cref="Snapshot()"/> produced.</param>
        /// <param name="nextId">What <see cref="NextId"/> was. Reusing identifiers would let a consumer join
        /// a new incident to a closed one's history.</param>
        /// <param name="now">Current time, for the staleness bound.</param>
        /// <param name="maxAge">
        /// How old a saved incident may be and still be adopted.
        ///
        /// <para><b>A bound is required, not optional.</b> A guard restarted after a week would otherwise
        /// resurrect week-old incidents and immediately close them, producing a burst of resolutions for
        /// problems nobody remembers — the same alert storm the tracker exists to prevent, wearing the
        /// opposite sign.</para>
        /// </param>
        /// <returns>How many were adopted.</returns>
        public int Restore(
            IReadOnlyList<PersistedIncident> incidents,
            long nextId,
            DateTimeOffset now,
            TimeSpan maxAge)
        {
            ArgumentNullException.ThrowIfNull(incidents);

            _open.Clear();
            _nextId = nextId < 1 ? 1 : nextId;

            var adopted = 0;
            var refused = 0;

            // Every saved record is visited, and that is the whole point of the loop's shape. It used to stop
            // as soon as MaxOpenIncidents was reached, so a payload larger than the cap left the highest
            // saved identifiers unseen — and the next new incident took an identifier a persisted one already
            // held. A consumer keyed on the identifier then joins two unrelated incidents into one history,
            // which is the failure the counter exists to prevent, reintroduced by an early exit that looks
            // like an optimisation. The bound is the saved list, which the guard itself wrote.
            for (var i = 0; i < incidents.Count; i++)
            {
                var saved = incidents[i];

                // Ahead of every filter below: a record dropped for age, or refused for capacity, has still
                // spent its identifier and must not have it handed out again.
                if (saved.Id >= _nextId)
                {
                    _nextId = saved.Id + 1;
                }

                if (now - saved.LastSeen > maxAge)
                {
                    continue;
                }

                if (_open.Count >= _options.MaxOpenIncidents)
                {
                    refused++;

                    continue;
                }

                var tracked = new Tracked
                {
                    Id = saved.Id,
                    Incident = Rebuild(saved),
                    FirstSeen = saved.FirstSeen,
                    LastSeen = saved.LastSeen,
                    CyclesSeen = saved.CyclesSeen,
                    CyclesMissing = saved.CyclesMissing,
                    PrimaryKey = saved.PrimaryKey,
                };

                for (var k = 0; k < saved.SubjectKeys.Count; k++)
                {
                    tracked.Keys.Add(saved.SubjectKeys[k]);
                }

                _open.Add(tracked);
                adopted++;
            }

            // Reported rather than left to look like "there were only that many". A caller logging
            // "adopted 5" cannot otherwise tell five saved from fifty saved and forty-five dropped.
            Truncated = refused;

            return adopted;
        }

        /// <summary>
        /// A restored incident, carrying its primary finding only.
        ///
        /// <para>The individual findings are gone with the process that gathered them, and reconstructing
        /// one from persisted fields is not the same as inventing it: every value here was written by the
        /// detector that produced the original. What a restored incident cannot do is list evidence it no
        /// longer holds, and it does not pretend to.</para>
        /// </summary>
        private static Incident Rebuild(PersistedIncident saved)
        {
            var primary = new SignalFinding(
                new IncidentSubject(
                    saved.Namespace, saved.Workload, saved.ReplicaSet, saved.Pod, saved.Node),
                saved.Signal,
                saved.Class,
                saved.Start,
                saved.End,
                saved.Severity,
                saved.Summary)
            {
                Novelty = saved.Novelty,
            };

            return new Incident(
                [primary],
                primary,
                saved.Start,
                saved.End,
                saved.Subjects,
                saved.Signals,
                saved.Summary);
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
        /// <param name="trace">
        /// Optional per-group explanation of the matching decision. Supplied only by a diagnostic: the
        /// ordinary path never computes the overlap of a candidate whose primary differs, and this makes it
        /// do so, which is the number that separates "the incident's centre moved" from "this is a different
        /// group". Null costs nothing.
        /// </param>
        public IReadOnlyList<TrackedIncident> Observe(
            IReadOnlyList<Incident> incidents,
            DateTimeOffset observedAt,
            Action<IncidentMatchTrace>? trace = null)
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
                _leftPrimary = SubjectKey(incident.Primary.Subject);

                var best = -1;
                var bestOverlap = 0.0;

                // Trace-only: the best overlap ignoring the primary requirement. The ordinary loop skips
                // those candidates entirely, so this is the one thing an outside observer cannot reconstruct.
                var anyBest = -1;
                var anyOverlap = 0.0;

                for (var o = 0; o < _open.Count; o++)
                {
                    if (trace != null && !_open[o].MatchedThisCycle)
                    {
                        var unconstrained = Overlap(_left, _open[o].Keys);

                        if (unconstrained > anyOverlap)
                        {
                            anyOverlap = unconstrained;
                            anyBest = o;
                        }
                    }

                    if (_open[o].MatchedThisCycle)
                    {
                        continue;
                    }

                    // The primary subject must be the same thing, not merely present in both groups.
                    // Without this the grouper's breadth defeats the matching: it merges every pod's
                    // findings into one incident, so a group about the degraded replica and a group about
                    // the surviving healthy ones share three subjects out of four and match at 0.75. The
                    // incident then never resolves — it silently changes what it is about while keeping its
                    // identity, which is worse than opening a new one. Measured on the recorded lab window.
                    if (!string.Equals(_leftPrimary, _open[o].PrimaryKey, StringComparison.Ordinal))
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

                // A matching primary subject is enough. There is deliberately no second gate on how much of
                // the periphery survived — see the type doc for the run that settled it.
                if (best >= 0)
                {
                    var tracked = _open[best];

                    trace?.Invoke(new IncidentMatchTrace(
                        _leftPrimary, _left.Count, IncidentMatchOutcome.Continued, tracked.Id,
                        anyOverlap, anyBest >= 0 ? _open[anyBest].Id : 0,
                        anyBest >= 0 ? _open[anyBest].PrimaryKey : string.Empty));

                    tracked.MatchedThisCycle = true;
                    tracked.Incident = incident;
                    tracked.LastSeen = observedAt;
                    tracked.CyclesSeen++;
                    tracked.CyclesMissing = 0;
                    tracked.PrimaryKey = _leftPrimary;
                    CopyKeys(_left, tracked.Keys);

                    results.Add(Snapshot(tracked, IncidentState.Ongoing));

                    continue;
                }

                if (trace != null)
                {
                    trace(new IncidentMatchTrace(
                        _leftPrimary,
                        _left.Count,
                        Explain(anyOverlap),
                        0,
                        anyOverlap,
                        anyBest >= 0 ? _open[anyBest].Id : 0,
                        anyBest >= 0 ? _open[anyBest].PrimaryKey : string.Empty));
                }

                results.Add(Open(incident, observedAt));
            }

            CloseAbsent(results);
            Evict();

            return results;
        }

        /// <summary>
        /// Which of the two very different reasons a group opened instead of continuing.
        ///
        /// <para>A substantial overlap with a different primary says the incident is the same and its centre
        /// moved — a matching problem. A low overlap everywhere says the group genuinely changed.</para>
        ///
        /// <para>Reached only when nothing with a matching primary was open, since that is now the whole test.
        /// <see cref="SubstantialOverlap"/> labels a trace and decides nothing.</para>
        /// </summary>
        private IncidentMatchOutcome Explain(double anyOverlap)
        {
            if (_open.Count == 0)
            {
                return IncidentMatchOutcome.NoOpenIncidents;
            }

            if (anyOverlap >= SubstantialOverlap)
            {
                return IncidentMatchOutcome.PrimaryChanged;
            }

            return IncidentMatchOutcome.NoResemblance;
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
            tracked.PrimaryKey = _leftPrimary;
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
        /// The identity of a group: the set of subjects it is about.
        ///
        /// <para>Subject is the pod where there is one and the workload otherwise, so a common-mode finding —
        /// which names no pod on purpose — is keyed to the deployment rather than colliding with every
        /// pod-less finding in the namespace, and a deployment-wide movement therefore keeps its own identity
        /// separate from any individual replica's.</para>
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

                destination.Add(SubjectKey(subject));
            }
        }

        /// <summary>
        /// A subject's identity for matching: the pod where there is one, the workload otherwise. A
        /// common-mode finding names no pod on purpose, so it keys to the deployment rather than colliding
        /// with every other pod-less finding in the namespace.
        /// </summary>
        private static string SubjectKey(IncidentSubject subject)
        {
            var who = subject.Pod.Length > 0 ? subject.Pod : subject.Workload;

            return string.Concat(subject.Namespace, "/", who);
        }

        private static void CopyKeys(HashSet<string> source, HashSet<string> destination)
        {
            destination.Clear();

            foreach (var key in source)
            {
                destination.Add(key);
            }
        }

        /// <summary>Jaccard similarity: shared subjects over total distinct subjects.</summary>
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

            /// <summary>Whose problem this is, as of the last cycle that observed it.</summary>
            public string PrimaryKey
            {
                get; set;
            } = string.Empty;

            public HashSet<string> Keys { get; } = new(StringComparer.Ordinal);
        }
    }
}
