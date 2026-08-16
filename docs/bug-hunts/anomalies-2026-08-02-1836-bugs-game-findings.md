# Bug hunt: Sources/Main/Anomalies (+ Statistics collaborators)

**Scope:** `Sources/Main/Anomalies` (Contracts, Incidents, Monitoring — with a focus on the code flagged
as written 2026-08-02), plus its immediate collaborators.
**Timestamp (UTC):** 2026-08-02-1836
**Commit:** `4059d0a` (branch `gimli`)
**Score:** 6 defects found = **12 points**
**Ended by:** the ten-minute cap, not by finishing the scope.
**README:** `Sources/Main/Anomalies/README.md` exists and was read first. Its claims (Cliff's delta
scale-freeness, the 20/60/240-minute window sweep, FloorCalibrator's 29-vs-44 false-incident result) were
treated as established and not re-litigated.

Reading-only per instructions; nothing below required running anything to confirm — each finding is a
static read of the control flow.

---

## Findings, ranked by damage

### 1. `IncidentTracker.Restore` still lets an identifier be reused, exactly what today's fix claims to prevent

**What breaks:** The loop that advances `_nextId` while restoring saved incidents stops early when
`MaxOpenIncidents` is reached, so identifiers for the incidents beyond the cap are never inspected and
never protected against reuse.

**Where:** `Sources/Main/Anomalies/Incidents/IncidentTracker.cs`, `Restore`:

```csharp
for (var i = 0; i < incidents.Count && _open.Count < _options.MaxOpenIncidents; i++)
{
    var saved = incidents[i];

    // Advanced for EVERY saved record, before the staleness filter and regardless of adoption.
    if (saved.Id >= _nextId)
    {
        _nextId = saved.Id + 1;
    }
    ...
}
```

The comment is true only for records the loop actually visits. The loop's own guard condition
(`_open.Count < _options.MaxOpenIncidents`) terminates the `for` entirely once the cap is hit, so any
saved incident after that point — including ones that are not stale and would otherwise have been
adopted — never reaches the `saved.Id >= _nextId` check. If the truncated tail contains an id at or above
the current `_nextId` (very likely, since ids grow over the run), a subsequently created incident can be
handed that same id. This is precisely the bug the surrounding comment and `OperatorLabel`'s own XML doc
say was fixed on 2026-08-02 ("That identity is only trustworthy because `IncidentTracker.Restore` stopped
recycling identifiers... before that fix an acknowledgement could have landed on an unrelated incident").
The fix closed the staleness-driven path but not the `MaxOpenIncidents`-driven one, which is the path most
likely to trigger during exactly the event this subsystem cares about — a storm large enough to overflow
`MaxOpenIncidents` on restart.

**How anyone would notice today:** Not by reading `Truncated` (it reports how many were dropped, not that
`_nextId` under-shot). The failure surfaces later and indirectly: an operator's acknowledgement, or a
suppression opened against "incident 42", lands on a different, unrelated incident that later reused id
42 — the exact silent misattribution `OperatorLabel`'s doc says is worse than no acknowledgement at all.
Nothing logs or counts it.

**What test would have caught it:** A `Restore` test with more saved incidents than `MaxOpenIncidents`,
where the *un-adopted* tail contains the highest ids, followed by opening a new incident and asserting its
id does not collide with any id in the original saved list (adopted or not).

---

### 2. `overfit_guard_state_failures_total` can never increment — the "store-error surfacing" fix from today isn't wired to anything

**What breaks:** `GuardTelemetry.StateWriteFailed()` exists, is documented as "the series because the
failure is otherwise perfectly silent", but nothing in the codebase calls it.

**Where:** `Sources/Main/Anomalies/Monitoring/GuardTelemetry.cs` (`StateWriteFailed`), cross-referenced
against `Sources/Main/Anomalies/Incidents/AnomalyGuard.cs` `RunCycleCore` (`_store?.Save(...)`,
`_historyStore.Save(...)`) and `Sources/Main/Anomalies/Incidents/Abstractions/IIncidentStore.cs`.

`FileIncidentStore.Save`/`Load` swallow every exception and record it on an implementation-specific
`LastError` property (`Sources/Main/Anomalies/Incidents/FileIncidentStore.cs`), but `IIncidentStore` — the
interface `AnomalyGuard` actually holds — exposes neither `LastError` nor a success/failure return from
`Save`/`Load`. There is therefore no code path by which `AnomalyGuard` could call
`Telemetry.StateWriteFailed()` even if it tried to, and grepping the whole `Sources/` tree confirms it
never does. The metric's own HELP text ("Incidents will not survive the next restart, and the restart is
when anyone would otherwise notice") describes exactly the failure this gap leaves undetectable.

**How anyone would notice today:** They would not. A full disk, a permissions change, or a deleted mount
point causes every `Save` to fail silently (per `FileIncidentStore`'s own documented contract — "nothing
here throws"), and `overfit_guard_state_failures_total` stays at zero the entire time. An operator who
follows the class's own advice and alerts on that series gets a false all-clear for as long as the disk
stays broken; they find out at the next restart, when every incident reopens — the exact scenario the
counter exists to give hours of warning about.

**What test would have caught it:** An `AnomalyGuard` test with a fake `IIncidentStore` whose `Save`
throws or returns failure, asserting `Telemetry.ToPrometheusText()` reports a non-zero
`overfit_guard_state_failures_total` after one cycle. As it stands no such wiring exists to test.

**Shared root cause with #1:** both are "the 2026-08-02 fix landed the mechanism but not every call site
that needed it" — `Restore`'s loop bound was overlooked, and `IIncidentStore`'s surface was never extended
to reach the telemetry that was built for it.

---

### 3. `OperatorLabelStore.Evict` protects the wrong invariant: the load-bearing minimum can be evicted for being old

**What breaks:** At `MaxLabels` capacity, `Evict` drops the oldest label, restricted to `Kind == Noise`
labels only if any exist — but among `Real` labels (once `Noise` ones are exhausted, e.g. a channel that
is mostly correctly flagged), it removes the *oldest* `Real` label regardless of magnitude. The class's own
purpose (per its header doc) is that `SmallestRealMagnitude` "answers 'what is the smallest finding on
this signal an operator has confirmed', and a proposed floor at or above that number would have silenced
it" — i.e. the *smallest-magnitude* `Real` label per signal is the one that matters, not the oldest.

**Where:** `Sources/Main/Anomalies/Monitoring/OperatorLabelStore.cs`, `Evict`:

```csharp
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
    _labels.RemoveAt(0);   // oldest overall, not smallest-magnitude
}
```

Over the lifetime of a busy deployment (2000 labels is not a large number across months of
acknowledgements on a multi-signal cluster), once `Noise` labels stop existing at the front of the queue,
age — not protective value — decides what is kept. The label carrying the smallest ever confirmed
magnitude for a given signal is exactly as likely to be evicted as any other `Real` label, silently
raising what `FloorCalibrator.Cap` computes as the ceiling for that signal (`SmallestRealMagnitude` is a
min over whatever remains). The doc's own words — "forgetting the constraint is worse than forgetting the
dismissal" — describe precisely what this does to the one label that constrained the most.

**How anyone would notice today:** Not from any counter — `Labels.Count` and `realLabels` in telemetry
stay the same shape either way; only a future floor proposal creeping above a magnitude someone once
confirmed real would reveal it, weeks or months later, with nothing pointing back at the eviction.

**What test would have caught it:** Fill the store past `MaxLabels` with `Real` labels of varying
magnitude for one signal (no `Noise` labels present), assert `SmallestRealMagnitude` after eviction still
returns the true minimum ever recorded — it currently does not, once that minimum ages to the front.

---

### 4. `FloorCalibrator` does not escape custom channel names before serialising them, unlike every other store in this module

**What breaks:** `OperatorLabelStore.Write`/`Read` and `SuppressionStore.Write`/`Read` both escape
tab/newline in every free-text field (`Escape`/`Unescape`). `FloorCalibrator.Write` does not apply the
same treatment to a custom channel's name, which is free text supplied by whoever configures
`CustomMetricBinding.Name` (validated only for non-blank, per `CustomMetricBinding.IsUsable`).

**Where:** `Sources/Main/Anomalies/Monitoring/FloorCalibrator.cs`, `Write`:

```csharp
foreach (var (name, channel) in _customChannels)
{
    text.Append(CustomMarker).Append(name).Append('\t')   // name is not escaped
        .Append(channel.PeerGaps.Write()).Append('\t')
        ...
}
```

A name containing a tab shifts every subsequent field on that line, and `Read`'s `parts.Length != 4` guard
then silently discards the whole line (`continue`), quietly resetting that channel's learned floors on the
next restore. A name containing a newline splits into two lines, both malformed, with the same silent
outcome. `LearnedState.Read` additionally locates the `### labels` / `### suppressions` section boundaries
with a plain `IndexOf` over the raw payload text — a custom channel name that happens to contain the
literal string `"### labels"` (nothing rejects it) would be found by that search *inside* the calibration
section and misplace every section boundary after it, corrupting history, labels and suppressions
together on the next `Read`. Narrower than #1/#2 but a real gap, and the exact category the other two
stores in this same directory were already hardened against today.

**How anyone would notice today:** Only by comparing calibration proposals before and after a restart for
a channel whose name happens to contain a tab/newline — nothing reports "this channel's history did not
round-trip."

**What test would have caught it:** A round-trip test on `FloorCalibrator.Write`/`Read` with a custom
channel name containing `\t` and `\n`, asserting the channel's samples survive — mirroring the existing
escape tests presumably already covering `OperatorLabelStore`/`SuppressionStore`.

---

### 5. `AnomalyGuard.FloorProposals` bypasses the `_gate` lock the class exists to provide

**What breaks:** `Acknowledge` and `ActiveSuppressions` both take `_gate` specifically so that state shared
between the cycle loop and a caller on another thread (the HTTP `/ack` and `/suppressions` handlers in
`Sources/Cli/GuardMetricsEndpoint.cs`) is never read mid-mutation. `FloorProposals` is the same shape of
call — it reads `_calibrator`, which `RunCycleCore` mutates every cycle via `_calibrator.Observe(window)`
and `Acknowledge` mutates via `_calibrator.UseLabels(Labels)` — but it is a bare property with no lock:

```csharp
public FloorProposal[] FloorProposals => _calibrator.Propose();
```

**Where:** `Sources/Main/Anomalies/Incidents/AnomalyGuard.cs`, `FloorProposals` property, vs. `Acknowledge`
/ `ActiveSuppressions` a few lines below it in the same file.

**How anyone would notice today:** Currently nothing calls `FloorProposals` from outside `Sources/Main`
(confirmed by grep across `Sources/`), so this is latent rather than active — no observable symptom exists
yet. It is included because it is a real inconsistency in a contract the class states explicitly ("A lock
around a cycle is cheap... and it keeps every store's contract intact"), and the moment a host wires a
`/floors` endpoint the way it already wired `/suppressions`, a concurrent read could race
`FloorCalibrator.Propose()`'s check-then-compute of `_cached` against `Observe`'s invalidation of the same
field from the cycle thread.

**What test would have caught it:** Not a unit test in the classic sense — this is a design-contract gap.
A concurrency stress test (two threads, one spinning `RunCycle`, one spinning `FloorProposals`) would
surface it once such a caller exists; today the honest statement is that it is unreachable, not that it is
safe.

---

### 6. `SignalSuppression.Magnitude`'s doc asserts a specific measured regression that, per the record, did not happen

**What breaks:** The XML doc on `SignalSuppression.Magnitude` presents a detailed, specific "measurement"
as the justification for the `Magnitude`/`Ceiling`/`Covers(..., magnitude)` mechanism:

> "Replaying the injected-fault panel after a simulated shadow week — 23 dismissals, each a seven-day
> mute — left `cpu 2.5x` undetected on one pod *and* on every pod at once."

This is written in the same voice and the same level of specificity as the module's genuinely-measured
claims (the README's Cliff's-delta and window-sweep numbers, `FloorCalibrator`'s 29-vs-44). Per the task
brief for this review, this particular mechanism "was added to fix a regression that later turned out not
to exist" — meaning the comment states a specific empirical event (a shadow-week replay with a concrete
dismissal count and a concrete undetected fault) as settled fact, in a codebase whose own README states
"No number in this subsystem is from the literature" and treats every claimed number as load-bearing
evidence for a design decision. A comment asserting a measurement that did not occur is worse than one
merely missing a citation — the next reader (or the calibration work described elsewhere in this module)
will treat it as fact and build on it.

**Judged on the mechanism's own merits, independent of the doc:** the `Ceiling`/`Covers(magnitude)` logic
itself is internally consistent — a suppression opened on a smaller confirmed finding does not cover a
materially larger later one, and `NaN`/non-positive magnitudes are treated as "unmeasurable, therefore
covered" consistently across `OperatorLabel`, `SignalSuppression`, and every `IncidentPipeline.Observe*`
call site. It is not dead code: `IncidentPipeline` exercises it on every muted finding. So the mechanism
is not obviously *wrong* to keep — but the comment justifying its existence is a false claim about the
project's own measurement history, which is exactly the category this module's README explicitly asks
readers to be able to trust ("no number in this subsystem is from the literature, and several contradict
it").

**Where:** `Sources/Main/Anomalies/Contracts/SignalSuppression.cs`, XML doc on the `Magnitude` parameter.

**How anyone would notice today:** Nobody would, short of someone asking (as this review's brief did)
whether the regression it describes ever happened. It reads exactly like the module's real measurements.

**What test would have caught it:** Not testable in the usual sense — this is a documentation-integrity
issue, not a behavioural one. The available control is process: any comment citing a specific measured
number in this module should point at the artifact (a benchmark run, a lab log) the way `FloorCalibrator`'s
29-vs-44 claim implicitly does, so a reviewer can verify rather than trust.

---

## Shared root causes

- **#1 and #2** are both partial fixes from today's session: the mechanism (id-reuse guard; a telemetry
  counter for state failures) was built, but a call site or an interface member needed to complete it was
  missed. Worth checking the other four "fixed today" items (workload derivation, `MaxRosterAge`, PromQL
  escaping, pipeline shedding) for the same shape of gap — this review read all four and found them
  complete, but the pattern recurring twice in six is worth a second pass by whoever owns this.
- **#3 and #4** are both "the eviction/serialisation policy protects the wrong thing" — #3 protects
  recency over magnitude-importance, #4 protects the fixed fields over the free-text ones. Both are in
  code that manages a bounded store and both are new today.

## Coverage

**Reviewed and found clean (read in full, no defect found):**
- `Sources/Main/Anomalies/README.md`
- `Sources/Main/Anomalies/Contracts/OperatorLabel.cs`, `SignalSuppression.cs` (mechanism, not the one
  comment flagged above), `ISignalSuppressor.cs`, `CustomMetricBinding.cs`, `AnomalyGuardOptions.cs`
  (`MaxRosterAge`, `WarmUpGrace` sections)
- `Sources/Main/Anomalies/Monitoring/OperatorLabelStore.cs` (Add/Write/Read/escaping — clean except #3),
  `SuppressionStore.cs`, `FloorCalibrator.cs` (clean except #4), `LearnedState.cs` (section-boundary logic
  itself is correct for well-formed input; only vulnerable via #4's unescaped names),
  `PromqlCatalog.cs` (label escaping, confirmed fixed), `PrometheusTopologySource.cs`,
  `GuardTelemetry.cs` (thread-safety of the counters themselves is correct — the gap is that one counter
  is never called, #2)
- `Sources/Main/Anomalies/Incidents/AnomalyGuard.cs` (read in full: `RunCycle`/`RunCycleCore`,
  `Acknowledge`, `ActiveSuppressions`, `IsWarmingUp`, `ResolveWorkload`, `RunSilentPods`, `RunPeer`,
  `RunTrend`, `RunCustom*` — clean except #5), `IncidentPipeline.cs` (full file — mute checks, capacity
  shedding, all clean), `IncidentTracker.cs` header doc and `Restore`/`Snapshot` region (clean except #1)
- `Sources/Cli/GuardMetricsEndpoint.cs` (`/metrics`, `/suppressions`, `/ack` handlers — correctly use the
  locked accessors)
- `Sources/Main/Anomalies/Incidents/FileIncidentStore.cs`, `Abstractions/IIncidentStore.cs`

**Not reached** (this is where the next reviewer should start):
- `Sources/Main/Anomalies/Rules/` in full (only `SustainedThresholdRule` call sites were seen, not its body)
- `Sources/Main/Anomalies/Incidents/IncidentGrouper.cs`, `IncidentReporter.cs`, `IncidentStateFormat.cs`
- `Sources/Main/Anomalies/Monitoring/MetricHistory.cs`, `PrometheusMetricSource.cs`,
  `PrometheusHistoricalSource.cs`, `SeasonalBaseline`/`CrossPeerBaseline`
- `Sources/Main/Anomalies/Gpt/`, `Baseline/`, `Neuro/`, `Adaptive/` (the learned family) — entirely unread
- `Sources/Main/Statistics/` (the rank-statistics collaborators — `TrendDetector`, `PeerGroupOutlierDetector`,
  `LevelShiftDetector`, `BoundedSamples` were exercised only through their callers, never opened directly)
- `Sources/Server.AspNet/Services/AnomalyGuardService.cs` beyond the two grep hits already shown

## What the score means

The hunt found 6 defects (12 points) against a target of 21, and it **ended by the ten-minute cap**, not
by finishing the scope. A large fraction of the module — the entire learned-family directories, the
grouping/reporting/state-format internals, and the `Statistics` package directly — was never opened. A low
remaining count here says nothing about that unopened code; it says the reviewed slice (the parts flagged
as written today, plus `AnomalyGuard`'s cycle core) had six real, independent issues in roughly six minutes
of reading, which is a denser hit rate than a clean pass would produce. The two most serious findings (#1,
#2) both undermine fixes the task description said landed today, in each case because a call site the fix
needed was missed rather than because the core mechanism was wrong — worth a second pass on the other four
claimed fixes before trusting this module's error/identity handling in production.
