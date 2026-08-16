STATUS: SIGNED
Author: overfit-architect
Date: 2026-08-10
Slug: an-d1-peer-novelty-plan

# AN-D1 — the peer family needs to learn a pod's own standing gap

## Process note

No `overfit-analyst` plan preceded this file — `docs/aiops/aiops-backlog.md` ("D1, diagnosed 2026-08-06")
already carries a measured problem statement and a target mechanism sentence ("judge a replica against its
peers after subtracting the gap it has held since it started, so that a stable difference is learned once and
only a change in that difference reports" — backlog line ~134). I am treating that as the problem statement
rather than re-deriving it. What the backlog does **not** settle is a business acceptance of the mechanism's
own trade-off — see Blocking questions.

## Architecture review (overfit-architect)

### 0. Verification note

Call-graph and implementation claims below were established with `mcp__overfit-navigator__find_callers` /
`find_references`, not `Grep`, after the correction in finding 4. Specifically:

- `find_callers(RunPeer)` — one caller, `RunCycleCore`, 77 transitive sites (tests + `AnomalyGuardService`).
- `find_callers(ObservePeerGroup)` — production callers are `AnomalyGuard.RunPeer` and
  `AnomalyGuard.RunCustomPeer`. Two, not one.
- `find_references(PeerGroupOutlierDetector.Detect)` — **three** hits in `Sources/Anomalies`, not two:
  `AnomalyGuard.cs:873` (`RunPeer`), `AnomalyGuard.cs:725` (`RunCustomPeer`), and
  `IncidentPipeline.cs:320`. The third is **not a call** — read directly, it is a `<see
  cref="PeerGroupOutlierDetector.Detect"/>` inside `ObservePeerGroup`'s own XML doc comment
  (`IncidentPipeline.cs:316-327`), which Roslyn resolves as a symbol reference. Nothing to gate there; the
  real count is two call sites, confirmed by two independent tools (`find_callers` on the callee side,
  `find_references` on the caller side) agreeing after the doc-comment hit is excluded.

### 1. Review verdict

1. **The three candidates are not independent alternatives — they nest.** Candidate 1 (suppress after N
   stable cycles) is a cruder version of candidate 3 (judge the *change* in gap): both carry the identical
   named risk ("a leak that plateaus goes silent"), and 3 adds nothing 1 doesn't already risk while being the
   mechanism the backlog's own measurement already points to. Build 3 directly.
2. **Candidate 2 (reclassify as "standing", stop reopening) needs no new incident-layer code once 3 exists.**
   `IncidentTracker.Observe`/`CloseAbsent` already resolves an incident after
   `ResolveAfterMissingCycles` consecutive cycles with no matching finding (`IncidentTracker.cs:479-503`). If
   the novelty gate simply stops forwarding a finding once a deviation is judged stable, the existing
   absence-based close does candidate 2's job for free. No `IncidentState`/`IncidentTracker` change is part
   of this design.
3. **`MetricHistory` keyed by workload does not bar a per-pod mechanism, but is the wrong type to extend.**
   Confirmed by reading it end to end (`MetricHistory.cs`): its reset boundary is a multi-day, per-workload
   seasonal one (`MaxDays = 7`, `Forget(maxAge)`), not the per-pod-restart boundary this needs. A new type is
   cheaper and clearer than overloading one whose whole doc-comment is about workload identity surviving a
   rollout while pod identity does not (`MetricHistory.cs:22-26`) — the opposite of what novelty state needs.
4. **`ObservePeerGroup` has a second caller, found by `find_callers`/`find_references`, not by the earlier
   text read** (section 0). `AnomalyGuard.RunCustomPeer` (`AnomalyGuard.cs:725`) calls it for
   `CustomMetricBinding`-declared channels — the vehicle every customer-added metric uses
   (`docs/specs/anomaly-guard-psi-cpu-channel-plan.md`'s own central design question), and on the deployed lab
   config that is five channels (`CpuPressure`, `LockContentions`, `Exceptions`, `ActiveRequests`,
   `GcCommittedBytes`). A plan scoped to `RunPeer` alone would leave every custom peer channel re-reporting a
   standing outlier forever — the exact defect surviving in the half of the system a client is most likely to
   extend. Both call sites are now in scope — see section 5's keying update.

### 2. System context

Touches: `AnomalyGuard.RunPeer` **and** `AnomalyGuard.RunCustomPeer` (`Sources/Anomalies/Incidents/AnomalyGuard.cs:822,725`
— see finding 4); a new type in `Sources/Anomalies/Monitoring/`; `AnomalyGuardOptions` (new config knobs);
`PeerDecisionTrace` (new field); `SignalFinding` (new `NoveltyKind` field, propagated through `Incident` /
`TrackedIncident` / `PersistedIncident` / `IncidentLogRecord` the same way `Class`/`Severity` already are);
`IncidentPipeline.ObservePeerGroup` (one new **optional** parameter — revised from the original review, see
section 4); `LearnedState` (one new section — see `docs/adr/0001-peer-novelty-state-persistence-format.md`).

Does **not** touch: `Sources/Main/Statistics/PeerGroupOutlierDetector` (stays pure, stateless, unchanged
public API); `DetectionStatus`; `IncidentTracker`/`IncidentState` (the reassert-as-`Standing` design, section
4, produces its periodic re-open through the tracker's existing absence-based close, unchanged).

Reused as-is: `TrendDetector` (`Sources/Main/Statistics`, public); `IPodTopology` /
`PodPlacement.CreatedAt` (already read by `AnomalyGuard` for `WarmUpGrace`, `AnomalyGuard.cs:1368-1383`).
**Not `BoundedSamples`** — proposed here originally and rejected during implementation for three reasons;
see the amendment in §5 before writing any storage.

### 3. Boundaries and responsibilities

- **Execution path**: neither inference nor training. This is the guard's per-cycle detection/orchestration
  loop (`Sources/Anomalies`, a `BackgroundService`) — a third execution path this repository already
  maintains alongside `InferenceEngine` and `ComputationGraph`, with its own allocation discipline (see below).
- **The boundary this must not cross**: `PeerGroupOutlierDetector` stays history-free. Two reasons already
  established elsewhere in the tree, not invented here: it is public API of the shipped `DevOnBike.Overfit`
  package and its entire pitch is "no history — works from the first minute after installation"
  (`PeerGroupOutlierDetector.cs:15-18`); and `Sources/Anomalies` was split out of `Main` on 2026-08-05
  specifically because "a zero-allocation CPU inference engine and a cluster anomaly detector... having
  nothing to do with each other" shared a project (`Anomalies.csproj` header). Per-pod cluster memory belongs
  on the Anomalies side of that split, full stop.
- **Where it lives**: `AnomalyGuard.RunPeer`, after `_peer.Detect` returns and before
  `pipeline.ObservePeerGroup` — exactly where `_floors.MinAbsoluteGap(metric)` is already injected into
  `options` for the same class of reason (a per-metric material-gap floor), one level down at pod
  granularity. `findings` is already a caller-owned array (`AnomalyGuard.cs:872`); the gate rewrites entries
  in place, the same pattern `PeerGroupOutlierDetector.Dominant` already uses to demote a `Deviation` to
  `None` without deleting the finding (`PeerGroupOutlierDetector.cs:424-435`).
- **Ownership/disposal**: not applicable — plain managed dictionaries and arrays, not the tensor/graph memory
  model (no `AutogradNode`, no pooled buffers on this path).
- **AOT reachability**: `Sources/Anomalies` is `IsAotCompatible=true` / `IsTrimmable=true` today and its own
  csproj header states this is a constraint kept on purpose, not a formality. The new type must obey the same
  bans as `Main` (no LINQ, no jagged arrays, one top-level type per file, no reflection) — a fixed-capacity
  value store plus `Dictionary<string, T>` keyed by pod name is the established idiom
  (`FloorCalibrator._customChannels`), so this is free, not a new risk. The store itself is the gap ring of
  §5, not `BoundedSamples`.
- **Allocation policy**: neither hot-path nor load-path. Runs once per 5-minute cycle, not per token/row;
  sibling code (`FloorCalibrator`, `MetricHistory`) allocates per cycle without concern, and this follows
  the same discipline.
- **Public surface**: `AnomalyGuardOptions` properties are public, but the assembly is `IsPackable=false` —
  not part of the `DevOnBike.Overfit` NuGet surface. It is still an operator-facing `guard.json` contract once
  documented, so knob names matter, but reversibility is medium, not Main's high.
- **Moat**: open/AGPL side, offline batch detection — no real-time/GPU claim, no concern.

### 4. The mechanism

Reuse `TrendDetector` on a **new per-cycle series**: each (pod, metric)'s `AbsoluteGap` (already computed by
`PeerGroupOutlierDetector` into `PeerOutlierFinding`, `PeerGroupOutlierDetector.cs:234-236`) becomes one
sample per cycle in a short rolling history. `TrendDetector.Detect` on that series answers exactly "is this
pod's *separation from its peers* changing" — reusing the existing significance test, materiality floor and
five-status vocabulary instead of inventing a bespoke novelty statistic. A flat gap-over-time series (the
`pj7r8` case) is `Healthy` on the change axis even while `Anomalous` on the level axis every cycle; a growing
gap (a leak widening the separation) is `Anomalous` on both.

**Revised 2026-08-10 — the client chose "accept the trade, but not silently": report once, suppress, then
reassert periodically at reduced severity, classified `Standing` rather than `New`.** Full suppression (the
original design, below the line) is superseded by this. Two per-(pod, metric) states, decided every cycle from
the change-detector:

- **`New`** — either insufficient history to judge (fail-open, unchanged from the original design) or the
  change-detector reads `Anomalous && Direction == Rising`. Forwarded every cycle, full severity, exactly as
  today.
- **`Standing`** — the change-detector reads `Healthy`, or `Anomalous && Direction == Falling` (see the
  direction-awareness paragraph below — a shrinking gap is stability, not a new event, for this purpose).
  Forwarded only once per `StandingReassertionInterval` of wall-clock time since it was last forwarded (a
  `TimeSpan`, tracked per pod/metric — see section 6 for why this is **not** expressed in cycles), at a
  reduced severity, tagged `Standing`. Suppressed every other cycle.

If the gap resumes rising, the state reverts to `New` on the next cycle that observes it — the change-detector
re-evaluates the rolling history every cycle, so there is no separate "un-suppress" step to get wrong.

**Carrying the distinction to an operator, not just the incident-open/absent behaviour.** `SignalFinding`
gains a `NoveltyKind` field (`New`/`Standing`) — a structured field, not prose in `Reason`, for the same
reason `Magnitude` is a field and not a sentence (`SignalFinding.cs:42-55`): a consumer that wants to filter
or badge on it needs to read it, not parse it. `IncidentPipeline.ObservePeerGroup` gets one new **optional**
parameter, `ReadOnlySpan<NoveltyKind> novelty = default` — empty/absent means "every entry is `New`", so
`RunCustomPeer`'s existing calls (and every test) are unaffected until a caller opts in. Inside the pipeline,
a `Standing` entry gets `NoveltyKind = Standing` and `Severity` scaled down by a fixed factor (see section 6)
rather than the caller pre-scaling it — severity is derived from `Comparison.EffectSize` **inside**
`ObservePeerGroup` today (`IncidentPipeline.cs:384`), so the reduction has to happen there too, not in
`AnomalyGuard`, or the two would need to agree on the same formula in two places.

`IncidentTracker` needs **no changes**: a `Standing` finding that stops being forwarded lets the open incident
resolve via the existing `ResolveAfterMissingCycles` absence-close; the next reassertion is a finding on the
same primary subject, which `IncidentTracker` opens as a new incident exactly as it would any other. The
`NoveltyKind = Standing` tag is what tells a consumer "this is not really new" — the incident-state machine
does not need to know the difference, matching finding 2's original point almost exactly, just with a
periodic re-open instead of silence.

The per-cycle `PeerGroupOutlierDetector` verdict itself is **unchanged** — still `Anomalous` every cycle for
`pj7r8`. Only what `NoveltyKind` and whether-to-forward-this-cycle the finding carries is gated. This
preserves `peerTrace` diagnostics and avoids redefining what `Anomalous` means (confirmed with the task owner,
see resolved question below).

**Direction-awareness, from the live measurement:** `TrendResult` already carries a `TrendDirection`
(`Rising`/`Falling`/`None`, `TrendDetector.cs:223-231`), so folding `Falling` into `Standing` above is free,
not new machinery. Without it, the −2.65 MB/3h drift the lab measured would read as "something changed" and
keep forwarding at full severity, which is backwards — a narrowing gap is the pod becoming *more* like its
peers.

<details><summary>Original design (full suppression, no reassertion) — superseded, kept for the record</summary>

Gate: in `RunPeer`, before calling `pipeline.ObservePeerGroup`, for every `findings[i]` with `IsOutlier` true,
record `AbsoluteGap` into the pod's rolling history and evaluate the change-detector. Only leave `Deviation`
set (so `ObservePeerGroup`'s existing `!findings[i].IsOutlier` check forwards it, `IncidentPipeline.cs:356`)
when the change-detector reads `Anomalous` — never when it reads `Healthy` (confirmed stable). Superseded
because it goes fully silent on a stable outlier, which the client rejected.

</details>

### 5. Bounding and lifetime

**AMENDMENT, 2026-08-10 — `BoundedSamples` rejected during implementation, ring buffer approved.** The
developer flagged this rather than working around it; verified in source before approving. `BoundedSamples`
cannot carry the per-pod gap history, for three specific reasons, not just "it didn't fit":

1. **It stores values only** (`private double[] _values`) — `TrendDetector.Detect` requires an index-aligned
   `timestampsSeconds` span, which `BoundedSamples` has nowhere to put.
2. **`Add` silently drops non-finite values**, so two parallel `BoundedSamples` instances (one for gaps, one
   for timestamps) would desynchronise the moment a sample is missing — precisely the cycle a caller most
   needs the pairing to hold.
3. **Its decimation is systematic halving spread across the whole stream** — the right semantics for "a floor
   set from a maximum, representative of all history", and the wrong semantics for "has the gap changed
   *recently*", which is what this mechanism asks every cycle.

**Replacement**: a fixed-capacity ring of `(timestampSeconds, gap)` pairs, serialised in the same
`CustomMarker`/column-guard style ADR 0001 already requires for the rest of this section's state. Two
correctness properties required of it, both direct consequences of the three reasons above: the ring's
capacity must be documented as a **wall-clock** statement (`capacity × cycle cadence` is the window the trend
sees — the `PS-3` unit trap, section 6, in a fourth place now inside the ring itself) and a non-finite gap
must drop **its timestamp atomically** with it, so the two arrays can never desynchronise the way two
`BoundedSamples` instances could.

New type (name TBD, e.g. `PeerGapHistory`), keyed by pod name like `AnomalyGuard._silent`
(`AnomalyGuard.cs:111`), one gap ring per metric per pod. Pruned the same way `_silent` already is —
against the live roster each cycle (`AnomalyGuard.cs:1229-1245`) — **and additionally** reset for a pod
whenever its `PodPlacement.CreatedAt` differs from what was last recorded for that name. This is required,
not optional: the backlog's own live evidence is that the heavy role "was reassigned to a different pod"
after a container restart *inside one generation*, i.e. same pod name, new incarnation (backlog "Note which
pod", line ~95-100) — a StatefulSet pod restarting keeps its name, so pruning by roster membership alone would
not catch it, but `CreatedAt` would. This is precisely why `_silent`'s roster-only pruning is *not* sufficient
precedent by itself and the design adds the second trigger.

Population-scaled like `_silent` (bounded by live pod count, not by a human-authored list), unlike
`FloorCalibrator._customChannels` (bounded only by however many channels an operator declares) — the two
existing per-key dictionaries in this codebase are bounded by two different mechanisms and this needs the
`_silent` one. Recommend also keeping a hard ceiling as defence in depth (mirroring `MetricHistory._maxBuckets
= 100_000`), in case a roster ever reports something malformed.

**Keying must be dual, matching `FloorCalibrator`'s own shape, because of finding 4.** `RunCustomPeer` names
its channel by string (`CustomMetricBinding`), not `MetricIndex`, so the per-pod history needs the same
gap-ring-array (indexed by `MetricIndex`) **plus** `Dictionary<string, GapRing>` (custom channel name) split
`FloorCalibrator` already carries per pod, rather than an index-only table that silently excludes every
custom channel.

**Collision, answered by an existing precedent rather than a new rule.** `FloorCalibrator` already solved "a
client's custom channel name must not be read back as a built-in one": its `CustomMarker` (`~`) prefixes every
custom-channel key in the serialised form specifically so a channel named like a `MetricIndex` member — or
like an integer, which also parses as one — cannot collide with it (`FloorCalibrator.cs:40-43`). The new
per-pod novelty type reuses the same marker convention rather than inventing a second one; two independent
collision-avoidance schemes for the same enum in the same assembly would be the defect waiting to happen.

**Persistence: custom-channel novelty state belongs in `LearnedState` too, not excluded.** `CustomMetricBinding`
is excluded only from the **learned family** (`MetricSnapshot`/`FeatureCount` — a trained model's fixed input
shape, `AnomalyGuardOptions.cs:270-274`). Peer-novelty state is not that family; it is the same category as
`FloorCalibrator`, which **already** persists custom-channel accumulators today
(`FloorCalibrator._customChannels`, confirmed by direct read of `FloorCalibrator.cs`). Excluding custom
channels here would mean every custom peer channel loses persistence specifically, silently, while its
built-in siblings keep it — a client's own channel would revert to `New` on every restart even though
`CpuUsageRatio` next to it would not. Include it, using the same `Dictionary<string, ...>` +
`CustomMarker`-style row as the rest of this section.

**Standing does not distinguish flat from shrinking.** Both `Healthy` and `Anomalous && Falling` map to the
same `Standing` state and the same reassertion cadence — no separate "improving" state. A shrinking gap that
later flattens or resumes rising is already handled by the per-cycle re-evaluation (section 4's "resumed
growth" case), and splitting `Standing` in two would add a state nobody asked for; the direction is still
available in `PeerDecisionTrace` (via `TrendResult.Direction`, already public) for a future consumer that
wants to render the distinction without it affecting the reassertion mechanics.

### 5a. Live measurement, 2026-08-10 — and what it settles about provability

Team lead measured the current live population: one pod tops the `MemoryWorkingSetBytes` ranking in 70% of
same-timestamp samples, but role changes hands **17 times in 180 transitions**, **median gap 3.27 MB against
a 9.52 MB floor** (max 26.69 MB), and drift over three hours is **negative** (−2.65 MB). Window contains an
injected CPU-starvation fault (22:16–22:26Z) and spans three ReplicaSet generations.

**This does not change the mechanism, and it does not need to** — the design already gates novelty-tracking
on `findings[i].IsOutlier` (section 4), so a pod whose gap sits under the 9.52 MB floor never reaches the
tracker at all regardless of how often the *ranking* reshuffles. The 17/180 churn is churn among pods that
mostly never clear the material gate in the first place; it is not 17 genuine reassignments of a reported
outlier. So it's evidence the ranking is noisy near the floor, not evidence against the mechanism.

**What it does settle: the live population right now cannot be relied on to produce the D1 symptom on
demand**, which matches "generation-dependent" from the original diagnosis and directly answers the question
below.

**Calibration caveat**: this window is not usable as a clean "healthy" reference for fitting the change-in-gap
floor (risk 2, section 7) — the CPU-starvation fault is inside it. Same rule as `LabWindowValidator` already
enforces elsewhere: fitting to a contaminated recording is worse than not fitting.

### 5b. What the positive fixture must look like

**Code-level (synthetic gap series, four cases, each proving one status transition):**

| case | shape | required verdict |
|---|---|---|
| flat offset | constant gap, N ≥ the (unmeasured) cycle-count floor | early cycles: `New`, forward every cycle (insufficient history — fail open); once past the floor: `Standing`, forwarded only on the `StandingReassertionInterval` cadence, reduced severity |
| growing offset | gap increasing, `Direction == Rising`, clears the change-in-gap floor | stays `New`, forwards every cycle, full severity, throughout |
| shrinking offset | gap decreasing, `Direction == Falling` (the −2.65 MB/3h live case) | becomes `Standing` once past warm-up, same cadence as flat |
| pod restart mid-series | flat offset, then `PodPlacement.CreatedAt` changes | history resets; post-restart cycles read as `New` again (insufficient history), exactly as if the pod were new |
| resumed growth after `Standing` | flat, then resumes rising | reverts to `New` on the next cycle the rise is observed, forwards every cycle again — proves there is no separate "un-suppress" step to fail |

**Live counterpart: yes, needs an injected fixed overhead — passive observation is not reliable on this
population, per 5a.** No new lab capability is required. `k8s/lab/anomaly-guard.yaml`'s own comment already
documents the mechanism: `POST /fault/leak?bytesPerSecond=N` briefly, then `POST /fault/clear` — clear stops
further growth but **does not release what already leaked**, which freezes the pod at a fixed, persistent
offset. That is exactly the flat-offset case, produced without waiting for one to occur naturally. Size the
injected offset well clear of both the floor and observed noise (9.52 MB floor, 26.69 MB observed max) so it
cannot be confused with the churn in 5a, and hold it for more cycles than whatever the cycle-count floor
(section 6) turns out to be — that number gates fixture duration and must come from the spike in section 7,
not be guessed here.

### 6. Quality requirements as parameters

| requirement | parameter | status |
|---|---|---|
| "Stop re-alerting on a stable gap, keep catching a real one" | per-metric change-in-gap floor, `MinAbsoluteGapChange`, in the metric's own units | **Still unmeasured — Spike 2, not run.** Decision (2026-08-10): ships **required and un-defaulted**, not a data-fitted default — the config's own `IsValid` must reject an unset value, mirroring `PeerOutlierOptions`/`IncidentTrackingOptions` rejecting `default`, rather than shipping a number nobody measured. The backlog's harvested peer-gap dispersion table remains the candidate seed for whoever fits it later, per `docs/autoresearch-program.md`'s "what value → search" method — but no default ships until it is. |
| "How many cycles before 'stable' is trusted" | cycle-count floor for the change-detector's own `MinimumSamples`, at **cycle cadence** | **Spike 1 passed, 2026-08-10 — measured, not assumed.** Replayed the real recorded `pj7r8` gap series: flat → `Healthy` (Kendall tau 0.16, p 0.232), rising → `Anomalous`/`Rising`, falling → `Anomalous`/`Falling`. The verdict was **insensitive to `MinimumSamples` at 8/12/15** — the p-value gate decides, not the sample-count gate — which is itself useful: it means this parameter is less load-bearing than the unit-trap warning below implied, though the warning stands. Do not reuse `MinimumHistoryDays = 2` (**days**) or `SustainedThresholdOptions`'s `MinimumSamples: 20` (**scrape-cadence within one window**) — this is a **third** unit, cycle cadence (default 5 min), and conflating it with either existing one is exactly the class of error `docs/aiops/aiops-task-protocol.md` step 8 records for `PS-3`. The gap-ring's own capacity (section 5 amendment) carries the same unit trap a **fourth** time and must be documented as `capacity × cadence`, not a bare count. |
| "How often does a `Standing` finding remind the operator it's still there" | `StandingReassertionInterval` (a `TimeSpan`, **hours**, a **fourth** unit distinct from the three above) | **No lab spike sets this, and none should be invented.** Unlike the two rows above, this is not a detection-accuracy question a healthy/faulted replay can answer — the underlying `PeerGroupOutlierDetector` verdict is `Anomalous` every cycle regardless of this value; the parameter governs only how often an *already-correct* detection reminds a human. That is an operator-attention/paging-cadence question, the same category every paging tool (PagerDuty, Opsgenie) treats as a deliberate customer choice, not a fitted constant. **Recommendation**: ship named presets requiring an explicit choice — mirroring `IncidentTrackingOptions.Balanced/Sticky/Strict` and `PeerOutlierOptions.Balanced/Strict/FastFeedback`, which already reject `default` rather than silently guessing zero (`IncidentTrackingOptions.cs:72-75`) — e.g. `PerShift` (8h), `Daily` (24h), `Weekly`. No single number should ship as *the* default; the client/operator picks one, the same way they must already pick a `PeerOutlierOptions` profile. |
| Severity reduction for a `Standing` finding | a scale factor on the effect-size-derived severity (`IncidentPipeline.cs:411-419`) | Same category as the row above — an ordering/priority knob, not a detection parameter, so it does not block a spike either. Recommend a simple, clearly-labelled placeholder (e.g. half of `New` severity) that the client can override; unlike the reassertion interval this one is lower-stakes (it changes notification *priority*, not whether anything is reported at all) so a reasoned default is acceptable where a silently-guessed reassertion interval was not. |
| Verdict | replay `AN-D1`'s own window (`Tests/bin/fp-run-guard-log-FINAL.txt`, or the `IMetricSource` replay seam) with the gate on vs off. Pass: `pj7r8`-class incident's open-time fraction drops materially from the measured ~97%/day baseline while still reasserting on the configured interval, **and** an injected/fixture growing-leak case still fires at full severity throughout. Both arms required per task-protocol step 8. |

### 7. Technical risks, in spike order

1. **`TrendDetector` reused at cycle cadence on a gap series — validated, 2026-08-10.** Replayed the recorded
   `pj7r8` gap sequence through `TrendDetector.Detect`: flat → `Healthy`, rising → `Anomalous`/`Rising`,
   falling → `Anomalous`/`Falling`, insensitive to `MinimumSamples` at 8/12/15 (table above). Closed — this
   shape is safe to build on.
2. **`MinAbsoluteGapChange` is still unmeasured** (table above) — decided to ship required/un-defaulted
   rather than guessed, not measured. Still open as a future fitting task, not as a blocker to shipping the
   mechanism.
3. **A plateaued leak is under-prioritised between reassertions, by design, not by defect — softened but not
   removed by the client's decision.** It still surfaces every `StandingReassertionInterval` at reduced
   severity rather than going permanently silent (client decision, 2026-08-10 — see Decisions), but between
   reassertions it carries the same reduced visibility a raised floor would give it. This is the accepted
   trade, not a residual bug.
4. **Persisted-state format extension is new on-disk surface** (`docs/adr/0001-peer-novelty-state-persistence-format.md`).
   Must follow `LearnedState`'s existing forward-compatible section convention exactly — verified by a test
   that reads a **real pre-this-change** payload (not one generated by the new code, per the ADR's own
   rationale) and confirms `MetricHistory` + `FloorCalibrator` still restore untouched.

### 8. Decisions

All below are now settled — both blocking questions from the first review round were answered by the client
2026-08-10 (see resolved questions at the end of this file).

- No change to `Sources/Main` public API or `DetectionStatus`.
- Gate lives in `AnomalyGuard.RunPeer` **and** `AnomalyGuard.RunCustomPeer` (finding 4), mutating the
  caller-owned `findings` array and classifying each surviving entry `New`/`Standing` before calling
  `pipeline.ObservePeerGroup`.
- `IncidentPipeline.ObservePeerGroup` gains one new **optional** parameter,
  `ReadOnlySpan<NoveltyKind> novelty = default` — additive, backward compatible, every existing caller and
  test unaffected until it opts in.
- `SignalFinding` gains a `NoveltyKind` field (`New`/`Standing`), propagated through `Incident` /
  `TrackedIncident` / `PersistedIncident` / `IncidentLogRecord` the same way `Class`/`Severity` already are.
- **Client decision, 2026-08-10: accept the plateaued-leak trade, but not as silence** — a `Standing` finding
  reasserts periodically (`StandingReassertionInterval`, section 6) at reduced severity rather than being
  suppressed forever. Full suppression (the original design) is superseded.
- New per-pod-per-metric state type in `Sources/Anomalies/Monitoring/`, keyed and pruned like `_silent`, plus
  a `PodPlacement.CreatedAt`-change reset, **and** dual-keyed (`MetricIndex` array + `Dictionary<string, ...>`
  for custom channels) matching `FloorCalibrator`'s own shape, per finding 4.
- No `IncidentTracker` / `IncidentState` change — the periodic reassertion rides the tracker's existing
  absence-based close/reopen, unchanged.
- **Client decision, 2026-08-10: persist novelty state through a guard restart.** Format: extends
  `LearnedState` with a new section — `docs/adr/0001-peer-novelty-state-persistence-format.md`, Accepted.
- `StandingReassertionInterval` and the standing-severity scale factor: no data spike sets either (section 6)
  — shipped as named presets requiring an explicit choice, not a silently-guessed default.

### 9. Operability

- Extend `PeerDecisionTrace` with a novelty-status field (mirrors why `IsOutlier`/`RelativeGap`/`AbsoluteGap`
  are already separate fields — "a single boolean cannot tell [gate causes] apart", `PeerDecisionTrace.cs`'s
  own doc comment, applied one gate further out).
- Add a `GuardTelemetry` counter for findings demoted by the novelty gate, so an operator can tell the gate is
  active rather than reading a quiet channel as either health or a broken detector — the same distinction the
  rest of this subsystem exists to preserve, applied to the gate itself. `Sources/Anomalies/Monitoring/GuardTelemetry.cs`
  is the existing composition-root-owned `Meter`; see [[project_guard_telemetry_meter]] for the ownership
  pattern already settled there.
- Restart: novelty state persists (Decisions), so a rollout does not reset the mechanism; state plainly in
  client-facing docs that a **reused pod name whose `CreatedAt` changed** does reset (by design — see the
  ADR), which is the one restart-adjacent case that still looks like "un-suppression" to an operator.

## RESOLVED QUESTIONS (were BLOCKING, answered by the client 2026-08-10)

**Client 1 — plateaued-leak trade**: accepted, but not as silence. Standing findings reassert periodically at
reduced severity (Decisions, section 4, section 6).

**Client 2 — persist across restart**: yes. `docs/adr/0001-peer-novelty-state-persistence-format.md`,
Accepted.

**Task owner 1 — diagnostic trace**: stays unchanged every cycle; only incident-forwarding and `NoveltyKind`
are gated. Confirmed by team lead 2026-08-10, no analyst round needed.

**Task owner 2 — config table**: new table, not reused from `MinAbsoluteTrendChange`. Confirmed by team lead
2026-08-10, no analyst round needed.

## Remaining, non-blocking, owed to the developer at implementation time

- `StandingReassertionInterval` and the standing-severity scale factor have no data spike (section 6) —
  implement as named presets requiring an explicit choice, per the recommendation there, rather than stalling
  on a number nobody can measure.
- Risks 1–2 (section 7) are still open spikes and should be the first two tasks, per the walking-skeleton
  order already stated there.

## SUGGESTED IMPROVEMENTS TO MY ROLE

None this run — the memory index, the existing plan-file examples (PSI, runtime-signals) and the semantic
navigator (once pointed at it — see finding 4, which a `Grep`-only pass had already missed once in this same
session) were enough to follow the convention without needing anything new.
