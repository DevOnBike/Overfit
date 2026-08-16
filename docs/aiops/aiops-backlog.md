# Anomaly guard — backlog, scored

Compiled 2026-08-05, after a day that closed five items and opened three. Every row carries a **cost paid
today** or is marked as not having one — that column exists because the largest single waste of this project
so far was chasing `GcGen2HeapBytes`, which dominates the *historical* statistics and produced **zero**
incidents in the 24-hour run.

**Scoring.** ROI is value per unit of work, not value. Difficulty is work. Risk is the chance the change
breaks detection or blinds a channel — a threshold raise is never zero-risk, because the way it fails is
silence, and silence is what this subsystem exists to distinguish from health.

---

## A. Verification — no production code changes

| # | Task | Why now | ROI | Difficulty | Risk |
|---|---|---|---|---|---|
| A1 | 🔴 **RUN 2026-08-06/07 — DID NOT PASS, and the tail is contaminated.** **11 incidents in 24.83 h = 10.63/day** against a 1-9/day criterion, plus **one cycle failure**. Zero `RequestsPerSecond` incidents, so that third condition held. See the section below. | | The 5/day figure belongs to a configuration that no longer exists: five floors and two code paths changed on 2026-08-05. A number quoted to a client must belong to the thing shipped. Pass: rate inside 1–9/day, zero cycle failures, zero `RequestsPerSecond` incidents from the diurnal ramp. | **highest** | trivial (one day of clock, no attention) | none |
| A2 | ✅ **DONE 2026-08-05 — and it found two defects, one of them ours.** (1) The stack's default route is `receiver: "null"`: a correct alert delivered nowhere. (2) **`time() - overfit_guard_last_cycle_timestamp_seconds > 900` cannot fire when the guard is gone** — the series vanishes with the pod, the expression evaluates over an empty vector, and the alert returns to `inactive`. Measured: `pending` for 60 s, then `inactive` for six minutes, zero notifications. With `absent()` added: **`firing` after 60 s, alert in Alertmanager, two notifications delivered, none failed.** Shipped as `k8s/lab/guard-alerts.yaml`. | **highest** | low | low |
| A3 | ✅ **DONE 2026-08-05.** Second instance: **42 Mi, 1m CPU**, Prometheus +12 Mi and no CPU change. But 42 Mi is a guard eleven minutes old — the steady-state figure measured over the 24-hour run is **130 Mi, peak 146**. **Quote 146, not 42**: fifty namespaces is **7.3 GB**, not 2.1. The script's own arithmetic multiplied the fresh number and was wrong by 3.5x. Prometheus at 50x is extrapolation (≈2.8 queries/s), not measurement. | high | trivial | none |
| A4 | **Evening diurnal check** | Closes the last two unverified floors (`RequestsPerSecond` gap and trend), raised 2026-08-05 and verifiable only by waiting — the fault panel cannot move deployment-wide traffic. Already scheduled. | medium | zero (scheduled) | none |
| A5 | ✅ **RESOLVED 2026-08-08 — both halves measured, and the second needed a control to mean anything.** HPA half: 0 incidents in 11 static cycles against 5 in the 7 spanning a 12→15→12 round trip. StatefulSet half: findings median **14** against **1** for the same twelve pods replaced under an unchanged topology — so the topology contributes an effect over and above pod age, which the first run alone could not have shown. See the section below. | | The lab is twelve identical stateless replicas. Peer comparison is structurally weakest exactly where a client is not: members with their own volumes and shards are not interchangeable, and HPA leaves ghost series and dilutes groups. Documented, never measured. | medium | medium | none |

## B. Client readiness — documentation, not code

| # | Task | Why now | ROI | Difficulty | Risk |
|---|---|---|---|---|---|
| B1 | ✅ **DONE 2026-08-05.** Folded into `aiops-client-readiness.md` rather than written as a second guide — two deployment documents are two versions of the truth. Added: measured cost per instance (**130 MiB steady state, not the 42 MiB a fresh one shows**) and the fleet arithmetic (**7.3 GB for fifty namespaces**, Prometheus at that scale marked as arithmetic not measurement); **both alerting traps with their measurements** — the expression that cannot fire when the guard is gone, and the `"null"` default route; the **first-day caveat** with the mechanism and the expected size (one extra incident per diurnal slope); the **5/day figure read as two results**, with the heap channel's 108 → 0 explicitly denied as credit for the fix; and the 250/day correction. The install protocol now ends by **killing the guard to prove the alert arrives**, because all three links in that chain have failed here. | **highest** | low | none |

## C. Tuning — configuration only

| # | Task | Why now | ROI | Difficulty | Risk |
|---|---|---|---|---|---|
| C1 | **Re-tune the heap floor on an aged population** | `GcGen2HeapBytes.minGap` was set from a population whose pods were hours old; three rollouts on 2026-08-05 reset that. The proposal grew 0.729 → 0.930 MB across one day as heaps diverged, so the current number is extrapolation. | medium | trivial (wait, read the proposal, edit) | **medium** — a floor raised past a real gap blinds the channel silently |
| C2 | **`GcPauseRatio.minTrendChange`** | Proposed at 0.0000345 and left unset: one finding is not evidence enough to arm a gate that has never misbehaved. Revisit only if it starts producing findings. | low | trivial | low |

## D. Detection gaps — real, measured, unfixed

| # | Task | Why now | ROI | Difficulty | Risk |
|---|---|---|---|---|---|
| D1 | **DIAGNOSED 2026-08-06 — the peer family reports a fixed pod property as a recurring anomaly.** Neither option below was right; see the section under this table. | medium | medium — needs a mechanism, not a number | low |
| D2 | 🟡 **The row predates its own fix by four hours — corrected 2026-08-08.** The peer half is true and unfixable: comparing replicas cannot see one dying. But the GUARD is not blind. `SustainedThresholdOptions.ForRareEvent` — whose doc names OOM kills explicitly, "one breach in twenty samples is enough, because these do not happen by accident" — was written at 23:40 on 2026-08-05 and wired to `OomEventsRate` in the same commit. This row was written at 19:21 the same evening, when the channel was genuinely attached to no rule (verified against the parent commit). **What remains is verification, not mechanism**: nobody has shown a real OOMKill producing a finding. The lab has `POST /fault/oom`; the experiment is one call and two cycles. | | Measured. One replica dying is invisible to a family that compares replicas against each other, and OOM is among the most common real failures. Needs a mechanism, not a threshold. | medium | high | medium |
| D3 | **A CPU rise on every replica at once is invisible to all four families** | Measured 2026-08-01 and never diagnosed to the end: peer is blind by construction, no threshold rule covers CPU, and a step that has finished has no slope left to fit. | medium | high | medium |
| D4 | **The heap oscillates with a period near the evaluation window** | Measured 2026-08-05: gen2 swings 3.31 MB inside a 20-minute window while replicas differ by 0.29 MB, and all twelve oscillate **in phase**, so the trend family sees a 1.2–1.5 MB alternation that is window alignment, not the cluster. Same class of error as the 240-minute window sitting on the daily slope, in the opposite direction. **No cost paid today — the channel produced zero incidents in the 24-hour run**, so this is a latent defect, not a live one. | low **now** | high | medium |
| D5 | **The `incidents=` cycle counter disagrees with the `opened`/`resolved` events** | Measured 2026-08-06: 3 of 74 cycles. An incident went open → absent from the count → `ongoing=1` again with **no `resolved` and no `opened`** in between. **Cost paid today:** every metric built on that field under-reports how long anything was open — the run's own open-time figure read 86% instead of 91% and split one 285-minute incident into "255 min, now 25 min", until it was caught by accident. | **high** — it corrupts a headline number | low — read the code that emits the field | low |

### D1, diagnosed 2026-08-06 — a persistent outlier is not an anomaly

The row above offered two explanations: the guard is right and those pods deserve attention, or the peer gate
needs the `MinRelativeGap` treatment `CpuUsageRatio` got. **Measured, and it is neither.**

Twenty-four hours of `container_memory_working_set_bytes` for `lab-workload-*`, split by ReplicaSet
generation — the split matters, because a 24-hour window spans every generation the deployment has had and
ranking pods across a rollout compares pods that never coexisted:

| generation | peer gap (heaviest − median) | clears the 9.52 MB floor | top-2 set changed |
|---|--:|--:|--:|
| `7f5fb9f88c` (the population the row was written about) | median **18.8 MB**, p90 20.4 | **34 / 65 samples (52%)** | **0 of 64 transitions** |
| `7765564ff6` (current) | median **5.9 MB**, max 9.3 | **0 / 108 samples** | 4 of 107 (4%) |

Two things follow, and the second is the finding.

**The symptom is generation-dependent, so the row's premise has expired.** Same workload, same config, same
floor: one generation sits at ~19 MB of spread and reports in half its cycles, the next sits at ~6 MB and is
silent in all 108 samples. Whatever makes two replicas heavy is assigned when the pods start and differs from
rollout to rollout. Re-checking a symptom before fixing it is why this took an hour instead of a day.

**In both generations the same two pods are the heaviest essentially always** — 100% and 96% of samples, with
the top-2 set changing 0 and 4 times respectively. That is not an event. **A pod that has been 19 MB heavier
than its peers since its first cycle, in every cycle, for hours, is not anomalous — it is that pod's
baseline.** The peer family has no notion of *novelty*: it re-derives the ranking every cycle from scratch and
so re-reports a fixed configuration difference forever.

**This is a missing mechanism, not a wrong number, and the trade proves it.** Raising the floor above 19 MB
would silence the current noise — and would simultaneously blind the channel to a genuine 50% memory leak on
a 40 MB baseline, which is exactly what the channel exists to catch. A parameter that cannot be moved without
buying silence at the price of detection is the signature of a degree of freedom that is absent, the same
diagnosis pattern recorded in `CLAUDE.md` for the synthetic generator's queueing term.


#### Confirmed live during the 2026-08-06 run — and it costs more than the rate suggests

The measurement above was historical: 24 hours of `container_memory_working_set_bytes` read back from
Prometheus. The 24-hour false-positive run that started 2026-08-06 07:59:19Z produced the same pattern **as
it happened**, which settles it.

`lab-workload-7765564ff6-pj7r8`, `MemoryWorkingSetBytes`, peer family:

> sits above the other 11 peers by **21% (9.99 MB)**: **Cliff's delta 1.0**

Cliff's delta of 1.0 is perfect separation — that pod is above every one of the other eleven in every sample.
The incident **opened at 08:29Z, closed at 08:59Z, reopened at 09:29Z and was still open at 11:14Z**, one
finding per cycle re-confirming it for an hour and three quarters.

**Note which pod.** The historical scan found generation `7765564ff6`'s gap sitting at a median of 5.9 MB and
never clearing the 9.52 MB floor in 108 samples, with `f7lrt` and `9nds7` as the persistent heavy pair. Those
samples predate the container restart at the run's start. After the restart the heavy role was **reassigned
to a different pod**, and its gap grew past the floor. That is the mechanism this section describes —
"assigned when the pods start and different from rollout to rollout" — playing out inside a single generation,
which is stronger than the between-generation evidence it was originally based on.

#### A second problem the historical scan could not see: the rate under-counts this

The run's false-positive figure counts **incidents opened per day**. An incident that opens once and stays
open for twenty hours is **one** by that measure, and a **permanently red entry on the operator's screen** in
reality. Those are different costs and only the first is being measured.

So the 24-hour report needs a second number beside the rate: **for how long was any incident open at all.**
On this run one incident looks likely to cover most of the day while contributing a single unit to the count
that everyone will quote.


**Measured, once the number existed.** `hourly_check.py` (the watcher for the 24-hour run; removed from the repository 2026-08-07 together with the run it served — the observations below stand, the script does not) gained an `OPEN` line on 2026-08-06 that counts
time rather than openings. At 3.8 h into the run, on 44 cycles:

| | |
|---|--:|
| incidents opened | **4** → 26/day |
| cycles with anything open | **35 of 44 (80%)** |
| longest unbroken open streak | **135 min**, still open |

**Both numbers are true and they say opposite things.** The rate says the guard rarely speaks. The open-time
says an operator is looking at a red entry for four fifths of the day. If `pj7r8` stays 21% above its peers —
and at Cliff's delta 1.0 nothing suggests it will stop — the 24-hour figures land near **4–5 openings per day
at ~97% of the day with an incident open**.

A client must be given both. The rate alone is technically correct and practically misleading, which is the
one kind of number this project does not ship.

This is a measurement gap, not a detection gap, and it is cheap to close — the guard already logs
`opened`/`ongoing`/`resolved` every cycle, so open-time is a sum over existing data rather than new
instrumentation.

The mechanism the family needs is a **per-pod offset learned over the pod's own history**: judge a replica
against its peers *after* subtracting the gap it has held since it started, so that a stable difference is
learned once and only a *change* in that difference reports. That is a design task, not a tuning task, and it
is shared with D2 and D3 — all three are the peer family lacking a notion the threshold cannot express.


### D5, measured 2026-08-06 — an incident that is open, absent, then ongoing again

The guard logs one line per cycle:

```
cycle: pods=12 findings=1 incidents=1 opened=0 ongoing=1 resolved=0
```

At **13:44:18**, mid-run, one cycle read:

```
cycle: pods=12 findings=0 incidents=0 opened=0 ongoing=0 resolved=0
```

and five minutes later the incident was back as `ongoing=1`, `opened=0`. **No `resolved` before it, no
`opened` after it.** Three cycles out of seventy-four behaved this way.

**Two readings, and only the code settles which:**

1. **`incidents=` means "incidents that produced a finding this cycle"**, not "incidents that are open". The
   field is then correct and its *name* is the defect — which is still worth fixing, because everything
   downstream reads it as a state.
2. **The incident state machine has a hole**: an incident can stop being counted without resolving. That
   would matter more, because incident lifetime is what an operator's screen shows.

**This already cost something, which is why it is not a curiosity.** `hourly_check.py`'s `OPEN` line — added
the same day precisely because the opened-per-day rate under-reports a permanently open incident — was itself
built on `incidents=` and inherited the flaw. It reported **86% of cycles with something open when the true
figure was 91%**, and cut a single continuous 285-minute incident into "longest 255 min, open now 25 min".
The number added to stop under-reporting was under-reporting.

Fixed in the script by latching on the **events** (`opened` minus `resolved`) instead of the snapshot, and it
now prints the disagreement count every run so this cannot go unnoticed again. **The guard-side question is
untouched by that** — the script only stopped trusting the field.

**Settled 2026-08-06 by reading the code — it is the first reading, and it is worse than it looked.**

`AnomalyGuard.RunCycleCore` builds the result as:

```csharp
var result = new GuardCycleResult(
    pipeline.Count, incidents.Count, opened, ongoing, resolved, blind, partial, unevaluable);
```

`Incidents` is `incidents.Count` — **the groups formed from THIS cycle's findings**, exactly as
`GuardCycleResult`'s own XML doc says ("Groups they formed"). `opened`/`ongoing`/`resolved` are counted by
walking `tracked`, the incidents touched this cycle. So when a window produces no finding, `tracked` is empty
and **`ongoing` is zero too**, while the incident is still open.

**So none of the five fields reports how many incidents are currently open.** They are all per-cycle activity
counters. The state machine is not broken — **the observable is missing**, and the field named `incidents`
invites precisely the misreading that cost a day's headline number here.

Two things follow:

1. **The guard should expose an open-incident count** (or an oldest-open-incident age) in the cycle line and
   in `GuardTelemetry`. An operator's real question is "is anything open right now, and for how long", and
   today it can only be answered by replaying the whole log and latching on `opened` minus `resolved` — which
   is what `hourly_check.py` now does.
2. **`Incidents` should be renamed** to something that cannot be read as state — `Groups`, or
   `GroupsThisCycle`. It is a `record struct` positional parameter, so this is a rename plus its call sites,
   not a design change.

Neither is urgent: nothing detects worse because of it. Both are cheap, and the second one prevents the next
reader repeating the mistake.

## E. Deferred by decision

| # | Task | Why deferred | ROI | Difficulty | Risk |
|---|---|---|---|---|---|
| E1 | **Multi-scope, slices 3–5** | One instance per namespace costs 130 MB and zero code, gives free failure isolation at a process boundary and allows per-namespace RBAC; fifty templated Deployments is ordinary Kubernetes practice. The memory saving is small because per-scope state dominates either way. **Build it when a client says the fleet is unmanageable**, not before. Slices 1–2 are done, behaviour-neutral and waiting. | low until asked | high | **high** — shared id counter, shared state file, and the design's own warning that `--real` labels must not cross scopes |
| E2 | **A separate CPU-limited deployment so the throttle channel is testable** | `container_cpu_cfs_throttled_periods_total` exists only on containers with a CPU limit, and the lab deliberately has none so that every CPU measurement has one explanation. This is test coverage, not a feature; its value appears the first time the channel misses something at a client. | low | medium | low |
| E3 | **Rebuild the guard image for the corrected startup message and the scope plumbing** | Cosmetic and already in the tree; ships free with the next deploy that has another reason to happen. | low | trivial | low |

---

## Elsewhere: the test gate that covers this subsystem

`docs/test-gate-backlog.md`, opened 2026-08-07 from the first execution the `[LongFact]` suite has ever had.
It is a separate file because most of what it holds is test infrastructure rather than detection, but two
items land squarely on this subsystem and are tracked there, not here:

- **Five `Anomalies/Diagnostics` lab tests** — `AnomalyGuardEndToEndDiagnostics`,
  `AnomalyGuardShadowRunDiagnostics`, `LabFixtureRecorderDiagnostics`, `LabFloorCalibrationDiagnostics`,
  `PrometheusMetricSourceLabDiagnostics` — all failed on a missing port-forward, and the documented
  `forward.cmd` did not fix it because it forwarded one of the three ports they use. Fixed 2026-08-07.
- **`GptAnomalyLoRATargetComparisonTests` is flaky by construction** (unseeded weight init), and its
  production counterpart **passed in under a second without loading the model** — the test whose purpose is
  to confirm the LoRA recommendation on the real trained artifact. That is one instance of a pattern
  affecting **62 of 256** `[LongFact]`, which is the highest-ranked item in that file.

## Order

**Everything that disturbs the lab happens BEFORE the run, not around it.**

1. **A3** — second-instance cost. Adds a Deployment, torn down after.
2. **A2** — alert delivery. Kills the guard on purpose.
3. **A1** — start the 24-hour run, then touch nothing for a day.
4. **B1** while the run goes — the paragraph that keeps a pilot from being surprised. Costs no cluster time.
5. **A5** after the run — StatefulSet and HPA are two separate experiments and each rewrites the workload.
6. Then **D1**, the only detection question with a cost being paid every cycle right now.

**This ordering is a correction, and the correction is the point.** The first version of this file put A2
*during* the run and A3 *after* it, reasoning that A2 "touches only the guard". That is true and irrelevant:
A2 kills the guard, which breaks the run's cycle continuity, and the run's whole product is an uninterrupted
count of cycles and incidents. A3 adds a second guard watching the same pods, which double-counts
everything while it lives. A5 replaces the workload. **Each of the three invalidates the measurement in a
different way, and none of them does it by touching the pods** — which is why "what does it touch" was the
wrong question and "what does it invalidate" is the right one.

Everything else waits for evidence that somebody is paying for it.


---

## A1, run 2026-08-06/07 — the number, and why it does not settle anything yet

Read out of `Tests/bin/fp-run-guard-log-FINAL.txt` on 2026-08-07 rather than from anyone's recollection,
because the run finished and its result was never written down — the watcher was removed the same day and
took the only record of the outcome with it.

| | |
|---|---|
| window | 2026-08-06 08:04 -> 2026-08-07 08:54 UTC (**24.83 h**) |
| cycles | 298 |
| incidents opened | **11** |
| incidents resolved | 10 (one still open at the end) |
| **rate** | **10.63 / day** — criterion was 1-9 |
| cycle failures | **1** — criterion was zero |
| `RequestsPerSecond` incidents | **0** — criterion met |
| cycles with at least one finding | 280 of 298 |
| max concurrent incidents | 1 |

**Two of the three pass conditions failed.** That is the headline and it is not softened below.

### The contamination, which is mine

The cycle failure at 2026-08-07 08:29 is `Connection refused` to
`overfit-lab-prometheus.monitoring.svc.cluster.local:9090`, immediately preceded by *"Pod topology could
not be refreshed; grouping this cycle used the previous snapshot of 12 pod(s)"*. Prometheus was evicted —
by the `[LongFact]` suite run happening on the same box that morning, the run whose single-process form
peaked at 21.7 GB and evicted the monitoring stack once before (recorded as A3).

Two of the eleven incidents (08:34 and 08:44) fall **after** that disturbance. Excluding them gives
8.85/day, which is inside the band.

**They are not excluded.** Dropping the two inconvenient points would require showing that the disturbance
CAUSED them, which has not been shown — proximity is not causation, and this file exists partly because
that kind of reasoning is how a measurement becomes a wish. The recorded rate is 10.63/day with the
disturbance documented alongside it, and a clean re-run is what would replace it.

### The protection existed and was bypassed

`Scripts/longfact_gate.py` opens with `refuse_if_the_box_is_an_instrument()`, which reads
`Tests/bin/fp-run-clean-start.txt` and exits 3 while a 24-hour measurement is in flight. It works. It was
simply not on the path taken: the suite was launched as `dotnet test` directly, which no marker check
guards. A guard that only covers one of two entrances is worth naming as such rather than trusting.

**Fix, when A1 is re-run:** put the marker check where the *test process* starts rather than where one
convenience script does — an xUnit assembly fixture that refuses to run the long suite while the marker is
fresh would cover every entrance, including a developer pressing Run in an IDE.


---

## Decision 2026-08-07 — replay and simulator first, the live day only when unavoidable

**What was decided.** Wire `AnomalyGuardService` to `IMetricSource` so a recorded or historical window can
drive it, then run everything that can be run off the simulator. A live 24-hour run happens only when
nothing else will answer the question.

**Why, and it is not impatience.** The A1 run produced **11 incidents**. For a counting process that is a
95% interval of roughly **5.5 to 19 per day** — and the pass criterion is "inside 1-9". The interval spans
both verdicts, so **one day at this rate cannot decide the criterion at all**. A second day gives about 22
events and an interval that still crosses the boundary. The limit is the event count, not the clock, which
is why repeating the day was the wrong instinct and why three levers beat it:

1. **Replay.** `PrometheusHistoricalSource` and `PrometheusHistoricalSourceConfig` already exist, as do
   `IMetricSource` / `IRawMetricSource`. What blocks replay is one seam: `AnomalyGuardService` holds a
   concrete `PrometheusMetricWindowSource`. Give it the interface and a virtual clock and a day of history
   evaluates in seconds — against data Prometheus already retains. Three wins at once: no waiting, a week
   of events instead of eleven, and **repeatability** — the same day can be re-run after every threshold
   change and the results compared, which a live run can never offer.
2. **Population.** The rate scales with pod count; 36 pods yield in eight hours what 12 yield in a day.
   Costs cluster resources rather than clock.
3. **Simulator.** `Tests/TestSupport/SyntheticCluster.cs` plus `SyntheticClusterCalibrationSearch`.

### What this decision puts on the critical path

Driving the work from the simulator makes the simulator's fidelity **the** question rather than a footnote.
The last recorded figure is ~250 false positives/day on 20 synthetic pods against **10.63/day on 12 real
ones** — roughly a 25x discrepancy per pod. That number is from 2026-07-30 and **has not been re-checked
since the threshold changes**; re-measuring it is step one, before any conclusion is drawn from a synthetic
run.

Two constraints carry over unchanged:

- **Calibrate only against a clean recording.** `LabWindowValidator` exists to reject a contaminated one,
  and the 2026-08-06/07 window is contaminated at the tail (Prometheus evicted at 08:29). Fitting a
  generator to a broken reference is worse than not fitting it: fast, repeatable and wrong.
- **A search fits a mechanism, it cannot invent one.** If the 25x gap is a missing degree of freedom in the
  generator, it will show up as a bad TRADE under fitting — one statistic bought at another's expense — not
  as a number that refuses to converge.

### Revised order

1. The `IMetricSource` seam in `AnomalyGuardService` (the enabler for everything below).
2. Re-measure the simulator's false-positive rate on the current thresholds — is 250/day still true?
3. Explain the simulator-versus-lab gap. Until it is explained, a synthetic run proposes hypotheses; it
   does not deliver verdicts.
4. Replay the week Prometheus already holds, and decide the 1-9 band on hundreds of events.
5. A live day only if 1-4 leave something genuinely unanswerable.

Also carried forward from A1: move the measurement-in-progress check from `Scripts/longfact_gate.py` to
where the **test process** starts, so no entrance bypasses it. Today's contamination came in through
`dotnet test` run directly, which the script's own guard never sees.


---

## Enriching the generator from data already on disk (2026-08-07)

The point of this section is that **another 24-hour run is not needed to improve the generator**. The run
that already happened left more behind than its headline rate, and none of it has been used yet. Source:
`Tests/bin/fp-run-guard-log-FINAL.txt`, 298 cycles, 2026-08-06 08:04 -> 2026-08-07 08:54.

### 1. The findings-per-cycle distribution, which is the most valuable thing in the file

| findings in a cycle | cycles |
|---|---|
| 0 | 18 |
| **1** | **272** |
| 2 | 4 |
| 3 | 1 |
| 5 | 2 |
| 8 | 1 |

**This is not noise, and the arithmetic says so.** A random process averaging one event per cycle would
put roughly 37% of cycles at zero, 37% at one and 18% at two. The observed shape is 6% / 91% / 1.3% —
nearly degenerate. Something **persistent** produces exactly one finding in almost every cycle; the eleven
incidents are not eleven independent events scattered through a day.

That independently corroborates **D1** — "the peer family reports a fixed pod property as a recurring
anomaly" — from a completely different reading of the same run. A standing outlier is exactly what
produces one finding per cycle, forever.

**What it means for the simulator, and it is not a small thing.** A generator that models false positives
as random draws cannot produce this shape at any parameter setting. If it currently reports ~250/day
against the lab's 10.63, the two numbers may not be measuring the same phenomenon at all — one a standing
outlier counted repeatedly, the other a stream of independent events. **Fitting the rate would then be
fitting the wrong quantity.** This is the "missing degree of freedom shows up as a bad trade, not a bad
number" rule from `docs/autoresearch-program.md`, arriving early enough to act on.

### 2. Real per-channel peer dispersion, twelve pods, twenty-four hours

Harvested from the guard's own floor proposals, which print the measured spread every time they fire.
These are parameters the generator currently has to guess.

| channel | peer gap (min..max) | typical magnitude | samples |
|---|---|---|---|
| `CpuUsageRatio` | 0.00051 .. 0.0043 | 0.00099 .. 0.0017 | 23 |
| `MemoryWorkingSetBytes` | 1.03e7 .. 1.08e7 | 4.39e7 .. 4.66e7 | 23 |
| `GcGen2HeapBytes` | 7.83e5 .. 1.04e6 | 3.47e6 .. 3.50e6 | 15 |
| `LatencyP50Ms` | 0.139 .. 0.476 | 37.5 (constant) | 20 |
| `LatencyP95Ms` | 0.264 .. 7.03 | 48.75 (constant) | 20 |
| `LatencyP99Ms` | 4.75 .. 24.25 | 49.75 (constant) | 20 |
| `GcPauseRatio` | 2.13e-5 | 0 .. 4.96e-6 | 23 |

Trend-change variants exist for the same channels and are in the log alongside these.

**The latency rows deserve a second look before use.** Typical magnitudes of 37.5 / 48.75 / 49.75 are
**constant across the whole day** while the peer gaps move by two orders of magnitude between p50 and p99.
That is the same arithmetically-suspicious shape which, at the previous calibration, revealed that the
latency quantiles were one series scaled by a constant. Whether that is the generator's artefact or the
lab workload's is unresolved — and it is a mechanism question, so no search will answer it.

### 3. What is NOT usable yet

A per-channel attribution of which channel produced which finding. A first pass counted channel names
across the whole log and was **contaminated by the floor-proposal lines themselves** (the proposals name
the same channels, 23 times each), so the counts described the proposals rather than the findings. It
needs a proper parse of the finding lines before any number from it is quoted.

### 4. Order for this material

1. Reproduce the findings-per-cycle **shape** before touching any rate. If the generator cannot produce a
   91%-exactly-one distribution, the rate discrepancy is a symptom and tuning it is wasted work.
2. Feed the measured per-channel dispersions in as starting values — `LabWindowValidator` gates the
   recording they came from, so run that check first.
3. Parse finding attribution properly, then compare per-channel false-positive shares against the lab's.


---

## A5, HPA half measured 2026-08-07 — and the lab cannot produce this failure at all

The row said "documented, never measured". Both claims in it are now measured. A third effect, not in the
row, turned out to be the largest.

**Method.** Manual `kubectl scale` on `lab-workload`, 12 -> 15 -> 12, rather than a real HPA. The guard
cannot tell what moved the replica count, only that it moved, so for dilution and ghost series the two are
the same event; an HPA would add a control loop whose timing is its own experiment. **What this substitution
does NOT cover** is HPA's own metric traffic and its scale-down stabilisation window.

| cycle (UTC) | pods | findings | opened |
|---|---|---|---|
| 20:34 – 21:24 | 12 | 1–3 | **0** across 11 cycles |
| 21:29 | 12 | 2 | 1 — scale-up command issued at 21:29:04 |
| 21:34 | **15** | 5 | 0 |
| 21:39 | **15** | **11** | 1 |
| 21:44 | **15** | 2 | 0 |
| 21:49 | **15** | 3 | 1 — *four minutes after the scale-DOWN* |
| 21:54 | 12 | 6 | 1 |
| 21:59 | 12 | 8 | 1 |

### Dilution — confirmed

Baseline holds a 1–3 band across eleven cycles. Two cycles after three replicas join: **5, then 11**
findings, over triple the previous maximum. New pods enter the peer group with young heaps and a different
profile from pods running for days — the same mechanism C1 describes for the heap floor, arriving as an
event rather than as drift.

### Ghost series — confirmed, and bounded

Scale-down at 21:45:04. The 21:49 cycle still counts **15 pods**, four minutes after three of them ceased
to exist; pods terminate in seconds, so this is Prometheus's lookback window, not slow deletion. By 21:54
it is 12 again. **One cycle of ghost, bounded between 4m08s and 9m08s** — which at a five-minute cycle is
one full evaluation window in which the guard compares live replicas against dead ones whose metrics are
frozen.

### The effect nobody predicted: scaling DOWN costs as much, and the tail outlives the change

Returning to exactly twelve replicas did not return the guard to baseline. Findings were **6 and 8** two
cycles later — still double to triple the band — and **two more incidents opened**. Fourteen minutes after
the scaling finished, the guard had not settled.

**Aggregate: 0 incidents in 11 static cycles, 5 in the 7 cycles spanning the change.**

### Why this matters more than the A5 row implied

A client running HPA scales several times a day. If every scaling event opens an incident and leaves a tail
of elevated findings behind it, that is **a false-positive source the lab structurally cannot contain** —
`lab-workload` has held twelve replicas for six days. It follows that the **10.63/day measured in A1 is a
floor for such a client, not a representative figure**, and no amount of re-running a static lab will
discover the difference.

It also names something the simulator must learn before it can stand in for the lab: it models a population
of **fixed size**. Without membership-change events it cannot reproduce what is plausibly the largest source
of false positives in a real deployment.

### Not done

The StatefulSet half. Members with their own volumes and shards are not interchangeable, so peer comparison
is structurally questionable there in a way scaling does not test. It rewrites the workload and is a
separate experiment.

### A note on the measurement itself

The first run of this experiment reported **empty result sections** while executing both scale operations
correctly. The parser required a timestamp on the guard's `cycle:` line; that line is an indented
continuation and carries none — the timestamp sits on the header line above it. Nothing failed loudly; the
report simply read as "no events". Fixed with `kubectl logs --timestamps`, which prefixes continuation lines
too, and the window was recovered from the logs without re-running anything. Third instance in one day of a
reporting bug that reads as good news.


---

## The replay is faithful, the learned state makes it worse, and the "unexplained" incidents were traffic

Three results from 2026-08-08, in the order they were found. Each overturned the hypothesis that produced
the previous one, which is why they are recorded together.

### 1. The replay reproduces A1 exactly — cold, with no learned state

The first historical replay compared **21.4 incidents per 100 completed cycles** against A1's **3.7**, a
5.8x gap, and the leading hypothesis was that replay runs cold while the deployed guard carries seasonal
history and floor calibration.

**Wrong.** Replaying A1's *own* window — its 298 cycle timestamps read out of
`Tests/bin/fp-run-guard-log-FINAL.txt` — cold, reproduced **all 11 incidents in the same 11 cycles**, nine
of them within one second and the worst within 24. Findings 296 against 301 live.

| window | state | completed | findings | opened | per 100 |
|---|---|---|---|---|---|
| A1's own window | cold | 298 | 296 | **11** | **3.7** |
| A1's own window | warm (deployed state) | 298 | 517 | **33** | 11.1 |
| recent 24 h | cold | 201 | 242 | 40 | 19.9 |
| recent 24 h | warm | 201 | 306 | 35 | 17.4 |
| recent 24 h | calibration only | 201 | 238 | 40 | 19.9 |
| recent 24 h | history only | 201 | 316 | 35 | 17.4 |

A1 live was 3.7. Cold replay of A1's window is 3.7. **Replay is a faithful instrument**, which is what the
2026-08-07 decision to prefer it over living through another day depended on and had not verified.

### 2. The learned state makes the guard NOISIER, and the seasonal history is the whole lever

Feeding the real deployed state into A1's window takes it from **11 opened to 33**, and findings from 296
to 517. Stripping one section of the payload at a time isolates it: **calibration-only moves nothing**
(40 opened, cycle-for-cycle identical to cold), **history-only reproduces the full warm result** (35, identical
to full-warm). Direction is not a contamination artefact — the deployed state contains the replayed days,
which biases it *quieter*, and it still came out louder.

**And the deployed guard was effectively cold too, by construction.** `MinimumHistoryDays = 2`
(`Contracts/AnomalyGuardOptions.cs:115`) and `MetricHistory` keeps one observation per
(workload, metric, hour) **per calendar day**. At A1's start the PVC held five hour-buckets from a single
day, so essentially nothing qualified — and **a 24-hour run cannot arm it at all**, because one day
contributes one observation per bucket. That is a property of the design, not of that particular run.

Open, and it is a product question rather than a measurement one: is the seasonal regression a defect, or
an expectation of the wrong shape? The hypothesis on offer — an hourly-anchored interpolation subtracted
from a smooth signal injects apparent trend — is reasoning, not measurement.

### 3. The 34 "unexplained" incidents were a 51% rise in traffic

Of the recent window's 40 incidents, five fell inside the 12 -> 15 -> 12 scaling experiment and one in the
hour after the scrape outage. The remaining 34, clustered 2026-08-07 12:00-22:15Z, had no explanation.

The hypothesis was host contention: this session drove the heavy `[LongFact]` group — Qwen-3B and
Bielik-4.5B loads, QLoRA fine-tunes, MNIST training — on the machine that hosts the cluster, over exactly
those hours. **Refuted by direct measurement.**

| | A1 (quiet) | suspect window | |
|---|---|---|---|
| request rate p50 | 14.17/s | **21.46/s** | **+51%** |
| CPU p50 | 0.0176 | 0.0224 | +27% |
| **CPU per request p50** | 0.00121 | **0.00106** | **-12%** |
| PSI cpu waiting p50 | 1.39e-5 | 1.46e-5 | unchanged |
| PSI memory / io | 0 | 0 | zero |
| p95 latency | 0.0488 s | 0.0488 s | identical |
| errors | 0 | 0 | zero |

`container_pressure_cpu_waiting_seconds_total` measures time a container was runnable and denied CPU. It
did not move. **Node-level metrics cannot settle this question** — `node_exporter` runs inside the
docker-desktop VM and cannot see host processes at all — but PSI measures the effect regardless of where
the cause lives, and there was no effect.

What happened is simply that the lab received half again as much traffic. CPU rose because work rose;
per-request cost *fell* 12% and kept falling through the afternoon (0.00169 at 06:00Z to 0.00099 at
19:00Z), consistent with JIT warm-up under sustained load.

**So the 34 are false positives of a class this backlog already names and shelved:** the "Optional" table's
*"Affine work-adjusted trend for load-sensitive signals — measured and NOT shipped"*.

**The quantified cost of not shipping it: 11 incidents at 14 req/s, 40 at 21 req/s.** Raising traffic by
half roughly quadrupled the alarm count on an unchanged, healthy cluster. That is a stronger argument for
work adjustment than anything previously recorded here, and it reframes A1's failure — the 10.63/day was
measured on a lab whose traffic was flat, and a client whose traffic varies gets worse, not better.

### One methodological note worth keeping

Three of four metric names guessed for this investigation did not exist, and Prometheus answers a query
over a non-existent metric with an empty result, not an error. Reading "no samples" as "the value did not
move" would have produced a confident wrong answer. The names came from
`/api/v1/label/__name__/values` with a `match[]` selector — ask the instrument what it has before asking
it what it says.


---

## A5, StatefulSet half measured 2026-08-08 — and the first run proved nothing until a control was added

The claim: peer comparison is structurally weakest where members are not interchangeable, and a
StatefulSet's are not. Documented since the row was written, never measured.

### What the code already handled, checked before deploying rather than after

Predicted failure: grouping keys on the ReplicaSet, and a StatefulSet has none. **The prediction was
wrong and the code is right.**

- `PrometheusTopologySource` resolves a StatefulSet pod's workload in **one hop** instead of two and leaves
  `ReplicaSet` empty on purpose — its own comment says inventing one "would be worse than reporting what
  is known".
- `IncidentGrouper.Matches` is `first.Length > 0 && string.Equals(...)`, so **empty does not match empty**.
  Members fall through to `SameWorkload` rather than being scored `SameReplicaSet` on a shared absence of
  information.
- `PeerCohorts` does not key on the ReplicaSet at all; that was tried and reverted within the hour because
  it left canaries alone in a cohort below the minimum group size.

Confirmed live: `kube_pod_owner` reported `StatefulSet` for all twelve members, and the guard held
`pods=12`.

### The measurement, and why the first table was not evidence

The lab was swapped onto a StatefulSet named `lab-workload`, so its pods are `lab-workload-0..11` and the
guard's existing `lab-workload-.*` selector matched them with **no configuration change**.

| arm | findings per cycle | median | max |
|---|---|---|---|
| Deployment, aged pods | 0 0 0 0 0 0 1 0 | 0 | 1 |
| **StatefulSet, fresh pods** | 0 0 12 14 26 25 16 | **14** | **26** |
| Deployment, **fresh** pods (`rollout restart`) | 0 1 0 1 6 4 3 | **1** | **6** |

**The first two rows alone would have been a false result.** Every one of the twelve pods was minutes old,
and A5's own first half had already measured that fresh pods produce findings — three of them gave 5 and
11. Twelve fresh pods reaching 12–26 is entirely consistent with pod age and needs no topology at all. The
shape agreed: 12 → 14 → 26 → 25 → **16**, a rise and the start of a fall, which is a warm-up transient and
not a stable structural property. `WarmUpGrace` is 15 minutes.

The third row is the control that separates them: **the same twelve pods replaced under an unchanged
topology**, changing pod age and nothing else. It produces a real but far smaller effect — median 1
against 14, peak 6 against 26.

**So both effects exist and they are separable.** Pod age is worth about 6; the StatefulSet topology adds
roughly a further four-fold on the peak and fourteen-fold on the median. A5's claim holds.

**Limits, stated rather than implied.** One run per arm, no repetition. The StatefulSet mounted per-member
PVCs that the Deployment does not, which is a difference beyond topology even though the workload never
writes to them. Every floor in use was calibrated on the Deployment population. And the load was **even
across members** — a real sharded service is uneven, so this measured the topology of sharding, not
sharding itself.

### Two findings the control produced that nobody asked for

**An ordinary `rollout restart` — that is, every deployment — lifts findings from zero to six for roughly
fifteen minutes.** Operationally that means a routine deploy lights the guard up, on a healthy cluster,
with nothing wrong.

**During the rollout the guard reported `pods=24`**, counting both generations at once. Peer comparison
therefore treats pods of two different software versions as equals for a cycle or two. This is not a
defect: `IncidentGrouper` scores `SameReplicaSet` above `SameWorkload` precisely to separate versions, but
the cohorts deliberately do not split on it, because splitting left canaries invisible. It is a known
trade — now with its cost measured, and it belongs beside the ghost-series result from the first half.
