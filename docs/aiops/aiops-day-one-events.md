# Day-one cluster events, measured

What a rollout, a scale-up, a scale-down and an HPA actually do to the guard. Every one of these happens at
a client in the first week and none had ever been shown to it — declared peer cohorts, the step detector and
the silent-pod check had all been validated against fixtures and never against a cluster doing the thing.

Run 2026-08-02 on `k8s/lab` (12 replicas of `Demo/LabWorkload`, paced load driver on a 1440-minute diurnal
curve, guard as a pod, five-minute cadence, twenty-minute window). Each phase stated its expectation **before
the action**, because a phase that can only confirm is not an experiment.

## Results

| Phase | cycles | quiet | findings | **opened** | step lines | **silent-pod** |
|---|---:|---:|---:|---:|---:|---:|
| P0 baseline, no action | 4 | 1 | 3 | 1 | 0 | **0** |
| P1 rollout, 12 replicas replaced | 6 | 1 | 12 | 4 | 1 | **0** |
| P2 manual scale 12 → 16 | 7 | 0 | 34 | 3 | 0 | **0** |
| P3 manual scale 16 → 8 | 6 | 4 | 3 | 1 | 0 | **0** |
| P4 HPA scales up on its own | 8 | 4 | 36 | 1 | 0 | **0** |
| P5 HPA scales down on its own | 7 | 2 | 31 | 1 | 0 | **0** |

## What held

**No pod was ever accused of silence.** This was the risk worth running the experiment for: the silent-pod
check compares the cluster's roster against who reported, and a scale-down removes pods that then, correctly,
report nothing. Eight deliberate deletions in P3 and eight more in P5 — the second with nobody having pressed
anything — produced zero accusations. The unit test `APodTheClusterHasForgottenIsNotReported` covers the
shape; this is the cluster doing it.

**Grouping is what makes the burst survivable.** P2 produced 34 findings and 3 incidents; P4 produced 36 and
1, with one incident absorbing 12 related findings across 9 subjects. The operator-facing number is `opened`,
and it stayed between 1 and 4 in every phase. Without grouping this experiment would read as a storm.

## What we learned, in order of how much it changes

### 1. Scale-UP is the noisy event, and the cause is warm-up, not the scaling

P2 had **zero quiet cycles out of seven**; P3, the opposite transition, had four out of six and three
findings in total. The asymmetry is in the incident text: `MemoryWorkingSetBytes rose by 13.2–17.5% of
typical, tau 0.70–0.94` on freshly created pods. A new pod starts cold and its memory climbs as it warms.

**The trend family is not wrong — the rise is real.** Raising the floor would be the wrong fix and an
actively dangerous one: on this lab the memory trend floor is 1.089 MB against pods of ~45 MB, so a floor
big enough to hide a 13% warm-up would also hide a real leak. What is missing is a rule, not a number:
**a pod with no history should not be judged by the trend family for its first few cycles**, the same shape
as `SilentPodCycles`. This is the one genuinely new defect the experiment found, and it will fire at every
client deploy and every autoscale event.

### 2. "The step detector never fired" was a measurement artefact

The harness counted lines matching the detector's own reason text and found zero in P2–P5, which read as a
silent detector. It was not.

Recomputed directly from Prometheus — the same series, the common component rebuilt the way
`CrossPeerBaseline` does it (median across pods per timestamp), and `LevelShiftDetector`'s arithmetic applied
to a window centred on each event:

| Event | median before → after | absolute | relative | p | \|delta\| | verdict |
|---|---|---:|---:|---:|---:|---|
| P1 rollout | 0.489 → 0.533 | 0.044 | 9.1% | 0.106 | 0.15 | silent, correctly |
| P2 12 → 16 | 0.494 → 0.411 | 0.083 | 16.9% | 2.5e-17 | 0.99 | silent — below the 25% and 0.194 gates |
| P3 16 → 8 | 0.444 → 0.789 | 0.344 | 77.5% | 2.8e-15 | 0.92 | **fires** |
| P4 12 → 16 | 0.533 → 0.578 | 0.044 | 8.3% | 4.8e-06 | 0.52 | silent — below both size gates |
| P5 16 → 8 | 0.578 → 1.245 | 0.667 | 115.4% | 1.6e-15 | 0.93 | **fires** |

And the guard's own log confirms it did:

```
Deployment-wide movement in lab/lab-workload on RequestsPerSecond (severity 0.76) — no individual replica is implicated
Deployment-wide movement in lab/lab-workload on RequestsPerSecond (severity 0.91) — no individual replica is implicated
```

`IncidentReporter` writes `incident.Summary` — built from the **primary** finding — into the row. A
non-primary finding's reason never appears there. At the scale-downs a CPU trend outranked the step on
severity, so the step became a related finding and its text was never printed, while the narrative line
naming deployment-wide movement was. **The instrument was wrong, not the detector.**

Two things follow. The behaviour is right: a scale-up spreads the same offered load over more replicas and
moves the per-pod median by 8–17%, below the 25% worth reporting; a scale-down concentrates it and moves the
median by 77–115%, which is worth saying. And an incident's log row is not a reliable census of what was
found — anything counting findings must read the findings, not the summary.

### 3. Two harness bugs, both of the same family

Neither is about the guard, and both would have produced a confident wrong answer:

- **A `kubectl patch` that did nothing.** P5 was first run with `spec.targetCPUUtilizationPercentage`, the
  `autoscaling/v1` field name on an `autoscaling/v2` object. kubectl warned `unknown field`, reported
  `patched (no change)` and exited 0. The phase then observed a stable 16-replica cluster for 38 minutes and
  produced **7 quiet cycles out of 7, zero findings** — the best result in the table, from an event that
  never happened. Rerun with the correct JSON path *and* a read-back assertion, which is the actual fix: the
  harness must confirm its own premise or abort.
- **A timezone conversion.** The Prometheus analysis first used `time.mktime` plus `time.timezone`, which
  reads the struct as local time against a non-DST offset — in a summer CEST zone the window landed an hour
  off the event, and every transition looked flat. `calendar.timegm` is the correct conversion.

### 4. The calibration on the lab is now contaminated, and must be reset

`FloorCalibrator` learns from what it observes and assumes the observed period was healthy. It observed this
experiment. The proposal for `RequestsPerSecond` moved from a peer gap of **0.0119** before the run to
**1.242** after it — a hundredfold, because deliberate scale transitions were folded in as "what this cluster
does when it is well".

This is precisely the hazard the calibrator documents about itself, arriving through the door nobody was
watching. **The learned-state volume must be wiped before the frozen 24-hour run**, or the guard will start
that measurement with a floor set above the events it is supposed to catch.

## Open items this produced

1. **Warm-up grace for pods without history** in the trend family. Needs measuring: does it silence the
   warm-up without hiding a leak in a freshly started pod?
2. **Reset the learned state before the frozen run** — and consider whether the calibrator should refuse to
   learn from a window in which the pod set changed, which is a cheap structural guard against exactly this.
3. **Nothing counting findings should count log lines.** The harness needs the guard's own counters, which
   `GuardCycleResult` already carries.

## Not covered by this run

Rolling updates with a failing new version (`CrashLoopBackOff` on the new ReplicaSet), scale-to-zero, node
drain, and any of it on a second technology stack. The first is the most valuable next step, because it is
the event where a client most needs the guard to be right and where the silent-pod check finally has
something real to find.
