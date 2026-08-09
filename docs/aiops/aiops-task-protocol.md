# How an anomaly task is run, start to finish

The working protocol for every `AN-*`, `RS-*` and `PS-*` task. `CLAUDE.md` carries the short version and
links here; this file is the detail, including the incidents each rule was written from.

**When this applies:** any change to *what the guard detects* — a channel, a binding, a threshold, a rule, a
detector. Not refactors, not a rename. **One task at a time.**

**Where the task lives:** `docs/TASKS.md` is the registry and the only place carrying status;
[`aiops-backlog.md`](aiops-backlog.md) is domain prose and its rows are commentary, not state. Editing a
status in both is how they diverge — that happened on 2026-08-08.

## Why this protocol exists at all

**A detector that works is silent almost all the time.** Every defect in it therefore *presents as silence*,
and silence is indistinguishable from success from the outside. Nothing else in this repository has that
property: a broken kernel returns wrong numbers, a broken loader throws, a broken test goes red. A broken
channel returns nothing, which is exactly what a healthy channel returns.

Every step below exists because that property defeated somebody — usually me — in a specific, recorded way.

## Before implementing

### 1. Read the code path that produces the number, and quote the decisive arithmetic

Read [`aiops-detection-pipeline.md`](aiops-detection-pipeline.md),
[`aiops-adding-a-metric.md`](aiops-adding-a-metric.md) and [`aiops-repair-plan.md`](aiops-repair-plan.md)
first, then the code — not the doc comment, the arithmetic. Also read any rule, profile or constant that
already exists for this signal, **including its calibration conditions**, because those name the environment
you have to reproduce before your measurement means anything.

**The task description is not a source of truth.** Earned three times on 2026-08-09:

- an `AN-D2` row written four hours before its own fix, describing a state that no longer existed;
- a `+1.00` correlation premise with no artefact computing it anywhere in the tree;
- a 200-line diagnostic reproducing `series - expectation` when `AnomalyGuard.Adjust` computes
  `series - expectation + median` twenty lines away — which made the recorded diagnosis an artefact of an
  operation the product does not perform.

And three hours of CPU-limit measurement answered a question `SustainedThresholdOptions.ForCpuThrottling`
states in its own doc comment.

### 2. Name or create the artefacts, before writing code

The query/binding, a **positive** fixture, a **negative** fixture, a **missing-data** fixture, and the
expected output for each.

The missing-data one is not optional and not a formality — it is the only artefact that distinguishes a
working detector from a dead one.

### 3. State the premise out loud

What produces this number, in what unit, and **what observation would refute the explanation**. Two lines,
written before the measurement runs.

The point is that it is catchable from outside by somebody who has not read the code. Nobody can challenge
a premise that was never stated.

## Implementing

### 4. Every change must distinguish `Detected`, `Healthy`, `WarmingUp`, `InsufficientData`, `QueryFailed`

**Absence of series is NEVER `Healthy`.** This is the single most important rule in the subsystem, for the
reason in *Why this protocol exists*.

Partial precedent exists and should be consolidated rather than duplicated: `GuardCycleOutcome`
(`Completed`/`Blind`/`Failed`) and `DiscoveryOutcome` (`Resolved`/`NotFound`/`Ambiguous`) carry the same
distinction at the cycle and the binding level. What each change adds is the per-signal version.

### 5. No threshold without a measurement IN THE MECHANISM'S UNIT

Measurement alone is not enough. On 2026-08-09 both bad thresholds *were* measured:

- a CPU limit sized at 83x the average usage, when CFS throttles on bursts inside a 100 ms period — average
  cores is not the unit that governs it, so the headroom was real and irrelevant;
- a floor calibrated over three minutes when the quantity is a maximum and the unit is time coverage.

Correct arithmetic about the wrong quantity.

### 6. No new metric without proving the workload actually emits it

Queried, non-empty, on the pods the guard watches. Two channels have been bound to series that were
structurally incapable of moving.

## Proving it

### 7. Run a mutation that should break the test

If the test does not fail, the task is not finished.

Assert the mutation anchor matched **exactly once** and print the count. Refuse to start if the target
already differs from `HEAD` — a harness killed mid-run leaves the source mutated, and the next run reads
that as its baseline.

### 8. Both arms

Healthy quiet AND faulted loud, same population, peers as control. For cluster-side work the positive
fixture needs a live counterpart: **a fixture proves the code path, an injected fault proves the chain.**

**Both arms have to be capable of a verdict, and that is a separate check from running them.** On
2026-08-09 `PS-3` was recorded as a failed positive arm when the replay had run 2 cycles: the window a rule
sees holds one sample per cycle, `AnomalyGuardConfigReader.BuildRule` pins `MinimumSamples: 20`, and
`SustainedThresholdRule` returns `WarmingUp` below that — so the arm could not have produced a finding
whatever the cluster did. `MinBreachFraction` is then a share of `cycles * cadence` of wall clock, which is
what decides the shortest fault a given harness can see at all. The negative arm has the same disease and
it is harder to notice there, because a harness incapable of a verdict and a healthy cluster produce
identical output.

Say what the harness could have detected before reporting what it did.

### 9. Read the deployed state back out of the cluster

`kubectl apply` reports success for a field it dropped, and silently removes anything the file does not
carry — that deleted a live `GcCommittedBytes` binding and reported `configured`. Use
`lab.apply_and_read_back`.

## The report

Changed files; artefacts read; test results; **mutation result**; **silence risk** — how this specific
change could fail without anyone noticing; and known limitations.

The silence-risk line is the one that earns its place: everything else says what works.

### Self-improvement notes

Close with what cost iterations on THIS task, and what would have prevented it. Not a ritual and not an
apology — a specific, checkable observation, or the honest sentence that nothing went wrong. Every rule in
this file exists because a mistake was named this way; the ones that were not named repeated.

Three examples of the right shape, all from the days this was written:

> the fault was sized for a 200m limit and not recomputed when the limit moved to 1000m — one variable
> changed and the consequence was not propagated

> my own test observed exactly `MinimumWindows`, which is also the legacy fallback, so both sides read 24
> and the test could not fail

> the Python side of one arm converted its window with `mktime - timezone`, which is an hour off under DST,
> so it scanned a different window than the replay it was checking and the disagreement was recorded as a
> detector failure

Two things to be strict about:

- **A rule in a file is weaker than a gate in code.** On the day this was written, the only two things that
  actually caught anything were a parity test and a mutation harness, not paragraphs. When a note is worth
  keeping, say whether it can become a test rather than a sentence — and when it is a helper rather than a
  rule, it belongs in `.claude/lab.py`, written once.
- **An agent's own account of its run is evidence about its instructions, never evidence that its output is
  sound.** One reported "the instructions worked as intended" in a run where it had failed.

Closing the task is then checking the list from step 3, not forming a judgement.
