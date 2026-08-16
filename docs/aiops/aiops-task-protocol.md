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

### 5b. A calibration window must be flat in the quantity being calibrated

Choosing a window for the absence of injected faults is not enough, and on 2026-08-10 that nearly produced a
useless number. `AN-D1`'s floor is a **change in the peer gap**; the window screened clean of faults, and the
gap inside it was moving at −3.94 MB/h — about 31 MB over the ring it would be fitted across, which would
have put the floor near 39 MB and blinded the channel to any leak smaller than that, against a 9.52 MB
detection floor. Measured correctly, describing the wrong thing.

**So: plot the calibrated quantity across the candidate window and require it flat, before fitting anything.**
Absence of a fault is a statement about the population; flatness is a statement about the number you are
about to set.

Two mistakes from the same hour, both cheap to repeat:

- **Do not screen a window with thresholds borrowed from a different measurement.** The CPU and latency
  screens that chose this window flagged it contaminated; the memory gap was then verified undisturbed at
  every flagged timestamp. A screen is only evidence about the quantity it screens.
- **A trend fitted across a step measures the step.** Regressing the gap from a sample taken before the
  offset was injected reported +26 MB/h; refitting from after it settled gave −3.94. Start the fit where the
  thing you are studying starts.

  **Broken again the same day, twice, which is why it is written this emphatically.** A slope over the whole
  window said +2.06 MB/h, over the last two hours −4.42, over the last hour −30.45 and over the last half
  +0.61 — because a fleet-wide gen2 collection sat inside the longer windows. **A slope whose SIGN depends on
  the window length is not a trend, it is a phase**, and the answer is to look at the shape, not to fit
  another line. The same run then computed a range across the injection step and concluded the gap inherited
  an oscillation it demonstrably did not (62.5 → 62.8 across the collection). Any statistic — slope, range,
  maximum — computed across a known discontinuity describes the discontinuity.

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

**A test whose fixture does not contain its subject passes for the wrong reason, and only a mutation finds
it.** On 2026-08-10 a test claiming "an unjudgeable cycle must not clear the counter" was green under the
mutation that removes exactly that behaviour — because the pod under test was **absent from the cycle being
tested**, so its counter was untouched whether the code existed or not. The code was right; the oracle was
empty. **When a test asserts that state carries across an event, assert the subject is present in the
event.** That is one line, and it is not covered by "prove the test can fail" — the test could fail, for a
different reason than its own description claimed.

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

**The sharpest version of this rule, and it has now caught three separate things: prove the mechanism can
say `Anomalous` on a synthetic input of the intended shape BEFORE you trust any healthy-arm result.** A
detector that is structurally silent and a cluster that is genuinely healthy produce identical output, so a
healthy arm alone is compatible with a completely dead mechanism.

On 2026-08-10 an `AN-D9` design passed every architecture check and was refuted by two lines in the file it
delegated to: `PeerOutlierOptions.Balanced` requires **30 samples per peer**, and the plan fed the detector
**one scalar per pod** — so every member was excluded and the verdict was `InsufficientData` every cycle,
healthy or faulted. The obvious repair failed on a different gate: peer gaps are measured between **medians**,
and the median of a 0/1 series is 1.0 for any pod above 50% coverage, so the verdict became `Healthy`. Both
would have sailed through a healthy-arm acceptance test.

So, concretely, before writing the code: **read the code the plan delegates to, find the gates the input has
to clear, and quote them.** If you cannot show the intended input clearing every gate, the design is refuted
and the task stops there — that is a result, not a setback.

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
  rule, it belongs in `Scripts/lab.py`, written once.
- **An agent's own account of its run is evidence about its instructions, never evidence that its output is
  sound.** One reported "the instructions worked as intended" in a run where it had failed.

Closing the task is then checking the list from step 3, not forming a judgement.
