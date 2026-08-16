---
name: overfit-anomalies-lab-window
description: Open, screen and close a calibration window on the anomaly-guard lab — choose a window that is clean of faults AND flat in the quantity being calibrated, state the premise before measuring, and refuse any statistic computed across a known discontinuity. Use before fitting any floor or threshold from lab data. Most of its rules exist because they were broken while writing it.
model: opus
color: yellow
---

# A calibration window, and the four ways it lies

Pick a stretch of lab history, prove it is fit to calibrate the specific quantity you are calibrating, then
fit. Reading a number off a chart looks like the whole job; every rule below was earned on 2026-08-09/10 by
skipping one of these steps, several of them within an hour of writing the rule down.

## Why a clean window is not automatically a usable one

| Problem | Symptom | Consequence |
|---|---|---|
| Not flat in the calibrated quantity | Screens clean of faults; the quantity is drifting | The floor absorbs the drift and blinds the channel |
| Screen borrowed from another quantity | Window rejected (or accepted) on unrelated thresholds | A verdict about something you are not measuring |
| Statistic across a discontinuity | A slope whose sign depends on the window length | You measured the step, not the trend |
| Subject read without the population | Gap moves; the subject looks like it is leaking | The fleet moved, not the pod |
| Own injection still running | Population contains a deliberate fault | The floor is fitted to a poisoned reference |

## When to Use

- Before fitting **any** floor, threshold or gap from recorded or live lab data
- Before quoting a slope, a range or an interquartile spread from a lab window
- After a rollout, a suite run or an injection, before re-using a window opened earlier
- When a measured number disagrees with what the mechanism should produce

## When Not to Use

- Verifying a detector fires or stays quiet (use `overfit-anomalies-lab-two-arms`)
- Comparing repo against cluster (use `overfit-anomalies-lab-config-drift`)
- Anything that needs a **benchmark**. BenchmarkDotNet pins every core and takes a global mutex; it would
  perturb the very window you are calibrating. Ordinary builds do not — measured, workload pods sit at
  0.002 cores
- Choosing a **mechanism** rather than a value. A search fits parameters; it cannot invent the term that is
  missing (see `docs/autoresearch-program.md`)

## Inputs

| Input | Required | Description |
|---|---|---|
| The quantity | Yes | Exactly what is being calibrated, in its unit — a gap, a change in a gap, a ratio |
| The mechanism | Yes | What produces that number, and **what observation would refute the explanation** |
| Candidate window | Yes | Start and end in UTC, plus the ring length the fit will actually see |
| Population | Yes | The peers, so the subject is never read alone |
| Screens | Where used | Which quantity each screen speaks about — a screen is evidence only about itself |

## Workflow

### Step 1: State the premise out loud, before any query runs

Two lines: **what produces this number, in what unit, and what observation would refute the explanation.**
It is catchable from outside by somebody who has not read the code, which is the entire point — nobody can
challenge a premise that was never stated.

### Step 2: Open the window and record its arithmetic

```python
import sys
sys.path.insert(0, r"D:\Overfit\Scripts")
from lab import ensure_prometheus, prom_range, utc, guard_cycles, inject
```

Record the opening timestamp and the ring length in wall clock: `RetainedCyclesPerSeries × cadence` is the
horizon the fit sees, and at the shipped profiles (96 × 5 min) that is **eight hours**, not twenty minutes.

### Step 3: Require the window flat in the calibrated quantity — not merely free of faults

**This is the rule that nearly produced a useless number.** `AN-D1`'s floor is a *change in the peer gap*.
The candidate window screened clean of injected faults, and the gap inside it was moving at **−3.94 MB/h** —
about 31 MB over the ring it would have been fitted across, which would have put the floor near **39 MB** and
blinded the channel to any leak smaller than that, against a 9.52 MB detection floor. Measured correctly,
describing the wrong thing.

**Plot the calibrated quantity across the candidate window and require it flat, before fitting anything.**
Absence of a fault is a statement about the population; flatness is a statement about the number you are
about to set.

### Step 4: Treat every screen as evidence only about the quantity it screens

The CPU-pressure, active-requests, exceptions and lock-contention thresholds that choose a *window* are
inherited from a different measurement. They flagged an `AN-D1` window as contaminated; the memory gap was
then verified **undisturbed at every flagged timestamp** (48.4–49.2 MB, ±1 MB neighbourhood). Do not discard
a window on a screen that has nothing to do with what you are measuring — and do not accept one either.

### Step 5: Refuse any statistic computed across a discontinuity

Broken three times in one day, in three shapes:

- a slope fitted from a sample taken **before** the offset was injected reported **+26 MB/h**; refitted from
  after it settled, **−3.94**;
- slopes over the same data gave +2.06 MB/h across three hours, −4.42 over two, **−30.45 over one** and
  +0.61 over the last half — because a fleet-wide gen2 collection sat inside the longer windows. **A slope
  whose SIGN depends on the window length is not a trend, it is a phase.** Look at the shape; do not fit
  another line;
- a *range* computed across the injection step concluded the gap inherited an oscillation it demonstrably did
  not — 62.5 → 62.8 across the collection.

### Step 6: Read the population, not only the subject

On 2026-08-10 the peer gap fell steadily and the injected offset had not decayed at all: the target rose
+15.1 MB while the population median rose +23.7. **The gap narrowed because the fleet grew faster.** Reading
only the subject would have produced a confident and wrong story about a leaking fixture.

### Step 7: Re-screen before using the window, and clear your own faults first

Builds, suites and agents run between opening and using. And **do not calibrate a population that contains
your own injected fault** — a floor derived from healthy peer gaps is poisoned by one pod carrying a
deliberate offset.

## Validation

- [ ] Mechanism, unit and the refuting observation were written down **before** the first query
- [ ] The calibrated quantity was plotted across the window and is flat, not merely fault-free
- [ ] Every screen used to accept or reject the window speaks about the quantity being calibrated
- [ ] No statistic spans a known step, collection or injection
- [ ] The population was read alongside the subject
- [ ] The ring length the fit sees was computed, not assumed
- [ ] No injection of your own was live in the population

## Common Pitfalls

| Pitfall | Solution |
|---|---|
| `mktime(...) - timezone` for UTC | An hour out under DST; it made one arm scan a different window than the replay it was checking. Use `lab.utc()` (`calendar.timegm`) |
| Dead port-forward reads as "no series" | Every query returns `[]`, indistinguishable from no data. `lab.ensure_prometheus()` raises instead |
| Assuming the ring is the cadence | `RetainedCyclesPerSeries × cadence` — 8 hours at the shipped profiles |
| Fitting a second line to explain the first | A sign that flips with window length is a phase. Look at the shape |
| Running a benchmark during the window | It pins every core. Ordinary builds are fine (0.002 cores measured); a benchmark is not |
| Starting Grafana to look at the window | Keep discretionary consumers off the instrument. Prometheus yes, Grafana only on explicit instruction |
| Hand-tuning a value a search could settle | If the question is "what value" and the objective is cheap, deterministic and **fair**, let a search run it |
