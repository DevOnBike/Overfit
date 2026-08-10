---
name: overfit-anomalies-lab-window
description: Open, screen and close a calibration window on the anomaly-guard lab — choose a window that is clean of faults AND flat in the quantity being calibrated, state the premise before measuring, and refuse any statistic computed across a known discontinuity. Use before fitting any floor or threshold from lab data. Most of its rules exist because they were broken while writing it.
---

# A calibration window, and the four ways it lies

Calibrating a threshold from lab data looks like reading a number off a chart. Every rule below was earned by
getting it wrong on 2026-08-09/10, several of them within an hour of writing the rule down.

## 1. State the premise before opening the window

Two lines, before any query runs: **what produces this number, in what unit, and what observation would
refute the explanation.** It is catchable from outside by somebody who has not read the code, which is the
entire point — nobody can challenge a premise that was never stated.

## 2. Absence of a fault is not enough — the window must be FLAT in what you are calibrating

**This is the rule that nearly produced a useless number.** `AN-D1`'s floor is a *change in the peer gap*.
The candidate window screened clean of injected faults, and the gap inside it was moving at **−3.94 MB/h** —
about 31 MB over the ring it would have been fitted across, which would have put the floor near **39 MB** and
blinded the channel to any leak smaller than that, against a 9.52 MB detection floor. Measured correctly,
describing the wrong thing.

**So plot the calibrated quantity across the candidate window and require it flat, before fitting anything.**
Absence of a fault is a statement about the population; flatness is a statement about the number you are
about to set.

## 3. A screen is evidence only about the quantity it screens

The CPU-pressure, active-requests, exceptions and lock-contention thresholds that choose a *window* are
inherited from a different measurement. They flagged an `AN-D1` window as contaminated; the memory gap was
then verified **undisturbed at every flagged timestamp** (48.4–49.2 MB, ±1 MB neighbourhood). Do not discard
a window on a screen that has nothing to do with what you are measuring — and do not accept one either.

## 4. Any statistic computed across a discontinuity describes the discontinuity

Broken three times in one day, in three different shapes:

- a slope fitted from a sample taken **before** the offset was injected reported **+26 MB/h**; refitted from
  after it settled, **−3.94**;
- slopes over the same data gave +2.06 MB/h across three hours, −4.42 over two, **−30.45 over one** and
  +0.61 over the last half — because a fleet-wide gen2 collection sat inside the longer windows. **A slope
  whose SIGN depends on the window length is not a trend, it is a phase.** Look at the shape; do not fit
  another line;
- a *range* computed across the injection step concluded the gap inherited an oscillation it demonstrably did
  not — 62.5 → 62.8 across the collection.

## 5. Watch what the population does, not only the subject

On 2026-08-10 the peer gap fell steadily and the injected offset had not decayed at all: the target rose
+15.1 MB while the population median rose +23.7. **The gap narrowed because the fleet grew faster.** Reading
only the subject would have produced a confident and wrong story about a leaking fixture.

## 6. Practicalities

```python
import sys
sys.path.insert(0, r"D:\Overfit\Scripts")
from lab import ensure_prometheus, prom_range, utc, guard_cycles, inject
```

- **`utc()`, never `mktime(...) - timezone`** — the latter is an hour out under DST, and it made one arm scan
  a different window than the replay it was checking.
- **`ensure_prometheus()`** — a dead port-forward makes every query return `[]`, which reads as "no series"
  rather than "no instrument". The helper raises instead.
- **Record the opening timestamp** and the ring length in wall clock: `RetainedCyclesPerSeries × cadence`
  is the horizon the fit sees, and at the shipped profiles that is 8 hours, not 20 minutes.
- **Re-screen before using the window**, not only when opening it. Builds, suites and agents run in between.
  Measured: ordinary builds do not perturb this lab (workload pods sit at 0.002 cores), **a benchmark
  would** — BenchmarkDotNet pins every core and takes a global mutex for exactly that reason.
- **Do not calibrate a population that contains your own injected fault.** A floor derived from healthy peer
  gaps is poisoned by one pod carrying a deliberate offset; that blocks `AN-C1` until the injection is gone.
