# Operator feedback — design

The last blocker on the must-have list that is a **product decision rather than an implementation**. Written
2026-08-02, before any code, because the three plausible meanings of one button produce three different
products and picking after the fact is expensive.

## The problem this exists to solve

Calibration is one-shot: `FloorCalibrator` watches a period believed to be healthy and proposes a floor above
what it saw. Everything after that is fixed until a human edits a ConfigMap. So when the guard reports
something the operator knows is routine, the only responses available today are:

1. edit the ConfigMap and restart — minutes of work, requires understanding the threshold model, and the
   operator has to invent a number;
2. ignore the alert;
3. mute the tool.

Option 3 is how products like this die, and it is the *rational* choice when 1 is expensive and 2 is
unbearable. A feedback path exists to make 1 cost one keystroke.

## What "this was noise" could mean — three different products

The button is easy. Its semantics are the decision.

### A. Raise the floor for that signal

Fold the reported magnitude into the floor: this incident's peer gap becomes the new bar. Cheap, global,
and it inherits the hazard the calibrator already documents in its own summary — **a floor set above a fault
blinds the guard to that fault at that size, permanently and silently**. Worse than the calibrator's version,
because the calibrator at least fits a maximum over hundreds of windows; one operator click is a sample of
one. A single mislabelled incident moves a production threshold.

### B. Suppression rule for that subject and signal

"This replica's GC pause, muted." Narrow, immediate, and exactly what the operator wants at 3 a.m. Its
failure is equally specific: a mute with no expiry becomes permanent, and the pod most likely to be muted is
the one that is genuinely misbehaving. This is the shape `MaintenanceWindow` already implements for a time
range, so most of the machinery exists — `IMaintenanceCalendar` is the seam, and a suppression is a window
scoped to a subject rather than to a clock.

### C. A label that feeds the calibrator

The incident's window is recorded as known-good and folded into calibration as a healthy observation. This
is the principled one: the calibrator's entire premise is *"the observed period was healthy"*, and that
premise is currently an assumption nobody can correct. Labels make it an input. Its weakness is latency — one
label barely moves a maximum taken over hundreds of windows, so the operator presses the button and nothing
visible happens, which reads as a broken feature.

## They are not alternatives — they are different time constants

- **B** is the response the operator needs in seconds.
- **C** is the mechanism that makes the threshold correct in days.
- **A** is what **C** eventually produces, but as a proposal a human accepts, never as an automatic write.

The recommended shape is therefore one button with two effects and one prohibition:

```
overfit anomaly ack <incident-id> --noise [--for 7d] [--reason "GC sawtooth on this replica"]

  1. opens a bounded suppression for (subject, signal) with an expiry     -> immediate relief   (B)
  2. records a label marking that window healthy for the calibrator       -> correct threshold  (C)
  3. never edits a configured floor                                       -> the prohibition    (A)
```

## The part that is easy to get wrong: a feedback loop with one sign converges to silence

Every mechanism above makes the guard quieter. Nothing makes it louder. A system that can only learn
"ignore" ends deaf, and it gets there gradually enough that nobody notices the day it stopped working — the
same pathology this whole subsystem exists to eliminate, arriving through the feature meant to build trust.

So the inverse label is not a nice-to-have, it is what makes the rest safe:

```
overfit anomaly ack <incident-id> --real [--reason "this was the leak"]
```

A `--real` label pins the window as one the guard **must** keep detecting. Concretely it becomes a regression
case: any proposed floor that would have silenced a `--real` window is rejected, and the proposal says so
rather than quietly clipping itself. That single rule is what stops a hundred `--noise` clicks from producing
a guard that reports nothing.

## What has to be visible

A mute nobody can see is indistinguishable from a detector that works. Whatever ships must expose:

- `overfit_guard_suppressions_active` and `overfit_guard_labels_total{kind="noise|real"}` on the existing
  `GuardTelemetry` endpoint;
- a listing (`overfit anomaly suppressions`) naming every active suppression, its subject, its signal, who
  set it and when it expires;
- expiry as the default, not an option. A suppression with no end date is a configuration change wearing the
  clothes of an acknowledgement.

## Where it hooks into what already exists

| Need | Existing seam |
|---|---|
| Stable incident identity to acknowledge | `PersistedIncident.Id` — durable across restarts, and the identifier-reuse hole was closed on 2026-08-02 (defect D), which this feature depends on: acknowledging a recycled id would acknowledge the wrong incident |
| Suppression evaluated per cycle | `IMaintenanceCalendar.IsDeclaredAbnormal` — same call site, extended from (time, workload) to (time, subject, signal) |
| Labels persisted | `ILearnedStateStore` — the payload is already sectioned (`### history`, `### calibration`), so `### labels` is additive and old files still read |
| Floor proposals that respect labels | `FloorCalibrator.Propose` — the `--real` regression rule belongs here, where the maximum is already computed |
| Reporting a suppressed row rather than dropping it | `IncidentLogRecord.SuppressedBy` — already carried, already non-empty on a suppressed row |

## What would need measuring afterwards

Not "did false positives fall" — they will, by construction, and measuring it would be circular. The
question worth an experiment is **whether detection survives**: replay the injected-fault fixtures against a
guard whose calibrator has absorbed a week of `--noise` labels, and check that all ten fault shapes are still
caught. If they are not, the label weighting is wrong, and it is better to learn that on a fixture than at a
client.

## Decided 2026-08-02 — B + C, with A prohibited

The button does the suppression **and** the label, and never writes a threshold. Recorded here rather than
left in a conversation, because the reasoning is what a future reader will need when someone proposes the
obvious simplification of dropping one half:

- **Dropping the label (B alone)** buys a year of accumulated mutes instead of a correct calibration, and the
  pod that eventually breaks is the one that has been on the mute list for months.
- **Dropping the suppression (C alone)** is conceptually the cleanest and reads as a broken button: one label
  barely moves a maximum taken over hundreds of windows, so the same alert arrives again five minutes later
  and the operator concludes the feature does nothing.
- **A stays prohibited** because a production threshold moved by a sample of one is how a guard goes blind to
  a real fault at exactly the size somebody once dismissed.

### Build order

1. `--real` and the regression rule in `FloorCalibrator.Propose` **first**. It is the smallest piece and it
   is the safety property everything else leans on; building the quieting half first means shipping a
   one-signed feedback loop, even briefly.
2. The label store (`### labels` section in `LearnedState`) and its effect on calibration.
3. Subject-scoped suppression through `IMaintenanceCalendar`, with a mandatory expiry.
4. `overfit anomaly ack` / `overfit anomaly suppressions`, and the two telemetry series.
5. The fixture replay: ten injected fault shapes still caught after a week of `--noise` labels.
