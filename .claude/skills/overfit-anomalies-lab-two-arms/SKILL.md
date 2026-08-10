---
name: overfit-anomalies-lab-two-arms
description: Verify a detector with two arms — healthy quiet and faulted loud — after first proving each arm is CAPABLE of a verdict. Prove the mechanism can say Anomalous on a synthetic input of the intended shape BEFORE trusting any healthy-arm result, and quote the gates the input must clear. Use for any change to what the anomaly guard detects; a fixture proves the code path, an injected fault proves the chain.
model: opus
color: yellow
---

# Both arms, and each one capable of a verdict

Run a detector against a healthy population and a faulted one, and require the two to differ. Before
believing either, prove the mechanism was **able** to produce a verdict at all. This is step 8 of
[`docs/aiops/aiops-task-protocol.md`](../../../docs/aiops/aiops-task-protocol.md) with the mistakes attached.

## Why one arm proves nothing

**A detector that is structurally silent and a cluster that is genuinely healthy produce identical output.**
A working detector is quiet almost all the time, so every defect in this subsystem presents as silence.

| Problem | Symptom | Consequence |
|---|---|---|
| Mechanism cannot fire | Healthy arm quiet, and so is everything else | Reads as a clean bill; the channel is dead |
| Input never clears a gate | `InsufficientData` or `WarmingUp` every cycle | Verdict is structural, not observational |
| Positive fixture outside the band | Fires, but on a case another check already covers | Cannot distinguish the new signal from the old one |
| Replay too short | Silence | Silence by arithmetic, not by evidence |
| Fixture mistaken for a live arm | "It works" | The code path is proved; the chain is not |

## When to Use

- Any change to **what the guard detects**: a channel, a binding, a threshold, a rule, a detector
- Before implementing a design whose mechanism you have not shown can fire
- Before accepting a healthy-arm result as evidence of anything
- When a replay or a fixture run comes back silent and you are about to call it a pass

## When Not to Use

- Refactors and renames — nothing about detection changed
- Calibrating a floor or a threshold from lab data (use `overfit-anomalies-lab-window`)
- Checking the cluster matches the repo (use `overfit-anomalies-lab-config-drift`)
- Proving a *test* can fail rather than a *detector* (use `overfit-mutate`)

## Inputs

| Input | Required | Description |
|---|---|---|
| The mechanism | Yes | The detector method the design delegates to, read end to end, with its gates quoted |
| Negative arm | Yes | Healthy population — fixture or live — that must stay quiet |
| Positive arm | Yes | Faulted input **inside the band you claim to detect**, not at the extreme |
| Refuting arm | Where an explanation is claimed | An arm that would land differently if the explanation were wrong |
| Cadence × cycles | For replays | The wall clock the replay covers, checked against `MinimumSamples` |

## Workflow

### Step 1: Before writing code, ask whether the mechanism can fire at all

**Read the code the design delegates to, find the gates the intended input must clear, and quote them.** If
you cannot show the input clearing every gate, the design is refuted and the task stops there — a result
worth more than an implementation of something silent.

Two designs died this way on 2026-08-10, both in fifteen minutes of reading:

- one scalar per pod handed to `PeerGroupOutlierDetector` — `PeerOutlierOptions.Balanced` requires
  **30 samples per peer**, so every member is excluded and the verdict is `InsufficientData` every cycle,
  healthy or faulted;
- the obvious repair, the raw `up` series, fails a *different* gate: peer gaps are measured between
  **medians**, and the median of a 0/1 series is 1.0 for any pod above 50% coverage, so the verdict is
  `Healthy` until coverage halves — near-total silence, and already another check's job.

Both would have passed a healthy-arm-only acceptance test perfectly.

### Step 2: Put the positive arm inside the band you claim to detect

`RS-6`'s evidence was 0 of 6 successful scrapes — total silence, a case already covered elsewhere. A fixture
there cannot distinguish a new signal from the check that already exists. **Put the positive fixture in the
partial band** — for coverage that meant 60–70%, not 0%. The shipped fixture sits at 0.65.

### Step 3: Add an arm that would refute the explanation

Include an arm that lands differently if you are wrong, and check where it lands. In the `AN-F1` work the
refuting arm was a flat fleet: if the two references had disagreed there, the difference had nothing to do
with common-mode movement and the whole explanation was wrong. They agreed, so the +15% meant something.

### Step 4: Say which claim you have — fixture or chain

**A fixture proves the code path; an injected fault proves the chain.** Several channels in
`docs/aiops/aiops-coverage-map.md` are marked fixture-only for exactly this reason, and the map keeps them
separate on purpose.

For the live arm use the lab's own fault endpoints —
`POST /fault/{latency|stall|errors|leak|cpu|contend|throw|oom|crash|clear}` on a workload pod, via
`lab.inject`. The list is `Demo/LabWorkload/Program.cs:19-27`; check it there rather than here, because a
fault added to the workload and not to this line reads as "no such fault".

```python
import sys
sys.path.insert(0, r"D:\Overfit\Scripts")
from lab import inject, guard_pod, guard_cycles, replay_signals, workload_pods
```

### Step 5: Check the replay could have produced a verdict

The window a rule sees holds **one sample per cycle**, and `AnomalyGuardConfigReader.BuildRule` pins
`MinimumSamples: 20`. Below 20 cycles `SustainedThresholdRule` returns `WarmingUp` and **cannot** produce a
finding, whatever the data does — so `cycles × cadence` is the wall clock the replay covers, and
`MinBreachFraction` is a share of that. An 8-minute fault replayed at the deployed 300 s cadence is at best
2 breaching samples out of 20: silent by arithmetic. Replay a short fault at a short cadence
(`cadence=30, cycles=24` covers 12 minutes) and only then is silence evidence about the detector.

This sank two `PS-3` runs. The first was recorded as a detector failure, the second as a healthy negative
arm. Both were the harness.

### Step 6: Say what the harness could have detected, before reporting what it did

That sentence is the whole skill. If you cannot write it, you do not yet know what your result means.

## Validation

- [ ] Every gate the intended input must clear was **quoted from the code**, not assumed
- [ ] The positive arm sits in the band the change claims to detect, not at the extreme
- [ ] Each arm was shown *capable* of a verdict before its result was read
- [ ] Fixture and live arm are reported as the different claims they are
- [ ] For a replay, `cycles × cadence` was checked against `MinimumSamples` and the breach fraction
- [ ] The report states what the harness could have detected

## Common Pitfalls

| Pitfall | Solution |
|---|---|
| Healthy arm quiet, called a pass | Quiet is the expected output of a broken detector too. Prove capability first |
| Fault sized against a stale limit | A fault sized for a 200m CPU quota cannot throttle a 1-core one. One variable changed and the consequence was not propagated — re-read the limit |
| `clear` that times out leaves the fault running | A starved pod cannot answer in 15 s. Use generous timeouts and verify by reading state back, not by the POST returning |
| Reading the wrong guard pod | Right after a rollout the terminating pod is still listed and its log is empty, which reads as "no cycles". `lab.guard_pod()` returns the live one |
| Scanning a different window than the replay | `mktime - timezone` is an hour out under DST. Use `lab.utc()` |
| Empty query result read as a negative answer | `kubectl` reports a malformed query on stderr and returns empty stdout with exit 0. Assert non-empty before building on it |
| Positive arm at the extreme | 0% coverage is `RunSilentPods`' case, not the new channel's. Choose the partial band |
