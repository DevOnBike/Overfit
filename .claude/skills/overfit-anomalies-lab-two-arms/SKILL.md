---
name: overfit-anomalies-lab-two-arms
description: Verify a detector with two arms — healthy quiet and faulted loud — after first proving each arm is CAPABLE of a verdict. Prove the mechanism can say Anomalous on a synthetic input of the intended shape BEFORE trusting any healthy-arm result, and quote the gates the input must clear. Use for any change to what the anomaly guard detects; a fixture proves the code path, an injected fault proves the chain.
model: opus
color: yellow
---

# Both arms, and each one capable of a verdict

**A detector that is structurally silent and a cluster that is genuinely healthy produce identical output.**
That is the failure mode this whole subsystem is built around, and it is why a healthy arm alone proves
nothing. This skill is step 8 of [`docs/aiops/aiops-task-protocol.md`](../../../docs/aiops/aiops-task-protocol.md)
with the mistakes attached.

## 1. Before writing code: can the mechanism fire at all?

**Read the code the design delegates to, find the gates the intended input must clear, and quote them.** If
you cannot show the input clearing every gate, the design is refuted and the task stops there — that is a
result worth more than an implementation of something silent.

Two designs died this way on 2026-08-10, both in fifteen minutes of reading:

- one scalar per pod handed to `PeerGroupOutlierDetector` — `PeerOutlierOptions.Balanced` requires
  **30 samples per peer**, so every member is excluded and the verdict is `InsufficientData` every cycle,
  healthy or faulted;
- the obvious repair, the raw `up` series, fails a *different* gate: peer gaps are measured between
  **medians**, and the median of a 0/1 series is 1.0 for any pod above 50% coverage, so the verdict is
  `Healthy` until coverage halves — which is near-total silence and already another check's job.

Both would have passed a healthy-arm-only acceptance test perfectly.

## 2. The positive arm has to sit in the band you claim to detect

`RS-6`'s evidence was 0 of 6 successful scrapes — total silence, a case already covered elsewhere. A fixture
there cannot distinguish a new signal from the check that already exists. **Put the positive fixture in the
partial band** — for coverage that meant 60–70%, not 0%.

## 3. Prove the arms differ for the reason you think

Include an arm that would **refute** the explanation, and check it lands where it must. In the `AN-F1` work
the refuting arm was a flat fleet: if the two references disagreed there, the difference had nothing to do
with common-mode movement and the whole explanation was wrong. They agreed, so the +15% result meant
something.

## 4. Fixture and live arm are different claims

**A fixture proves the code path; an injected fault proves the chain.** Say which you have. Several channels
in `docs/aiops/aiops-coverage-map.md` are marked fixture-only for exactly this reason, and the map keeps them
separate on purpose.

For the live arm use the lab's own fault endpoints — `POST /fault/{latency|stall|errors|leak|cpu|contend|throw|oom|crash|clear}`
on a workload pod, via `lab.inject`. The list is `Demo/LabWorkload/Program.cs:19-27`; check it there rather
than here, because a fault added to the workload and not to this line reads as "no such fault". Two things
learned the hard way:

- **size the fault against the CURRENT limit.** A fault sized for a 200m CPU quota cannot throttle a 1-core
  one; one variable changed and the consequence was not propagated.
- **a `clear` that times out leaves the fault running.** A starved pod cannot answer in 15 seconds. Use
  generous timeouts and verify the fault cleared by reading state back, not by the POST returning.

## 5. Replay arms must be capable too

The window a rule sees holds **one sample per cycle**, and `AnomalyGuardConfigReader.BuildRule` pins
`MinimumSamples: 20`. Below 20 cycles `SustainedThresholdRule` returns `WarmingUp` and **cannot** produce a
finding, whatever the data does — so `cycles × cadence` is the wall clock the replay covers, and
`MinBreachFraction` is a share of that. An 8-minute fault replayed at the deployed 300 s cadence is at best
2 breaching samples out of 20: silent by arithmetic. Replay a short fault at a short cadence
(`cadence=30, cycles=24` covers 12 minutes) and only then is silence evidence about the detector.

This sank two `PS-3` runs. The first was recorded as a detector failure; the second as a healthy negative
arm. Both were the harness.

## 6. Say what the harness could have detected, before reporting what it did

That sentence is the whole skill. If you cannot write it, you do not yet know what your result means.
