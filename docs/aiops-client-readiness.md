# The anomaly guard at a client site: what it sees, what it cannot, and how to install it

This is the document to read before deciding whether to deploy this at a customer, and the one to hand them
once you have. It says what has been measured, what has not, and what the guard is knowingly blind to.
Everything with a number behind it names where the number came from; everything without one says so.

---

## The short answer

**Deploy it in shadow mode, on a defined protocol. Do not arm it on day one.**

Detection is not the open problem. What is open is the false-positive rate at a site nobody has measured, and
the only honest way to find that out is to run quietly for a week and count.

---

## What it needs

One HTTP route to Prometheus. That is the entire dependency list: **no API-server access, no RBAC, no CRDs,
no operator, and nothing inside the customer's application.** It runs as one pod, configured from a ConfigMap,
and this claim is tested by running it that way rather than asserted.

One replica per monitored scope. Two would keep two independent incident trackers and page twice for one
problem; naive high availability here is worse than none.

A volume that outlives the pod. It holds open incidents (so a restart does not re-page for problems already
reported) and the learned state — the per-hour baseline and the calibrated floors, which take days to rebuild.
128 MiB is enough and does not grow with the cluster: the history keeps seven days per bucket and the
calibration keeps a capped sample per signal.

**A scrape interval that fits the window.** The detectors need 30 samples per replica, so
`window_minutes × 60 / scrape_seconds ≥ 30`. At a 60-second scrape a twenty-minute window yields 20 samples
and every verdict comes back `InsufficientData` — silence that reads exactly like health. Say this at install
time; it is not discoverable later without knowing to look.

---

## What it detects

Measured on a synthetic population with faults injected one at a time, with per-family attribution by
ablation (`DetectionMatrixDiagnostics`). Latency is time from the fault to the first cycle that named the
affected subject.

| Fault | Detected | Latency | Which family catches it alone |
|---|---|---|---|
| Memory leak, 5 MB/min, one replica | yes | 10 min | peer + trend |
| Memory leak, 20 MB/min, one replica | yes | 5 min | peer + trend |
| Memory leak, every replica together | yes | 5 min | trend + step |
| Latency 3× on one replica | yes | 0 min | peer + trend |
| Error rate 15% on one replica | yes | 0 min | peer + trend |
| CPU 2.5× on one replica | yes | 5 min | peer + trend |
| CPU 2.5× on **every** replica | yes | 0 min | **step only** |
| CPU throttling 30%, one replica | yes | 0 min | rules only |
| OOM kill, one replica | yes | 0 min | peer + rules |
| Crash-restart, one replica | yes | 0 min | peer + rules |

Two rows carry the architecture. **CPU rising on every replica at once is caught by the step detector and by
nothing else** — peer comparison has no outlier when everybody moves together. **Throttling is caught by the
rules family and by nothing else.** A guard with fewer families is not a simpler guard; it is a blind one.

A note on the "step only" row: a step is visible while it passes through the window and not afterwards. Once
both halves of a window sit at the new level there is nothing to compare, exactly as a finished ramp has no
slope. That is the shape of the fault, not a weakness — and it is why the incident tracker, which keeps the
incident open afterwards, matters.

---

## What it is blind to

Stated because an operator should be handed this list at install time rather than infer it from a month of
silence.

| Blind to | Why | Mitigation |
|---|---|---|
| A signal the cluster does not export | No query returns anything, and empty is indistinguishable from healthy | `overfit anomaly-discover` names every unbindable channel **before** deployment; the guard logs each blind channel every cycle and exports `overfit_guard_blind_metrics` |
| A single OOM kill, via peer comparison | One event yields a non-zero rate across ~10% of the window; Cliff's delta lands under the materiality gate | The rules family and the restart channel catch it — measured, and the two are deliberately separate signals |
| CPU throttling without a CPU limit | CFS throttling does not occur, so the metric is absent or zero | Nothing to do; say so rather than report a series of zeroes as health |
| Volume saturation | `kubelet_volume_stats_*` is unavailable in the development cluster, so this is untested rather than unsupported | Bind it at a site that exports it; treat as unverified until then |
| A workload's dependencies | There is no service-to-service model | Out of scope. The most common real root cause is a downstream dependency and this will not find it |
| A shift that finished before the window opened | Both halves sit at the new level | Caught while it passes through; the tracker keeps it open after |

---

## What has been measured, and what has not

**Measured.** Detection across ten fault shapes, with family attribution. Calibrated floors beating
hand-reasoned ones on a held-out population — 29 against 44 false incidents a day at identical detection.
A false-positive rate on a live twelve-replica cluster, moving from 0.56 to 0.25 incidents per cycle as the
floors and detectors were corrected, with quiet cycles going from **0 of 18 to 20 of 48**.

**Not measured, and these are the honest gaps.**

The lab is **nearly idle** — 0.93 requests per second per replica and CPU at 0.0012 of a core. That is the
hardest case for relative gates, so the numbers are more likely pessimistic than optimistic, but they are
**not representative** of a busy cluster and must not be quoted as if they were.

Everything is validated against **.NET workloads**. Container-level channels are stack-independent and cover
about half the signals; runtime channels are not. A JVM grows its heap to `-Xmx` by design, so a rising memory
trend is normal there until it plateaus; PHP-FPM has no long-lived heap at all and its memory **is**
load-sensitive, which the signal catalogue currently states globally rather than per workload type.

**A day-one figure for a customer's own cluster does not exist and cannot** — that is what the shadow week
produces.

---

## The install protocol

**Week 0 — coverage, before anything runs.** `overfit anomaly-discover --prometheus … --namespace …` proposes
a mapping from what the cluster actually exports and prints, per channel, whether it is bound, ambiguous, or
blind. Ambiguous channels are reported and **deliberately left out** of the generated configuration: whether a
4xx counts as an error is a business decision, and a guess there is a guard confidently measuring the wrong
thing. A human resolves those and reviews the file. "This guard will not see these four things" is a decision
to make knowingly.

**Week 1 — shadow, learning.** The guard runs, counts, explains and wakes nobody. It reports hourly what
floors its own observations imply, for every gate whose configured value sits below what a healthy period
produced. Nothing is applied automatically except where **no** floor was configured at all, because an absent
floor means the gate is off and that is the worst available default.

**Week 2 — thresholds accepted, still quiet.** A human accepts the proposed floors, adds the operational ones
the data cannot produce — a calibrated floor is a *noise* floor and says what to ignore, never what is worth
waking for — and the guard runs again. This week produces the incidents-per-day figure for **this** customer.

**Then, and only then, wire it to paging.**

Two things to configure before the first deployment lands: declare maintenance windows for planned rollouts,
and alert on the guard itself with `time() - overfit_guard_last_cycle_timestamp_seconds > 900`. A guard that
has stopped is worse than one that never started, because somebody is relying on it.

---

## Risks, in the order they will hurt

**1. Silent blindness from a mis-scaled threshold.** This is the top risk and it has already happened here: a
256 MiB memory floor measured on a population whose pods carry 1.23 GB was carried to a lab whose pods carry
43 MB. A floor six times larger than the whole signal cannot fire, so the gate meant to catch a memory leak
was switched off — and it looked exactly like health. The coverage report and the calibrator reduce this; they
do not eliminate it.

**2. Calibrating on a period that was not healthy.** A fault inside the observation window raises the floor
above that fault and blinds the guard to it at that size, permanently and quietly. There is no automatic
detection of this yet. Declare maintenance windows for anything known to be abnormal — they suppress learning
as well as reporting, which is the half that is easy to forget.

**3. The false-positive rate at this site is unknown until week 2.** Arming before then is how a tool gets
muted, and a muted tool detects nothing.

**4. One stack validated.** See above. A second stack is a day of work and removes the largest unknown in the
"works with any application" claim.

**5. Cluster events not yet exercised**: rollout, scale-up, scale-down, HPA, StatefulSets, leader/follower
cohorts. Declared peer cohorts exist and are tested in unit form; they have not met a real rollout.

**6. Restart duplicates without a persistent volume.** Known and bounded: one duplicate notification per open
incident. With the volume, none — and the learned state survives too, which matters more.

---

## What is deliberately not built

**Learned detector families** need training data a customer does not have on day one, and the statistical
families already answer the questions customers ask. **A service dependency graph** has the highest diagnostic
value on the list and is a separate product. **Anything requiring an agent inside the customer's application**
would trade away the strongest property this has: that the only dependency is an HTTP route to Prometheus.
