# The anomaly guard at a client site: what it sees, what it cannot, and how to install it

This is the document to read before deciding whether to deploy this at a customer, and the one to hand them
once you have. It says what has been measured, what has not, and what the guard is knowingly blind to.
Everything with a number behind it names where the number came from; everything without one says so.

---

**For the engagement itself — what happens on which day, what they hand over, what comes out of each stage
and how each one is known to have worked — see `aiops-client-flow.md`.** This document is about capability;
that one is about a week.

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

### What one instance costs, measured

| | measured |
|---|---|
| memory, steady state | **130 MiB resident, 146 MiB peak** — over a 24-hour run, 292 cycles |
| memory, first minutes | 42 MiB — **do not quote this one**, it is a guard that has not accumulated anything yet |
| CPU | 1 m — thousandths of a core |
| Prometheus, per extra instance | +12 MiB, no measurable CPU change |
| request / limit to configure | 128 MiB / 512 MiB |

**One guard watches one namespace and one pod selector, so a fleet is N of these.** Fifty namespaces is
**≈7.3 GB and 50 m of CPU**, plus roughly 850 queries per five-minute cadence (~2.8/s) against the client's
Prometheus. The memory figure is measured; **the Prometheus figure at fifty is arithmetic from one extra
instance and should be checked against their Prometheus rather than assumed.**

Several scopes in one process is designed (`aiops-multi-scope-design.md`) and deliberately not built: an
instance per namespace costs no code, gives failure isolation at a process boundary and allows per-namespace
RBAC, and the saving would mostly be runtime overhead because the per-scope state dominates either way.

### The alerting, which is where this is most often installed wrong

Two independent ways to end up with a correct detector nobody hears from. Both were **measured on 2026-08-05**
by killing the guard on purpose and watching what arrived.

**1. The obvious alert expression cannot fire in the only case that matters.**

```promql
time() - overfit_guard_last_cycle_timestamp_seconds > 900     # WRONG
```

When the guard's pod is gone, its series is gone with it. The expression evaluates over an empty vector,
produces no result, and the alert returns to `inactive` — it reports "fine" precisely when the cluster has
stopped being watched. Measured: `pending` for 60 seconds while Prometheus still returned the last sample,
then `inactive` for the following six minutes, zero alerts in Alertmanager, zero notifications.

```promql
absent(overfit_guard_last_cycle_timestamp_seconds)
or (time() - overfit_guard_last_cycle_timestamp_seconds > 900)     # RIGHT
```

Same cluster, same guard, same receiver: **firing after 60 seconds, alert in Alertmanager, two notifications
delivered, none failed.** The wrong version passes review — it reads correctly and it does fire when the
guard is merely *slow*. Ship `k8s/lab/guard-alerts.yaml`, which also carries two further rules: failing
cycles, and durable state that cannot be written.

**2. kube-prometheus-stack routes everything to `receiver: "null"` by default.** A client who installs it
with defaults gets a correct rule, a correct alert, and delivery to nowhere. Check before believing any of
this works:

```
kubectl exec -n monitoring alertmanager-<release>-alertmanager-0 -c alertmanager -- \
  wget -qO- localhost:9093/api/v2/status | grep -A3 '"route"'
```

### The first day is noisier than every day after it

`SeasonalBaseline` compares a signal against the same phase of previous days, and it needs
`MinimumHistoryDays` before it can. A guard installed today has no previous day, so **the daily traffic curve
looks like a change on its first pass**. In the 24-hour run, two of five false incidents were exactly this:
the same signal rising 10.6% of typical at 19:03Z and falling 15.2% at 23:43Z — one phenomenon counted twice,
correctly described both times as deployment-wide movement with no replica implicated.

**Expect roughly one extra incident per diurnal slope in the client's traffic on day one**, and expect it to
stop on day two. Saying this in advance costs a paragraph; letting them discover it costs the pilot.

---

## What it detects

Measured on a synthetic population with faults injected one at a time, with per-family attribution by
ablation (`DetectionMatrixDiagnostics`). A fault counts as detected when a cycle named the affected subject
**on a channel the fault actually moved**; the affected channels are diffed out of the injection rather than
listed by hand. Latency is time from the fault to the first such cycle.

> **Re-measured with the signal check, then again after two floor defects were fixed. 2026-08-03.** The
> previous version counted any incident about the right pod as a detection and carried a warning saying so.
> The criterion is now strict, the affected channels are derived from the injector rather than listed by
> hand, and the strict criterion **agreed with the loose one on every row** — the worry that the table was
> crediting coincidences was unfounded, and both are still reported so the answer stays visible.
>
> The strict re-run put **both CPU rows at `no`**, which sent the investigation to the size gates and found
> two defects, both since fixed and both measured. See *Why the CPU rows used to read `no`* below.

| Fault | Detected | Latency | Which family catches it alone |
|---|---|---|---|
| Memory leak, 5 MB/min, one replica | yes | 10 min | peer + trend |
| Memory leak, 20 MB/min, one replica | yes | 5 min | peer + trend |
| Memory leak, every replica together | yes | 5 min | trend + step |
| Latency 3× on one replica | yes | 0 min | peer + trend |
| Error rate 15% on one replica | yes | 0 min | peer + trend |
| CPU 2.5× on one replica | yes | 5 min | trend |
| CPU 2.5× on **every** replica | yes | 0 min | **step only** |
| CPU throttling 30%, one replica | yes | 0 min | rules only |
| OOM kill, one replica | yes | 0 min | peer + rules |
| Crash-restart, one replica | yes | 0 min | peer + rules |
| **No fault at all (control)** | **no** | – | – |

**The control row is the one to read first.** Nothing is injected, and the table says `no`: the criterion is
not matching background noise, so every row above it means what it says. Over the same 69 cycles the control
opened **one** incident across twelve replicas — and the faulted runs opened two or three, the difference
being the fault itself.

**Two rows carry the architecture.** CPU rising on every replica at once is caught by the step detector and
by nothing else — peer comparison has no outlier when everybody moves together. Throttling is caught by the
rules family and by nothing else, because the CFS counters exist only on containers with a CPU limit, so a
peer group can hold exactly one member, on precisely the pod being throttled. A guard with fewer families is
not a simpler guard; it is a blind one.

A step is visible while it passes through the window and not afterwards: once both halves sit at the new
level there is nothing to compare, exactly as a finished ramp has no slope. That is the shape of the fault,
not a weakness, and it is why the incident tracker — which keeps the incident open afterwards — matters.

### Why the CPU rows used to read `no`

Two defects, stacked, and the second was hidden by the first.

**The step gate was calibrated on the wrong quantity.** It was fed `MinAbsoluteTrendChange`, which
`FloorCalibrator` accumulates from a Theil-Sen slope fitted to each pod **individually**, and applied it to a
step in the **cross-pod common component** — a median over twelve replicas, roughly √N less scattered. On CPU
the borrowed floor landed near 1.5× the signal's own level, so no step below 150% was reportable. Measured:
the step detector reported the cluster-wide rise at delta 0.47, p = 0.00017, and the gate threw it away at
0.39 against a floor of 0.81. The step gate now has its own accumulator, fed by the same function the gate
compares against, so the two cannot describe different quantities again.

**The calibrator learned from the window it was about to judge.** `Observe` ran at the top of the cycle and
invalidates the proposal cache, so every gate below read a floor that already contained that window. With the
proposal set from the maximum times a 1.25 margin, the floor was never below 1.25× the very quantity being
gated — so the gate could not fire, for any fault, once thirty samples existed. This survived because the
floor was calibrated on a *different* quantity from the one it gated, which is loose enough that the
inequality did not always hold; fixing that mismatch made the self-reference exact and therefore visible.
Calibration now runs after the detectors, which is what the code's own comment had claimed all along.

**What this cost in noise: nothing measurable.** The control row opened one incident before the fix and one
after. That is a single seed over 69 cycles, so it is not a false-positive *rate* — the rate comes from the
lab, and a lab re-run is owed before this is quoted as a noise figure at a client.

**One caveat on the attribution column.** `Options(trend: false)` ablates the trend family by setting
`MinimumSamples` to 100 000, and `FloorCalibrator` fits its own samples with **the same options object**, so
turning the trend family off also switches off every calibrated floor. Any row whose family is gated by a
calibrated floor is comparing configurations that differ in more than one thing, until the ablation is
separated from the calibration.

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

**The false-positive number to quote, and what it belongs to.** A 24-hour run on the live twelve-replica
cluster measured **5 incidents per day (95% Poisson 1–9)** across 292 cycles with **zero cycle failures** and
zero cycles overlapping a recorded build window. Read it as two results rather than one:

- Channels other than the heap went **4/day → 5/day, i.e. unchanged** — that was the acceptance question
  after a round of floor repairs and it passes.
- The heap channel went **108/day → 0**, and that is *not* clean credit for the repair: the calibrator was
  proposing a 0.729 MB floor against the 0.641 MB configured, meaning peer heap gaps peaked at 583 kB and sat
  just under the threshold, where in the older, longer-lived pods of the baseline run they sat well above it.
  **Do not quote 22× as the effect of a fix.**

On a synthetic twenty-replica population run through the same shipped code path, the untuned rate is
**5–15/day depending on the population** — three seeds at the lab's shape gave 11.7, 4.7 and 5.0, so a 2.5×
spread between draws of the same generator is normal and a single number is not a property of the product.

> An earlier figure of **250/day** circulated for the synthetic population and was wrong in a way worth
> recording: it came from a harness that hand-drives the detectors and **never constructs the guard** — no
> calibrated floors, no common-mode decomposition, no level-shift gate, no threshold rules. Running the same
> generator through the real cycle gave 15.1/day, a 16.5× difference produced by nothing but calling the
> shipped code.

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

Two things to configure before the first deployment lands.

Declare maintenance windows for planned rollouts.

And **alert on the guard itself, then prove the alert arrives by killing the guard.** Apply
`k8s/lab/guard-alerts.yaml`, scale the deployment to zero, and confirm three things in order: the rule is
loaded (`/api/v1/rules`), an alert appears in Alertmanager (`/api/v2/alerts`), and
`alertmanager_notifications_total` increments for the receiver. Then scale it back and watch the alert
resolve.

**Every one of those three has failed here.** The rule silently not loading, the alert going `inactive`
exactly when the guard disappears, and Alertmanager routing to `"null"` — all three look identical from the
outside, and all three look like a healthy cluster. A guard that has stopped is worse than one that never
started, because somebody is relying on it; an alert that cannot fire is worse still, because somebody is
relying on the alert.

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
