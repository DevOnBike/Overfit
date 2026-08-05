# The engagement, stage by stage

What happens at a client site, in order: what they hand over, what we do with it, what comes out, and how
each stage is known to have worked. Companion to `aiops-client-readiness.md`, which says what the guard can
and cannot see; this one says how a week actually goes.

## The direct answer first

**A shadow pilot on one workload can start today.** Nothing in the shadow path is unfinished, every piece of
it has been run on a live cluster, and the failure mode of getting it wrong is a log line nobody reads.

**Arming it — routing an incident to a human — should wait for two things**, both in flight as of
2026-08-02:

1. **A false-positive rate measured over a full day** on the current build. Iteration runs are four hours,
   and the number quoted to a client has to come from an observed day, because the load driver runs a real
   1440-minute curve and four hours is 17% of it. A run is in progress.
2. ~~The acknowledgement path reachable from outside the process.~~ **Done 2026-08-02** — CLI and HTTP
   endpoint, with the replay above showing a week of dismissals costs no detection.

One caveat that belongs in this section rather than buried: the detection table in
`aiops-client-readiness.md` comes from `DetectionMatrixDiagnostics`, whose sink matches a reported incident
on the **subject alone**. The replay harness above originally did the same and it inflated its results — an
unrelated incident on the right pod counted as detecting the fault. The matrix should be re-run with a signal
check before its table is quoted to anyone.

Everything below describes the finished flow and marks what is not there yet.

---

## Stage 0 — before anyone travels (one hour, remote)

**They provide**

| Thing | Why it decides something |
|---|---|
| A Prometheus URL the guard's pod can reach | The only dependency. No agent, no sidecar, no operator, no RBAC beyond reading a Service |
| Whether kube-state-metrics is installed | Without it there is no pod ownership, no roster, and no pod age — so no peer grouping, no silent-pod check and no warm-up rule. Everything else still works |
| One namespace and one pod regex | The scope of a single guard instance. Fifty namespaces means fifty instances today — see the limits |
| Which workload is the pilot | One Deployment or StatefulSet with **at least three replicas**. Below three there is no "rest of the group" and the peer family is undefined |
| A pod label naming peer cohorts, if any | `role`, `tier`, whatever they already publish. Optional; without it every replica compares against every other |

**We provide**: a go/no-go on the spot. Fewer than three replicas, or no Prometheus, is a "not this
workload" and it is cheaper to say so on a call than on site.

---

## Stage 1 — discovery (thirty minutes)

**We run** `overfit anomaly-discover` against their Prometheus. It reads what the cluster actually exports
and proposes a mapping from their metric names to the guard's thirteen channels.

**It produces** three lists, and the third is the one worth their attention:

- bound automatically — exact-name or suffix-shape matches (on our lab: 8 of 13 with no human input);
- needs a decision — several candidates, or an ambiguous shape (4 of 13);
- **blind** — nothing in this cluster exports it (1 of 13).

**They do**: confirm or correct the middle list. Ten minutes of a person who knows their stack.

**How you know it worked**: the guard logs `metric coverage: N of 13 known features mapped` at startup and
prints a `blind:` warning per unbound channel. A channel nobody binds is a channel the guard is blind on for
ever, and blindness is indistinguishable from health — so it is said out loud at every start.

---

## Stage 2 — shadow install (one hour)

**We deploy** four objects, all in their namespace: a ConfigMap with the mapping, a Deployment running one
pod, a PersistentVolumeClaim of 128 Mi for durable state, and a Service plus ServiceMonitor so Prometheus
scrapes the guard itself.

**Shadow means**: incidents go to the guard's log and to its own metrics endpoint, and nowhere else. Nobody
is paged. The route to a human is deliberately not connected yet.

**They do**: nothing. This is the point of a shadow week.

**How you know it worked**: `overfit_guard_cycles_total` climbing, and
`overfit_guard_last_cycle_timestamp_seconds` fresh. If the guard's own queries start failing it looks
exactly like a healthy cluster — that series is what makes the difference visible.

---

## Stage 3 — the quiet week (five to seven days, unattended)

The guard evaluates a twenty-minute window every five minutes and learns three things it cannot be told:

- **what normal difference between replicas looks like**, per signal, in that cluster's own units;
- **what this workload does at this hour**, per hour of day, over seven days;
- **which channels never report anything**, which is a coverage problem, not a health one.

**They do**: glance at the log once a day and tell us which entries are noise. That feedback is the input
the calibrator never had — its whole premise is that the observed period was healthy, and until an operator
says otherwise that premise is an assumption nobody can correct.

**What comes out**: hourly floor proposals in the log, in the signal's own units and in plain words —

```
Floor proposal for GcGen2HeapBytes (peer gap): healthy peers differed by up to 707832
(typical magnitude 3797024), so a gap below 884790 is something this cluster does when it is well.
```

**The one thing that can ruin this stage** is a real fault inside the observation window: the floor is then
set above the fault and the guard is permanently blind to it at that size. It is why the proposals are
suggestions for a human and never applied on their own, and why a week with a known incident in it should be
restarted rather than used.

**Measured, so the shape of the week is not a guess**: on our lab, calibrated floors took false incidents
from **124 a day to 29**, at identical detection, against 44 for hand-reasoned numbers.

---

## Stage 4 — arming (one hour, with them in the room)

**We do** three things:

1. **Copy the proposed floors into the ConfigMap.** An explicit value always wins over a calibrated one,
   even a lower one, so this is the moment the numbers stop being ours and become theirs.
2. **Add the operational floors only they can state.** A calibrated floor is a *noise* floor — it says what
   to ignore. It never says what is worth waking someone for. Measured on our lab, the calibrator proposed a
   latency floor of **0.01 ms**: arithmetically correct, and not a threshold anybody would act on. "We do
   not get up for less than fifty milliseconds" is a policy, and only they have it.
3. **Declare their deploy windows**, or wire their pipeline to declare them. A rollout *is* a level shift
   and the step detector will say so, correctly. Without a declared window the first deploy after install
   generates noise and spends the trust before the guard has caught anything.

**Two alerts must exist before the route to a human is connected**, and they are about the guard, not about
the cluster:

| Alert | Why it is not optional |
|---|---|
| `overfit_guard_last_cycle_timestamp_seconds` older than 15 minutes | A stopped guard reports no incidents, which is exactly what a healthy cluster looks like |
| `overfit_guard_suppressions_active` climbing without bound | Every response an operator can give makes the guard quieter. This is how a team notices it has silenced its way to a green dashboard |

---

## Stage 5 — steady state

**Per cycle** the guard evaluates five families over thirteen channels and groups what it finds. The
operator-facing number is incidents, not findings: on a live scale-up that produced 34 findings, they were
grouped into **3 incidents**, and one incident absorbed 16 related findings across 4 subjects.

**An incident carries**: the subject (namespace, workload, ReplicaSet, pod, node), the signal, an effect
size, the window it was observed in, a reason in the detector's own words, and a narrative naming what moved
with it.

**The operator's response** is one of three, and this is the loop the whole product rests on:

```
overfit anomaly ack <id> --noise --for 7d --reason "GC sawtooth on this replica"
    -> stops reporting that signal on that subject, with an expiry
    -> records that this window was healthy, which the calibrator folds in
    -> never touches the configured floor

overfit anomaly ack <id> --real --reason "this was the leak"
    -> pins the observation: no future floor proposal may silence it
```

**The `--real` half is not a nicety.** Every other mechanism makes the guard quieter, and a feedback loop
with one sign converges on a detector that reports nothing — gradually enough that nobody notices the day it
stopped working. That is the failure this product exists to remove, arriving through the feature meant to
build trust in it.

**Reachable as of 2026-08-02**, over the guard's own metrics port:

```
overfit anomaly-ack <id> --noise --for 7d --reason "..."   [--url http://guard:9469]
overfit anomaly-ack <id> --real  --reason "..."
overfit anomaly-suppressions
```

Three details that are decisions rather than defaults. It talks to the **running process**, not to the state
file — the guard rewrites that file from memory every cycle, so a command that edited it would lose the
operator's judgement within one cadence, silently. `--noise` and `--real` have **no default**: one silences a
signal and the other pins it, and guessing between them is not something to do quietly. And `/ack` is a
**POST**, because a GET that mutates gets replayed by every proxy and prefetch that ever saw the URL, and the
change it makes is silence.

**Measured, so the loop is not merely intended.** A simulated shadow week — 23 incidents dismissed, each with
a seven-day mute — was replayed against the injected-fault panel in three arms: a clean guard, one carrying
only the week's calibration, and one carrying the acknowledgements as well. Detection latency was
**identical in all three** for every fault the configuration catches. See
`Tests/Anomalies/Diagnostics/OperatorFeedbackRegressionDiagnostics.cs`.

---

## Stage 6 — the thirty-day review

Four numbers, all already exported:

| Question | Where it is answered |
|---|---|
| Is it still running? | `overfit_guard_cycles_total`, `overfit_guard_cycle_failures_total` |
| Is it still able to see? | `overfit_guard_blind_metrics`, and the `blind:` lines at startup |
| Are we silencing our way to quiet? | `overfit_guard_suppressions_active`, `overfit_guard_findings_muted_total` |
| Did it catch anything real? | `overfit_guard_labels_real` — how many findings a human confirmed |

The fourth is the one that decides whether the tool stays. It is also the only one that cannot be produced
by the tool alone, which is why the acknowledgement path matters more than any detector on the list.

---

## What we will not know until we are there

Stated plainly, because each of these is a thing a client can hit in the first month:

- **A rollout with a failing new version.** Every scaling event we have run was healthy. `CrashLoopBackOff`
  on a new ReplicaSet is the case where the silent-pod check finally has something to find, and it has never
  been shown one.
- **A second technology stack.** Everything known about signal semantics is known about .NET.
  `PeerSignalCatalog` is one static table for the whole world: PHP-FPM memory *is* load-sensitive and .NET's
  is not, and nginx has no GC at all.
- **More than one namespace per instance.** Fifty namespaces is fifty Deployments today. The history is
  already keyed by workload, so what is missing is a loop, not a structure.
- **Volume saturation.** Untested rather than unsupported — the development cluster does not export
  `kubelet_volume_stats_*`.

---

## When to stop the engagement

Three conditions, agreed before starting, so that stopping is a decision and not an argument:

1. **Coverage below eight bound channels.** Below that the guard is guessing about a cluster it mostly
   cannot see, and the honest move is to fix the exporters first.
2. **A confirmed miss of a fault the matrix says is detectable.** That is a defect, not a tuning problem,
   and tuning it would hide the defect.
3. **More than a handful of false incidents a day after arming.** The measured figure on our lab is 29 a day
   before the warm-up rule and the operator loop; if their cluster produces an order of magnitude more, the
   difference is a mechanism we have not modelled and no threshold will fix it.
