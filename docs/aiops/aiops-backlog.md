# Anomaly guard — backlog, scored

Compiled 2026-08-05, after a day that closed five items and opened three. Every row carries a **cost paid
today** or is marked as not having one — that column exists because the largest single waste of this project
so far was chasing `GcGen2HeapBytes`, which dominates the *historical* statistics and produced **zero**
incidents in the 24-hour run.

**Scoring.** ROI is value per unit of work, not value. Difficulty is work. Risk is the chance the change
breaks detection or blinds a channel — a threshold raise is never zero-risk, because the way it fails is
silence, and silence is what this subsystem exists to distinguish from health.

---

## A. Verification — no production code changes

| # | Task | Why now | ROI | Difficulty | Risk |
|---|---|---|---|---|---|
| A1 | **24-hour run on the current build and config** | The 5/day figure belongs to a configuration that no longer exists: five floors and two code paths changed on 2026-08-05. A number quoted to a client must belong to the thing shipped. Pass: rate inside 1–9/day, zero cycle failures, zero `RequestsPerSecond` incidents from the diurnal ramp. | **highest** | trivial (one day of clock, no attention) | none |
| A2 | ✅ **DONE 2026-08-05 — and it found two defects, one of them ours.** (1) The stack's default route is `receiver: "null"`: a correct alert delivered nowhere. (2) **`time() - overfit_guard_last_cycle_timestamp_seconds > 900` cannot fire when the guard is gone** — the series vanishes with the pod, the expression evaluates over an empty vector, and the alert returns to `inactive`. Measured: `pending` for 60 s, then `inactive` for six minutes, zero notifications. With `absent()` added: **`firing` after 60 s, alert in Alertmanager, two notifications delivered, none failed.** Shipped as `k8s/lab/guard-alerts.yaml`. | **highest** | low | low |
| A3 | ✅ **DONE 2026-08-05.** Second instance: **42 Mi, 1m CPU**, Prometheus +12 Mi and no CPU change. But 42 Mi is a guard eleven minutes old — the steady-state figure measured over the 24-hour run is **130 Mi, peak 146**. **Quote 146, not 42**: fifty namespaces is **7.3 GB**, not 2.1. The script's own arithmetic multiplied the fresh number and was wrong by 3.5x. Prometheus at 50x is extrapolation (≈2.8 queries/s), not measurement. | high | trivial | none |
| A4 | **Evening diurnal check** | Closes the last two unverified floors (`RequestsPerSecond` gap and trend), raised 2026-08-05 and verifiable only by waiting — the fault panel cannot move deployment-wide traffic. Already scheduled. | medium | zero (scheduled) | none |
| A5 | **Behaviour on StatefulSets and under HPA** | The lab is twelve identical stateless replicas. Peer comparison is structurally weakest exactly where a client is not: members with their own volumes and shards are not interchangeable, and HPA leaves ghost series and dilutes groups. Documented, never measured. | medium | medium | none |

## B. Client readiness — documentation, not code

| # | Task | Why now | ROI | Difficulty | Risk |
|---|---|---|---|---|---|
| B1 | ✅ **DONE 2026-08-05.** Folded into `aiops-client-readiness.md` rather than written as a second guide — two deployment documents are two versions of the truth. Added: measured cost per instance (**130 MiB steady state, not the 42 MiB a fresh one shows**) and the fleet arithmetic (**7.3 GB for fifty namespaces**, Prometheus at that scale marked as arithmetic not measurement); **both alerting traps with their measurements** — the expression that cannot fire when the guard is gone, and the `"null"` default route; the **first-day caveat** with the mechanism and the expected size (one extra incident per diurnal slope); the **5/day figure read as two results**, with the heap channel's 108 → 0 explicitly denied as credit for the fix; and the 250/day correction. The install protocol now ends by **killing the guard to prove the alert arrives**, because all three links in that chain have failed here. | **highest** | low | none |

## C. Tuning — configuration only

| # | Task | Why now | ROI | Difficulty | Risk |
|---|---|---|---|---|---|
| C1 | **Re-tune the heap floor on an aged population** | `GcGen2HeapBytes.minGap` was set from a population whose pods were hours old; three rollouts on 2026-08-05 reset that. The proposal grew 0.729 → 0.930 MB across one day as heaps diverged, so the current number is extrapolation. | medium | trivial (wait, read the proposal, edit) | **medium** — a floor raised past a real gap blinds the channel silently |
| C2 | **`GcPauseRatio.minTrendChange`** | Proposed at 0.0000345 and left unset: one finding is not evidence enough to arm a gate that has never misbehaved. Revisit only if it starts producing findings. | low | trivial | low |

## D. Detection gaps — real, measured, unfixed

| # | Task | Why now | ROI | Difficulty | Risk |
|---|---|---|---|---|---|
| D1 | **DIAGNOSED 2026-08-06 — the peer family reports a fixed pod property as a recurring anomaly.** Neither option below was right; see the section under this table. | medium | medium — needs a mechanism, not a number | low |
| D2 | **Peer is structurally blind to a single OOMKill** | Measured. One replica dying is invisible to a family that compares replicas against each other, and OOM is among the most common real failures. Needs a mechanism, not a threshold. | medium | high | medium |
| D3 | **A CPU rise on every replica at once is invisible to all four families** | Measured 2026-08-01 and never diagnosed to the end: peer is blind by construction, no threshold rule covers CPU, and a step that has finished has no slope left to fit. | medium | high | medium |
| D4 | **The heap oscillates with a period near the evaluation window** | Measured 2026-08-05: gen2 swings 3.31 MB inside a 20-minute window while replicas differ by 0.29 MB, and all twelve oscillate **in phase**, so the trend family sees a 1.2–1.5 MB alternation that is window alignment, not the cluster. Same class of error as the 240-minute window sitting on the daily slope, in the opposite direction. **No cost paid today — the channel produced zero incidents in the 24-hour run**, so this is a latent defect, not a live one. | low **now** | high | medium |

### D1, diagnosed 2026-08-06 — a persistent outlier is not an anomaly

The row above offered two explanations: the guard is right and those pods deserve attention, or the peer gate
needs the `MinRelativeGap` treatment `CpuUsageRatio` got. **Measured, and it is neither.**

Twenty-four hours of `container_memory_working_set_bytes` for `lab-workload-*`, split by ReplicaSet
generation — the split matters, because a 24-hour window spans every generation the deployment has had and
ranking pods across a rollout compares pods that never coexisted:

| generation | peer gap (heaviest − median) | clears the 9.52 MB floor | top-2 set changed |
|---|--:|--:|--:|
| `7f5fb9f88c` (the population the row was written about) | median **18.8 MB**, p90 20.4 | **34 / 65 samples (52%)** | **0 of 64 transitions** |
| `7765564ff6` (current) | median **5.9 MB**, max 9.3 | **0 / 108 samples** | 4 of 107 (4%) |

Two things follow, and the second is the finding.

**The symptom is generation-dependent, so the row's premise has expired.** Same workload, same config, same
floor: one generation sits at ~19 MB of spread and reports in half its cycles, the next sits at ~6 MB and is
silent in all 108 samples. Whatever makes two replicas heavy is assigned when the pods start and differs from
rollout to rollout. Re-checking a symptom before fixing it is why this took an hour instead of a day.

**In both generations the same two pods are the heaviest essentially always** — 100% and 96% of samples, with
the top-2 set changing 0 and 4 times respectively. That is not an event. **A pod that has been 19 MB heavier
than its peers since its first cycle, in every cycle, for hours, is not anomalous — it is that pod's
baseline.** The peer family has no notion of *novelty*: it re-derives the ranking every cycle from scratch and
so re-reports a fixed configuration difference forever.

**This is a missing mechanism, not a wrong number, and the trade proves it.** Raising the floor above 19 MB
would silence the current noise — and would simultaneously blind the channel to a genuine 50% memory leak on
a 40 MB baseline, which is exactly what the channel exists to catch. A parameter that cannot be moved without
buying silence at the price of detection is the signature of a degree of freedom that is absent, the same
diagnosis pattern recorded in `CLAUDE.md` for the synthetic generator's queueing term.

The mechanism the family needs is a **per-pod offset learned over the pod's own history**: judge a replica
against its peers *after* subtracting the gap it has held since it started, so that a stable difference is
learned once and only a *change* in that difference reports. That is a design task, not a tuning task, and it
is shared with D2 and D3 — all three are the peer family lacking a notion the threshold cannot express.

## E. Deferred by decision

| # | Task | Why deferred | ROI | Difficulty | Risk |
|---|---|---|---|---|---|
| E1 | **Multi-scope, slices 3–5** | One instance per namespace costs 130 MB and zero code, gives free failure isolation at a process boundary and allows per-namespace RBAC; fifty templated Deployments is ordinary Kubernetes practice. The memory saving is small because per-scope state dominates either way. **Build it when a client says the fleet is unmanageable**, not before. Slices 1–2 are done, behaviour-neutral and waiting. | low until asked | high | **high** — shared id counter, shared state file, and the design's own warning that `--real` labels must not cross scopes |
| E2 | **A separate CPU-limited deployment so the throttle channel is testable** | `container_cpu_cfs_throttled_periods_total` exists only on containers with a CPU limit, and the lab deliberately has none so that every CPU measurement has one explanation. This is test coverage, not a feature; its value appears the first time the channel misses something at a client. | low | medium | low |
| E3 | **Rebuild the guard image for the corrected startup message and the scope plumbing** | Cosmetic and already in the tree; ships free with the next deploy that has another reason to happen. | low | trivial | low |

---

## Order

**Everything that disturbs the lab happens BEFORE the run, not around it.**

1. **A3** — second-instance cost. Adds a Deployment, torn down after.
2. **A2** — alert delivery. Kills the guard on purpose.
3. **A1** — start the 24-hour run, then touch nothing for a day.
4. **B1** while the run goes — the paragraph that keeps a pilot from being surprised. Costs no cluster time.
5. **A5** after the run — StatefulSet and HPA are two separate experiments and each rewrites the workload.
6. Then **D1**, the only detection question with a cost being paid every cycle right now.

**This ordering is a correction, and the correction is the point.** The first version of this file put A2
*during* the run and A3 *after* it, reasoning that A2 "touches only the guard". That is true and irrelevant:
A2 kills the guard, which breaks the run's cycle continuity, and the run's whole product is an uninterrupted
count of cycles and incidents. A3 adds a second guard watching the same pods, which double-counts
everything while it lives. A5 replaces the workload. **Each of the three invalidates the measurement in a
different way, and none of them does it by touching the pods** — which is why "what does it touch" was the
wrong question and "what does it invalidate" is the right one.

Everything else waits for evidence that somebody is paying for it.
