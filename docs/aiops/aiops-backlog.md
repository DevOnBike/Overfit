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
| A2 | **Prove an alert actually arrives** | `overfit_guard_last_cycle_timestamp_seconds` going stale is the one alert every deployment needs, and **it has never notified anyone**. Write the rule, wire the Alertmanager route, kill the guard, confirm delivery and recovery. An alert written against a series that never arrives is itself silence that looks like health. | **highest** | low (Alertmanager already deployed) | low — touches the guard only |
| A3 | **Cost of a second instance** | The fleet answer (one guard per namespace) rests on multiplying 130 MB by N *on paper*. One extra instance turns extrapolation into measurement, and fleet cost is the first question a client asks. Pass: under 150 MB, no visible load on Prometheus. | high | trivial (20 minutes) | none |
| A4 | **Evening diurnal check** | Closes the last two unverified floors (`RequestsPerSecond` gap and trend), raised 2026-08-05 and verifiable only by waiting — the fault panel cannot move deployment-wide traffic. Already scheduled. | medium | zero (scheduled) | none |
| A5 | **Behaviour on StatefulSets and under HPA** | The lab is twelve identical stateless replicas. Peer comparison is structurally weakest exactly where a client is not: members with their own volumes and shards are not interchangeable, and HPA leaves ghost series and dilutes groups. Documented, never measured. | medium | medium | none |

## B. Client readiness — documentation, not code

| # | Task | Why now | ROI | Difficulty | Risk |
|---|---|---|---|---|---|
| B1 | **Deployment guide: resource requests, the first-day caveat, the known blind spots** | The first day after deployment has a structurally raised false-positive rate — `SeasonalBaseline` needs `MinimumHistoryDays` and a fresh guard has no previous day. Two of five incidents in the 24-hour run were the two slopes of one diurnal curve. Telling a client in advance costs a paragraph; letting them find it costs the pilot. | **highest** | low | none |

## C. Tuning — configuration only

| # | Task | Why now | ROI | Difficulty | Risk |
|---|---|---|---|---|---|
| C1 | **Re-tune the heap floor on an aged population** | `GcGen2HeapBytes.minGap` was set from a population whose pods were hours old; three rollouts on 2026-08-05 reset that. The proposal grew 0.729 → 0.930 MB across one day as heaps diverged, so the current number is extrapolation. | medium | trivial (wait, read the proposal, edit) | **medium** — a floor raised past a real gap blinds the channel silently |
| C2 | **`GcPauseRatio.minTrendChange`** | Proposed at 0.0000345 and left unset: one finding is not evidence enough to arm a gate that has never misbehaved. Revisit only if it starts producing findings. | low | trivial | low |

## D. Detection gaps — real, measured, unfixed

| # | Task | Why now | ROI | Difficulty | Risk |
|---|---|---|---|---|---|
| D1 | **`MemoryWorkingSetBytes` peer reports two replicas every cycle** | Two of twelve hold ~60 MB against ~40 MB and are named in every cycle, while the calibrator considers the 9.52 MB floor adequate. Either the guard is right and those pods deserve attention, or the peer gate needs the treatment `CpuUsageRatio` got. **Nobody has determined which**, and that is the reason to look. | medium | medium — diagnosis before any change | low |
| D2 | **Peer is structurally blind to a single OOMKill** | Measured. One replica dying is invisible to a family that compares replicas against each other, and OOM is among the most common real failures. Needs a mechanism, not a threshold. | medium | high | medium |
| D3 | **A CPU rise on every replica at once is invisible to all four families** | Measured 2026-08-01 and never diagnosed to the end: peer is blind by construction, no threshold rule covers CPU, and a step that has finished has no slope left to fit. | medium | high | medium |
| D4 | **The heap oscillates with a period near the evaluation window** | Measured 2026-08-05: gen2 swings 3.31 MB inside a 20-minute window while replicas differ by 0.29 MB, and all twelve oscillate **in phase**, so the trend family sees a 1.2–1.5 MB alternation that is window alignment, not the cluster. Same class of error as the 240-minute window sitting on the daily slope, in the opposite direction. **No cost paid today — the channel produced zero incidents in the 24-hour run**, so this is a latent defect, not a live one. | low **now** | high | medium |

## E. Deferred by decision

| # | Task | Why deferred | ROI | Difficulty | Risk |
|---|---|---|---|---|---|
| E1 | **Multi-scope, slices 3–5** | One instance per namespace costs 130 MB and zero code, gives free failure isolation at a process boundary and allows per-namespace RBAC; fifty templated Deployments is ordinary Kubernetes practice. The memory saving is small because per-scope state dominates either way. **Build it when a client says the fleet is unmanageable**, not before. Slices 1–2 are done, behaviour-neutral and waiting. | low until asked | high | **high** — shared id counter, shared state file, and the design's own warning that `--real` labels must not cross scopes |
| E2 | **A separate CPU-limited deployment so the throttle channel is testable** | `container_cpu_cfs_throttled_periods_total` exists only on containers with a CPU limit, and the lab deliberately has none so that every CPU measurement has one explanation. This is test coverage, not a feature; its value appears the first time the channel misses something at a client. | low | medium | low |
| E3 | **Rebuild the guard image for the corrected startup message and the scope plumbing** | Cosmetic and already in the tree; ships free with the next deploy that has another reason to happen. | low | trivial | low |

---

## Recommended order

1. **A1** — start the 24-hour run. It gates the client conversation and costs no attention.
2. **A2** during the run — it touches the guard only, not the pods being measured.
3. **B1** while waiting — the paragraph that keeps a pilot from being surprised.
4. **A3** after the run, so it does not perturb it.
5. Then **D1**, because it is the only detection question with a cost being paid every cycle right now.

Everything else waits for evidence that somebody is paying for it.
