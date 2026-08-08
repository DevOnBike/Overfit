# Task registry

One place to look, and one way to name a task in conversation.

## The ID scheme, and why it is shaped this way

`AREA-localid`. The area prefix is new; **the local id is whatever that document already called it**, so
nothing is renumbered and every existing cross-reference inside those documents keeps working.

| prefix | area | source of truth |
|---|---|---|
| `AN-` | anomaly guard | `docs/aiops/aiops-backlog.md` |
| `TG-` | `[LongFact]` release gate | `docs/test-gate-backlog.md` |
| `MS-` | metric-source seam (replay) | `docs/specs/anomaly-guard-metric-source-seam-plan.md` |
| `PS-` | PSI CPU-pressure channel | `docs/specs/anomaly-guard-psi-cpu-channel-plan.md` |
| `RS-` | .NET runtime signals | `docs/specs/anomaly-guard-runtime-signals-plan.md` |
| `XC-` | cross-cutting, no plan of its own | this file |

The problem it solves: three plan documents each have a "Task 1", so "do task 3" was ambiguous. `MS-3`
is not.

**Status vocabulary**, chosen so that the uncomfortable states have names:

| status | means |
|---|---|
| `DONE` | finished and, where a gate applies, it passed |
| `PART` | genuinely half-done, with the remaining half named |
| `OPEN` | not started |
| `FAILED` | ran, did not meet its own criterion |
| `UNGATED` | **code is committed but never verified or reviewed** — the state that otherwise hides |
| `DEFER` | deliberately not now, with a recorded reason |

---

## Anomaly guard — `AN-`

| id | status | task | note |
|---|---|---|---|
| `AN-A1` | **FAILED** | 24-hour run on a frozen configuration | 10.63/day against a 1-9/day criterion, plus one cycle failure. Tail contaminated by a test run on the same box. 11 events cannot decide the band anyway — the interval is 5.5-19 |
| `AN-A2` | DONE | alert delivery | found two defects, one ours |
| `AN-A3` | DONE | second-instance cost | 42 Mi, 1m CPU |
| `AN-A4` | OPEN | evening diurnal check | closes the last two unverified floors; verifiable only by waiting |
| `AN-A5` | **PART** | StatefulSet and HPA | HPA half measured 2026-08-07: 0 incidents in 11 static cycles, **5 in the 7 spanning a 12→15→12 round trip**. StatefulSet half not started |
| `AN-B1` | DONE | client-readiness documentation | folded into `aiops-client-readiness.md` |
| `AN-C1` | OPEN | re-tune the heap floor on an aged population | current number is extrapolation |
| `AN-C2` | DEFER | `GcPauseRatio.minTrendChange` | one finding is not evidence enough to arm a gate |
| `AN-D1` | OPEN | peer reports a fixed pod property as a recurring anomaly | diagnosed 2026-08-06, unfixed. **Independently corroborated 2026-08-08**: 272 of 298 cycles carried exactly one finding, which is not a noise distribution |
| `AN-D2` | OPEN | peer is structurally blind to a single OOMKill | measured |
| `AN-D3` | OPEN | a CPU rise on every replica at once is invisible to all four families | measured 2026-08-01, never diagnosed to the end |
| `AN-D4` | OPEN | the heap oscillates with a period near the evaluation window | measured 2026-08-05 |
| `AN-D5` | DONE | `incidents=` counter disagreed with the events | |
| `AN-E1` | DEFER | multi-scope, slices 3-5 | |
| `AN-E2` | DEFER | a CPU-limited deployment so throttle is testable | **now justified rather than optional**: PSI covers starvation without a limit, but throttle itself still cannot be tested any other way |
| `AN-E3` | DEFER | rebuild the guard image | ships free with the next deploy |
| `AN-F1` | **OPEN** | the learned seasonal history makes the guard 3x noisier | 2026-08-08: 11 → 33 opened on the same window; floor calibration moved nothing, history is the whole lever. **Unexplained** |
| `AN-F2` | **OPEN** | work-adjusted trend measured and not shipped | 2026-08-08: a 51% traffic rise on a healthy cluster took incidents from 11 to 40. This is the quantified cost |
| `AN-F3` | PART | the calibrated floors lived only in a gitignored file | rescued to `k8s/anomaly-guard/guard.lab-workload.calibrated.json`; reconciling the three guard configs is still open, and `guard.lab.json` targets a superseded deployment |

## Release gate — `TG-`

| id | status | task |
|---|---|---|
| `TG-T1` | **PART** | silent passes: 65 of 72 converted, **7 left**, each needing a judgement |
| `TG-T2` | DONE | weight initialisation is seedable |
| `TG-T3` | DONE | split; the half that survived became `TG-T8` |
| `TG-T4` | DONE | the tiled-prefill flag is restored |
| `TG-T5` | DONE | replica forwards documented and verified |
| `TG-T6` | **PART** | heavy group: **21 tests have still never executed** |
| `TG-T7` | DONE | lab absence is reported, gate exits 4 |
| `TG-T8` | PART | tiled prefill: mechanism resolved, **one unexplained event** kept open |
| `TG-T9` | DONE | loader parity — the size gap was by design |
| `TG-T10` | DONE | **a real defect**: both converters wrote Q/K in the wrong RoPE layout |
| `TG-T11` | **PART** | speculative decoding is not bit-identical. Tests fixed; **the product decision is open** — `ChatSession` dispatches to it and greedy implies a determinism it does not have |

## Metric-source seam — `MS-`

| id | status | task |
|---|---|---|
| `MS-1` | DONE | extract `IMetricWindowSource`, wire the field. Verified, reviewed, committed |
| `MS-2` | DONE | parameterise `now` out of the cycle. Verified, reviewed, committed |
| `MS-3` | **UNGATED** | historical replay driver + discriminated `GuardCycleOutcome`. **Committed, never verified or reviewed** — the gate was deferred while the tree moved and not resumed |
| `MS-4` | OPEN | live-loop regression guard. No test anywhere constructs `AnomalyGuardService` and drives `ExecuteAsync` |

## PSI channel — `PS-`

`ANALYSIS_READY`. Four questions open; the recommendation is config binding first, enum member after the
floor is measured.

| id | status | task |
|---|---|---|
| `PS-1` | OPEN | decide the vehicle: `CustomMetricBinding` now, `MetricIndex` member later |
| `PS-2` | OPEN | bind `container_pressure_cpu_waiting_seconds_total`, observable-only |
| `PS-3` | OPEN | calibrate the floor — needs induced contention, folds into `AN-E2` |

## .NET runtime signals — `RS-`

`ANALYSIS_READY`. **None of the five instrument names exist in the repo**; exposing them needs
`MeterListener` code this repository has never written. Three spikes ordered first.

| id | status | task |
|---|---|---|
| `RS-1` | OPEN | spike: confirm the instrument names live, and measure `MeterListener` overhead on the lab-as-instrument |
| `RS-2` | **OPEN, cheap** | bind `dotnet_gc_committed_bytes` — **already emitted by the workload and bound nowhere**. Committed-vs-used, zero code |
| `RS-3` | OPEN | `http.server.active_requests` — the hung-request blind spot: latency percentiles only count completed requests |
| `RS-4` | OPEN | `dotnet.monitor.lock_contentions` — a latency cliff with every current channel quiet |
| `RS-5` | OPEN | `dotnet.exceptions` — caught exceptions precede 5xx |
| `RS-6` | OPEN | `kestrel.queued_connections` / `rejected_connections` — saturation before the app sees it |

## Cross-cutting — `XC-`

| id | status | task |
|---|---|---|
| `XC-1` | DONE | the Native-AOT gate reached ILCompiler again — 34 XML-doc warnings, an orphaned doc block, a banned `Array.Copy` and a disarmed `RS0030` |
| `XC-2` | DONE | six hardcoded `D:\Overfit\` paths removed; `ModelFact` now resolves relative fixtures against the repository root |
| `XC-3` | **OPEN** | three `.bak` files from the RS0030 experiment were committed. Mine, and they need `git rm` |
| `XC-4` | OPEN | three copies of the repository-root walk exist (`FixtureFact`, `TelemetryInstrumentWiringTests`, `HybridVsDenseOnDocsCorpusTests`); `RepositoryPaths` is the fourth and the one meant to be shared |
| `XC-5` | OPEN | `gh` is not authenticated, so no CI job status has been checked all session. The AOT guard may have been red |
