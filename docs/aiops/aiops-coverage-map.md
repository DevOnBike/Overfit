# Detection coverage map — what the guard catches, cannot catch, and hasn't been shown to catch

The question a client asks first: *"which failures will this catch?"* Every answer to it today is scattered
across `docs/TASKS.md` as forty-odd rows, each phrased as a defect against a detector rather than as an
answer about a failure mode. This document inverts that: one row per failure a real workload actually has,
not per detector.

**How to read a row.** Four statuses, and the difference between the first two is the point of this
document:

| Status | Means |
|---|---|
| **Detected — live-verified** | An injected fault on the real lab cluster produced the expected finding, end to end, this session or a prior one, and the result is recorded with a date |
| **Detected — fixture-proven only** | The code path is unit- or mutation-tested and its arithmetic is read and correct, but nobody has injected the real fault on the running guard. A fixture proves the code path; an injected fault proves the chain from Prometheus scrape to logged incident |
| **Structurally undetectable** | A property of the mechanism, not a missing feature. Building this would need a different data source or a different product (a dependency graph, an in-process agent) |
| **Uncovered today** | Nobody built it, or it exists but isn't armed/deployed. Distinguished below as *no mechanism* vs. *mechanism exists, not live* |
| **Unknown / untested** | Neither measured nor reasoned about with enough confidence to place elsewhere |

**Source of truth.** `docs/TASKS.md` as read 2026-08-10 (every `AN-*`/`RS-*`/`PS-*`/`XC-*` row cited below was
open in that file at that date — re-check before quoting a status as current), `Sources/Anomalies/**` read
directly for the rows marked *verified by reading*, and the live cluster (`kubectl -n lab get configmap
anomaly-guard-config`, `kubectl -n lab get pods`) read this session, 2026-08-10, without applying or
injecting anything — the lab was mid-experiment (`AN-D1`'s memory-offset calibration window) and the brief
that opened this task said not to touch it. Per-channel binding detail (source series, `MetricSourceKind`,
floor, known failure mode of the *channel itself*) already lives in
[`aiops-metrics-catalogue.md`](aiops-metrics-catalogue.md) — this document links to it rather than repeating
it; read that one for "what does this channel measure," read this one for "what failure does that answer."

---

## 1. Container & process lifecycle

| Failure mode | Status | Family | Evidence | What its silence looks like |
|---|---|---|---|---|
| OOM kill, single replica | **Detected — live-verified** | Rules (`OomEventsRate`, `ForRareEvent`) + `ContainerRestarts` | `AN-D8`, 2026-08-09: `POST /fault/oom` → `OOMKilled` exit 137 in 5s → channel moved 0→1.14, incident led by `OomEventsRate`, all 11 peers at 0 | Peer comparison alone would report nothing — one event yields a non-zero rate over ~10% of the window, under the 0.33 Cliff's-delta materiality gate (`OomDetectionTests.PeerComparison_MissesASingleOomKill`). If the rule ever gets un-armed, an OOM kill goes fully silent, because peer cannot pick up the slack |
| Crash-restart without OOM (e.g. unhandled exception at startup) | **Detected — live-verified (as part of a rollout measurement)** | Rules (`ContainerRestarts`, `ForRareEvent`) | `AN-A5`, 2026-08-08: a plain `rollout restart` lifted findings 0→6 | Same shape as OOM — a single restart is a spike, not a level, so peer alone would miss it |
| Bad rollout / topology-changing deployment (StatefulSet or Deployment replace) | **Detected — live-verified** | Peer (per-pod findings during the transition) + trend (level shift on the cross-peer common component, §4.5 of `aiops-architecture.md`) | `AN-A5`, 2026-08-08: StatefulSet rollout findings **median 14** vs **1** for the same pods replaced under an unchanged topology — the control that made the number mean something | Nothing today distinguishes "rollout in progress" from "something is wrong" except an operator-declared maintenance window — the guard correctly finds real movement during a rollout and reports it as such |
| Guard process itself stops | **Detected — live-verified** | Prometheus alert on `overfit_guard_last_cycle_timestamp_seconds`, not a detector family | `AN-A2`, DONE, 2026-08-05: `time() - ts > 900` alone goes `inactive` (not firing) once the guard's own series disappears with it — measured zero alerts. `absent(...) or (...)` fires in 60s, verified with real Alertmanager delivery | The naive alert expression is the textbook example of silent failure here: it evaluates over an empty vector and reports "fine" in exactly the one case that matters. `k8s/lab/guard-alerts.yaml` ships the corrected rule |

---

## 2. Resource exhaustion — CPU, memory, GC

| Failure mode | Status | Family | Evidence | What its silence looks like |
|---|---|---|---|---|
| Memory leak, single replica | **Detected — live-verified (as a fixture-scale measurement)** | Peer + trend | `aiops-client-readiness.md` fault table, 2026-08-03: 5 MB/min leak detected in 10 min, 20 MB/min in 5 min, on a synthetic population run through the real shipped code path | **Real, unfixed noise problem sits next to this, not on top of it**: `AN-D1`, PART — a pod's own fixed offset from its peers is re-reported as a *new* finding every cycle (272 of 298 cycles carried exactly one finding — corroborated, not a noise distribution, 2026-08-08). A genuine leak is still caught; a standing difference drowns it in repeat pages of the same non-event. The fix (`MinAbsoluteGapChange` per custom binding) is blocked twice: the calibration window is contaminated by a moving target (a fitted −3.94 MB/h drift in the population itself, found 2026-08-10), and the config surface to express the fixed value doesn't exist yet (`AnomalyGuardConfigFile.CustomEntry` has no field for it — found by the `AN-D9` developer reading for something else) |
| Memory leak, every replica simultaneously (fleet-wide) | **Detected — fixture-proven** | Trend + step (level-shift on the cross-peer common component) | `aiops-client-readiness.md` table: detected in 5 min on a synthetic population. `AN-D4`, PART: watched live for 3 hours 2026-08-10 — a real fleet-wide 23 MB gen2 collection at 09:15Z moved the peer gap by 0.3 MB (62.5→62.8), confirming common-mode cancels out of the gap as designed | A common-mode step that is NOT decomposed (fewer than `CrossPeerBaseline.MinimumPeers` = 3 pods) reads as a genuine trend on every pod's raw series and pages once per replica — this is the exact failure `DecomposeCommonMode` exists to prevent, and it is on by default |
| CPU throttling, replica carries a CPU limit | **Detected — live-verified** | Rules only (`SustainedThresholdOptions.ForCpuThrottling`, 5% held over 25% of window) | `AN-E2`, DONE: 71.6%→100% throttling, incident at "0.05 for 52% of the window", peer independently flagging the same pod, no peer wrongly accused | Trend correctly finds nothing (a persistent level, not a drift) — only the rule catches this, and only the rule ever will, by construction |
| CPU throttling, replica carries **no** CPU limit | **Structurally undetectable via `CpuThrottleRatio`** | — | `AN-D7`, DONE: `container_cpu_cfs_throttled_periods_total` has **0 series** on a container with no quota — CFS accounting does not run | Mitigated, not fixed, by a different channel: see PSI row below |
| CPU starvation, no CPU limit set (denial without a quota) | **Detected — live-verified** | Rules + peer + trend on custom channel `CpuPressure` (PSI) | `PS-3`, DONE: healthy peak 0.00005, starved minimum 0.167 — **16485x separation**; live guard opened an incident from this channel unprompted, 2026-08-10 lab session | Needs a scrape step ≤ 45s at the deployed 15-minute recent window (`RecentSamples = ceil(RecentWindow/step)` needs ≥20 samples) — raise the step and the channel goes permanently `WarmingUp`, indistinguishable from a quiet cluster. This already happened once during development |
| CPU rise, one replica | **Detected — live-verified (fixture-scale)** | Trend | `aiops-client-readiness.md` table: 5 min latency | — |
| CPU rise, every replica at once | **Detected — fixture-proven only** | Step (level-shift on cross-peer common component) — **peer structurally cannot see this by construction** (nobody is an outlier if everyone moves together) | `AN-D3`, refuted-as-written 2026-08-10: `AnomalyGuard.RunTrend` builds the common component and tests it via `ObserveLevelShift` for **every** channel, not only heap (`AnomalyGuard.cs:944, 1291`, gated on `podCount >= CrossPeerBaseline.MinimumPeers`). The 2026-08-01 comment in the code names this exact case as the reason the gate has its own accumulator now. **Still owed: nobody has injected a fleet-wide CPU step on the live guard and watched a finding come out** | Below `MinimumPeers` (3), decomposition is skipped and the row's original claim — invisible to all four families — is true again. A peer group under 3 (small namespace, or most pods filtered by maintenance) silently loses this coverage |
| GC managed-heap growth | **Detected — live-verified (fixture-scale)** | Peer + trend + learned (feature 9, not wired into the deployed cycle) | Floor `minGap 0.93MB / minTrendChange 3.93MB`, verified live-bound today (`GcGen2HeapBytes` in the live ConfigMap) | Floor is `AN-C1`, OPEN: "extrapolation from a trend, not yet re-verified on an aged fleet" — a floor set too low on a young population reads real growth as noise once the population ages past what set it |
| GC pause-time pathology | **Uncovered today — mechanism exists, not armed** | Peer + trend wired, **no default rule**, `minTrendChange` deliberately unset | `AN-C2`, DEFER: "one finding is not evidence enough to arm a gate." Live config confirms: `GcPauseRatio` has `minGap` only, no `minTrendChange` | A GC-pause spike that isn't also a level shift on the raw series (peer's job) produces nothing — trend cannot fire without a change-detection floor |
| GC committed-bytes growth (native + managed) | **Uncovered today — mechanism exists, not armed** | Peer + trend (custom channel `GcCommittedBytes`) | Live config: bound, has `minGap`/`minTrendChange`, but **no `ruleThreshold`** — confirmed in the live ConfigMap read this session | Same shape as GC pause: relative families can fire, but there is no absolute "this is just too high" backstop |
| Thread-pool queue depth (generic starvation, e.g. blocked async work) | **Uncovered today — mechanism exists, not armed, and never load-tested** | Peer (`LoadSensitive`) + trend + learned (feature 11) — `ThreadPoolQueueLength` is a `MetricIndex` member | Live config confirms the channel is bound (`dotnet_threadpool_queue_length`, `Gauge`) but has **no entry in `thresholds` at all** — the metrics catalogue's "None shipped" is current | Whatever depth the pool reaches, nothing is armed to notice it — a queue growing without bound looks exactly like a queue holding steady, because there's no floor to compare against |
| Lock contention | **Detected — live-verified, one open question** | Rules + peer + trend (custom channel `LockContentions`) | `RS-4`, PART: 152x separation under fault (0.076→11.6/s, peers flat); floor recalibrated once in production after a 3-minute calibration window produced a false positive within the hour | Whether CPU is blind to contention that blocks *without spinning* is unmeasured — the injected fault so far also moved CPU 8x, so it is not proven this channel is load-bearing on its own for every contention shape |

---

## 3. Serving-path symptoms

| Failure mode | Status | Family | Evidence | What its silence looks like |
|---|---|---|---|---|
| Latency regression, single replica | **Detected — live-verified (fixture-scale)** | Peer + trend + learned (features 4-6) | `aiops-client-readiness.md` table: 0 min latency, 3x on one replica | **Saturates at both ends.** A request that never finishes never enters the histogram (see hung-request row below); a stalled p95 pins to the top bucket edge (30000ms) and carries no further magnitude once there — a 10x-worse stall and a 2x-worse stall look identical past that point |
| Hung / stalled requests (never complete) | **Structurally undetectable via any latency channel** — closed by a different channel | Rules + peer + trend on custom channel `ActiveRequests` | `RS-3`, DONE: fires 5→40 rows, **two minutes before** any latency percentile moves, because in-flight count doesn't wait for completion | If `ActiveRequests` is ever un-deployed, this failure mode goes fully invisible to the guard — latency, error rate and peer-on-latency all structurally cannot see a request that is still open |
| Error rate rise (5xx responses) | **Uncovered today — deliberately, as of 2026-08-10, verified live this session** | would be peer + trend + learned (feature 8) if bound | `AN-D12`: the deployed binding measured errors/sec (an unbounded rate) against a channel documented and calibrated as a `[0,1]` fraction of responses — a real unit-contract defect. **Fixed by dropping the binding, not by correcting it.** Verified this session: neither `guard.lab-workload.json` nor the **live** ConfigMap (`kubectl -n lab get configmap anomaly-guard-config`) bind `ErrorRate` in `metrics` — a leftover `thresholds.ErrorRate` entry remains in both but is inert with no binding to gate. **Corrects the brief's own list**: `docs/TASKS.md`'s `AN-D12` row still reads "not yet applied to the live cluster" — that is stale; it has been applied | An operator who has read the row expects this channel to be blind and it is, correctly — but the leftover threshold entry, if ever mistaken for an active gate during a config review, would be a false reassurance |
| Application exceptions, caught (never surface as a 5xx) | **Detected — live-verified** | Rules + peer + trend on custom channel `Exceptions` | `RS-5`, DONE: fires "0.0167 for 44% of the window" on the injected pod; measured with **every other channel blind by construction** — `ErrorRate` (counts responses, not throws), peers, p95 (flat 48.8-48.9ms) | This is the one custom channel where a healthy cluster reading **exactly zero** on all twelve pods is expected and correct, not a defect — `InertChannels()` will flag it as ambiguous constant-zero, and only an operator who knows the event hasn't happened can resolve that correctly |
| Scrape-exporter self-saturation (a pod cannot report the very thing overwhelming it) | **Structurally undetectable, refuted as specified** | — | `RS-6`: Kestrel `queued`/`rejected_connections` only move once the process is *at* a concurrency limit; measured with the limit as the only variable — no limit, 40 holders → 6/6 scrapes succeed, queue always 0; limit set, same load → 0/6 scrapes succeed, nothing observable | A channel that goes dark exactly when it has something to report is worse than no channel — not bound, by decision, and no concurrency limit is set on the lab workload |
| A pod silently stops answering scrapes at all (coverage loss short of a full outage) | **Uncovered today — mechanism exists in code, not deployed** | Peer + trend on custom channel `ScrapeCoverage` (`avg_over_time(up[...])`) | `AN-D9`, PART: implemented, 13 mutations all caught, 91/91 lines covered — but **verified absent from the live ConfigMap this session** (`customMetrics` keys are `GcCommittedBytes, ActiveRequests, LockContentions, Exceptions, CpuPressure` — no `ScrapeCoverage`), matching `aiops-metrics-catalogue.md`'s "five shipped." No live faulted arm exists either | This is a real gap in what's *running*, not in what's *built* — the distinction matters because "we built this" and "this protects the lab today" are different claims and `docs/TASKS.md`'s PART status can read as either |

---

## 4. Fleet-wide traffic and seasonality — noise that threatens the rows above, not a missing detector

| Situation | Status | Evidence | What its silence looks like |
|---|---|---|---|
| Day-one deployment, no history for `SeasonalBaseline` | **Known, documented cost, not a defect** | `aiops-client-readiness.md`: 2 of 5 false incidents in a 24h run were the same diurnal traffic curve counted twice — "expect roughly one extra incident per diurnal slope on day one" | Reads exactly like a real deployment-wide finding on day one; distinguishable only by knowing the guard has no prior day yet |
| Young pod during a rollout misread as a leak (peer) or an anomaly (trend) | **Covered for trend, deliberately NOT covered for peer** | `WarmUpGrace`/`IsWarmingUp` gates `RunTrend` and `RunSilentPods` (`AnomalyGuard.cs:982, 1340`). `XC-9`, DONE 2026-08-10: the custom-channel trend path (`RunCustomTrend`) was missing this gate and has been fixed to mirror it | Peer is deliberately ungated here — a replica that differs from its peers *right now* is worth reporting whatever its age, and during a rollout every pod is young, so a peer-wide grace would blind the guard exactly when a bad version is going out (documented reasoning in `WarmUpGrace`'s own remarks) |
| A metric bound to the wrong series *that still moves plausibly* | **Unknown / untested — no guard exists for this class today** | `aiops-architecture.md` §8: "nothing catches one that moves plausibly but means the wrong thing" — the `AN-D3` shape, which every existing self-check (`InertChannel`, `BlindMetrics`/`PartialMetrics`/`UnevaluableMetrics`, config-parity gates) is structurally unable to catch, because all of them detect *flat* or *absent*, not *wrong* | This is the honest ceiling of the self-check machinery described in §8 of `aiops-architecture.md` — a plausible-looking wrong signal produces plausible-looking findings, indistinguishable from real ones without an independent source of truth |

---

## 5. Structurally out of scope

| Failure mode | Why | What would be needed instead |
|---|---|---|
| Root cause in a downstream/dependency service | No service-to-service model exists; the guard sees one pod's metrics, not a call graph | A separate product, explicitly named as such in `aiops-client-readiness.md`: "the most common real root cause is a downstream dependency and this will not find it" |
| Disk / volume saturation | `kubelet_volume_stats_*` is unavailable in the development cluster — **untested, not unsupported** | Bind it at a site that exports the metric; treat as unverified until then |
| Anything needing an in-process agent | Would trade away the guard's core property — the only dependency is one HTTP route to Prometheus | Out of scope by product decision, not a gap |
| Non-.NET workload correctness of the runtime-signal channels | Every runtime-channel floor and behavioural claim (`GcGen2HeapBytes` growth semantics, `LockContentions`, `Exceptions`, `ActiveRequests`) is validated against .NET only. A JVM grows its heap to `-Xmx` by design — a rising trend there is normal until the plateau. PHP-FPM has no long-lived heap and its memory *is* load-sensitive, opposite the current global classification | A second stack's validation pass — `aiops-client-readiness.md` names this as "removes the largest unknown in the 'works with any application' claim," not yet done |
| The learned/GPT detection family | Not wired into the deployed cycle (`AnomalyGuard.RunCycle` never calls it); lives in a separate execution path, `LiveMonitoringPipeline`, that shares only ingestion with the deployed guard. Turning it on in replay made the guard **3x noisier**, cause not fully isolated (`AN-F1`, OPEN) | Not part of "what the guard detects" until it is wired in and that regression is understood |

---

## Live-cluster findings from this pass (2026-08-10, read-only)

Checked against `kubectl -n lab get configmap anomaly-guard-config` and `kubectl -n lab get pods`, without
applying or injecting anything, respecting the mid-experiment constraint:

1. **`ErrorRate`'s `AN-D12` fix is live**, not merely committed — corrects `docs/TASKS.md`'s own "not yet
   applied to the live cluster" note (see §3 table).
2. **All twelve `lab-workload` replicas now carry a `1`-core CPU limit** (`kubectl -n lab get pods -o
   jsonpath=...resources.limits.cpu` → `1` on every pod). The checked-in manifest,
   `k8s/lab/workload.yaml:64-66`, still specifies `limits: cpu: 200m` — **live/repo drift**, not touched here.
   Practical effect on this map: `CpuThrottleRatio`'s peer comparison is no longer "a group of one," which
   `aiops-metrics-catalogue.md` and `aiops-architecture.md` both still describe as the state from `AN-D7`
   (when only one throttle-test pod carried a limit). The rule family was always the one that actually fires
   on throttling regardless of peer-group size, so this does not change any status above, but the peer-group-
   of-one caveat in the linked documents is stale for this cluster as it stands today.
3. **`ScrapeCoverage` (`AN-D9`) is confirmed absent from the live `customMetrics` block** — five custom
   channels are bound (`GcCommittedBytes`, `ActiveRequests`, `LockContentions`, `Exceptions`, `CpuPressure`),
   matching `aiops-metrics-catalogue.md`'s count exactly. The mechanism is real and mutation-tested; it is
   not protecting this cluster today.
4. **`rules` is unset in the live config** (`null`), so the deployed guard runs on `AnomalyGuardOptions.
   DefaultRules` (`CpuThrottleRatio`, `OomEventsRate`, `ContainerRestarts`) rather than a config-supplied
   override — consistent with every row above that cites those three as rule-armed.

## What I could not verify

- **No live-injected fault exists yet** for: a fleet-wide CPU step (`AN-D3`'s open half), `ScrapeCoverage`
  (not deployed, so nothing to inject against), or the CPU-vs-lock-contention isolation `RS-4` still owes.
  Their rows above are marked fixture-proven or explicitly flagged as still owing a live arm; do not read
  them as equivalent to the live-verified rows.
- **No independent check that the live `guard.json` matches what `DeployedConfigParityTests` last verified**
  beyond the fields read above (`metrics`, `customMetrics`, `thresholds`, `rules`) — did not diff the full
  document byte-for-byte against the checked-in `guard.lab-workload.json`.
- **Whether the CPU-limit drift (finding 2 above) was a deliberate operational change or an unrecorded
  `kubectl edit`** — nothing in `docs/TASKS.md` or the checked-in manifests documents raising it from `200m`
  to `1`. Flagged, not resolved.
- **Volume/disk saturation and the second (non-.NET) stack** — both remain untested for the reasons stated in
  §5; nothing done here changes that.

---

Route any question about a specific channel's PromQL, unit or floor to
[`aiops-metrics-catalogue.md`](aiops-metrics-catalogue.md). Route "why this threshold" to
[`aiops-detection-pipeline.md`](aiops-detection-pipeline.md). Route "should we run this" to
[`aiops-business-case.md`](aiops-business-case.md) / [`aiops-client-readiness.md`](aiops-client-readiness.md).
