# Metrics catalogue — everything the guard can watch, and in what unit

Every channel the anomaly guard supports today, what it binds to, what the resulting PromQL actually is, and
how it fails. This is a reference, not a tutorial — for how to add a new one, see
[`aiops-adding-a-metric.md`](aiops-adding-a-metric.md); for the arithmetic behind a threshold, see
[`aiops-detection-pipeline.md`](aiops-detection-pipeline.md).

**Two vocabularies, and they are different sizes on purpose.** `MetricIndex` (`Sources/Anomalies/Contracts/MetricIndex.cs`)
lists 13 ingestion channels — what the sources can scrape. `MetricSnapshot.FeatureCount` (`Sources/Anomalies/Contracts/MetricSnapshot.cs`)
is 12 — the learned model's fixed input contract. The gap is deliberate: `ContainerRestarts` (index 12) was
added last, is read by the rules/peer/trend families straight off the raw series, and was kept out of the
token vocabulary because moving it would invalidate every trained checkpoint. Beyond the known 13, a
deployment can add **custom channels** (`CustomMetricBinding`) for anything the enum does not model — five are
shipped in the lab (`GcCommittedBytes`, `ActiveRequests`, `LockContentions`, `Exceptions`, `CpuPressure`).
Custom channels reach the rules, peer and trend families and **never** the learned one, for the same reason.

**How a binding becomes a query, in one line** (full mechanism in `aiops-adding-a-metric.md`): a deployment
names a source series and a `MetricSourceKind` (`Gauge`/`Counter`/`EventCount`/`Ratio`/`HistogramSeconds`);
`MetricMap.Build` (`Sources/Anomalies/Monitoring/MetricMap.cs:219-240`) wraps it in `sum by (pod) (...)`,
`rate(...)`, `increase(...)`, or `histogram_quantile(..., sum by (pod, le) (rate(..._bucket...)))`
accordingly — or, when a verbatim `Query` is supplied on the binding, that string wins outright and the
kind is ignored for query-building purposes (still used elsewhere, e.g. sampling semantics).

**Source of truth used to build this document**: `Sources/Anomalies/Monitoring/MetricNameCatalog.cs`,
`MetricMap.cs`, `PromqlCatalog.cs`, `Sources/Anomalies/Contracts/{MetricIndex,MetricSnapshot,MetricBinding,
CustomMetricBinding,MetricSourceKind}.cs`, `Sources/Anomalies/Monitoring/PeerSignalCatalog.cs`,
`Sources/Anomalies/Incidents/SignalCatalog.cs`, `Sources/Anomalies/Contracts/SustainedThresholdOptions.cs`
and `AnomalyGuardOptions.cs`, the deployed configs (`k8s/anomaly-guard/guard.lab-workload.json`,
`k8s/anomaly-guard/guard.lab.json`, `k8s/lab/anomaly-guard.yaml`), `Demo/LabWorkload/RuntimeSignalListener.cs`
+ `WorkloadMetrics.cs`, and `docs/TASKS.md`. Every number below is quoted, not summarised — read the cited
file before acting on one.

---

## Infrastructure / resource — container-level, stack-neutral

Sourced from cAdvisor and kube-state-metrics, which report the same series names whatever language the pod
runs. Roughly half the known channels are this shape (`MetricNameCatalog` remarks), and it is the whole
reason automatic discovery works on a workload nobody has met — no application instrumentation required.

| Channel | Source series | Kind | Unit | Detector families | Floor / rule (calibration conditions) | Known failure mode |
|---|---|---|---|---|---|---|
| `CpuUsageRatio` | `container_cpu_usage_seconds_total` | Counter → `rate()` | **Cores consumed**, not the `[0,1]` fraction of the pod's CPU limit the field's own doc comment (`MetricSnapshot.cs:52-59`) claims. `PromqlCatalog.OverfitServerQueries` states the deviation explicitly: a limit is optional, and this binding never divides by one. | rules: none armed by default; peer: `LoadSensitive` (needs a work metric — see `RequestsPerSecond`); trend; learned (feature 0) | minGap `0.00041`, minTrendChange `0.000326` — 24-hour false-positive run, 292 cycles, nothing wrong with the cluster. A 24-*window* (2-hour) sample proposed `0.000159`, too low; the full day's evening ramp pushed healthy peer spread up to `0.000328` | Name promises a ratio, binding delivers raw cores. A consumer that assumes `[0,1]` on an unlimited pod is simply wrong. Peer comparison additionally requires a work metric or reports `InsufficientData` |
| `CpuThrottleRatio` | `container_cpu_cfs_throttled_periods_total` / `container_cpu_cfs_periods_total` | **Must be a verbatim `query`** (`rate(throttled)/rate(periods)`) — binding it as `kind: Ratio` alone renders as `{name}{selector}`, the raw cumulative counter with no `rate()` or division at all | Fraction `[0,1]` of CFS periods throttled | rules: `SustainedThresholdOptions.ForCpuThrottling` (threshold `0.05`, breach fraction `0.25`, min 20 samples) is in `AnomalyGuardOptions.DefaultRules`; peer; trend; learned (feature 1) | Threshold+fraction measured on a pod limited to 1 core: median throttling 1.4%, p90 11.8%, peak **19.8%** while siblings burst to ~1.8 cores — the commonly-cited literature threshold of 25% **never fired** on this fault. 33% of samples sat ≥5% in the loaded window vs 13% idle, hence the 25% persistence gate | **Exists only on containers carrying a CPU limit** — `0` series in the lab namespace until one pod was given `1000m` (`AN-D7`). Peer comparison over a group of one is undefined, not merely hard. Binding it as `Ratio` without a verbatim query silently queries the wrong quantity and was live in production for part of a day (`AN-E2`): 12 false trend findings across one replay window where the true ratio was `0.0000` on all twelve pods  **Peer-group caveat updated 2026-08-10**: `AN-D7` recorded this channel as having 0 series because no `lab-workload` pod carried a CPU limit, which made any peer comparison a group of one. All twelve replicas now carry `cpu: 1` — verified against the live cluster, and the checked-in manifest was corrected the same day after it was found still saying `200m` (`XC-14`). The channel therefore has full peer coverage here. **The structural point is unchanged for a client**: on pods without a quota the series does not exist, and absence of a series is not health. |
| `MemoryWorkingSetBytes` | `container_memory_working_set_bytes` | Gauge | Bytes (RSS, excludes reclaimable page cache — this is what Kubernetes evicts on) | rules: none armed; peer: `LoadIndependent` (corrected — see below); trend; learned (feature 2) | minGap `9.52 MB`, minTrendChange `25.85 MB` — full 24-hour run. Two earlier floors were both wrong in opposite directions: `256 MiB` was sized for a *different, synthetic* population with 1.23 GB pods and silently disabled the gate on this 43 MB one; `1.089 MB` (from a 24-window sample) then over-corrected and became the **largest single false-positive source**, 7 findings/day, because healthy replicas move working set by up to 20.7 MB across a window on their own | `PeerSignalKind` originally classified memory as `LoadSensitive` (divide by work); measured wrong — on a synthetic population this made memory the single largest source of false peer findings (317/269/35 across three seeds) at a gap matching the traffic spread exactly. Reclassified `LoadIndependent`. **`AN-D1` open**: peer still reports a fixed per-pod offset as a recurring anomaly (272 of 298 cycles carried exactly one finding — not a noise distribution) |
| `OomEventsRate` | **Not** `container_oom_events_total` — verbatim join: `increase(kube_pod_container_status_restarts_total[range]) * on(namespace,pod,container) group_left() kube_pod_container_status_last_terminated_reason{reason="OOMKilled"}`, `or (...restarts... * 0)` fallback | EventCount, verbatim query only | Events in window (an `increase`, decays with window length: `0` at 2/5/15 min, `1.00` at 30 min for a kill 17 minutes old) | rules: `SustainedThresholdOptions.ForRareEvent`; peer: `LoadIndependent`; trend; learned (feature 3) | Not calibrated — `PeerSignalCatalog.IsCountedEvent` opts restarts/OOM out of floor calibration entirely (see `ContainerRestarts` below for why) | `container_oom_events_total` reads **structurally zero on this runtime** — measured against a pod Kubernetes reported `OOMKilled`/exit 137: that series stayed 0, and across 43 series cluster-wide over an hour every value was `0.0`. **The peer family is structurally blind to a single OOMKill** (`OomDetectionTests.PeerComparison_MissesASingleOomKill`) — one event yields a non-zero rate over ~10% of the window, below the 0.33 Cliff's-delta materiality gate; only the rule family and `ContainerRestarts` catch it. The `or ... * 0` fallback exists because a join returns **no series at all** for a pod whose OOM-reason side is absent — the first fix without it turned the channel from "always zero" into "always blind" |
| `ContainerRestarts` | `kube_pod_container_status_restarts_total` | EventCount (`increase`, not the raw counter — the total says how old a pod is, the increase says whether it just restarted) | Restarts in window | rules: `ForRareEvent`; peer: `LoadIndependent`; trend — **not learned** (this is the 13th `MetricIndex` member, deliberately excluded from `MetricSnapshot`; see the top-of-document note) | Not calibrated — `FloorCalibrator` run naively over healthy data would propose **`1.25`**, because pods in this population restart about once a day, which would make a genuine single restart permanently unreportable. `IsCountedEvent()` opts this signal out of calibration by design | Same "unit is arbitrary vs. unit is the thing you care about" trap as `OomEventsRate` — never calibrate a floor for a rare-event counter from observed frequency |

---

## .NET runtime — named after this project's own exporter, cross-stack candidates exist

`MetricNameCatalog` lists JVM/Go/Python/PHP-FPM equivalents for every one of these (the *concept* is
cross-stack — heap size, GC pause share, a work queue), but no exact series name is universal, so each
deployment supplies its own binding. The lab binds the .NET-runtime-exporter names.

| Channel | Source series (lab) | Alt. stacks known | Kind | Unit | Detector families | Floor (calibration conditions) | Known failure mode |
|---|---|---|---|---|---|---|---|
| `GcGen2HeapBytes` | `dotnet_gc_heap_size_bytes` | jvm, go, python | Gauge | Bytes — **whole managed heap on this exporter**, not gen-2 specifically (`dotnet_gc_heap_size_bytes` carries no generation label; stated as an approximation in `PromqlCatalog` remarks) | peer: `LoadIndependent`; trend; learned (feature 9); no default rule | minGap `0.93 MB`, minTrendChange `3.93 MB` — raised **before it cost anything**, on a measured trend (proposal grew `0.729 → 0.930 MB`, 1.27x, across a day) rather than a false positive. Same channel produced 108 false incidents/day in an earlier baseline on an older population | Floor is extrapolation from a trend, not yet re-verified on an aged fleet — `AN-C1` open: "current number is extrapolation" |
| `GcPauseRatio` | `dotnet_gc_pause_seconds_total` | jvm, go (no python candidate) | Counter → `rate()` | Fraction `[0,1]` of wall-clock time in GC | peer: `LoadSensitive` (pause time tracks allocation, allocation tracks requests served); trend; learned (feature 10); no default rule | minGap `0.0000209` (raised from `0.0000082` off one finding in 292 cycles); minTrendChange **left unset** — one finding was judged not enough evidence to arm a change-detection gate (`AN-C2`, deliberately deferred) | Needs `RequestsPerSecond` as its work metric or reports `InsufficientData` |
| `ThreadPoolQueueLength` | `dotnet_threadpool_queue_length` | micrometer/jvm (`executor_queued_tasks`), php-fpm, nginx | Gauge | Pending work items | peer: `LoadSensitive` (a queue is work waiting; comparing depths under different arrival rates compares the rates); trend; learned (feature 11); no default rule | **None shipped** — absent from every threshold block in `guard.lab-workload.json`/the deployed ConfigMap | Uncalibrated: whatever an operator sets, or effectively unbounded |
| `GcCommittedBytes` *(custom)* | `dotnet_gc_committed_bytes` | .NET-specific (`GC.GetGCMemoryInfo().TotalCommittedBytes`) | Gauge | Bytes | rules: **no `ruleThreshold` set** — not armed; peer: `LoadIndependent` (`loadSensitive: false`); trend (minTrendChange `3.0 MB`); **not learned** (custom channel) | minGap `1.0 MB`, minTrendChange `3.0 MB` | Committed-vs-used distinction from `GcGen2HeapBytes`; emitted by `Demo/LabWorkload/WorkloadMetrics.cs:178` and bound in both the lab JSON and the deployed ConfigMap today — `docs/TASKS.md`'s `RS-2` row still reads "OPEN, cheap: bind `dotnet_gc_committed_bytes`" and is **stale** against the code and config |

---

## Request / HTTP symptoms — application-instrumented, multi-stack candidates

| Channel | Source series (lab) | Alt. stacks known | Kind | Unit | Detector families | Floor (calibration conditions) | Known failure mode |
|---|---|---|---|---|---|---|---|
| `LatencyP50Ms` / `P95` / `P99` | `labapp_request_duration_seconds` histogram, quantile 0.50/0.95/0.99 | otel, prometheus-client, micrometer all have their own histogram name | HistogramSeconds → `histogram_quantile(q, sum by (pod, le) (rate(name_bucket{sel}[range]))) * 1000` | Milliseconds | peer: `LoadIndependent`; trend; learned (features 4-6); no default rule | Only `minTrendChange = 50 ms` set, and it is **deliberately not the calibrated value** — the calibrator correctly proposes ≈0.01 ms (latency here is near-constant), which is a noise floor, not an actionable threshold; the rule is `max(calibrated, operational)` and 50 ms is the operational number, not derived from this population | **Saturates at both ends.** A hung request never enters the histogram at all — it only counts on completion — so `/fault/stall` is invisible to every latency channel (closed instead by the custom `ActiveRequests` channel, `RS-3`). At the other end, a stalled p95 pins to the top bucket edge (`30000 ms`) and carries no further magnitude once there. Dropping the `pod` label through the quantile (`sum by (le)` without `pod`) silently discards every sample |
| `RequestsPerSecond` | `labapp_requests_total` | prometheus-client, micrometer, nginx, php-fpm all have a candidate | Counter → `rate()` | Requests/sec | Also **the work metric** every `LoadSensitive` channel above needs; peer: `LoadIndependent` itself; trend; learned (feature 7) | minGap `0.0238`, minTrendChange `0.333` — **partly circular, and stated as such rather than hidden**: calibrated over the same 24h run that contained the two false incidents it now suppresses (the up/down slopes of a 1440-minute diurnal load curve). Sound only because that period was independently verified healthy | **Stated cost**: the channel is now blind to any genuine deployment-wide traffic change smaller than 0.333 — including a real outage that halves traffic on a quiet night. The documented correct fix is `SeasonalBaseline` (compare same phase of the previous day); not armed, needs `MinimumHistoryDays` and a day of history the guard did not have when this floor was set |
| `ErrorRate` | `labapp_errors_total` | prometheus-client, micrometer, nginx | **Counter**, bound to the raw error count only | **Errors/sec, an absolute rate — not the `[0,1]` fraction of 5xx responses `MetricSnapshot.cs:129-136` documents the field as.** No division by `RequestsPerSecond` appears in the binding | peer: `LoadIndependent`; trend; learned (feature 8) | minGap `0.01`, minTrendChange `0.01` — sized as if the value were a `[0,1]` fraction (1%), which is inconsistent with the bound quantity being an absolute rate | **Contradiction found while writing this document, not resolved by it.** `guard.lab-workload.json`'s own top-of-file comment says *"ErrorRate is derived from the app's own error counter over its request counter, which the map cannot express as a single source, so it is left out and reported blind."* — but the file's own `metrics` block **does** bind `ErrorRate`, to `labapp_errors_total` alone, `kind: Counter`. This is deployed as-is in `k8s/lab/anomaly-guard.yaml` too, so it is not a stray leftover. Either the header comment is stale documentation of an earlier state, or the binding is wrong; neither is resolved here and it should be checked before trusting this channel's number. By contrast `guard.lab.json` (the `overfit` namespace config) genuinely leaves `ErrorRate` unbound, matching what its own comment says — only the `lab-workload` config disagrees with itself. `OverfitServerQueries`'s in-code default (`5xx rate / total rate`) is the version that actually produces the documented `[0,1]` ratio, and is what the deployed binding should probably match |

---

## Custom channels shipped in the lab — rules/peer/trend only, config-only vehicle

None of these reach the learned family (`CustomMetricBinding`'s own doc: "the enum cannot grow per
customer"). Five of the six ship in `k8s/anomaly-guard/guard.lab-workload.json`'s `customMetrics` block and the
matching `k8s/lab/anomaly-guard.yaml` ConfigMap; only `ActiveRequests` also ships in the `overfit`-namespace
config (`guard.lab.json`). **`ScrapeCoverage` is the sixth and is in the repo only** — added 2026-08-10,
mutation-proven, and absent from the live ConfigMap, which still carries five. Checked against the cluster,
not inferred from the file.

| Channel | Source series | Kind | `SignalKind` / `Class` | Unit | Rule armed? | Floor (calibration conditions) | Known failure mode |
|---|---|---|---|---|---|---|---|
| `CpuPressure` (`PS-2`/`PS-3`) | `container_pressure_cpu_waiting_seconds_total` (PSI) | Counter | `LoadIndependent` / `Infrastructure` | Fraction `[0,1]` of time a task waited for CPU — measures **denial**, not usage; needs no CPU quota to exist, unlike `CpuThrottleRatio` | Yes — threshold `0.01`, breach fraction `0.25` | minGap `0.00004` (healthy max × 1.25). Measured on 12 pods, one starved at ~2 cores of demand against a 1-core quota: healthy max `0.00003-0.00005`, starved max `0.48682` — separation **16485x**, the sharpest of any channel here. Rule threshold `0.01` sits 200x above the healthy peak and 16x below the starved minimum sample (`0.167`) — three orders of magnitude of empty space, so the exact value inside it is not load-bearing | **Silence risk, and it already happened once**: the rule needs 20 samples, and `RecentSamples` (`AnomalyGuard.cs:1321-1332`) derives that as `ceil(RecentWindow / window.Step)`. At the deployed 15-minute recent window this holds only while the scrape step stays ≤ 45 s; raise the step and the channel goes permanently `WarmingUp`, which reads exactly like a quiet cluster. `waiting` was chosen over `stalled` because for a single-container cgroup they are the same event (measured side by side: `0.48682` vs `0.48681` under starvation) — binding both would duplicate one signal |
| `LockContentions` (`RS-4`) | `dotnet.monitor.lock_contentions` (via `RuntimeSignalListener`) | Counter, **observable** — reports nothing until `RecordObservableInstruments()` is polled, and reports an absolute value, not a delta | `LoadIndependent` / `Resource` | Contentions/sec | Yes — threshold `0.119`, breach fraction `0.25` | Separates 152x under fault (`0.076 → 11.6/s`, peers flat). Floor **recalibrated once, in production**: first value `0.064` came from a 3-minute window on freshly-started pods (peak `0.0514 × 1.25`) and produced a false positive on an unfaulted pod within the hour; re-measured over a settled 20-minute window, healthy peak was `0.0952/s` — above the old floor — giving the current `0.119` (`0.0952 × 1.25`), ~97x below the fault | Being **observable** is the trap: before `RecordObservableInstruments()` was wired in, the instrument published, the subscription looked correct, and it read a clean, permanent `0` through sixteen contending threads. CPU also moves under this fault (8x, not the sole tell); whether CPU is blind to contention that blocks without spinning is unmeasured |
| `Exceptions` (`RS-5`) | `dotnet.exceptions` (via `RuntimeSignalListener`) | Counter (pushed) | `LoadIndependent` / `Symptom` | Exceptions/sec | Yes — threshold `0.0167`, breach fraction `0.25` | Healthy is **exactly 0** on all twelve pods, so the usual `max × 1.25` says nothing; floor derived as "one exception in 2 minutes" (`0.0083/s`) doubled to `0.0167/s`, ~56x below the injected `0.99/s` | The one of the three runtime channels where **every pre-existing channel is genuinely blind**: measured with half of requests throwing-and-catching, `ErrorRate` stayed at 0 (nothing becomes a 5xx), p95 stayed flat (48.8-48.9 ms), no non-200 response occurred. `InertChannels()` will flag this as constant-zero/`AMBIGUOUS` on a healthy cluster — correct, because that is what a working rare-event channel looks like before the event happens |
| `ActiveRequests` (`RS-3`) | `http.server.active_requests` (via `RuntimeSignalListener`, ASP.NET Core built-in) | Gauge, **pushed** (deltas — `+1` on request start, `-1` on end; must be summed, not last-value-read) | `LoadIndependent` / `Symptom` | In-flight request count | Yes — threshold `3`, breach fraction `0.25` | At rest `0` on all twelve pods; ten-minute maxima `0/1/2` observed | Closes the hung-request gap `LatencyP50/95/99Ms` cannot: fires (`5 → 40`) *two minutes before* p95 moves, and once p95 does move it pins to its saturating bucket edge and carries no magnitude — the two channels' blind spots are complementary, not redundant. The self-scrape must be subtracted at the source, or every pod reads a flat `1.0` permanently and dominates the signal — done, but a fragile assumption to carry forward |
| `GcCommittedBytes` | see the .NET-runtime table above | Gauge | `LoadIndependent` / `Resource` | Bytes | **No** — no `ruleThreshold` configured | minGap `1.0 MB`, minTrendChange `3.0 MB` | See above; `docs/TASKS.md`'s `RS-2` status is stale |
| `ScrapeCoverage` (`AN-D9`) | `up` — Prometheus's OWN series, written on every scrape attempt regardless of outcome, so it has none of a data series' staleness lag | `Ratio` with a **verbatim query**, `avg_over_time(up{%selector%}[15m])` | `LoadIndependent` / `Symptom` | Fraction `[0,1]` of scrape attempts that succeeded | **No** — `requirePersistence: true` instead (two cycles, reusing `SilentPodCycles`) | `calibrated: false` — **exempt from floor fitting on purpose**: a healthy fleet's coverage medians are all 1.0, so `gapMax × 1.25` proposes exactly **0.0**, the value that means "gate off"; and the first time it moves, that event becomes the maximum and the floor lands at 1.25x the only thing the channel exists to report. Useless healthy, harmful faulted | **Three, and the first two are the reason the design was rewritten twice.** A per-pod SCALAR cannot clear `MinimumSamplesPerPeer` (30) — verdict `InsufficientData` every cycle, healthy or faulted. The RAW `up` series cannot clear the materiality gate either, because peer gaps are measured between MEDIANS and the median of a 0/1 series is 1.0 for any pod above 50% coverage — so it only fires below half, which is near-total silence and already `RunSilentPods`' case. Hence the smoothed `avg_over_time`. Third: **it is implemented and NOT deployed** — the live ConfigMap still carries five custom channels, so "built" and "protecting this cluster" are different claims here |

---

## Explored and rejected — `kestrel.queued_connections` / `kestrel.rejected_connections` (`RS-6`)

Both instruments publish cleanly (5 of 5 subscribed via `RuntimeSignalListener`), but **the channel cannot be
delivered by a pull exporter under the very condition it measures**, and this is structural rather than a
missing binding: Kestrel only queues or rejects connections once it is *at* a concurrency limit. Measured with
the limit as the only variable: no limit set, 40 concurrent holders → `6/6` `/metrics` scrapes succeed, queued
always `0`; limit set to 8, same 40 holders → `0/6` scrapes succeed, nothing observable at all — the pod stops
answering the scrape exactly when it has something to report. **Not bound.** Instruments are left exposed
(cost nothing, read 0) but no channel or floor exists for either name, and the lab sets no concurrency limit.

## Not covered at all

- **No `MetricIndex` member is entirely unbound in the lab-workload deployment** — all 13 known channels have
  an entry in `guard.lab-workload.json`. The `overfit`-namespace config (`guard.lab.json`) leaves
  `CpuThrottleRatio` and `ErrorRate` unbound on purpose, "so that list has something in it on the first run".
- No custom channel besides the five above exists in either shipped config.
- `PS-1` (whether PSI CPU pressure should eventually become a `MetricIndex` member rather than staying a
  custom channel) is explicitly undecided — `docs/TASKS.md`.
- Kestrel connection saturation (`kestrel.queued_connections`/`rejected_connections`) has no channel, by
  design, per the section above.

## Contradictions found while writing this — verify before relying on the summary above

The following disagreed with the brief this document was written from, or with each other in the source
material, and are called out rather than silently resolved:

1. **`ErrorRate` in `guard.lab-workload.json` contradicts its own file header.** The header says the channel
   is "left out and reported blind"; the `metrics` block binds it anyway, to the raw error counter with no
   division by request count — see the table entry above. Not something this document's author fixed.
2. **`docs/TASKS.md`'s `RS-2` row ("OPEN, cheap: bind `dotnet_gc_committed_bytes`") is stale.** The series is
   emitted (`Demo/LabWorkload/WorkloadMetrics.cs:178`) and bound as the custom channel `GcCommittedBytes` in
   both the standalone lab config and the deployed ConfigMap today.
3. **The brief's "LockContentions floor recalibrated from 3 minutes to 20 minutes, 0.0514 → 0.0952, floor
   0.119" is confirmed correct** against `docs/TASKS.md` `RS-4` and the config comments verbatim, as are
   every other measured number quoted in the brief (`CpuPressure`, `CpuThrottleRatio`, `MemoryWorkingSetBytes`
   floor, `OomEventsRate`/`ContainerRestarts` peer-blindness, Kestrel's scrape-under-saturation refutation).
4. **`MetricIndex.Count` = 13, `MetricSnapshot.FeatureCount` = 12 — confirmed current**, not merely as stated
   in the brief.
