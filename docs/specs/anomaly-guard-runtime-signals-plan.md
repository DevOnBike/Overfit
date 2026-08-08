STATUS: ANALYSIS_READY
Author: overfit-analyst
Date: 2026-08-08
Slug: anomaly-guard-runtime-signals-plan

# Five ASP.NET Core / .NET runtime signals for the anomaly guard

## Why a sibling file, not a section of the PSI plan

Judged separately because the shape of the work is different, not just the count of metrics. The PSI plan
(`docs/specs/anomaly-guard-psi-cpu-channel-plan.md`) is entirely inside `Sources/Anomalies` — a metric that
already exists on the lab, reached through catalogs this repository already has a precedent for extending.
**None of the five metrics below exist anywhere in this repository today** (grep across the whole tree found
zero hits for any of the five instrument names): the prerequisite is new code in `Demo/LabWorkload`, a
project I am read-only on and outside `Sources/Anomalies` entirely, using an exposition mechanism
(`System.Diagnostics.Metrics.MeterListener` reading built-in framework meters) this repository has not used
before. That prerequisite dominates the risk and the ordering of this plan; keeping it separate stops it from
diluting the PSI plan's much narrower, much closer-to-ready scope. Both plans reuse the same guard-side
playbook (`CustomMetricBinding` vs. `MetricIndex`, `PeerSignalCatalog`, calibration) established there and
cross-reference it rather than repeating it.

## What the client asked for

Relayed by the coordinator, five metrics, each with the gap it closes that the current 13 channels
structurally cannot:

| Metric | Gap it closes |
|---|---|
| `http.server.active_requests` | Latency percentiles only count *completed* requests; a hung request never enters the histogram. In-flight count moves immediately. |
| `dotnet.monitor.lock_contentions` | A latency cliff with normal CPU and normal GC — every current channel quiet. |
| `dotnet.gc.last_collection.memory.committed_size` | Committed-vs-used divergence precedes OOM; `GcGen2HeapBytes` measures used and hides it. Coordinator: measured 0.43x spread between pods on this lab. |
| `dotnet.exceptions` (tagged `error.type`) | First-chance exceptions, distinct from `ErrorRate` — a caught-and-retried exception never becomes a 5xx but often precedes one. |
| `kestrel.queued_connections`, `kestrel.rejected_connections` | Saturation before the application sees it — a rejected connection is invisible to every current channel. |

Instrument names given as verified by the coordinator against current Microsoft specifications
(`System.Runtime` meter, ASP.NET Core built-in metrics), not from memory. This plan treats that as a strong
starting point and still calls out live confirmation as a required task (Scope, below) — this repository's
own convention (`MetricNameCatalog.cs`'s own doc: "the mapping ... left two channels unbound, reporting
blind for hours") is not to trust a name until it has been seen coming out of the actual process.

## Inventory

### Already exists

| Capability | Type / file |
|---|---|
| The whole guard-side ingestion playbook | Identical to the PSI plan: `CustomMetricBinding` (zero source changes, config-only) vs. a new `MetricIndex` member, `PeerSignalCatalog` classification, `FloorCalibrator`, blindness accounting. Nothing new to establish here — see `docs/specs/anomaly-guard-psi-cpu-channel-plan.md` for the full investigation. |
| A hand-rolled Prometheus exposition precedent for a *self-owned* `Meter` | `Sources/Server.AspNet/Services/ServerMetrics.cs` and `Demo/LocalAgentAspNetDemo/Observability/MetricsCollector.cs` — both create their own `Meter` and record into it. **Not the same shape as what this needs** (see Prerequisite, below): those record measurements this project's own code takes; this needs to *read* meters the .NET runtime and ASP.NET Core populate on their own. |
| Confirmation that `Meter` (not the OTel Prometheus exporter) is real-AOT-safe in this repo | `docs/specs/guard-telemetry-meter-plan.md` (a separate, unrelated plan about the guard's *own* self-monitoring telemetry — flagged in its own banner as containing some fabricated content, but its `Fact` rows are independently verified) records that the `Meter` API is compiled under a real Native-AOT publish via the `aot-guard` CI job through `Sources/Cli`. Relevant here only as precedent that the BCL `Meter` type itself carries no AOT risk; `Demo/LabWorkload` is not part of that CI job, so this does not by itself clear the new code. |
| A load-generation and fault-injection surface on the current lab | `Demo/LabWorkload` (`/fault/latency`, `/fault/stall`, `/fault/errors`, `/fault/leak`, `/fault/cpu`, `/fault/oom`, `/fault/crash`) and `Demo/LabLoadDriver` (in-cluster, RPS-paced). Relevant to which of the five channels can actually be calibrated on this lab — see per-metric table below. |

### Partially exists — the one concrete, cheap finding

**`Demo/LabWorkload/WorkloadMetrics.cs:111` already emits `dotnet_gc_committed_bytes`** (`GC.GetGCMemoryInfo().TotalCommittedBytes`), and **neither `guard.lab.json` nor `guard.lab-workload.json` binds it.** This is not the exact instrument the client named (`dotnet.gc.last_collection.memory.committed_size` is generation-tagged and scoped to the last collection; `TotalCommittedBytes` is the current whole-process committed figure) but it is a measured proxy for the same phenomenon — committed-vs-used divergence — **available today with zero code changes**, the same "add a config line" move as the cheap half of the PSI plan. Recommend binding it under `CustomMetricBinding` immediately, independent of and prior to the exposition work below, as a fast, honest partial answer to that one row of the client's table while the real instrument is built.

### Does not exist

- All five instrument names, anywhere in the repository (grep, zero hits).
- Any code in `Demo/LabWorkload` that reads a foreign `Meter` (`System.Runtime`, `Microsoft.AspNetCore.Hosting`, `Microsoft.AspNetCore.Server.Kestrel`) via `MeterListener` or any other mechanism. `WorkloadMetrics.Render()` is entirely hand-computed from direct API calls (`GC.GetTotalMemory`, `GC.GetGCMemoryInfo`, `ThreadPool.PendingWorkItemCount`, `Environment.WorkingSet`) — a different, simpler shape than subscribing to a `Meter` neither this project nor the .NET team's own instrumentation exposes as a public API surface beyond the `Meter`/`MeterListener` pair.
- Any fault-injection endpoint that specifically drives lock contention, connection queueing/rejection, or a thrown-and-caught exception. The closest existing fault (`/fault/errors`) returns a deliberate 500 status code (`Results.StatusCode(500)`), not a `throw` — it will not move `dotnet.exceptions`.

## The prerequisite, established rather than assumed

**`Demo/LabWorkload` uses `WebApplication.CreateSlimBuilder(args)`** (`Program.cs:37`) — Kestrel is still the
server, so the built-in ASP.NET Core Hosting and Kestrel meters populate automatically regardless of builder
type (this is framework behaviour, not opt-in code); the `System.Runtime` meter is likewise always-on since
.NET 8. **Populating these meters requires no code change. Reading and exposing them does.** The idiomatic,
AOT-clean way to read a foreign `Meter` without the reflection-heavy OTel exporter package (consistent with
this repository's own stated reason for avoiding that package in `ServerMetrics.cs`) is
`System.Diagnostics.Metrics.MeterListener`: subscribe to `"System.Runtime"`, `"Microsoft.AspNetCore.Hosting"`
and `"Microsoft.AspNetCore.Server.Kestrel"`, register a typed `SetMeasurementEventCallback<T>` per numeric
type actually emitted, and accumulate into the same kind of interlocked fields `WorkloadMetrics` already
renders. This is a genuinely different code shape from anything in this repo today (nothing currently
consumes a *foreign* `Meter` — both existing precedents create and record into their own).

**Three things this plan cannot verify from here and states as required spikes, not facts:**

1. **Exact instrument names on this repo's actual runtime.** The coordinator verified the names against
   current Microsoft specifications; this repo targets `net10.0` (`LabWorkload.csproj:4`), and runtime meter
   instrument names/units have moved between preview and GA before. The oracle is empirical:
   `dotnet-counters monitor -p <pid> --counters System.Runtime,Microsoft.AspNetCore.Hosting` (or a throwaway
   `MeterListener` dump) against the actual published `lab-workload` binary, before any rendering code is
   written against a guessed name.
2. **Cost of adding a `MeterListener` and whether it disturbs the lab as a measurement instrument.** `Program.cs`'s
   own comments show this project already measures the workload's resource profile carefully (the leak-fault
   comment block documents an 8 KB-vs-542%-heap-rise investigation). A `MeterListener` subscribed to
   `Microsoft.AspNetCore.Hosting`'s per-request histogram adds a callback on every request — plausibly small
   next to what `WorkloadMetrics.Observe()` already does per request, but "plausibly small" is exactly the
   kind of claim this codebase's own house rule (`CLAUDE.md`, "Performance work — measure, don't assume")
   says must be measured, not asserted. Recommend the developer benchmark request throughput/allocations
   before/after with `MemoryDiagnoser`, the same discipline as any other perf-adjacent change, even though
   `Demo/LabWorkload` is outside `Sources/Main`'s formal benchmark obligations.
3. **Whether the current load shape can calibrate `queued_connections`/`lock_contentions` at all.**
   `Demo/LabLoadDriver/Program.cs:24-28` documents, in its own comments, a deliberate design choice: paced by
   requests-per-second rather than fixed concurrency, specifically so that "a replica that slows down
   receives LESS traffic" rather than backing up a queue of waiting workers. That is the right choice for the
   experiments it was built for and is simultaneously the wrong shape for saturating a connection queue or
   provoking lock contention through sheer concurrent volume — those need a concurrency *spike*, which the
   current driver does not produce and was not designed to. Calibrating either signal needs either a new load
   shape or a new fault endpoint; neither exists today.

## Per-metric classification

Applying the same discipline the PSI plan used for `PeerSignalCatalog`, and the coordinator's own reasoning
("Lock contention and active requests both scale with traffic; committed heap does not"):

| Metric | `MetricSourceKind` | `SignalKind` (recommended) | Reasoning | `SignalClass` | Calibratable on this lab today? |
|---|---|---|---|---|---|
| `http.server.active_requests` | `Gauge` (current value) | `LoadSensitive` | In-flight count scales with traffic by construction (more concurrent requests, more in-flight) — the anomaly is active-requests *relative to* `RequestsPerSecond`, not the raw count. Matches `ThreadPoolQueueLength`'s precedent exactly ("a queue is work waiting; comparing depths under different arrival rates compares the rates"). | `Symptom` (what the caller experiences — a stall shows up here before anywhere else) | **Yes, plausibly** — `/fault/stall` already makes requests sit unfinished; this is the one metric of the five with an existing fault to calibrate against. |
| `dotnet.monitor.lock_contentions` | `Counter` (monotonic, rate-wrapped) | `LoadSensitive` | Coordinator: "scale with traffic." Matches `GcPauseRatio`'s precedent (collection time is driven by allocation, allocation by requests served) — contention is driven by concurrent access, concurrency by requests served. | `Resource` (the workload's own internal consumption) | **No** — no existing fault produces contention; the RPS-paced driver does not create a concurrency spike (see Prerequisite, point 3). Needs new lab infrastructure. |
| `dotnet.gc.last_collection.memory.committed_size` | `Gauge` | `LoadIndependent` | Coordinator: "committed heap does not [scale with traffic]." Matches `MemoryWorkingSetBytes`/`GcGen2HeapBytes`'s precedent — a fixed-cost class PeerSignalCatalog's own doc warns against dividing. | `Resource` | **Partially, already** — the existing `/fault/leak` fault already produces committed/used divergence via LOH growth (per `Program.cs`'s own comments on the leak mechanism); the coordinator's cited 0.43x spread was presumably observed from this without a dedicated fault. The cheap proxy (`dotnet_gc_committed_bytes`, see Inventory) can be bound and calibrated **now**, ahead of the real instrument. |
| `dotnet.exceptions` (`error.type`) | `Counter`, **design choice flagged below** | See note | Coordinator did not classify; the closest existing sibling, `ErrorRate`, is `LoadIndependent` — but only because it is built as a ratio (`rate(5xx)/rate(total)`) at the PromQL layer, not because errors are inherently load-independent. `dotnet.exceptions.count` is a raw count. **Recommend the same treatment as `ErrorRate`**: wrap it as a rate-over-rate at the query layer and classify `LoadIndependent`, for consistency with the one metric already in this codebase that answers the same kind of question — rather than leaving it a raw `LoadSensitive` count divided at peer-comparison time. This is a developer/architect call, not settled here. | `Symptom` (an exception is closer to the user-visible failure than to the resource that caused it, matching `ErrorRate`'s own class) | **No** — `/fault/errors` returns a deliberate status code, not a `throw`; nothing in the lab currently causes a first-chance exception on demand. |
| `kestrel.queued_connections` | `Gauge` | `LoadSensitive` | Direct match to `ThreadPoolQueueLength`'s precedent — a connection queue is work waiting for a scarce resource (an accepted-connection slot), and its depth scales with arrival rate for the same reason a thread-pool queue does. | `Infrastructure` (platform-level denial-before-the-app-sees-it, same class as `CpuThrottleRatio`/PSI) | **No** — see Prerequisite, point 3; needs a concurrency spike the current driver does not produce. |
| `kestrel.rejected_connections` | `EventCount` | `LoadIndependent` + `IsCountedEvent = true` | A healthy Kestrel server rejects ~0 connections; any non-zero is itself the finding, exactly `ContainerRestarts`/`OomEventsRate`'s precedent, not something to divide by traffic. | `Infrastructure` | **No** — same gap as `queued_connections`. |

## Which of the five earn a channel now, and which stay config-level (or later)

Applying the PSI plan's own rule — no invented floors, ship what can be honestly calibrated, defer the rest:

- **`http.server.active_requests`**: earns a channel in this slice. It is the highest-value gap per the
  client's own framing, and it is the one signal here with an existing fault (`/fault/stall`) to calibrate
  against. Ship via `CustomMetricBinding` first, same sequencing recommendation as PSI.
- **Committed-vs-used divergence**: ship the existing `dotnet_gc_committed_bytes` proxy **now**, as a
  `CustomMetricBinding`, zero code changes, calibratable against the existing leak fault. Treat the exact
  `dotnet.gc.last_collection.memory.committed_size` instrument as a **Could** — a refinement once the
  `MeterListener` exposition work happens anyway for the other metrics, not a reason to delay the proxy.
- **`dotnet.monitor.lock_contentions`, `kestrel.queued_connections`, `kestrel.rejected_connections`,
  `dotnet.exceptions`**: recommend these stay **config-level only, unfloored, reported without a calibrated
  rule** even after exposition ships — none has an existing fault to validate against, so any threshold
  would repeat the exact `ContainerRestarts`-floor-of-1.25 mistake this codebase has already paid for once.
  Whether any of them is worth a permanent `MetricIndex` member (vs. staying a `CustomMetricBinding` forever)
  is a question for after a calibration path exists, not before.

## Gate answers

- **Execution path**: neither, for the guard-side half (same as PSI). The exposition-layer half touches
  `Demo/LabWorkload`, which has no `InferenceEngine`/`ComputationGraph` involvement at all — it is a plain
  ASP.NET Core minimal API.
- **Verification oracle**: for exposition — a `dotnet-counters`/`MeterListener` dump against the actual
  running binary confirming instrument names and units exist as claimed (Prerequisite, point 1), and a
  scrape of `/metrics` showing the rendered series with plausible values under the corresponding fault where
  one exists. For guard-side ingestion — identical oracle shape to the PSI plan (PromQL string equality,
  end-to-end peer-finding test, blindness accounting).
- **AOT reach**: `Demo/LabWorkload` carries `IsAotCompatible=true` and is explicitly "verified AOT-clean,
  published JIT" per its own `.csproj` comment — new code here must preserve that (the trim/AOT analyzers run
  on every ordinary build per that comment, so a reflection-heavy addition would be caught, not silently
  shipped). `MeterListener` itself is plain BCL with no reflection in its typed-callback API, so it should be
  compatible in kind with the existing discipline — not yet verified by an actual analyzer run against new
  code, because that code does not exist yet.
- **Allocation policy**: `Demo/LabWorkload` is not part of `Sources/Main`'s formal zero-allocation contract,
  but it is explicitly used as a *measurement instrument* for the guard (`README.md`: "Ground truth was
  verified before it was trusted"), so a `MeterListener` that meaningfully changes its own resource profile
  would contaminate every experiment run against it after — this is why Prerequisite point 2 is treated as a
  required measurement, not a nice-to-have.
- **Moat side**: open/offline-batch, same as PSI — lab tooling and detection-quality work, not real-time or
  GPU.

## Not-settled-fact table

| Type | Item |
|---|---|
| Fact | Zero references to any of the five instrument names exist anywhere in this repository — verified by `grep` across the whole tree. |
| Fact | `Demo/LabWorkload/WorkloadMetrics.cs:111` already emits `dotnet_gc_committed_bytes`, unbound in both committed guard configs — verified by reading the file and both JSON configs. |
| Fact | `Demo/LabLoadDriver` is deliberately RPS-paced, not concurrency-fixed, per its own code comments (`Program.cs:24-28`) — a design choice that works against calibrating connection-queue or lock-contention signals without new lab infrastructure. |
| Fact | `Demo/LabWorkload`'s only current fault that touches HTTP error responses (`/fault/errors`) returns a status code directly, not a thrown exception — verified by reading `Program.cs:158-163,121`. It will not exercise `dotnet.exceptions`. |
| Fact | `Demo/LabWorkload` uses `WebApplication.CreateSlimBuilder` with Kestrel as the server (`Program.cs:37`) — the built-in ASP.NET Core Hosting and Kestrel meters populate automatically without code changes; only *reading* them needs new code. |
| Assumption | The five instrument names and their exact semantics match the coordinator's citation of current Microsoft specifications. **Not verified against this repo's actual `net10.0` runtime** — treated as a required spike (Scope), not a fact. |
| Assumption | `dotnet.exceptions` should be built as a pre-divided ratio (like `ErrorRate`) rather than a raw `LoadSensitive` count. Recommended by the analyst for consistency with the one existing sibling; not yet confirmed by the developer/architect. |
| Assumption | A `MeterListener` subscription's overhead on `Demo/LabWorkload` is small enough not to disturb its role as a measurement instrument. **Not measured** — flagged as a required benchmark before shipping (Prerequisite, point 2), consistent with `CLAUDE.md`'s "measure, don't assume" rule. |
| Constraint | Same `MetricSnapshot`/`FeatureCount` (12) constraint as PSI — none of these five reach the learned family in this plan. |
| Constraint | Changing `Demo/LabWorkload` requires a host `dotnet publish` + `docker build` + redeploy cycle (`Demo/LabWorkload/Dockerfile`'s own header comments) — not a hot-reload; each iteration of the exposition work has a real, if unquantified-in-hours, cycle cost. |
| Open question | **For the client** — priority among the five. All five are asked for; only `active_requests` and the committed-bytes proxy have an existing fault to calibrate against today. Does the client want the other three shipped unfloored-and-uncalibrated now, or held until lock-contention/connection-saturation fault injection exists? |
| Open question | **For the developer/architect** — is a new fault-injection endpoint (a deliberate concurrency spike, a lock-contention generator, a `throw`-based error mode) in scope as part of this work, or a separate follow-on? It is the only way three of the five channels get calibrated at all. |
| Open question | **For the developer/architect** — confirm the `dotnet.exceptions`-as-ratio design choice (table above) before writing the PromQL/config for it. |

## Scope

### Must (ordered — highest technical uncertainty first, per this repo's own ordering rule)

1. **Spike: confirm the five instrument names and units on a running `net10.0` `lab-workload` binary** via
   `dotnet-counters` or a throwaway `MeterListener` dump. Settles Prerequisite point 1. No further exposition
   code should be written against a guessed name.
2. **Spike: measure whether a `MeterListener` subscription changes `Demo/LabWorkload`'s own resource
   profile** meaningfully (throughput, allocations per request, working set) — before shipping it as the
   lab's measurement instrument for anything else. Settles Prerequisite point 2.
3. Bind the existing `dotnet_gc_committed_bytes` proxy via `CustomMetricBinding` — zero code changes, ships
   independent of and ahead of everything else in this plan.
4. Implement the `MeterListener`-based renderer in `Demo/LabWorkload`, following `WorkloadMetrics`'s existing
   Prometheus-text conventions, for `http.server.active_requests` first (the metric with an existing fault to
   calibrate it against).
5. Bind `http.server.active_requests` via `CustomMetricBinding`, calibrate against `/fault/stall`.

### Should

- Extend the renderer to the remaining four instruments once the `MeterListener` mechanism is proven (steps
  1-4 above de-risk the mechanism itself), shipped observable/unfloored per the per-metric table.
- Resolve the `dotnet.exceptions`-as-ratio design question with the developer/architect before wiring its
  PromQL.

### Could

- A new fault-injection endpoint for lock contention and/or a concurrency-spike load mode in
  `Demo/LabLoadDriver`, to make the remaining three channels calibratable. Real work, not a config change —
  flagged as a `Could` rather than a `Must` because the client did not ask for new fault infrastructure,
  only for the signals.
- Promoting any of the five from `CustomMetricBinding` to a `MetricIndex` member, once calibrated and once
  the client confirms (as in the PSI plan) that it should ship as a permanent product default rather than
  lab-only config.

### Won't (this time)

- No calibrated `SustainedThresholdRule`/absolute floor for `lock_contentions`, `queued_connections`,
  `rejected_connections`, or `exceptions` — no existing fault validates any of them.
- No change to `MetricSnapshot`/`FeatureCount`/the learned family — same constraint as PSI.
- No new fault-injection endpoint in this slice (see Could) — the client asked for signals, not for new lab
  infrastructure to calibrate them, and that is a separate, larger piece of work the client has not yet
  scoped.
- No production (non-`Sources/Main`) benchmark obligation beyond the one spike above — `Demo/LabWorkload`
  is not under the formal `Sources/Benchmark` regime, but the spike is still required because it is the
  guard's own measurement instrument.

## Priorities (MoSCoW) — see Scope above; repeated here for the template's sake

Must/Should/Could/Won't as listed in Scope. No performance target was attached to any of the five metrics by
the client, so none of this carries a benchmark *claim* obligation beyond the resource-profile spike, which
exists to protect the lab as an instrument, not to prove a speed number.

## Tasks (user stories)

### T1 — As the developer about to write exposition code, I want the five instrument names confirmed against the actual running binary, so that the renderer is not built against a name that does not exist on this runtime.

**Acceptance criteria**:
- Given a published, running `lab-workload` process on `net10.0`, when `dotnet-counters monitor` (or an
  equivalent throwaway listener) is run against it, then each of the five instruments is observed by its
  exact name and unit, or its absence/rename is recorded plainly.

**NOT READY as a coding task** — it is the spike itself; nothing downstream should be built before it runs.

### T2 — As the operator running the lab, I want to know a new `MeterListener` has not changed what the lab-workload measures, so that every experiment run against it afterwards is still trustworthy.

**Acceptance criteria**:
- Given the `MeterListener` code added, when request throughput and per-request allocations are measured
  before/after under identical load, then the difference is reported with a number, not asserted — matching
  this repo's own "measure, don't assume" rule.

**Ready** — small, oracle is a before/after comparison, no dependency on T1's outcome for the harness itself
(only on the code it measures).

### T3 — As the operator watching a deployment, I want a hung-request signal that is not hidden by completed-request latency percentiles, so that a stalled dependency is visible immediately rather than after enough requests time out to move a histogram.

**Acceptance criteria**:
- Given `/fault/stall` set to a nonzero probability and duration, when the guard runs a cycle, then
  `http.server.active_requests` (bound as `CustomMetricBinding`, `LoadSensitive`) shows a materially higher
  value on the faulted pod than its healthy peers, and a peer finding is reported naming it.
- Given no fault set, when the guard runs a cycle, then the channel does not fire — the calibration check
  against a healthy baseline this repository always requires before trusting a threshold.

**NOT READY** until T1 confirms the instrument name/shape and T4 (the `MeterListener` renderer) exists.

### T4 — As the guard operator, I want the committed-vs-used memory divergence signal available now, without waiting for the full exposition work, so that one of the five gaps closes immediately.

**Acceptance criteria**:
- Given `dotnet_gc_committed_bytes` bound via `CustomMetricBinding` (`Gauge`, `LoadIndependent`, `Resource`),
  when `/fault/leak` is active, then the channel shows the divergence the coordinator's 0.43x figure
  describes, and a peer or trend finding is reported.

**Ready** — zero code changes, only a config entry and a fixture test in the shape of
`CustomMetricEndToEndTests`.

## Traceability

| Goal | User need | Task | Acceptance criterion | Verified by |
|---|---|---|---|---|
| No unverified instrument name reaches production code | Developer needs ground truth before building | T1 | Names/units confirmed live | `dotnet-counters` / listener dump |
| The lab stays a trustworthy measurement instrument | Operator needs experiments run after this change to remain comparable to before it | T2 | Before/after resource-profile delta reported | Manual benchmark, `MemoryDiagnoser` or equivalent |
| Hung requests are visible before they age into a latency percentile | Operator needs the highest-value gap the client named, closed first | T3 | Peer finding on `/fault/stall`, silence on healthy baseline | End-to-end test |
| Committed-vs-used divergence visible without waiting on exposition work | Operator gets partial value immediately | T4 | Finding on `/fault/leak` | End-to-end test |

## SUGGESTED IMPROVEMENTS TO MY ROLE

**One real gap, surfaced by this task specifically.** My default instructions for round one point at
`overfit-navigator`'s semantic tools first; for this request the highest-value finding — `guard.lab.json`
targets a superseded deployment, and the actually-live config is a third, untracked file at
`Tests/bin/lab-guard.json` — came entirely from reading plain-text config files, a README, and a code
comment, none of which the navigator (a C#-symbol tool) can see. The instructions already say to fall back
to `Grep`/`Glob` when the navigator server is down; they do not say to reach for that fallback by *content
type* (JSON config, Markdown, code comments) even when the server is up and answering fine. Worth a line
noting that config/doc archaeology is always a `Grep`/`Read` job, navigator or not — I did not lose time to
this today because I happened to follow the thread anyway, but a narrower reading of "use the navigator
first" could plausibly have stopped short of `Tests/bin/lab-guard.json` and reported the wrong live config
with high confidence.
