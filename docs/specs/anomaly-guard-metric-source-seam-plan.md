# Spec/Plan: `AnomalyGuardService` metric-source seam (enable replay)

STATUS: APPROVED - Tasks 1 and 2 IMPLEMENTED + VERIFIED + REVIEWED and COMMITTED 2026-08-08 (e6398fa, e69645e). Tasks 3 and 4 not started, both lab-dependent. OPEN for Task 3: RunCycleAsync returns null for both "no window" and "cycle threw" - decide the shape before a consumer exists. Success metric (determinism) NOT yet measured end to end: it needs Task 3, and the plan's replay-duration claim still lacks a pod count (architect Finding 9).

Owner of this file: `overfit-analyst` (this document) → `overfit-architect` signs DESIGN before
`overfit-developer` writes source. See `.claude/skills/overfit-spec/SKILL.md` for the gate.

## What the client asked for

> "Wire `AnomalyGuardService` to the `IMetricSource` abstraction so a **recorded or historical
> window** can drive the anomaly guard, instead of only a live Prometheus."

Backed by `docs/aiops/aiops-backlog.md`, **"Decision 2026-08-07 — replay and simulator first, the
live day only when unavoidable"**: the A1 24-hour lab run produced 11 incidents; for a counting
process that is a 95% interval of ~5.5–19/day against a pass criterion of 1–9/day, so **one day at
this event rate cannot decide the criterion, and neither can a second day** — the limit is the
event count, not the clock. The decision's own **"Revised order"** lists this exact task as item 1,
labelled *"the enabler for everything below."*

## Inventory — verified via the semantic navigator (not grep), 2026-08-08

**Navigator queries run and what each returned** (per the hard requirement in the task):

| query | result |
|---|---|
| `find_implementations(IMetricSource)` | **0 implementations, anywhere.** This interface is not implemented by any type in the repository — not `PrometheusMetricWindowSource`, not a test double, nothing. |
| `find_implementations(IRawMetricSource)` | 2: `PrometheusMetricSource` (production) and `Tests/Anomalies/LiveMonitoringPipelineTests.ScriptedRawMetricSource` (test double). Both belong to `LiveMonitoringPipeline`, a **different** pipeline (uses `IAlertSink`/`AlertEvent`, not `IIncidentSink`/incidents). |
| `find_references(PrometheusMetricWindowSource)` | 13: two DI registration lambdas in `AnomalyGuardRegistration.cs`, three sites in `AnomalyGuardService.cs` (field, ctor param, doc comment), and 8 test sites (`PrometheusMetricWindowSourceTests`, two `[LabFact]` diagnostics). No other production consumer. |
| `find_references(PrometheusHistoricalSource)` | 11: consumed internally by `PrometheusMetricWindowSource.ReadAsync`/`FillCustomAsync`, by `HistoricalCsvLoader`'s XML doc (not code), by `PromqlCatalog`, and by 3 test/diagnostic files that construct it directly for one-shot historical range queries. |
| `find_callers(AnomalyGuardService)` (type) | 0 — `find_callers` targets methods; re-ran as `find_references` below. |
| `find_references(AnomalyGuardService)` | 6: `AddGuardCore` (`services.AddSingleton<AnomalyGuardService>()` + `AddHostedService(provider => …)`), the options-doc XML comment, two ctor-body self-references, and **one CLI resolution site** — `Sources/Cli/AnomalyGuardCommand.cs:177`, `host.Services.GetRequiredService<AnomalyGuardService>()`. **Zero direct `new AnomalyGuardService(...)` calls anywhere** (confirmed separately by `Grep`, which found none) — every construction goes through DI. |

### The central finding: neither existing interface has the right *shape*, not just the wrong *implementer*

This matters more than "which interface to point the field at" — it changes what "wire it to the
interface" actually requires.

| | `IMetricSource` | `IRawMetricSource` | what `AnomalyGuardService` actually calls |
|---|---|---|---|
| method | `ValueTask<MetricSnapshot> ReadAsync(ct)` | `Task<List<RawMetricSeries>> ReadAsync(ct)` | `Task<MetricWindow?> ReadAsync(DateTimeOffset end, TimeSpan window, ct)` |
| scope | one pod, one instant | all pods, "now" | all pods, an explicit historical **range** |
| takes a time range? | **no** | **no** | **yes — this is the whole point of replay** |
| implementers | **0** | `PrometheusMetricSource`, a test double | `PrometheusMetricWindowSource` only |

Both existing interfaces are **point-in-time reads by construction** — neither method takes a
start/end or "as of" argument. A recorded/historical window cannot be requested through either one
regardless of what implements it. `IMetricSource` is additionally unimplemented dead code today.
**"Give it the interface" undersells the work**: a *new* interface, shaped around
`PrometheusMetricWindowSource.ReadAsync(DateTimeOffset end, TimeSpan window, ct) -> MetricWindow?`,
is what the seam actually needs — extracting an interface from the existing concrete class, not
retrofitting either of the two named ones. `MetricSourceKind` (Gauge/Counter/EventCount/
HistogramSeconds/Ratio) is unrelated to any of this despite the name — it governs PromQL shape for
`PromqlCatalog`, not source pluggability.

### The second finding: the seam alone does not deliver "a day evaluates in seconds"

`AnomalyGuardService.RunOneCycleAsync` (private, `AnomalyGuardService.cs:380`) computes
`var now = DateTimeOffset.UtcNow;` **internally**, then reads `end = now - EndOffset` and passes
`now` into `_guard.RunCycle(window, now, ...)`, where it is documented as *"Cycle timestamp, used
for incident ages"* and also drives `ProposeFloors`'s hourly cadence gate. Swapping only the field
type does not let a caller feed historical timestamps through the loop — `RunOneCycleAsync` would
still ask for data at real wall-clock "now" no matter what implementation the field holds.

This is smaller than it sounds, and the codebase already has the pattern to copy:
`AnomalyGuard.Acknowledge(..., DateTimeOffset now)` already takes `now` as an explicit parameter,
with the comment *"Clock, supplied so tests do not depend on the wall clock"* — i.e. `AnomalyGuard`
itself was **already designed to avoid an internal wall-clock dependency**; the dependency is
localised entirely to the host loop in `AnomalyGuardService`. `Grep` across `Sources/Anomalies` for
`DateTime.UtcNow`/`DateTimeOffset.UtcNow` found exactly one call inside the cycle logic
(`AnomalyGuardService.cs:386`); the others are in the unrelated `LiveMonitoringPipeline`/
`AlertEngine`, in a one-time template default in `AnomalyGuardRegistration.cs` (overwritten every
cycle, so inert), and in `AnomalyGuard.cs:238`'s `restoredAt ?? DateTimeOffset.UtcNow` fallback used
once at construction when no explicit `restoredAt` is supplied (`AnomalyGuardService` currently
passes `restoredAt: null`).

So: **making replay actually replay needs `observedAt`/`now` to become a parameter of a per-cycle
method, not just the source field to become an interface.** This is what the task description's
"virtual clock" is pointing at. Given the existing `Acknowledge(..., now)` precedent, it is **not**
a new clock abstraction (no `TimeProvider`, no `IClock`) — it is the same small, already-idiomatic
move applied to one more method. See "Gate answers" below for the recommended shape and why it is
separable from `ExecuteAsync`'s `PeriodicTimer`.

### Already exists, do not rebuild

- `IMetricSource` / `IRawMetricSource` — `Sources/Anomalies/Monitoring/Abstractions/` (wrong shape for this seam, see above; do not extend either).
- `PrometheusMetricWindowSource` — `Sources/Anomalies/Monitoring/PrometheusMetricWindowSource.cs`. Already does the alignment/staleness/grid work a replay source needs; its `ReadAsync(end, window, ct)` **already supports arbitrary historical ranges** — it builds a fresh `PrometheusHistoricalSourceConfig` with `RangeStart/RangeEnd` per call, so pointing `end` at a day-old timestamp against a live Prometheus that still retains that data **already works today**, once something other than `AnomalyGuardService.RunOneCycleAsync` drives it with a non-`UtcNow` timestamp. `AnomalyGuardShadowRunDiagnostics.RunsInShadowAgainstTheLab` (`Tests/Anomalies/Diagnostics/`) proves this pattern by hand-rolling the cycle loop outside `AnomalyGuardService` entirely — which is itself evidence the seam is missing: that test duplicates `RunOneCycleAsync`'s logic because there was no reusable, parameterisable entry point to call instead.
- `PrometheusHistoricalSource` — `Sources/Anomalies/Monitoring/PrometheusHistoricalSource.cs`. The XML doc's claim *"nothing in `Sources/Main` consumes this class"* is true as written (Anomalies ≠ Main) but reads as stronger than that — it **is** consumed, by `PrometheusMetricWindowSource` internally and by three test/diagnostic files directly. Not a drift finding worth its own task, but noted so nobody re-derives "is this dead code" from the comment alone.
- `HistoricalCsvLoader` — `Sources/Anomalies/Monitoring/HistoricalCsvLoader.cs`. Loads/saves `IReadOnlyList<MetricSnapshot>` (one row per pod per instant, all metrics as named columns) to/from CSV. **A different shape from what a windowed replay source needs** — it has no notion of a multi-sample time series per pod/metric, only discrete snapshots, so it cannot serve `ReadAsync(end, window, ct) -> MetricWindow?` without new aggregation logic.
- The lab fixture recorder — `Tests/Anomalies/Diagnostics/LabFixtureRecorderDiagnostics.cs`, `[LabFact]`. Writes a **third, different** CSV-like format (`# comment` header, `timestamps_ms,...` row, then one row per `(metric, pod)` with NaN for gaps) specifically so calibration decisions "can be repeated, extended and run in CI." This shape is much closer to what a replay source needs (multi-sample, multi-pod, multi-metric, gap-aware) but nothing reads it back into a `MetricWindow` today — it exists to be compared against by hand ("six numbers copied off a Grafana panel found three simulator bugs"), not to drive the guard.
- `AnomalyGuardEndToEndDiagnostics` / `AnomalyGuardShadowRunDiagnostics` (`Tests/Anomalies/Diagnostics/`) — both hand-roll "read a historical Prometheus window, run detectors" outside `AnomalyGuardService`, which is exactly the duplication a proper seam removes. Good precedent for the new interface's test double (`ScriptedRawMetricSource` in `LiveMonitoringPipelineTests` is the sibling pattern to copy, one level down).
- DI wiring test — `Tests/Anomalies/WorkloadIdentityTests.cs` builds a real `ServiceProvider` via `AddOverfitAnomalyGuard(...)` and resolves `AnomalyGuardServiceOptions`, but **does not** resolve `AnomalyGuardService` itself — so today nothing catches a broken DI graph for this exact type. Cheap to add; see acceptance criteria.

### Does not exist

- Any interface shaped `ReadAsync(DateTimeOffset end, TimeSpan window, ct) -> MetricWindow?`.
- Any unit test that constructs `AnomalyGuardService` at all — **zero today** (`Grep`/navigator both confirm). Its cycle logic (blind-metric reporting, floor proposals, stale-pod logging) is exercised only by `[LabFact]` diagnostics against a live cluster.
- Any offline, fully-disconnected ("recorded", no live Prometheus reachable) implementation of the seam. Two incompatible fixture shapes exist (`HistoricalCsvLoader`'s snapshot CSV, the lab recorder's custom CSV) and neither is wired to feed the guard.
- A replay driver/CLI command that actually walks a window across many cycles without wall-clock delay. `AnomalyGuardShadowRunDiagnostics` is the closest thing and it still `Task.Delay`s between cycles against a live cluster — it demonstrates the shape of a driver, not a fast one.

## Problem / user need / business goal / proposed solution

| | |
|---|---|
| **Problem** | A1's 24-hour live run produced 11 incidents; the pass criterion (1–9/day) cannot be decided by that count or by repeating the day — the interval spans both verdicts at any single-day sample size. Every threshold change today can only be evaluated by running the cluster for another day (or more), which is both slow and non-repeatable (no two live days see the same traffic). |
| **User need** | Whoever tunes the guard's thresholds (today: the person iterating on `MinAbsoluteGap`/`MinAbsoluteTrendChange`/window/cadence) needs to re-evaluate a fixed body of history after each change and compare results directly — the same question `AnomalyGuardEndToEndDiagnostics`'s doc comment names: *"a comparison that can be repeated, extended and run in CI."* |
| **Business goal** | Decide the 1–9/day criterion on hundreds of events instead of eleven, in seconds instead of a day, so threshold tuning stops being gated on live-cluster clock time. |
| **Proposed solution (client's)** | Wire `AnomalyGuardService` to `IMetricSource` so a recorded/historical window can drive it. **Treated as one candidate, not the requirement** — see the finding above: the two named interfaces are the wrong shape, and a new one is what actually satisfies the goal. |
| **Success metric** | **Decision — client, 2026-08-08**: determinism is the bar, confirmed as stated — "byte-identical results replaying the same window/config twice" is correct; wall-clock replay time is explicitly *not* the metric. (Originally inferred from "repeatability," which the backlog decision names as the win a live run "can never offer"; now confirmed rather than assumed.) The testable form is unchanged: **replaying the same historical window with the same `AnomalyGuardOptions` twice must produce byte-identical incident counts, open/resolve timestamps and floor proposals.** |

## Uncertainty table

| type | item | detail |
|---|---|---|
| Fact | `IMetricSource` has 0 implementations anywhere in the repo. | `find_implementations(IMetricSource)`, 2026-08-08. |
| Fact | `IRawMetricSource` belongs to `LiveMonitoringPipeline`, a pipeline `AnomalyGuardService` does not use (different sink type, `IAlertSink` vs `IIncidentSink`). | `find_implementations(IRawMetricSource)` + read of `LiveMonitoringPipelineTests.cs`. |
| Fact | Neither existing interface's `ReadAsync` takes a time range; `AnomalyGuardService` needs one. | Read of `IMetricSource.cs`, `IRawMetricSource.cs`, `PrometheusMetricWindowSource.cs:140`. |
| Fact | `AnomalyGuardService` is constructed only via DI in this repo; zero direct `new AnomalyGuardService(...)` call sites. | `find_references(AnomalyGuardService)` + `Grep "new AnomalyGuardService\("` (0 hits). |
| Fact | `Sources/Anomalies/Anomalies.csproj` sets `IsPackable=false`, with an explicit comment that this is deliberate until "somebody wants to reference the guard from their own host." | Read of `Anomalies.csproj`. |
| Fact | `AnomalyGuard.Acknowledge` already takes `now` as an explicit parameter for the stated reason "so tests do not depend on the wall clock" — the pattern this plan proposes extending to `RunOneCycleAsync` already exists one class down. | Read of `AnomalyGuard.cs`. |
| Fact | Zero unit tests construct `AnomalyGuardService`; its cycle logic is covered only by `[LabFact]` diagnostics needing a live cluster. | `Grep "new AnomalyGuardService("` across `Tests/` (0 hits) + read of `WorkloadIdentityTests.cs`. |
| Fact | `Sources/Anomalies` is explicitly **not** a hot path — "the guard runs once every five minutes and none of those rules is about its hot path" (project comment); allocations per cycle are the existing norm. | `Anomalies.csproj` header comment; corroborated by existing per-cycle `new List<string>`, `Array.Clear` etc. in `AnomalyGuardService.cs`/`PrometheusMetricWindowSource.cs`. |
| Decision — client, 2026-08-08 | The constructor signature change is a clean break (no back-compat overload). | Client confirmed: no out-of-repo consumer of `AnomalyGuardService` exists. This was originally raised as an architect/developer question (nothing in this checkout could rule out a private fork); the client has since answered it directly, so it is settled rather than merely assumed. |
| Decision — client, 2026-08-08 | "Historical" (query a live-but-past Prometheus range) is the mode this plan targets now; fully offline "recorded" replay stays in **Could**, deferred. | Client confirmed historical is sufficient for this change. Explicitly *aware of the trade*: an offline, fixture-backed implementation is what a **cluster-free CI run or a simulator-side replay** would eventually need (no live Prometheus reachable at all), and chose to defer it rather than settle the fixture-format question now. Recorded in the Could section below with this framing. |
| Decision — client, 2026-08-08 | Success metric is determinism: replaying the same window/config twice must be byte-identical. Wall-clock replay speed is not the metric. | See "Problem / user need / business goal / proposed solution" above; originally inferred, now confirmed as stated with no change to the testable form. |
| Assumption | The new per-cycle method (`RunCycleAsync(DateTimeOffset now, ct)` or similar) should be `internal` for this plan, not `public`, since no CLI replay driver ships in this change and `Sources/Cli` is not in `Anomalies.csproj`'s `InternalsVisibleTo` list. Widening later is a 1-line, non-breaking change. | Read of `Anomalies.csproj`'s `InternalsVisibleTo` (`Tests`, `Benchmarks` only). Still open for the architect — not covered by the client's answers. |
| Decision | Scope is the seam + the `now`-parameterisation needed to make it usable, proven by a scripted-source unit test and one historical-replay proof-of-capability test. A CLI-facing replay command, an offline "recorded" fixture format, and re-measuring the simulator gap are separate backlog items (2–5 in the decision's own revised order) and out of this plan. | Matches the backlog decision's own item 1 vs items 2–5 split; client's task frames item 1 as "the enabler," not the whole programme. |
| Risk | Delivering only the interface swap (no `now`-parameterisation) would satisfy the letter of "wire to an interface" but not the stated purpose — a caller could inject a fake source and still get real wall-clock `now`, silently corrupting incident ages and the floor-proposal cadence gate rather than failing loudly. | Cheapest retirement: make the `now`-parameterisation part of Must, not a follow-on, and add the twice-replayed-determinism acceptance test below. |
| Risk | `PrometheusTopologySource` (pod ownership for grouping) refreshes against the **current** cluster, not a historical one. Replaying an old window still groups findings using today's topology, which may have drifted (renamed workloads, scaled replicas) since the replayed data was captured. | Already tolerant of absence (`topology: null` falls back to a name heuristic); not a defect this task introduces, but worth naming so nobody is surprised when a replay of last week's data groups by this week's ReplicaSets. Out of scope to fix here. |
| Risk | `AnomalyGuard`'s constructor falls back to `DateTimeOffset.UtcNow` for `restoredAt` when none is supplied, and `AnomalyGuardService` always passes `restoredAt: null`. Irrelevant when replay starts with no durable store (the common case), but would silently use real wall-clock "now" against replayed data if a store *were* supplied. | Flagged as an open question for the developer/architect rather than fixed here — cheapest retirement is a one-line follow-up once a replay driver actually needs a store. |
| Open question (architect/developer) | Interface name, member list, method visibility of the new per-cycle entry point, and whether to keep both DI registration overloads updated identically. | See "For the developer or architect" below. |

## Scope

**Must**
- Extract a new interface (proposed name `IMetricWindowSource`, `Sources/Anomalies/Monitoring/Abstractions/`) shaped exactly around what `AnomalyGuardService` consumes today: `Task<MetricWindow?> ReadAsync(DateTimeOffset end, TimeSpan window, CancellationToken ct = default)` and `IReadOnlyList<string> StalePodsExcluded { get; }`, plus `IDisposable` (matches `PrometheusMetricWindowSource`'s existing lifetime, and DI singleton disposal already relies on this). This member list — and the `var now = DateTimeOffset.UtcNow;` call site at `AnomalyGuardService.cs:386` — has been **independently cross-checked by the coordinating session** (2026-08-08) against the same two call sites (`ReadAsync` at line ~388, `StalePodsExcluded` at line ~456); no change to the proposed shape resulted, this is a confirmation.
- `PrometheusMetricWindowSource` implements the new interface (adding an interface to an existing sealed class — no behaviour change).
- `AnomalyGuardService`'s field/constructor parameter changes from `PrometheusMetricWindowSource` to the interface. Both `AnomalyGuardRegistration.AddOverfitAnomalyGuard` overloads' DI registration lines updated to register the concrete type under the interface.
- `RunOneCycleAsync`'s internal `var now = DateTimeOffset.UtcNow;` becomes a parameter of a per-cycle method (`internal Task<GuardCycleResult?> RunCycleAsync(DateTimeOffset now, CancellationToken ct)` or equivalent — final name/shape for the architect), with `ExecuteAsync`'s `PeriodicTimer` loop supplying `DateTimeOffset.UtcNow` unchanged, so **live behaviour is provably identical** (same value, same call site, just moved up one frame).
- A scripted/fake `IMetricWindowSource` test double (sibling pattern to `LiveMonitoringPipelineTests.ScriptedRawMetricSource`) and the **first-ever unit test constructing `AnomalyGuardService` directly**, proving a cycle can be driven with an arbitrary `now` and arbitrary `MetricWindow` data with no live Prometheus and no wall-clock wait.
- A DI-wiring regression test resolving `AnomalyGuardService` itself (not just `AnomalyGuardServiceOptions`) through `AddOverfitAnomalyGuard(...)`, closing the gap noted in the inventory.

**Should**
- One proof-of-capability test/diagnostic that replays a **historical** window (live Prometheus, past range) through `RunCycleAsync` several times back-to-back with no `Task.Delay`, demonstrating the "day into a bounded, measured run" claim end to end (30-minute failable ceiling, measured number reported — see Task 3, corrected 2026-08-08 per architect Rulings 1–2) using only the Must-scope changes plus a discriminated `RunCycleAsync` return type — no new source implementation needed, since `PrometheusMetricWindowSource` already accepts arbitrary historical ranges.
- The twice-replayed-determinism acceptance test (see success metric above), run against the scripted source from Must (fast, no live cluster needed) rather than the live lab.

**Could**
- A fully offline "recorded" `IMetricWindowSource` implementation backed by a checked-in fixture file — requires first settling which of the two existing incompatible fixture shapes (or a third) to standardise on; a genuinely separate design task. **Deferred by client decision, 2026-08-08** (historical is sufficient for this change) — noted here because this is specifically **what a cluster-free CI run or a simulator-side replay would eventually need**, since neither can reach a live Prometheus; the client is aware of that trade and chose to defer it rather than settle the fixture format now.
- A CLI-facing `overfit anomaly-guard replay <window>` command exposing this to an operator — needs the per-cycle method to become `public` (currently scoped `internal` per the Must-scope assumption above) and a decision on output shape (log stream vs summary).
- Parameterising `AnomalyGuard`'s `restoredAt` fallback for full replay determinism when a durable store is in play.

**Won't (this time)**
- Re-measuring the synthetic-cluster false-positive rate (backlog item 2) — separate, already-tracked task with its own success criterion.
- Explaining the ~25x simulator-vs-lab gap (backlog item 3) — depends on item 2's re-measurement.
- Any live 24-hour re-run (backlog item 5) — explicitly the option of last resort per the decision.
- A `TimeProvider`/general clock abstraction touching `ExecuteAsync`'s `PeriodicTimer`. Not needed: a replay driver bypasses `ExecuteAsync`'s host loop entirely and calls the per-cycle method directly in a tight loop, so the live cadence machinery is untouched. Flagged for the architect to overrule if there's a reason to want live-mode replay-speed testing through the actual hosted-service loop (e.g. a `FakeTimeProvider`-driven integration test) — but nothing in this request needs it.
- Fixing `PrometheusTopologySource`'s "current topology against replayed data" mismatch (see risk row above).
- Touching `LiveMonitoringPipeline`/`IRawMetricSource`/`AlertEngine` — confirmed a separate pipeline; nothing in this plan changes it.

## Gate answers (overfit-spec template)

- **Execution path**: Neither inference nor training. `Sources/Anomalies` is a separate, host/monitoring domain (`AnomalyGuardService : BackgroundService`) with its own allocation norms; the inference/training split does not apply here.
- **Verification oracle**: (1) existing `PrometheusMetricWindowSourceTests` / `WorkloadIdentityTests` stay green unchanged — proves no behaviour regression; (2) the new scripted-source unit test proves the seam is load-bearing, not decorative; (3) the twice-replayed-determinism test is the oracle for "is `now` actually decoupled from the wall clock," which is the whole point of the change and the one thing a green build cannot prove by itself.
- **AOT reach**: `Tests/AotSmokeTest` does **not** reference `Sources/Anomalies` (confirmed in prior session, capability-map memory). CI's `aot-guard` job **does** publish `Sources/Cli/Cli.csproj` (which references Anomalies) under `PublishAot=true`, so this change is reached by a real Native-AOT publish in CI even without touching AotSmokeTest. A plain interface + a parameter move introduces no reflection, no `Activator`, no LINQ — `Anomalies.csproj` already declares `IsAotCompatible=true`/`IsTrimmable=true`, and this change should not disturb that; the developer should still run the CI `aot-guard` job (or the equivalent local `dotnet publish` command below) once, since it cannot be verified from this analysis alone.
- **Allocation policy**: Not hot-path (5-minute cadence, `Anomalies.csproj`'s own words). Per-cycle allocations at the level already present (`new List<string>`, `Array.Clear`, etc.) are acceptable; no new allocation-policy obligation from this change. A future replay driver running hundreds of cycles in seconds is the one place allocation volume might matter in aggregate — flagged for the developer to keep an eye on if/when that Should-scope proof-of-capability test is written, not a hard requirement here.
- **Moat side**: Firmly the open/offline side — this makes the guard's own tuning loop reproducible and faster to iterate, which is developer/operator tooling, not a real-time or GPU capability. Nothing here belongs on the commercial side.

## Commands

```
dotnet build -c Release
dotnet test -c Release --filter "FullyQualifiedName~Anomalies"
dotnet test -c Release --filter "FullyQualifiedName~AnomalyGuardService"
dotnet publish ./Sources/Cli/Cli.csproj -c Release -r linux-x64 -p:PublishAot=true -p:TreatWarningsAsErrors=true   # AOT reach, requires local C++ toolchain per CLAUDE.md
```

## Tasks (user stories, dependency-ordered — highest uncertainty first)

### 1. Extract the interface and wire the field (Must)

**As** `AnomalyGuardService`, **I want** to depend on an interface shaped for a windowed historical
read instead of a concrete Prometheus-only class, **so that** a future implementation (historical
replay, eventually a recorded fixture) can be substituted without changing the host loop.

- Given `PrometheusMetricWindowSource.ReadAsync(end, window, ct)` and `.StalePodsExcluded`,
  When a new `IMetricWindowSource` interface is extracted with exactly those members and the class
  is marked as implementing it,
  Then `PrometheusMetricWindowSourceTests` (all of them) pass unchanged, proving no behavioural
  change from adding the interface.
- Given `AnomalyGuardService`'s constructor and field retyped from `PrometheusMetricWindowSource` to
  `IMetricWindowSource`,
  When both `AnomalyGuardRegistration.AddOverfitAnomalyGuard` overloads are updated to register the
  concrete instance under the interface (`services.AddSingleton<IMetricWindowSource>(_ => new
  PrometheusMetricWindowSource(...))`),
  Then `WorkloadIdentityTests` (existing) still passes, **and** a new test resolves
  `AnomalyGuardService` itself via `provider.GetRequiredService<AnomalyGuardService>()` without
  throwing.
- **Dependencies**: none. **Breaks**: nothing outside this repo per the "0 external references" Fact
  above; the client confirmed 2026-08-08 there is no out-of-repo consumer, so the clean constructor
  break is approved (see Decision row in the uncertainty table) and needs no further confirmation.
- Verify: `dotnet test -c Release --filter "FullyQualifiedName~Anomalies"`

### 2. Parameterise `now` out of the per-cycle method (Must)

**As** a future replay caller (test or driver), **I want** to supply the cycle's `now` explicitly
instead of the method reading `DateTimeOffset.UtcNow` internally, **so that** incident ages, floor-
proposal cadence and fetched data all agree with the timestamp I intended, not the wall clock.

- **Decision — client, 2026-08-08 (resolving an `overfit-verifier` BLOCKED finding on this
  criterion, option B)**: the original wording of this bullet named a `[LabFact]` regression guard
  as the oracle. **No such diagnostic exists, and none ever has** — verified independently by the
  developer and the verifier: `new AnomalyGuardService(` appears in `Tests/` only in Task 1's and
  Task 2's own new test files, and all five `[LabFact]` diagnostics in `Tests/Anomalies/Diagnostics/`
  construct `AnomalyGuard` + `PrometheusMetricWindowSource` directly, never `AnomalyGuardService`,
  `ExecuteAsync` or `RunCycleAsync`. The criterion was **structurally untestable as worded** — the
  lab being down (port-forwards are currently down) was never the blocker, and bringing it up would
  not have satisfied it either. Replaced below with what is actually true and checkable, and the
  limitation is stated in the criterion itself rather than in prose beneath it.
- Given `RunOneCycleAsync` renamed/reshaped to accept `DateTimeOffset now` as a parameter and marked
  `internal` (see open question on visibility),
  When `ExecuteAsync`'s loop calls it with `DateTimeOffset.UtcNow` exactly where it used to compute
  it internally,
  Then a code-level review confirms the *value* is unchanged (still one `DateTimeOffset.UtcNow`
  call, still fed into the same `_guard.RunCycle`/`ProposeFloors` calls) — **but Task 2 explicitly
  does NOT claim the live loop's *behaviour* is unchanged, and does not claim to verify it.**
  Specifically: `now` is now read at the `ExecuteAsync`/`RunCycleAsync` call boundary, i.e. **before**
  `RefreshTopologyAsync` runs; the previous code read it **after**. This ordering flip is **recorded
  and unmeasured** — not regression-guarded, not asserted negligible, just named. Its plausible
  magnitude is one topology-refresh HTTP round trip against a two-minute default `EndOffset`, but
  that is reasoning, not a measurement. **Task 2 ships with live-loop behaviour explicitly
  unverified as a stated limitation of this task**, not grown to cover it (client decision, option
  B, over growing Task 2 to build the missing guard itself) — Task 4 below is the guard this
  criterion originally assumed existed, and closes this gap and the ordering question together.
- Given a scripted `IMetricWindowSource` test double serving fixed `MetricWindow` data regardless of
  the `end` argument (mirroring `ScriptedRawMetricSource`'s pattern),
  When the new per-cycle method is called directly, twice, with the same historical `now` and the
  same `AnomalyGuardOptions`, and no `store`,
  Then the two `GuardCycleResult`s are equal on every field, and any incident timestamps recorded are
  identical between the two calls — this is the acceptance test for the success metric.
- **Dependencies**: Task 1 (needs the interface to inject the scripted source).
- Verify: `dotnet test -c Release --filter "FullyQualifiedName~AnomalyGuardService"`

### 3. Historical replay proof-of-capability (Should)

**As** whoever tunes the guard's thresholds, **I want** one example of replaying a real historical
Prometheus window through several cycles with no wall-clock delay between them, **so that** the
24-hour live run is replaced by a bounded, measured run — not asserted as "seconds," but proven
inside a ceiling that can actually fail, with the real number reported afterward.

- **Decision — architect, 2026-08-08 (fourth-pass addendum, Ruling 1 — `overfit-reviewer`'s
  null-return finding)**: before the driver is built, `RunCycleAsync`'s return type must be widened
  from `Task<GuardCycleResult?>` to a discriminated outcome — proposed shape a `GuardCycleKind`
  enum (`Completed`/`Blind`/`Failed`) paired with the `GuardCycleResult`, valid only when
  `Completed`. Placed at the front of Task 3, not as a reopening of Tasks 1–2 (committed and
  verified at `e6398fa`/`e69645e`, and neither needed the distinction) — Task 3's driver is the
  first real consumer, and widening the contract after a consumer exists is a breaking change.
  **Named trap, so it is not rediscovered**: do not make the result readable without going through
  the kind — a plain public `Result` field defaulting to an all-zero `GuardCycleResult` for
  `Blind`/`Failed` recreates the exact ambiguity being fixed, one field over, since a caller reading
  counts would see zeros and could not tell "nothing happened" from "nothing ran." Prefer something
  that makes skipping the check awkward (pattern-matching on `Kind`, or a
  `TryGetResult(out GuardCycleResult)` that returns `false` for anything but `Completed`).
- Given `RunCycleAsync` returns the new discriminated outcome,
  When Task 2's existing tests (`AnomalyGuardServiceCycleTests.cs:65,122,137,142,149`, currently
  reading `GuardCycleResult?`/`.Value`/`.HasValue`) are migrated to the new shape,
  Then they still pass unchanged in substance — this is a mechanical signature update, not a
  redesign, and is the regression guard that the widening itself did not silently change Task 2's
  already-verified behaviour.
- Given a live (or lab) Prometheus that still retains a window of history, and the Task 1+2 changes
  (plus the widened return type above),
  When a small driver loop calls the per-cycle method repeatedly with `now` walking backward-to-
  forward through that window (no `Task.Delay`), using the **existing** `PrometheusMetricWindowSource`
  unchanged, against the lab's actual scale of **12 `lab-workload` replicas**,
  Then it produces one outcome per cycle (`Completed`/`Blind`/`Failed`, from the widening above) for
  a day's worth of 5-minute cadence cycles (~288 cycles), and the `Completed`-cycle incident counts
  are compared against the recorded A1 headline (11/day) as a sanity check — reported **alongside**
  the `Blind` and `Failed` counts, not folded into it, since a run with several blind or crashing
  cycles can match a clean run's incident-count total and is not the same finding about the replay.
- **Decision — architect, 2026-08-08 (fourth-pass addendum, Ruling 2 — timing criterion corrected)**:
  "low seconds" is replaced because it named no scale and the benchmark behind it does not cover
  what this task actually does. `AnomalyGuardScaleBenchmark` measures `_guard.RunCycle` alone against
  an in-memory `MetricWindow`, with **no I/O** — it is a lower bound on Task 3's wall time, not an
  estimate of it, because Task 3 drives `PrometheusMetricWindowSource` against a **live** Prometheus
  for all ~288 cycles, and this repository has no measured number for that round trip yet. Corrected
  criterion:
  Given the lab's actual scale (12 pods, strictly between the benchmark's measured 4-pod and 20-pod
  points),
  When the ~288-cycle replay runs to completion,
  Then the **in-memory guard-processing component alone is bounded between ≈1.2 s and ≈8.8 s** by
  monotonic interpolation of `AnomalyGuardScaleBenchmark`'s measured 4-pod (4.174 ms/cycle) and
  20-pod (30.669 ms/cycle) points — labelled explicitly as a **lower bound that excludes Prometheus
  round trips**, not the pass criterion — and the **actual pass/fail ceiling is 30 minutes
  wall-clock total** for the full ~288-cycle replay (fetch + processing together): generous enough
  not to assert a number nobody has measured, ~48× faster than living through the day, and still
  genuinely failable (a driver that regressed into per-cycle sleeping or serial re-fetching would
  land near the original ~24 h, ~48× over the ceiling). **The task must measure and report the real
  wall-clock number** (and the fetch/processing split, if cheap to obtain) rather than merely passing
  under the ceiling — this run is this repository's first opportunity to have a measured figure for
  `PrometheusMetricWindowSource.ReadAsync`'s live round-trip cost at all, and a task that passes
  without producing it wastes that opportunity. The reported number is a candidate for
  `docs/measured-baselines.md`, replacing this 30-minute placeholder with a tighter, measured ceiling
  for any future replay task.
- **Dependencies**: Tasks 1–2, plus the return-type widening above (now the first step of this task,
  not a reopening of them). This task needs a live Prometheus (lab or otherwise) reachable, so it is
  naturally `[LabFact]`-gated like its siblings, not part of the fast default suite. **The lab is up**
  as of 2026-08-08 (12 `lab-workload` replicas, guard and load driver running, Prometheus
  port-forwards restored) — this task is unblocked on infrastructure grounds as of this decision.
- Verify: run manually per `Tests/README.md`'s `[LabFact]` instructions; not part of `dotnet test -c
  Release`.

### 4. Live-loop regression guard for `AnomalyGuardService` (Should) — **new task, client decision,
2026-08-08, option B**

**Why this exists and why it is a new task rather than growing Task 2**: `overfit-verifier` returned
BLOCKED on Task 2's original first acceptance criterion, which named a `[LabFact]` regression guard
that does not exist and never has (see the corrected Task 2 criterion above). The client's own
reasoning for keeping this separate rather than folding it into Task 2: **this gap is not introduced
by this change** — `AnomalyGuardService` has had zero tests of any kind since it was written,
confirmed independently by the developer and verifier. Bolting a lab-backed guard onto Task 2 would
conflate "this change is sound" (provable today, in-process, no lab needed — see Task 2's two tests)
with "this subsystem was never tested" (a pre-existing gap needing lab infrastructure Task 2 does not
have). **Should, not Must** — same justification as Task 3: it needs a live/lab Prometheus reachable,
so it is naturally `[LabFact]`-gated and cannot be part of the fast default suite, and — unlike Tasks
1–2 — it is not required for this plan's own correctness claim (Task 2's determinism test already
proves that in-process); it closes a longstanding testability gap and answers the ordering question
Task 2 left open, which is valuable but not blocking.

**As** whoever next changes `AnomalyGuardService`'s host loop (`ExecuteAsync`, cadence, topology
refresh ordering, or anything else inside it), **I want** a test that actually drives the live loop
against a real Prometheus, **so that** a future change to that ordering — the exact kind Task 2 just
made, un-guarded — has something that can go red instead of shipping silently, the way this one did.

- Given a lab (or otherwise reachable) Prometheus and a real `PrometheusMetricWindowSource` pointed
  at it,
  When an `[LabFact]` diagnostic constructs `AnomalyGuardService` itself (not `AnomalyGuard` directly,
  which is what every existing diagnostic does instead) and drives it through `RunCycleAsync` and/or
  `ExecuteAsync` for at least one real cadence tick,
  Then it completes without throwing and produces a `GuardCycleResult` consistent with the lab's
  known state, giving a future host-loop change something to break.
- Given the same diagnostic run before and after a change to the ordering between reading `now` and
  calling `RefreshTopologyAsync`,
  When the two runs are compared,
  Then this closes the open ordering question Task 2 left unmeasured (see corrected Task 2 criterion
  above) — either by showing the flip is observably inert at lab scale, or by surfacing a real
  difference worth investigating. This is the first time that question will have an actual answer
  rather than a reasoned guess.
- **Dependencies**: Tasks 1–2 (needs the interface and the parameterised per-cycle method to exist).
  **Needs the local lab's Prometheus port-forwards up** — currently down, per the same precondition
  every existing `[LabFact]` diagnostic in this file already states; this task cannot run until they
  are.
- Verify: run manually per `Tests/README.md`'s `[LabFact]` instructions, following the same
  port-forward preconditions as `AnomalyGuardShadowRunDiagnostics`/`AnomalyGuardEndToEndDiagnostics`;
  not part of `dotnet test -c Release`.

## Decision table — which mode needs what

| mode | data origin | needs a new `IMetricWindowSource` implementation? | needs `now` parameterised? | in this plan? |
|---|---|---|---|---|
| Live (today, unchanged) | live Prometheus, wall-clock now | No | No (uses `UtcNow`, byte-identical) | Unchanged, regression-guarded |
| Historical replay | live-but-retained Prometheus, moving `now` | **No** — `PrometheusMetricWindowSource` already accepts any range | **Yes** | Must (task 2) + Should (task 3 proof) |
| Recorded (offline) replay | checked-in fixture, no live Prometheus | **Yes** — new implementation, format undecided | Yes (already delivered by task 2) | Could — separate task |

## Traceability

| goal | user need | task | acceptance criterion | verified by |
|---|---|---|---|---|
| Decide the 1–9/day criterion on hundreds of events, in seconds | Repeatable re-evaluation after a threshold change | Task 1 | Interface extracted, field retyped, existing tests green | `dotnet test --filter Anomalies` |
| — | — | Task 2 | Twice-replayed same-input run is byte-identical | New `AnomalyGuardService` unit test |
| — | — | Task 3 | Discriminated `RunCycleAsync` outcome shipped first; 12-pod, ~288-cycle replay completes within a 30-minute failable ceiling (not "low seconds" — corrected 2026-08-08) and reports the measured wall-clock number | `[LabFact]` diagnostic, manual run |

## Open questions

### Answered by the client, 2026-08-08 — no longer open

All three of the following are now **Decisions** (see the uncertainty table), recorded with date and
attribution rather than left as assumptions, per the coordinator's instruction:

1. Success metric confirmed as determinism (byte-identical twice-replayed results); wall-clock replay
   time is explicitly not the metric.
2. Offline "recorded" fixture-backed replay deferred to **Could**; historical (live-but-past
   Prometheus) is sufficient for this change. The client is aware that a cluster-free CI run or a
   simulator-side replay will eventually need the offline form, and chose to defer it anyway.
3. No out-of-repo consumer of `AnomalyGuardService` exists; the clean constructor break (no
   compatibility overload) is approved. (This one was originally routed to "for the developer or
   architect" below — the client answered it directly, so it moves here rather than staying open.)

### For the developer or architect — still open

1. Interface name and exact member list — proposed `IMetricWindowSource` with `ReadAsync(end,
   window, ct)` + `StalePodsExcluded` + `IDisposable`. Should `SeriesReturned(MetricIndex)` and
   `CustomChannels` also move onto the interface (a future recorded implementation could report
   them too), or stay concrete-only since `AnomalyGuardService` never calls them today (confirmed —
   only `PrometheusMetricWindowSourceTests` and two diagnostics touch them)?
2. Visibility of the new per-cycle method — this plan assumes `internal` (no `Sources/Cli`
   `InternalsVisibleTo`, and no CLI driver ships in this change). Confirm, or make it `public` now
   if a CLI replay command is imminent enough to plan the visibility once rather than widen it later.
3. Should `AnomalyGuard`'s `restoredAt ?? DateTimeOffset.UtcNow` fallback also be parameterised now,
   or left until a replay driver actually needs a durable store (the common replay case has none)?
   This plan defers it (see Risk row).

## SUGGESTED IMPROVEMENTS TO MY ROLE

None this run. The semantic navigator tools worked as expected and were decisive here — in
particular `find_implementations(IMetricSource)` returning zero implementations is the kind of fact
a grep-based pass could easily have gotten wrong (grep would find the interface *referenced* in XML
docs and comments and could be misread as "in use"), and it materially changed the plan's shape
(from "point the field at an existing pluggable seam" to "extract a new one"). No stale instruction,
missing tool, or noisy guidance surfaced this round.

---

## Architecture review — `overfit-architect`, 2026-08-08

STATUS: **APPROVED**. Verified against the code, not the analyst's description of it — every file
named below was read directly (`PrometheusMetricWindowSource.cs`, `AnomalyGuardService.cs`,
`AnomalyGuard.cs`, `AnomalyGuardRegistration.cs`, `IncidentTracker.cs`, `IncidentPipeline.cs`,
`IncidentGrouper.cs`, `IMetricSource.cs`, `IRawMetricSource.cs`, `Anomalies.csproj`), and the
inventory claims (reference counts, "0 implementations", DI wiring, `InternalsVisibleTo`) were
independently re-run through `find_references`/`find_implementations` and matched what the analyst
recorded. One correctness finding below (Finding 1) changes how Task 2's test must be built; nothing
found requires going back to the client.

### Review verdict — numbered findings

**1. Task 2's acceptance criterion, as worded, admits a reading that is provably wrong — this is the
one finding that must be read before implementation starts.**

The Given/When/Then for Task 2 says the new per-cycle method is "called directly, twice, with the
same historical `now`… Then the two `GuardCycleResult`s are equal on every field." Read as "call the
method twice **on the same `AnomalyGuardService`/`AnomalyGuard` instance**," this is not a
determinism test — it is a test of an idempotency property `AnomalyGuard` does not have, by design,
and its own doc comments say so: "Stateful through its `IncidentTracker`… not thread-safe." Concretely,
with a fixed scripted window and the same `now` supplied twice on one instance: call 1 opens an
incident (`IncidentState.Opened`, `GuardCycleResult.Opened = 1`); `IncidentTracker.Observe` then
re-runs its greedy primary-subject match on call 2 against the *same* still-open incident from call 1
and reports it `Ongoing` instead (`GuardCycleResult.Opened = 0`, `Ongoing = 1`) — see
`IncidentTracker.cs:387-407` (`Observe`) and the class doc's own account of why matching is by primary
subject. `_calibrator.Observe(window)` (`AnomalyGuard.cs:523`) also accumulates a second sample on the
second call, which can move `FloorProposal` output even when nothing else does. **`GuardCycleResult`
is a `readonly record struct` (`Sources/Anomalies/Contracts/GuardCycleResult.cs:38`), so the equality
check is real and cheap — which means this reading of the test would either fail for a reason that has
nothing to do with the seam being built, or — if the scripted fixture happens to produce zero findings —
pass vacuously without exercising the one thing the success metric cares about.**

The only reading that actually tests "byte-identical results replaying the same window and config
twice" (the client's own words) is: **construct two independent instances** (fresh
`IncidentTracker`, fresh `FloorCalibrator`/`MetricHistory`, fresh `_recent`/`_silent` state — i.e. two
separate `new AnomalyGuardService(...)` or `new AnomalyGuard(...)` graphs, both built with `store:
null`), feed each the **same ordered sequence** of scripted windows and `now` values from cold start,
and compare the two runs' full sequence of `GuardCycleResult`s (and, per the plan's own wording, any
incident IDs/timestamps observed through the sink) for exact equality. A single cycle from two cold
starts is sufficient to prove the detectors and grouper are deterministic (see the code-level
determinism check below), but it does **not** exercise `IncidentTracker`'s cross-cycle continuation —
the greedy best-overlap matching in `Observe` — which is the most state-heavy, least obviously
deterministic part of this subsystem. **The Must-scope determinism test should walk at least two
cycles per run** (e.g. cycle 1 opens an incident, cycle 2 either continues or resolves it,
deterministically the same way in both runs), not one, so the oracle actually touches the part of the
system a same-instance misreading would have exercised by accident and for the wrong reason.

This is a correction to the verification oracle, which is squarely architecture's call to make
("Name the verification oracle before approving any design"), not a change to scope, a business rule
or a success metric — the success metric stands exactly as the client stated it. Recorded here as the
binding version; the analyst's Task 2 text is left as written per the rule against editing another
agent's section, but the developer should build the test to **this** description.

**2. The analyst's recommendation against a `TimeProvider`/clock abstraction is correct — confirmed,
not overruled, with one addition to the reasoning.** `PeriodicTimer` does have a
`TimeProvider`-accepting constructor since .NET 8, so adopting `TimeProvider` *would* let a test drive
`ExecuteAsync`'s cadence with a fake clock. But that solves a different problem than the one this
plan has: "make the wait between cycles fast in a test" (scheduling) is not "let a caller choose which
moment of history a cycle evaluates" (a domain parameter — what `end`/`observedAt` the cycle is
computed *against*). `TimeProvider` would still need `TimeProvider.GetUtcNow()` read somewhere to
produce that timestamp, which is functionally the same move as passing `now` explicitly, just wired
through a heavier, DI-wide abstraction for no capability this plan needs — replay bypasses
`ExecuteAsync` and calls the per-cycle method directly, exactly as the Won't-scope row says. The
existing `AnomalyGuard.Acknowledge(..., DateTimeOffset now)` precedent is the right-sized pattern to
extend. No ADR: this doesn't touch published API (`Anomalies` is `IsPackable=false`, confirmed by
reading `Anomalies.csproj`), doesn't move a capability to a new assembly, doesn't change AOT reach, and
isn't a wire-format commitment.

**3. Interface member list — recommend `ReadAsync` + `StalePodsExcluded` + `IDisposable` only; leave
`SeriesReturned(MetricIndex)` and `CustomChannels` off the interface.** Confirmed by
`find_references(PrometheusMetricWindowSource)`: `AnomalyGuardService` calls neither today (only
`PrometheusMetricWindowSourceTests` and two `[LabFact]` diagnostics do, and they already hold a
concrete `PrometheusMetricWindowSource` reference, so nothing is lost). Adding them speculatively for
"a future recorded implementation might want to report them too" is exactly the kind of
sophistication this repository's own measured history argues against (see CLAUDE.md's proportionality
section) — there is exactly one implementer today, adding a member later when a second implementer
actually needs it is a same-day change, and a narrower interface is more honest about what
`AnomalyGuardService` actually depends on.

**4. Visibility of the new per-cycle method — `internal`, confirmed.** `Anomalies.csproj`'s
`InternalsVisibleTo` lists only `DevOnBike.Overfit.Tests` and `Benchmarks` (verified by reading the
csproj); no CLI-facing replay command ships in this change (Could-scope, explicitly deferred), so there
is no consumer today that would need `public`. Widening later is a one-line, non-breaking change in a
project that is `IsPackable=false` — there is no shipped-API cost to deferring it.

**5. `restoredAt ?? DateTimeOffset.UtcNow` — deferred, confirmed as a Decision rather than left as an
open Risk.** The Must-scope determinism test (as corrected in Finding 1) runs with `store: null`, and
`restoredAt` is read only when `store is not null` (`AnomalyGuard.cs:233-239`) — so it cannot affect
the determinism claim as this plan bounds it. This is a scope boundary worth stating explicitly rather
than leaving implicit: **the byte-identical-replay guarantee this plan delivers is proven for
store-less replay only.** A replay driver that later adds a durable store (there is no reason for one
in a replay — the store exists to survive a *restart*, and a replay driver is one process, one run)
would need `restoredAt` parameterised too; until then, parameterising it is waste against a
requirement nobody has.

**6. Zero-unit-test scope is the right boundary, and the DI-wiring test is a real regression guard,
not a decorative one.** `provider.GetRequiredService<AnomalyGuardService>()` resolving without
throwing is exactly the kind of check that silently rots — `WorkloadIdentityTests` already resolves
`AnomalyGuardServiceOptions` but not `AnomalyGuardService` itself (confirmed by reading the test), so
today a broken constructor-parameter order or a missing DI registration for a new dependency would not
be caught until a real deployment. Combined with Finding 1's correction, the scripted-source test also
becomes a real regression guard for the tracker's continuation logic, not just the seam.

**7. Interface dispatch where a concrete call used to be — unmeasurable here, not merely unmeasured, and
the margin is enormous by direct measurement (audited by `overfit-perf-claim-auditor`, verdict SUPPORTED).**
`_source.ReadAsync(...)` and `_source.StalePodsExcluded` move from a concrete `PrometheusMetricWindowSource`
call to an interface call. `AnomalyGuardServiceOptions.Cadence` defaults to `TimeSpan.FromMinutes(5)`
(`AnomalyGuardServiceOptions.cs:20`), consumed by `PeriodicTimer` at `AnomalyGuardService.cs:260` — a
period of 3×10¹¹ ns, code-verified. No BenchmarkDotNet run in this repository could resolve a
single-digit-nanosecond interface-dispatch effect against that period above its own `RatioSD` noise floor
— the correct verdict is **unmeasurable, not merely unmeasured**, which is a different and more useful
statement than "we did not measure it."

The stronger, directly-relevant bound was sitting unused rather than absent: `AnomalyGuardScaleBenchmark`
(`Sources/Benchmark/AnomalyGuardScaleBenchmark.cs`) measures the **whole cycle** — `_guard.RunCycle`, not a
kernel — at `[Params(4, 20, 50, 100, 200)]` pods. Measured 2026-08-01 and re-verified directly against the
artifact (`BenchmarkDotNet.Artifacts/results/Benchmarks.AnomalyGuardScaleBenchmark-report-github.md`):
**4.174 ms at 4 pods (the lab's own scale) up to 700.961 ms at 200 pods**. Even at the fast end this is
4–9 orders of magnitude above anything an interface call could contribute at this call frequency (once per
`ReadAsync`/`StalePodsExcluded` pair per cycle), and that margin — not the paragraph below — is the actual
basis for "does not matter here." `IncidentGrouper`'s own benchmark table (20 µs–172 ms per cycle at
realistic finding counts) is consistent with this whole-cycle number and corroborates it as a component
cost, but the whole-cycle figure is the one that directly bounds the question asked.

The "single-digit nanoseconds" (interface dispatch) and "HTTP round-trip in milliseconds" (Prometheus)
figures are **reasoning about typical .NET interface-call and HTTP-request costs, not measurements taken
in this repository**, and should be read that way rather than as cited baselines. No allocation-policy or
hot-path obligation follows from this change.

**8. No ADR required.** Checked against all six triggers: no published-API surface changes (`Anomalies`
is `IsPackable=false`); no capability moves assembly; `Tests/AotSmokeTest` does not reference
`Anomalies` today and still won't (confirmed by re-reading `Tests/AotSmokeTest/AotSmokeTest.csproj`),
so AOT reachability is unaffected in the sense that matters (the real AOT proof for this assembly is
the `aot-guard` CI job publishing `Sources/Cli/Cli.csproj`, which already includes `Anomalies` — this
change does not change that fact either way); no on-disk/on-wire format changes; stays entirely on the
open/offline side (see Moat side below); no dependency added to `Main`. Design reasoning for the
adjacent multi-scope telemetry decision in this same file/assembly went through the identical checklist
with the same "no ADR" outcome (`.claude/agent-memory/overfit-architect/project_guard_telemetry_meter.md`)
— consistent precedent, not re-derived from nothing.

**9. Task 3's "low seconds" claim is pod-count-dependent, and the plan states neither — added by
`overfit-perf-claim-auditor`'s review, recorded here rather than by editing the analyst's Task 3 text.**
~288 cycles (a simulated day at the default five-minute cadence) against `AnomalyGuardScaleBenchmark`'s
whole-cycle numbers (Finding 7): 288 × 4.174 ms ≈ **1.2 s at 4 pods** (the lab's own scale, and the scale
the A1 24-hour run this whole plan exists to speed up was run at) — "low seconds" holds. 288 × 700.961 ms
≈ **202 s at 200 pods** — two orders of magnitude apart, and "low seconds" does not hold. Task 3's Given
clause ("a live (or lab) Prometheus that still retains a window of history") names no pod count, so its
Then clause ("completes in low seconds… ~288 cycles") is true at one real scale in this repository's own
measurements and false at another, and nobody can say today which one a run will hit. Task 3 is not built
(`STATUS` line: "Tasks 2-3 not started"), so pinning this down now costs nothing and prevents an acceptance
criterion nobody can meet. See Decision 6.

### System context

- **Touches**: `Sources/Anomalies/Monitoring/Abstractions/` (new interface file),
  `Sources/Anomalies/Monitoring/PrometheusMetricWindowSource.cs` (implements it, no behaviour change),
  `Sources/Anomalies/Hosting/AnomalyGuardService.cs` and `AnomalyGuardRegistration.cs` (field/ctor/DI
  retype), plus new test files under `Tests/Anomalies/`.
- **Depends on**: `Sources/Main` only (unchanged — `Anomalies.csproj` has one `ProjectReference`).
- **Depends on this** (`find_references(AnomalyGuardService)`, re-run and matched to the analyst's
  count of 6): `AnomalyGuardRegistration.AddGuardCore` (DI registration, this plan's own target),
  `AnomalyGuardServiceOptions`'s doc comment, and **one external caller**,
  `Sources/Cli/AnomalyGuardCommand.cs:177` (`host.Services.GetRequiredService<AnomalyGuardService>()`).
  The CLI resolves the concrete `AnomalyGuardService` type, not the new interface, and nothing in this
  plan changes that call site's signature — the CLI is unaffected by this change and needs no update.
- **Moat side**: open/offline, confirmed. This makes the guard's own tuning loop reproducible and
  faster to iterate — developer/operator tooling on a monitoring subsystem, not a real-time,
  performance or GPU capability. Nothing here approaches the boundary from
  `project_moat_public_private.md`.

### Boundaries and responsibilities (gate answers, confirmed)

- **Execution path**: neither inference nor training — `Sources/Anomalies` is a separate host/monitoring
  domain with its own allocation norms. Confirmed by reading `Anomalies.csproj`'s header comment and
  `AnomalyGuardService : BackgroundService`.
- **AOT reachability**: `Tests/AotSmokeTest/AotSmokeTest.csproj` references `Main` only (re-verified by
  reading the csproj directly, matching `.claude/agent-memory/overfit-architect/project_assembly_graph.md`).
  The real AOT proof for this change is the `aot-guard` CI job's `Sources/Cli/Cli.csproj` publish step,
  which already carries `Anomalies` under `PublishAot=true`. A plain interface extraction plus a
  parameter move introduces no reflection, no `Activator`, no LINQ, and does not touch the one
  intentionally-reflective method in this file (`AddOverfitAnomalyGuard(IConfiguration, ...)`, already
  annotated `[RequiresUnreferencedCode]`/`[RequiresDynamicCode]`).
- **Allocation policy**: not hot-path (five-minute cadence). Per-cycle allocations at the level already
  present are the existing norm; no new obligation from this change (see Finding 7).
- **Ownership/disposal**: `IMetricWindowSource` carries `IDisposable`, matching
  `PrometheusMetricWindowSource`'s existing lifetime (one instance per process, owned by the DI
  container as a singleton, disposed at host shutdown). Nothing here is an `AutogradNode` — the
  ownership-tag model does not apply to this assembly.
- **Public surface**: `Anomalies` is `IsPackable=false` (verified in `Anomalies.csproj`), so "public API
  creep" carries none of the shipped-NuGet-package weight it would in `Main`. `internal` for the new
  per-cycle method is still the right default (Finding 4) because it is the narrower commitment, not
  because publishing it would be unsafe.

### Key flow — the corrected determinism oracle (Finding 1)

```mermaid
sequenceDiagram
    participant T as Test
    participant A as AnomalyGuardService (run A, cold)
    participant B as AnomalyGuardService (run B, cold)
    participant S as Scripted IMetricWindowSource

    Note over T: Same scripted windows, same now values, both runs; store: null
    T->>A: new AnomalyGuardService(..., source: S, store: null)
    T->>B: new AnomalyGuardService(..., source: S, store: null)
    loop cycle 1..N, same now sequence
        T->>A: RunCycleAsync(now_i)
        A-->>T: GuardCycleResult_A_i
        T->>B: RunCycleAsync(now_i)
        B-->>T: GuardCycleResult_B_i
        T->>T: assert GuardCycleResult_A_i == GuardCycleResult_B_i
    end
```

N ≥ 2 so the assertion exercises `IncidentTracker.Observe`'s cross-cycle continuation (Opened →
Ongoing → Resolved), not just single-cycle detection.

### Quality requirements as parameters

| requirement | parameter | how measured | baseline comparison |
|---|---|---|---|
| Determinism (success metric) | Two cold-start replays of the same scripted window sequence + `now` sequence + options produce field-equal `GuardCycleResult`s per cycle, N ≥ 2 cycles | New xUnit test (Finding 1's corrected shape) | No prior baseline — this is the first test of the property. Code-level check (this review): detector loops iterate fixed-order lists (`window.Pods` from a `SortedSet<string, Ordinal>`, `MetricIndex` enum, configured `Rules`/`CustomMetrics` lists); `IncidentGrouper`'s only two `Dictionary`/`HashSet`-driven orderings (`Build`'s `groups.Values`, `ResolveWorkload`'s `counts`) are both explicitly tie-broken by name/signal so output does not depend on enumeration order — see `IncidentGrouper.cs`'s own doc comment on `SortBySeverityDescending` ("a report that reorders itself between two runs over the same data is one nobody can diff") and `AnomalyGuard.cs`'s comment on `ResolveWorkload` ("Ties broken by name, so the resolved workload does not depend on dictionary ordering"). No `Guid`, no `Random`, no wall-clock read found inside the cycle path (`Grep` for `Guid.NewGuid|new Random|GetHashCode` in `Sources/Anomalies` found only `Training/OfflineTrainingJob.cs`, an unrelated offline-training subsystem that already seeds explicitly). This is why the shape is expected to deliver the metric — but Finding 1's test is what actually proves it rather than this review's reading of the code. |
| AOT reach | `dotnet publish Sources/Cli/Cli.csproj -c Release -r linux-x64 -p:PublishAot=true -p:TreatWarningsAsErrors=true` succeeds | CI `aot-guard` job (already runs this) | Unaffected — no reflection/LINQ/Activator introduced |
| Allocation policy | Not applicable (not hot-path) | — | `Anomalies.csproj`'s own stated norm |

### Technical risks and spikes

| risk | spike to retire it | order |
|---|---|---|
| Finding 1's misreading ships as written | Write the Must-scope determinism test per the corrected shape (two cold instances, N ≥ 2 cycles) **before** anything else in Task 2 — it is cheap (in-process, scripted source, no I/O) and is the acceptance oracle for the whole plan | First, inside Task 2 |
| `PrometheusTopologySource` grouping replayed history against today's topology (analyst's own Risk row) | Not this plan's problem to retire — already correctly out of scope; flagged so nobody is surprised by drifted grouping during the Should-scope proof-of-capability replay | N/A — Won't, correctly |
| Task 3's "low seconds" claim is true at 4 pods (~1.2 s) and false at 200 (~202 s), per `AnomalyGuardScaleBenchmark` (Finding 9) | State the target pod count in Task 3 before it starts; at lab scale (4 pods) the existing wording holds without further work — if the proof-of-capability is meant to run against a larger namespace, its Then clause needs a scale-appropriate bound, not "low seconds" unqualified | Before Task 3 starts — not built yet, so free to fix now |

### Decisions

1. No `TimeProvider`/clock abstraction — confirmed (Finding 2). No ADR (checklist in Finding 2).
2. `IMetricWindowSource` carries exactly `ReadAsync(DateTimeOffset end, TimeSpan window, ct)`,
   `StalePodsExcluded`, `IDisposable` — `SeriesReturned`/`CustomChannels` stay concrete-only (Finding 3).
3. New per-cycle method is `internal` (Finding 4).
4. `AnomalyGuard`'s `restoredAt` fallback stays deferred; the determinism guarantee this plan delivers
   is explicitly scoped to store-less replay (Finding 5) — this is now a stated boundary of the success
   metric, not an open risk.
5. Task 2's acceptance criterion is the corrected shape in Finding 1 / "Key flow" above, not the literal
   text of the analyst's Given/When/Then (which is ambiguous and, under one reading, wrong).
6. Task 3's acceptance criterion needs a stated pod count before it starts (Finding 9). Read as scoped to
   lab scale (4 pods, matching A1) unless the analyst or client says otherwise — at that scale "low
   seconds" (~1.2 s, `AnomalyGuardScaleBenchmark`) holds as written and nothing else changes. If Task 3 is
   meant to run against a larger namespace, its Then clause needs a bound that names the scale — "low
   seconds" is false at 200 pods (~202 s) by the same benchmark. Not a rewrite of the analyst's Task 3
   text, per the rule against editing another agent's section; the developer should not start Task 3
   without this being settled one way or the other.

No ADR filed — none of the six triggers apply (Finding 8).

### Operability

No change to the live path's operability: `ExecuteAsync`'s loop, logging, telemetry and restart
behaviour are byte-for-byte unchanged (Task 1/2's own regression guard is exactly this — same `UtcNow`
call, same call site, moved up one frame). A future replay driver (Should/Could scope, not this plan)
would need its own operability answers — none are owed here since nothing new runs as a result of this
change; the new per-cycle method is only reachable from tests until a driver exists.

### Open questions

None blocking. All three questions the analyst routed to "for the developer or architect" are closed
as Decisions above (2, 3, 4). Finding 1 is a correction the developer must apply, not a question — it
is answered in this section, not routed anywhere.

**For the analyst**: none.

**For the client**: none. Nothing in this review depends on business input beyond what was already
supplied on 2026-08-08.

Signed: this plan is **APPROVED** for `overfit-developer`. Build against Finding 1's corrected Task 2
acceptance criterion, not the literal text under "Tasks" above.

### Addendum — 2026-08-08, second pass (`overfit-verifier` BLOCKED on Task 2 → analyst correction,
client decision option B → new Task 4)

**Signature holds.** This is a read of the delta only (Task 2's rewritten first criterion, new Task 4),
not a re-review — per the coordinator's framing and because nothing outside those two spots changed.

**Does the corrected Task 2 criterion still support the success metric? Yes — provably, not merely
plausibly.** The ordering flip only has a code path to matter through `RefreshTopologyAsync`, and that
method returns immediately, doing nothing, when `_topology is null`
(`AnomalyGuardService.cs:284-287`). `ExecuteAsync` now captures `now` at line 269, before
`RunCycleAsync` calls `RefreshTopologyAsync` internally at line 417 — confirmed by reading the shipped
code, not the plan's description of it. Task 2's determinism test
(`Tests/Anomalies/AnomalyGuardServiceCycleTests.cs:193-197`, already written) constructs
`AnomalyGuardService` with four positional arguments (`options, source, sink, logger`); `topology`
takes its default of `null` (`AnomalyGuardService.cs:174`). **The ordering flip and the determinism
test do not share a code path** — the flip is inert, not "negligible," for every input the
success-metric oracle exercises. This is stronger than, and independent of, the store-less-replay
boundary already stated in Decision 4 / Finding 5.

**Should is the right level for Task 4 — confirmed, with the reasoning written down rather than left
as "plausibly."** Two independent points:
1. *Structural*: as above, the ordering flip cannot touch anything Task 2's oracle checks, so Task 4
   does not gate this plan's own correctness claim — the client's stated reasoning for keeping it
   separate holds up under inspection, not just under the client's assertion of it.
2. *Magnitude, live path only*: `RefreshTopologyAsync` is one HTTP round trip
   (`PrometheusTopologySource`, kube-state-metrics-backed), and the shift it introduces moves the
   window's anchor **earlier** — further from the moment `EndOffset`'s two-minute margin exists to
   stay clear of (`PrometheusMetricWindowSource.ReadAsync`'s own doc comment: leave a margin behind
   "now" because rate expressions are still filling in at the trailing edge, per the lab incident that
   motivated `EndOffset` in the first place). A topology refresh taking Δt now means `end`/`observedAt`
   sit Δt **further back** in history than before — the safer direction relative to that specific,
   documented failure mode, not the dangerous one. Under a healthy cluster, Δt is
   milliseconds-to-low-hundreds-of-ms against a 120 s margin; under a degraded/slow Prometheus it could
   grow, but that is a pre-existing failure mode of `RefreshTopologyAsync` itself (already handled:
   previous snapshot stands, logged as stale) that this change does not make more likely or more
   severe — only the order relative to `now` changed. This is reasoning, not measurement, exactly as
   the analyst's correction states, and Task 4 is what turns it into one — an argument for building
   Task 4 soon once the lab is up, not for blocking Task 2 on it now.

**Finding 7 and Finding 9 stand unchanged.** Neither is about `now`'s ordering relative to
`RefreshTopologyAsync` — Finding 7 is interface-dispatch cost, Finding 9 is Task 3's pod-count-dependent
wall-clock claim. The ordering flip touches neither.

`STATUS:` line left exactly as given.

### Addendum — 2026-08-08, fourth pass (pre-Task-3 rulings: `overfit-reviewer`'s null-return finding,
Finding 9's scale gap)

Two rulings, both mine to make, both bearing on Task 3 before it is dispatched. `STATUS:` and the
analyst's task text left untouched; where task text needs to change, said explicitly below rather than
edited here.

**Ruling 1 — the ambiguous `null` return. Introduce a discriminated outcome now, before Task 3's
driver becomes the first consumer.** The reviewer is right, and precisely for the reason given:
`RunCycleAsync`'s own doc comment justifies returning a value at all as "so a replay driver need not
read it back out of a log," and `Task<GuardCycleResult?>` only delivers half of that once a driver
exists to care — `null` from a blind cluster (`:427`) and `null` from a caught exception (`:531`) are
already logged differently (`_blind` vs `_cycleFailed`, confirmed by reading both call sites) but are
the same value to any caller. Task 3's own acceptance criterion (incident counts compared against the
A1 headline as a sanity check) is exactly the case that needs the distinction: a replay with several
blind cycles and a replay with several crashing cycles can produce the identical incident-count
sequence and are not the same finding about the replay.

**Shape**: widen the return type to a small discriminated outcome — e.g. a `GuardCycleKind` enum
(`Completed`/`Blind`/`Failed`) paired with the `GuardCycleResult`, valid only when `Completed`. One
concrete trap to name so the fix does not quietly recreate the bug it fixes: **do not make the result
field readable without going through the kind.** A `GuardCycleOutcome` whose `Result` is a plain
public field defaulting to an all-zero `GuardCycleResult` for `Blind`/`Failed` looks exactly like "a
clean cycle that found nothing" to any caller that forgets to check `Kind` first — the same ambiguity,
one field over. Prefer something that makes skipping the check awkward (pattern-matching on `Kind`, or
a `TryGetResult(out GuardCycleResult)` that returns `false` for anything but `Completed`) over a plain
struct with two public fields and an honour system.

**Where this lands**: Task 1 and Task 2 are committed and VERIFIED (`e6398fa`, `e69645e`) — this is not
a defect in either, since neither needed the distinction, and reopening a verified commit to widen a
return type is a heavier move than doing it once, ahead of the first real consumer. Recommend this be
the **first step of Task 3** ("before building the driver, widen `RunCycleAsync`'s return type") rather
than a Task 2 amendment — Task 2's own tests need only mechanical updates (`AnomalyGuardServiceCycleTests.cs:65,122,137,142,149`
currently read `GuardCycleResult?`/`.Value`/`.HasValue`; migrating them to the new shape is small and
contained, not a redesign). **This needs task text — the analyst's, not mine — to add it as Task 3
scope. Please route it.**

**Ruling 2 — Task 3's wall-clock criterion still names no scale, and Finding 9's numbers need a caveat
before they're used to pick one.** Correcting my own earlier framing first: `AnomalyGuardScaleBenchmark`
(cited in Finding 7 and Finding 9) measures `_guard.RunCycle` alone — the in-memory detection/grouping/
tracking pipeline — against a pre-built `MetricWindow` with **no I/O**. It does **not** measure
`RunCycleAsync`'s `_source.ReadAsync` call, and Task 3 explicitly drives the **existing**
`PrometheusMetricWindowSource` against a **live** Prometheus for all ~288 cycles — real network round
trips this benchmark never takes. Finding 7's conclusion is unaffected and if anything strengthened (a
larger true per-cycle cost only widens the margin over interface-dispatch overhead), but Finding 9's
specific figures (1.2 s @ 4 pods, 202 s @ 200 pods) are a **lower bound** on Task 3's real wall time,
not an estimate of it — this repository has no measured number for `PrometheusMetricWindowSource.ReadAsync`'s
round-trip cost against a live cluster, run back-to-back. Recording this as a caveat on Finding 9 rather
than a rewrite: its pod-count-sensitivity conclusion still holds (more pods still costs more, monotonically),
it just isn't the whole story for a task that does real I/O.

Given that, a single precise "N seconds" bound would be false precision — asserting a number this repo
has not measured, the exact trap CLAUDE.md's performance section exists to catch. The lab's own history
is reassuring but not a tight bound either: the live guard has run 5-minute-cadence cycles against real
Prometheus for extended periods with zero cycle failures (`docs/measured-baselines.md`'s "false
positives, 24h to 2026-08-05… 292 cycles, 0 failures"), which is existence evidence that a cycle's
total cost — processing *and* fetch — comfortably fits inside 300 s, not evidence of where inside it.

**Ruling**: replace "low seconds" with a bound that (a) names the lab's actual scale, (b) is generous
enough not to assert a number nobody has measured, and (c) is still genuinely failable — the coordinator's
own bar. Concretely: **at the lab's 12 replicas, the in-memory guard-processing component of a
288-cycle replay is bounded between ≈1.2 s and ≈8.8 s by monotonic interpolation of `AnomalyGuardScaleBenchmark`'s
measured 4-pod (4.174 ms/cycle) and 20-pod (30.669 ms/cycle) points (12 sits strictly between them, and
the benchmark's cost is monotonically increasing in pod count at every measured point, so this is
interpolation between two real measurements, not extrapolation past them). Set the task's pass/fail
ceiling at 30 minutes wall-clock total for the ~288-cycle replay** — about 48× faster than replaying it
live (24 h), comfortably inside reach even under a Prometheus round trip an order of magnitude slower
than the in-memory component per cycle, and still hard enough to fail a real regression (a driver that
resurrected a per-cycle `Task.Delay(Cadence)`, for instance, would land at ~24 h, ~48× over). **Task 3
should also record and report the actual measured wall-clock time (and, if cheap, the fetch/processing
split) as an observation** — this run is the first opportunity in this repository to get a real number
for `PrometheusMetricWindowSource.ReadAsync`'s live round-trip cost, and that number belongs in
`docs/measured-baselines.md` once it exists, replacing the 30-minute placeholder ceiling with a tighter
one for any future replay task. **This also needs task text — the analyst's — to carry the number and
the reporting requirement. Please route it alongside Ruling 1.**

`STATUS:` line left exactly as given; neither ruling touches Task 1 or Task 2's already-verified text.

### SUGGESTED IMPROVEMENTS TO MY ROLE

One worth recording. The task brief asked me to check whether the proposed tests "can actually fail" —
that check is what surfaced Finding 1, and it did so only because I traced `IncidentTracker.Observe`'s
matching logic by hand against the literal wording of the acceptance criterion rather than trusting
that "call it twice" obviously meant "two independent runs." Nothing in my standing instructions
prompts that trace specifically for a *stateful* subject under a *twice-called* test — the guidance
says "confirm the tests proposed can actually fail," which caught it here, but only because I happened
to read the target class's own doc comments closely enough to notice it documents its own statefulness.
A sharper version of that instruction — check, for any test that calls the same method twice on the
same instance, whether the class's own documentation asserts state carries between calls — would make
this reproducible rather than depending on how closely one class's comments get read. No change made
to any instruction file; recording this for my own memory instead, since it is a pattern rather than a
one-off fact about this codebase.
