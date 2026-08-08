STATUS: ANALYSIS_READY
Author: overfit-analyst
Date: 2026-08-08
Slug: anomaly-guard-psi-cpu-channel-plan

# Add a CPU-pressure (PSI) channel to the anomaly guard

## What the client asked for

> Add `container_pressure_cpu_waiting_seconds_total` as a new metric channel — **alongside**
> `CpuThrottleRatio`, not replacing it.

Context supplied with the request (not re-derived here, per instruction): `CpuThrottleRatio`'s source
(`container_cpu_cfs_throttled_periods_total`) is permanently blind on the lab because no pod there carries a
CPU limit; PSI needs no limit and is verified present (15 pods, values vary); PSI's `waiting` series is the
one that settled the "34 unexplained incidents" investigation on 2026-08-08 (`docs/aiops/aiops-backlog.md`,
section "The replay is faithful..."). The client first asked to *replace* throttle, then accepted *alongside*
after the ordinal-13 / model-contract argument below — recorded as **Decision, client, 2026-08-08**.

## Inventory

### Already exists

| Capability | Type / file |
|---|---|
| The exact precedent for this change | `MetricIndex.ContainerRestarts` (`Sources/Anomalies/Contracts/MetricIndex.cs:42`) — added at the end, deliberately excluded from `MetricSnapshot`/`FeatureCount`, documented as "scraped but not a model feature." Every touch point below is copied from this precedent, verified by grep across `Sources/Anomalies`. |
| A config-only path that needs **zero source changes** | `CustomMetricBinding` (`Sources/Anomalies/Contracts/CustomMetricBinding.cs`) + `AnomalyGuardConfigFile.CustomMetrics` (`Sources/Anomalies/Contracts/AnomalyGuardConfigFile.cs:86`). Fully wired end-to-end today: `Tests/Anomalies/CustomMetricEndToEndTests.cs` shows a custom metric reaching a reported peer finding, an absolute rule, and blindness accounting, entirely from a `CustomMetricBinding` value — no enum change. |
| PromQL wrapping for a monotonic counter | `MetricSourceKind.Counter => rate(name[range])` (`MetricMap.cs:226`) — exactly the shape PSI needs (`rate(container_pressure_cpu_waiting_seconds_total[...])`); no second "total periods" counter required, unlike throttle. |
| Floor calibration for a channel outside `FeatureCount` | `FloorCalibrator` already calibrates and persists `CustomMetricBinding` channels (`_customChannels`, `CustomMarker`, `FloorCalibrator.cs:391-406`) as well as `MetricIndex` ones — the calibration mechanism does not care which route is chosen. |
| Loop/array sites that auto-adjust to a new `MetricIndex` member | Verified by grep: `FloorCalibrator`, `MetricMap`, `AnomalyGuardConfigReader`, `AnomalyGuard.RunCycleCore`, `AnomalyGuardService`, `MetricDiscovery`, `PrometheusMetricSource`/`PrometheusHistoricalSource`/`PrometheusMetricWindowSource`, and ~15 test files all iterate `for (m = 0; m < (int)MetricIndex.Count; m++)` or size arrays off it — none of these needs editing for a new member. |
| A safe-by-construction persistence format | `FloorCalibrator.Write/Read` and `MetricHistory` (seasonal baseline) both key by **enum name** via `Enum.TryParse<MetricIndex>` (`FloorCalibrator.cs:384,456`; `MetricHistory.cs:354`), not by ordinal — a state file written before the new member exists simply has no line for it, exactly as happened for `ContainerRestarts`. |
| A worked classification precedent for "the same phenomenon" | `PeerSignalCatalog` classifies `CpuThrottleRatio` as `LoadIndependent` and this is pinned by a test (`Tests/Anomalies/PeerSignalCatalogTests.cs:48`, `EventsAndFractions_AreComparedRaw`). Measured evidence for PSI specifically points the same way: `docs/aiops/aiops-backlog.md` records PSI-waiting p50 **unchanged** (1.39e-5 → 1.46e-5) while request rate rose 51% and CPU rose 27% on the same window — the opposite of what a load-sensitive signal would do. |
| The default-blind-metric UX | `MetricMap.Unmapped`, guard startup log, `k8s/anomaly-guard/guard.lab.json`'s own comment ("CpuThrottleRatio and ErrorRate are left out on purpose so that list has something in it") — adding an unmapped PSI channel to the lab config is the same mechanism already in daily use. |

### Partially exists

| Capability | What's there | What's missing |
|---|---|---|
| PromQL default template | `PromqlCatalog.DefaultTemplate` has cases for all 13 current `MetricIndex` members and `_ => throw ArgumentOutOfRangeException` otherwise (`PromqlCatalog.cs:384-424`). A new enum member with no explicit case here **throws at runtime** the first time a caller resolves it without an override — this is the one **mandatory** touch point if the enum-member route is chosen, verified by reading `ResolveTemplate` (`PromqlCatalog.cs:293-305`), which falls back to `DefaultTemplate` whenever `QueryOverrides` (built from `MetricMap.ToQueryOverrides` / the client's config) has no entry for that key. |
| Discovery candidate names | `MetricNameCatalog.For(metric)` — no case for a PSI-shaped metric; falls to `_ => []`, i.e., zero candidates. Confirmed **not** fatal: `MetricDiscoveryTests.EveryChannelIsAccountedFor` shows every channel gets a report entry regardless of candidate count (`Tests/Anomalies/MetricDiscoveryTests.cs:109-116`). Missing candidates only degrade auto-discovery UX for a client who hasn't named the metric explicitly. |

### Does not exist

- Any `MetricIndex` member for CPU pressure, in either the enum or any catalog.
- Any starvation-floor measurement — the lab has never starved a container (see Scope below).
- A `SustainedThresholdOptions` shape for PSI (the throttle one, `ForCpuThrottling`, is explicitly documented as "calibrated on one fault, on one single-node lab... no threshold should be promoted to a product default without that repeat" — `SustainedThresholdOptions.cs:40-44`).

## Problem / user need / business goal / proposed solution

- **Problem**: `CpuThrottleRatio` is permanently blind on any deployment without CPU limits (the lab, and plausibly other clients that don't set them). Measured cost: it produced 34 false-positive-adjacent incidents traceable to an unadjusted-for-load family gap the client's own PSI ad-hoc query settled in minutes, while the guard's own channel set had no equivalent instrument running continuously.
- **User need**: the operator watching the guard needs a CPU-contention signal that works whether or not the deployment sets CPU limits, without waiting on backlog item E2 (a CPU-limited test deployment, itself only test infrastructure, not a client-facing fix).
- **Business goal**: the guard should not have a channel that is structurally blind by design on a class of deployments (no CPU limit) with no substitute.
- **Proposed solution (client's)**: add `container_pressure_cpu_waiting_seconds_total` as a new channel alongside throttle.
- **Success metric**: **not stated by the client.** Candidates to propose, pending their answer (see Open questions): (a) the guard reports zero blind CPU-contention metrics on the lab's own unlimited pods, where today it reports one (throttle) permanently blind; (b) on a future CPU-starved scenario (real or synthetic), the new channel produces a finding where throttle alone would produce none. Neither is measurable today without an answer to the starvation-floor question below.

## The central design question: enum member vs. `CustomMetricBinding`

Both were investigated because the task explicitly asked which is the better vehicle; the client's own framing (ordinal 13, `MetricIndex.Count > FeatureCount`) already leans toward the enum route, so this section states plainly what each costs and what each buys, and asks the client to confirm the choice knowingly rather than by default.

| | `CustomMetricBinding` (config-only) | New `MetricIndex` member |
|---|---|---|
| Source changes required | **None.** Add an entry to `k8s/anomaly-guard/guard.lab.json`'s (new) `customMetrics` section; `AnomalyGuardConfigReader` already parses it. | `MetricIndex.cs` (1 member) + `PromqlCatalog.DefaultTemplate` (mandatory, else `ArgumentOutOfRangeException`) + `PeerSignalCatalog.Classify`/`IsCountedEvent` (recommended, defaults are safe) + `MetricNameCatalog.For` (optional, discovery UX) + `AnomalyGuardOptions.DefaultRules`/`OverfitServerQueries` (optional, if a built-in rule/query is wanted) — six files, all following the `ContainerRestarts` precedent exactly. |
| Reaches rules, peer, trend families | Yes, identically — this is the mechanism `CustomMetricBinding` exists for. | Yes. |
| Reaches the learned family (Gpt/Baseline/Neuro) | No (by design — neither does `ContainerRestarts`). | No, if excluded from `MetricSnapshot`/`FeatureCount` as the client's own reasoning requires. |
| Floor calibration (peer-gap / trend-change history) | Yes — `FloorCalibrator` already persists custom channels. | Yes. |
| Ships as a product default (every deployment gets it without config) | No — must be declared per deployment. | Yes — available the moment a client upgrades, `Unmapped` until they bind a source name. |
| Auto-discovery proposes it from a name convention | No. | Only if `MetricNameCatalog.For` gets a case (cheap, optional). |
| Blast radius / review size | One JSON config entry + a fixture test. | Touches shared enum + shared catalogs used by every deployment; risk is low (mechanical, precedented) but the diff is wider and the enum is a public contract. |
| Reversible | Trivially — delete the config entry. | An enum member, once shipped, is public API; removing it is a breaking change for anyone who wrote a floor-state file or config referencing it by name. |

**Recommendation**: given the client said "alongside `CpuThrottleRatio`" (a named, permanent channel) rather than "watch this on the lab", and reasoned specifically about `MetricIndex` ordinal 13, the enum-member route matches stated intent and its cost is low — the `ContainerRestarts` precedent shows the whole change is mechanical and six files, none of them behaviourally risky. `CustomMetricBinding` is flagged as available **today, at zero source cost**, in case the actual near-term need is "make PSI visible on this lab while E2/scope questions settle" rather than "ship a permanent product channel" — those are different asks with the same PromQL. This is **Open question 1** below, not decided here.

## What must be established (round two)

### 1. `MetricIndex` switch/iteration sites — verified by grep + `find_callers`/`find_references`

Ran `find_references(MetricIndex)` (result too large to inspect in full — 639 lines/~20 files; textual `grep -r "MetricIndex\."` across `Sources/Anomalies` was used instead and cross-checked against `find_callers(PeerSignalCatalog.Classify)` and `find_references(CustomMetricBinding)`, both of which returned manageable, fully-read results). Findings:

- **Mandatory**: `PromqlCatalog.DefaultTemplate` (throws on an unhandled member).
- **Recommended, safe if skipped**: `PeerSignalCatalog.Classify` (defaults to `LoadIndependent`, which is almost certainly correct here — see below) and `IsCountedEvent` (defaults to `false`, correct — PSI is a ratio, not a discrete event like a restart).
- **Optional**: `MetricNameCatalog.For` (discovery candidates), `AnomalyGuardOptions.DefaultRules`/`OverfitServerQueries` (a built-in `SustainedThresholdRule` and a default PromQL — both require a measured threshold the lab cannot currently produce, see Scope).
- **Confirmed no-op**: `LiveMonitoringPipeline.MetricTypeIdToFeatureIndex` already defaults unhandled ids to `-1` (the exact `ContainerRestarts` comment explains why); `FloorCalibrator`, `MetricMap`, `AnomalyGuardConfigReader`, `AnomalyGuard.RunCycleCore`, `MetricDiscovery`, all three Prometheus sources, and ~15 test files iterate `MetricIndex.Count` generically.

### 2. `PeerSignalCatalog` classification

**Not currently classified explicitly** (`CpuThrottleRatio` is likewise not in the `LoadSensitive` list and is pinned `LoadIndependent` by test). Recommend the same for PSI, on two grounds: (a) codebase convention treats the sibling signal for "the same phenomenon" this way; (b) direct measurement (`docs/aiops/aiops-backlog.md`, PSI-waiting p50 unchanged across a 51% traffic increase on this lab) is evidence against a load-proportional relationship, at least in the unstarved regime this lab can produce. This is explicitly **not** conclusive for the starved regime (nobody has measured PSI under contention on this lab — see Scope), and the "Affine work-adjusted trend for load-sensitive signals" backlog item stays unshipped regardless of this classification. Recommend an explicit `MetricIndex.CpuPressureWaiting => PeerSignalKind.LoadIndependent` case (not relying on the silent default), with a code comment citing the measurement, so a future reader does not have to re-derive it.

### 3. Persisted ordinals — no hazard found

- `RawMetricSeries.MetricTypeId` (byte) is constructed fresh every cycle from a live Prometheus read (`PrometheusMetricSource.cs:128`, `PrometheusHistoricalSource.cs:273`) — never written to disk in the production path.
- `FloorCalibrator`'s state file and `MetricHistory`'s seasonal file both key by **enum name** (`Enum.TryParse<MetricIndex>`), not ordinal (`FloorCalibrator.cs:456`, `MetricHistory.cs:354`) — appending a member is exactly as safe as `ContainerRestarts` was.
- `HistoricalCsvLoader` is `MetricSnapshot`-shaped only (12 fixed named columns) and never touches `MetricIndex` — irrelevant, since PSI (like `ContainerRestarts`) is deliberately excluded from `MetricSnapshot`/`FeatureCount`.
- One write-only recorder exists (`LabFixtureRecorderDiagnostics`'s bespoke CSV, per prior-session memory) that nothing reads back today — flagged as out of scope for this change, not a hazard.
- **Constraint, not a risk**: `MetricIndex.cs`'s own doc comment states insertion **in the middle** would "silently reinterpret every stored sample" — the new member(s) must be appended after `ContainerRestarts` (ordinal 13+), never inserted earlier. This is a hard constraint on the diff shape, not a judgement call.

### 4. `CustomMetricBinding` as the alternative vehicle

Covered above. It is unambiguously the cheaper artefact; the question of which the client wants is Open question 1.

## Gate answers

- **Execution path**: neither — this is metric ingestion/config (`Sources/Anomalies/Contracts`, `Sources/Anomalies/Monitoring`), not `InferenceEngine`/`ComputationGraph`. No autograd, no inference buffers.
- **Verification oracle**: parity is not the right frame here (nothing is being computed against a reference implementation). The oracle is **behavioural**: (a) a `[Fact]` proving the PromQL built for the new channel is `rate(container_pressure_cpu_waiting_seconds_total{...}[range])` (string equality, following `PromqlCatalog`'s existing test pattern); (b) an end-to-end test in the shape of `CustomMetricEndToEndTests` or a `MetricMapTests`-style test proving the channel reaches a peer finding and counts as blind when unmapped; (c) if the enum-member route is chosen, `MetricIndex.Count`-dependent tests (`AnomalyGuardTests.cs:49,102`, `MetricMapTests.cs:129`, `MetricDiscoveryTests.cs:115`) must still pass unmodified — they compute expectations off `MetricIndex.Count` dynamically, so they are the regression guard for "did this break blindness accounting", not something to edit.
- **AOT reach**: `Sources/Anomalies` is reached by the `aot-guard` CI job via `Sources/Cli` (verified in memory, 2026-08-06 session) — a new enum member and catalog cases are plain switch expressions, the same shape as the existing 13, so no new AOT risk. No LINQ, no reflection introduced by this change in either route.
- **Allocation policy**: not a hot path (guard cycles run on a multi-second/minute cadence, not per-inference-token); the existing per-cycle array allocations sized off `MetricIndex.Count` are unaffected in kind, only in size (+1 or +2 `double`s per array, per cycle — immaterial).
- **Moat side**: open/offline-batch. This is detection-quality tooling for the AGPL guard, not real-time or GPU work — no moat concern.

## Scope

### Must decide before implementation (client)

1. **Vehicle**: `CustomMetricBinding` (config-only, today) vs. new `MetricIndex` member (product default, ~6-file mechanical change). See table above.
2. **One channel or two**: `waiting` only, or `waiting` + `stalled`. Measured on this lab they track closely (p50 1.39e-5 vs 1.38e-5 in one window — a single data point, not a distribution comparison). `stalled` is the stronger, rarer signal (full-CPU-denial); `waiting` is the more sensitive one (any denial). Recommend **`waiting` only** for the first slice — it is what already settled a real investigation — with `stalled` as a `Should`/`Could` follow-on once `waiting` is in and its behaviour under load is understood, rather than shipping two unvalidated channels at once.
3. **Threshold / floor**: **this lab cannot calibrate a starvation floor** — it has never starved a container, so every distribution the lab can produce (p50 ≈1.4e-5, p95 ≈7.5e-5, max 2.6e-4) is "nothing happened". A `SustainedThresholdRule` shipped from this data would be exactly the `ContainerRestarts`-floor-of-1.25 mistake `PeerSignalCatalog.IsCountedEvent`'s own doc warns about — fitting a rare-event signal to a day with zero events sets the bar below a real one. **What would calibrate it**: a synthetic CPU-contention fault (e.g., a CPU `stress`/`stress-ng` sidecar or a deliberately over-subscribed node) run on the lab, the same way the memory-leak floor (256 MiB) and the throttle floor were each calibrated on an induced fault, not a healthy day. Recommend: ship the channel unfloored (reported but not rule-gated) until that fault is run, exactly as `CpuThrottleRatio` itself currently has no calibrated absolute floor beyond the one-lab-one-fault caveat already on `ForCpuThrottling`.
4. **Does E2 become unnecessary?** No — recommend keeping it. E2 ("a CPU-limited deployment") is what would let `CpuThrottleRatio` itself finally be tested and calibrated properly; PSI does not need a CPU limit but a starvation floor for *either* channel needs an induced-contention scenario, which is a superset of what E2 already proposed. Suggest E2's scope note be updated (by the developer/architect, not here) to say it now serves both channels, rather than closing it.

### Won't (this time)

- No exporter/format work — this is ingestion only, consistent with the project's one-directional loading stance (not directly relevant here, cited for completeness).
- No `stalled` channel in the first slice (see point 2).
- No calibrated `SustainedThresholdRule`/absolute floor for PSI in the first slice (see point 3) — ship observable-only.
- No change to the learned family (`Gpt`/`Baseline`/`Neuro`) or `MetricSnapshot.FeatureCount` — matches the client's own reasoning and the `ContainerRestarts` precedent exactly.
- No change to `k8s/anomaly-guard/guard.lab-workload.json` or the production ServiceMonitor beyond what's needed to scrape the new series, unless the client says the workload-scope config needs it too (ask).
- No retrofitting `CpuThrottleRatio`'s blindness away — it stays blind on this lab by design (matches the JSON config's own stated intent) unless E2 ships separately.

## Not-settled-fact table

| Type | Item |
|---|---|
| Fact | `PromqlCatalog.DefaultTemplate`'s unhandled-member arm throws `ArgumentOutOfRangeException` at first resolution without an override — verified by reading `PromqlCatalog.cs:293-305,384-424`. Mandatory touch point if the enum route is chosen. |
| Fact | `FloorCalibrator`/`MetricHistory` persist by enum **name**, not ordinal — verified by reading `FloorCalibrator.cs:378-456` and `MetricHistory.cs:354`. Appending a member at the end is safe. |
| Fact | `CpuThrottleRatio` is classified `LoadIndependent` and pinned by `PeerSignalCatalogTests.EventsAndFractions_AreComparedRaw` — verified by reading `Sources/Anomalies/Monitoring/PeerSignalCatalog.cs` and `Tests/Anomalies/PeerSignalCatalogTests.cs:45-53`. |
| Fact | On this lab, PSI-waiting p50 was unchanged (1.39e-5 → 1.46e-5) across a 51% request-rate increase — quoted from `docs/aiops/aiops-backlog.md`, "The 34 'unexplained' incidents were a 51% rise in traffic" table. One measured window, not a distribution. |
| Fact | `CustomMetricBinding` reaches rules, peer, trend and floor calibration with zero source changes, already tested end-to-end — verified by reading `Tests/Anomalies/CustomMetricEndToEndTests.cs` and `AnomalyGuardConfigReader.cs:65-115`. |
| Assumption | PSI is not load-proportional (`LoadIndependent`), based on the one measured window above plus the `CpuThrottleRatio` convention. **Not validated under actual contention** — the lab has never starved a pod. Client/architect should treat this as provisional until a contention experiment runs. |
| Assumption | `waiting` alone (not `stalled`) is the right first slice — the client's request named `waiting` explicitly; `stalled` was raised by the analyst as an option, not requested. |
| Assumption | No built-in `SustainedThresholdRule`/absolute floor ships in this slice — observable-only until an induced-contention measurement exists. |
| Decision | client, 2026-08-08 — add PSI **alongside** `CpuThrottleRatio`, not replacing it. |
| Constraint | A new `MetricIndex` member, if that route is chosen, must be appended after `ContainerRestarts` (ordinal ≥13) — inserting earlier silently reinterprets every persisted sample (`MetricIndex.cs:26-28`, the file's own warning). |
| Constraint | `MetricSnapshot`/`FeatureCount` (12) must not change — moving it invalidates trained checkpoints (`MetricSnapshot.cs:24,31-36`), and the client's own reasoning already excludes PSI from it. |
| Open question | **For the client** — vehicle: `CustomMetricBinding` (today, config-only, this lab) vs. new `MetricIndex` member (product default, ~1-week-visible mechanical change across 6 files). See table above. Owed before any code is written. |
| Open question | **For the client** — one channel (`waiting`) or two (`waiting` + `stalled`) in this slice? |
| Open question | **For the client/architect** — is an induced-CPU-contention experiment (stress sidecar or over-subscribed node) in scope now, to calibrate a starvation floor, or is "observable, unfloored" acceptable for launch and the floor a follow-on once E2-adjacent infrastructure exists? |
| Open question | **For the client** — does `k8s/anomaly-guard/guard.lab-workload.json` (the multi-scope/workload variant) need the same binding, or is the lab's single-scope config (`guard.lab.json`) sufficient for this slice? |
| Open question | **For the developer/architect** — confirm the recommended `SignalClass` for PSI. `CpuThrottleRatio`'s sibling reasoning in `SignalClass.Infrastructure`'s own doc comment explicitly lists "CFS throttling" as a worked example of that class ("platform acting on the workload... load-independent, actionable end of an incident") — PSI is the same kind of platform-level denial-of-CPU signal, so `Infrastructure` is the analyst's recommendation, not yet confirmed by the developer against `SignalClass`'s full semantics. |

## Priorities (MoSCoW)

- **Must**: land the `waiting` channel via the chosen vehicle (Open question 1), reaching rules/peer/trend and blindness accounting, verified by the tests in "Gate answers → Verification oracle". No performance target attached, so no benchmark obligation.
- **Should**: an explicit `PeerSignalCatalog.Classify` case for `waiting` (rather than relying on the silent default) with the measurement cited in comment, matching this codebase's convention of stating the reasoning rather than leaving it implicit.
- **Could**: `stalled` channel; `MetricNameCatalog.For` discovery candidates; a built-in `OverfitServerQueries`/`DefaultRules` entry once a starvation floor exists.
- **Won't (this time)**: calibrated absolute floor/`SustainedThresholdRule` for PSI; any change to `MetricSnapshot`/`FeatureCount`/the learned family; closing backlog item E2; exporting or converting anything (not applicable here, stated for completeness per house style).

## Tasks (user stories)

### T1 — As the operator watching a deployment with no CPU limits, I want a CPU-contention signal that is not permanently blind, so that CFS-limit-less pods are still monitored for CPU denial.

**Acceptance criteria** (vehicle-agnostic; concretised once Open question 1 is answered):

- Given a healthy pod with the PSI series present and no CPU limit, when the guard runs a cycle, then `MetricMap.IsMapped`/blindness accounting for the new channel is **true** on the lab config, and `CpuThrottleRatio` remains **blind**, unchanged from today.
- Given the same pod, when the guard resolves the PromQL for the channel, then the query string is exactly `sum by (pod) (rate(container_pressure_cpu_waiting_seconds_total{<selector>}[<range>]))`, matching `MetricSourceKind.Counter`'s existing wrapping rule.
- Given a synthetic window where one pod's PSI-waiting rate is materially above its peers (fixture: 8 pods, 7 at ~1.5e-5, 1 at ≥5e-4 — two orders of magnitude above the measured healthy max of 2.6e-4), when the guard runs a cycle, then a finding is reported naming the new channel, following the pattern of `Tests/Anomalies/CustomMetricEndToEndTests.ACustomMetricProducesAPeerFinding` (if the config vehicle is chosen) or an equivalent `MetricIndex`-based test (if the enum vehicle is chosen).
- Given the enum-member route is chosen, when `dotnet test -c Release` runs, then `AnomalyGuardTests.cs:49,102`, `MetricMapTests.cs:129`, `MetricDiscoveryTests.cs:115` (all `MetricIndex.Count`-relative) pass unmodified — they are the regression guard, not code to edit.

**NOT READY** until Open questions 1–2 are answered: the acceptance criteria above cannot be made concrete (which files change, which test class houses them) without knowing the vehicle and the channel count.

### T2 — As the incident-grouping pipeline, I want the new channel's peer-comparison behaviour to reflect what was actually measured about it, so that it does not inherit the load-sensitivity gap the backlog already named and priced (34 incidents at +51% traffic).

**Acceptance criteria**:

- Given the classification is set explicitly (not left to the silent default), when `PeerSignalCatalogTests`-style tests run, then the new channel is asserted `LoadIndependent`, with a test citing the aiops-backlog measurement in its doc comment, following the existing pattern (`Tests/Anomalies/PeerSignalCatalogTests.cs:45-53`).
- Given this classification is provisional (see Assumption in the table), when the client or architect later runs a load-variation experiment on PSI specifically (not just the one incidental window already recorded), then the classification is revisited — this task does not close the "Affine work-adjusted trend" backlog item, only avoids repeating its known failure mode on day one.

**Ready** — the acceptance criteria do not depend on the vehicle decision (a `PeerSignalCatalog.Classify` case is only meaningful for the enum route; for the `CustomMetricBinding` route, the equivalent is `SignalKind: PeerSignalKind.LoadIndependent` in the config entry itself — either way the *value* is settled, only its location moves).

### T3 — As the client's compliance/ops reader of the guard's own documentation, I want to know that this channel ships unfloored, so that an absent alert is read correctly as "not yet calibrated," not "confirmed healthy."

**Acceptance criteria**:

- Given the channel ships in this slice, when the guard's own docs or the `k8s/anomaly-guard/guard.lab.json` comment block are updated (by the developer, not this plan), then they state plainly that no absolute rule is attached and why — mirroring the file's existing convention of explaining every deliberate omission in its own comment header.

**Ready** — small, no code, but real: this codebase's own house style (per every file read in this investigation) is to write down *why* a threshold is or isn't there, and skipping that here would be inconsistent with everything else in `Sources/Anomalies`.

## Traceability

| Goal | User need | Task | Acceptance criterion | Verified by |
|---|---|---|---|---|
| Guard has no permanently-blind CPU channel on unlimited pods | Operator needs CPU-contention visibility regardless of limits | T1 | PromQL/mapping/blindness criteria above | New `[Fact]`s per chosen vehicle; existing `MetricIndex.Count`-relative tests pass unmodified |
| Don't repeat the 34-incident false-positive class | Grouper needs correct load-sensitivity for the new signal | T2 | Explicit `LoadIndependent` classification, cited measurement | `PeerSignalCatalogTests`-style test |
| Guard's own docs stay honest about calibration state | Ops/compliance reader needs to know absence-of-alert ≠ health | T3 | Doc/config comment states "unfloored, pending contention measurement" | Manual review against house style (no automated oracle for prose) |

## BLOCKING QUESTIONS

1. **Vehicle**: ship this as a `CustomMetricBinding` config entry (available today, zero source changes, this lab only) or as a new `MetricIndex` member (product default for every deployment, ~6-file mechanical change following the `ContainerRestarts` precedent, ready in about the same effort but touching shared code)? If unanswered, I will assume the **`MetricIndex` member** route, because the request said "alongside `CpuThrottleRatio`" (a named, permanent product channel) and reasoned about ordinal 13 specifically.

2. **Channel count**: `waiting` only, or `waiting` + `stalled`? If unanswered, I will assume **`waiting` only**, since that is what the request named and what already resolved a real investigation; `stalled` becomes a `Could`.

3. **Calibration scope**: is running an induced-CPU-contention experiment (to calibrate a starvation floor) in scope for this delivery, or does the channel ship observable-only with the floor as a follow-on? If unanswered, I will assume **observable-only for this slice**, because the lab genuinely cannot calibrate a starvation floor today without such an experiment, and fabricating one from a healthy-day distribution would repeat the `ContainerRestarts`-floor-of-1.25 mistake this codebase has already paid for once.

4. **Scope config**: does `k8s/anomaly-guard/guard.lab-workload.json` need the same binding as `guard.lab.json`, or is the single-scope lab config sufficient for this slice? If unanswered, I will assume **`guard.lab.json` only**, matching where the client's own PSI measurement was taken.

5. **`SignalClass`** (for the developer/architect to confirm, not the client): is `Infrastructure` the right class for PSI, by the same reasoning `SignalClass.Infrastructure`'s own doc comment gives for CFS throttling? If unanswered, I will assume **`Infrastructure`**, on that stated precedent.

## SUGGESTED IMPROVEMENTS TO MY ROLE

None this run. The task's own framing (capability map, backlog pointers, `ContainerRestarts` precedent) matched the repository closely enough that round one was largely confirmation rather than discovery; the one genuinely new finding — that `CustomMetricBinding` already delivers the requested capability today with zero source changes — came from following "what already exists" past the enum into `AnomalyGuardConfigReader`/`AnomalyGuardConfigFile`, which the task's own bullet list had already pointed at ("Whether `CustomMetricBinding` is the better vehicle... it exists precisely to add metrics without touching the enum"). No instruction gap found.
