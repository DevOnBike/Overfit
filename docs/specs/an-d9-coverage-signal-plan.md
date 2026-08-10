STATUS: SIGNED
Author: overfit-architect
Date: 2026-08-10
Slug: an-d9-coverage-signal-plan

# AN-D9 — per-pod scrape coverage as a saturation signal

## Process note

No `overfit-analyst` plan preceded this file. `docs/TASKS.md`'s `AN-D9` row and the team lead's dispatch
message carry a measured problem statement and a target mechanism ("what separates the two cases is the
peers... checkable from data the guard already has"). Treated as the problem statement rather than
re-derived. The lead has already corrected the task row once (`RunSilentPods` already covers total silence;
the gap is partial reporting) — verified below, not re-derived.

**On the missing-analyst-plan gap, confirmed by the lead as the dispatch pattern rather than my own gap** —
this is the second time; noted, not re-litigated here.

**Amendment 1, 2026-08-10**: the lead answered the two original blocking questions and asked five follow-up
points; §1, §3, §5 and the blocking-questions section were updated in place.

**Amendment 2, 2026-08-10 — §3's mechanism is wrong, and the developer was right to refuse it.** The lead
verified two arithmetic defects in the signed `RunCoverage` design against the source directly, I independently
re-verified both plus a third the lead's own proposed fix does not clear, and **§3 below is a full redesign,
not a patch.** Superseded claims are marked, not deleted — see §1 findings 7-9 and the new §3.

## Architecture review (overfit-architect)

### 0. Verification note

Amendment-2 claims and their source, each re-derived independently against the current source rather than
taken from the lead's message: `PeerOutlierOptions.cs` (full read), `PeerGroupOutlierDetector.cs` (`Detect`,
`Exclude`, `MeasureGaps`, full read of the decision path), `FloorCalibrator.cs` (`ObserveChannel`,
`InertChannels`, `Propose`, full read of the accumulation and proposal paths), `CustomMetricBinding.cs` (full
read — no `Query` field exists), `MetricMap.cs` (`CustomQueries`, `Build`).

### 1. Review verdict

1. **The task row's premise is correct but its scope claim is not.** "Needs no new channel" is true in the
   sense that matters to the lead's framing — no change to the workload's own instrumentation — but false
   literally: no bound channel today can answer "did Prometheus successfully scrape this pod," and one new
   PromQL binding is required (finding 3). This is a scope correction, not a disagreement.
2. **`MetricNameCatalog`'s own `Container(...)` helper (`MetricNameCatalog.cs:263-264`) proves the five
   container-level channels — `CpuUsageRatio`, `CpuThrottleRatio`, `MemoryWorkingSetBytes`, `OomEventsRate`,
   `ContainerRestarts` — are scraped by kubelet/cAdvisor, a different target from the app's own `/metrics`
   endpoint that `RS-6` found saturates.** A saturated Kestrel connection queue stops that endpoint answering;
   it does not touch kubelet's cgroup read. **Coverage computed from a container-level channel would read
   100% throughout the fault** — the exact silent-failure shape this protocol exists to catch. The signal
   must be sourced from the same scrape target as the one that saturates. **Unaffected by amendment 2.**
3. **That target already publishes the exact right series: Prometheus's own `up`.** Verified live
   (`up{namespace="lab"}`, 2026-08-10): one series per pod, `job="lab-workload"`, `container="workload"`,
   `pod=<name>`, all `1` on the healthy fleet at the time. `up` is written by Prometheus **on every scrape
   attempt, success or failure** — unlike an application series, it never goes missing when a scrape fails,
   it goes to `0`. **Unaffected by amendment 2**; what changed is *how* `up` is turned into the value the
   detectors compare — see finding 9 and the new §3.
4. **SUPERSEDED by amendment 2, keep for the record.** The original finding 4 claimed a coverage value
   computed as `(samples where up==1) / window.Length` needed a `CreatedAt`-bounded denominator to avoid
   reading a young pod as falsely degraded, and that `RunCustomPeer` provided no such exemption. **The premise
   under which this mattered — a bespoke C#-computed per-pod scalar — no longer exists**; see finding 9. The
   underlying observation about `RunCustomPeer` itself is separated out and restated in finding 6 below,
   because it is still true and still not about coverage specifically.
5. **`FloorCalibrator.Propose` treats every custom channel as fittable by default** (`FloorCalibrator.cs:565-569`,
   "nothing here can tell a restart counter from a latency" for a customer-named channel). Coverage is
   healthy at a **constant ceiling** (1.0, essentially always) — the same shape `PeerSignalCatalog.IsCountedEvent`
   exists to exempt `ContainerRestarts`/`OomEventsRate` from, and the exact trap `AN-C3` paid for on a
   different channel. There is **no equivalent opt-out for a custom channel.** **Unaffected by amendment 2**
   — inertness is a property of the *value distribution*, independent of which mechanism produces it.

   **Amendment 1, in answer to the lead's point 5, with the mechanism traced end to end and re-verified in
   amendment 2's pass**: `RunCoverage`/its replacement feeds `FloorCalibrator.Observe` → `ObservedRange.Fold`
   (`ObservedRange.cs:59-85`) folds every raw sample into a running `Min`/`Max` → on a healthy channel every
   sample is `1.0`, so `Min == Max == 1.0` and `ObservedRange.IsConstant => HasSamples && Min.Equals(Max)`
   (`ObservedRange.cs:56`) is `true` → `FloorCalibrator.InertChannels()` (`FloorCalibrator.cs:641-668`)
   already iterates custom channels (`foreach (var pair in _customChannels)`, line 656) and calls
   `IsInert(observed, count, minimumObservations) => count >= minimumObservations && observed.IsConstant`
   (`FloorCalibrator.cs:670-673`) — `true` once `count` (see the corrected timing below) is reached →
   `InertChannel.IsConclusive => Value != 0.0` (`InertChannel.cs:52`) is unconditional, and the folded value
   is `1.0`, non-zero. **Every step of that chain is satisfied by a correctly-working coverage channel, so
   `InertChannels()` will report it as `IsConclusive = true` — a confirmed defect — on every healthy
   deployment, deterministically.**

   **Amendment 2 correction, verified against the source, not the lead's number alone**: my "~20 hours" was
   wrong. `InertChannels()` gates on `channel.Magnitudes.Count`
   (`FloorCalibrator.cs`: `IsInert(channel.Observed, channel.Magnitudes.Count, minimumObservations)`), and
   `ObserveChannel` calls `channel.Magnitudes.Add(...)` **once per pod, inside the per-pod loop**
   (`FloorCalibrator.cs:354-369`), not once per cycle. At 12 pods and the default `minimumObservations = 240`,
   that is **20 cycles — 100 minutes at the deployed 5-minute cadence**, not 20 hours.

   **Amendment 2 correction on the exemption's shape, verified**: it **cannot** be a flag on
   `CustomMetricBinding`. `FloorCalibrator._customChannels` is keyed by name, discovered from
   `window.CustomChannels` (`FloorCalibrator.cs:250-260`) — the calibrator never receives a
   `CustomMetricBinding` at all. The exemption has to be a name set threaded onto `AnomalyGuardOptions` and
   passed into `FloorCalibrator`'s construction/read path.

   **Amendment 2 correction on the exemption's granularity, verified**: it must be **binary**, not "still fit
   the floor but suppress only the inert report" as I'd left open. `Propose`'s `gapMax = channel.PeerGaps.Max`
   (`FloorCalibrator.cs:585`), and `PeerGaps` is folded from `|median(pod) − median(others)|`
   (`FloorCalibrator.cs:384-397`). On a healthy fleet every pod's median is `1.0`, so every folded gap is
   `0.0`, `gapMax` stays exactly `0.0` forever, and `Propose` returns `proposedGap = 0.0` — which
   `PeerOutlierOptions.MinAbsoluteGap`'s own doc states means "gate off" (zero disables the gate). The first
   time the channel *does* move — the fault this whole signal exists to report — that single event becomes
   `PeerGaps.Max`, and a live calibrator would then propose a floor at `1.25×` **that one event**, permanently
   raising the bar against future faults of the same or smaller size. Useless while healthy, actively harmful
   once it has seen the only thing it is for. This is required-before-ship, not a spike — see §5.
6. **`RunCustomPeer` (`AnomalyGuard.cs:762-810`) does not check `IsWarmingUp`, and this is a live defect
   today, independent of this task, for the five channels already flowing through it in the deployed
   config** (`CpuPressure`/`LockContentions`/`Exceptions`/`ActiveRequests`/`GcCommittedBytes`) — confirmed by
   reading it end to end: no call to the warm-up gate that `RunTrend`'s per-pod loop uses
   (`AnomalyGuard.cs:1210`). A freshly-created pod's cold-start value (empty caches, no request history yet)
   is genuinely different from steady state and gets compared as a real outlier with nothing to catch it.
   **Not fixed here — and, after amendment 2, not needed for coverage either** (finding 9: Prometheus's own
   `avg_over_time` excludes samples from before a pod existed, so a young pod's value is never artificially
   low). **Recommend a separate `XC-` row** for the general `RunCustomPeer` warm-up gap; not opened here,
   since assigning it is the registry owner's call.
7. **Defect 1, verified against the source, not merely relayed: `RunCoverage`'s per-pod scalar is structurally
   silent under `PeerGroupOutlierDetector.Detect`.** `PeerOutlierOptions.Balanced` is
   `new(0.05, 0.33, 30, 3, 0.08)` — `MinimumSamplesPerPeer = 30` (`PeerOutlierOptions.cs:74`). `Detect` calls
   `Exclude(peers, options.MinimumSamplesPerPeer, ...)`, and `Exclude`'s gate is
   `usable = bounds[i + 1] - bounds[i]; if (usable >= minimumSamples) { ... }`
   (`PeerGroupOutlierDetector.cs:495-497`) — `usable` is the count of normalised values that peer contributed.
   **One scalar per pod per cycle means `usable == 1` for every member**, which is below 30 for all of them,
   so every peer is excluded and `comparable < options.MinimumPeers` fires every cycle
   (`PeerGroupOutlierDetector.cs:158-169`): the verdict is `InsufficientData`, healthy or faulted, always.
   Lowering `MaxPValue` cannot rescue it either — `correctedAlpha = options.MaxPValue / (2.0 * comparable)`
   (`PeerGroupOutlierDetector.cs:197`) is a family-wise correction on top of a comparison that never runs
   because the sample gate already returned before it.
8. **Defect 2, verified: feeding the raw per-slot `up` series instead clears the sample gate but fails
   materiality.** With ~30 raw `0`/`1` samples per pod, `Exclude` passes (`usable == 30`), but
   `medians[i] = MedianSelector.MedianInPlace(...)` (`PeerGroupOutlierDetector.cs:186-191`) is the **median**
   of that peer's own samples, and `MeasureGaps` (`PeerGroupOutlierDetector.cs:561-587`) compares medians, not
   raw values. **The median of a binary series is `1.0` for any pod with more than 50% of its samples at `1`**
   — a pod at 100%, 90% or 65% coverage all report the identical median `1.0` as a fully healthy peer, so
   `absolute[i] = |medians[i] − median(others)| = 0`, the size gate never clears, and the verdict is `Healthy`
   whatever the true coverage is above the 50% line. Raw `up` only ever moves the median once coverage falls
   *below* 50%, which is near-total silence — `RunSilentPods`' territory, not the partial-degradation case
   this task exists for.
9. **Accepted redesign, with one gap the proposal itself did not surface: `CustomMetricBinding` has no
   verbatim-query escape hatch, so the fix cannot ship as config alone.** The repair both Defect 1 and
   Defect 2 point to — bind the per-slot value itself as a continuous fraction,
   `avg_over_time(up{%selector%}[<range>])`, as an ordinary `Ratio` custom channel through the existing
   `RunCustomPeer`/`RunCustomTrend` — is accepted (see the new §3) and is a materially simpler design than the
   signed one: it needs no `RunCoverage`, no `CreatedAt` denominator (Prometheus's `avg_over_time` excludes
   samples from before a target existed rather than treating them as `0`, so a young pod reads `1.0` from its
   first real sample onward — this also resolves finding 4/6 for coverage specifically, though not generally),
   and it rides `RunCustom`'s existing `blind`/`partial` tracking. **But it cannot be expressed through
   `CustomMetricBinding` as it stands today** — verified by reading the full record
   (`CustomMetricBinding.cs:77-92`): there is **no `Query` field**, only `Source` (a bare metric name) +
   `Kind` (which picks exactly one of `Gauge`/`Counter`/`EventCount`/`Ratio`/`HistogramSeconds` wrapping in
   `MetricMap.Build`, `MetricMap.cs:219-238`, reused for custom channels by `CustomQueries`,
   `MetricMap.cs:92-112`). `Ratio` renders `{name}{{{token}}}` — a bare series name with the selector
   appended — which cannot produce `avg_over_time(up{selector}[range])`; the range vector and the selector
   would end up in the wrong place. **This needs a small, precedented code change**: a `Query` field on
   `CustomMetricBinding`, mirroring `MetricBinding.Query` for built-ins exactly (same verbatim-wins-over-
   kind pattern, same `%selector%`-substitution mechanism `FillCustomAsync` already performs downstream), and
   `CustomQueries()` preferring it when set. This is the same class of fix `OomEventsRate` and
   `CpuThrottleRatio` already forced onto the built-in path — the custom path never got the equivalent.
   **§7's "config-only, additive" decision in the signed version was wrong and is retracted below.**
10. **Amendment 3, from the lead, verified precisely rather than taken at the stated wording: the config-file
    surface for custom entries has the identical class of gap one layer down, and it is slightly more specific
    than "`CustomEntry` needs a `query` field."** Checked `AnomalyGuardConfigFile.cs` and
    `AnomalyGuardConfigReader.cs` directly: `AnomalyGuardConfigFile.CustomEntry : MetricEntry`
    (`AnomalyGuardConfigFile.cs:133`) **already inherits a `Query` string property** from `MetricEntry`
    (`:129`) — the JSON key `query` is already syntactically acceptable on a `customMetrics` entry today. The
    actual gap is in `AnomalyGuardConfigReader.ReadMap`: the `file.Metrics` loop reads, validates
    (`%selector%` presence, `AnomalyGuardConfigReader.cs:54-63`) and passes `entry.Query` into
    `MetricBinding`'s constructor (`:74`); the `file.CustomMetrics` loop (`:77-129`) **never references
    `entry.Query` at all**, and could not wire it anywhere even if it did, because `CustomMetricBinding` has
    no field to receive it (finding 9). So a `query` key set on a custom entry today is silently read into the
    DTO and then dropped on the floor — the exact `minGapChange` shape (`AnomalyGuardConfigReader.cs:121-124`'s
    own comment: "Reading it was missing entirely until 2026-08-10 — the binding carried the property, the
    file could not set it"), just with the DTO field present and the *reader* missing instead. Three things
    are needed, not one: the `CustomMetricBinding.Query` field (finding 9); a wiring change in the
    `CustomMetrics` loop passing `entry.Query.Trim()` into it, with the **same `%selector%` validation** the
    `Metrics` loop already applies (`:56-63`) — currently unenforced for custom entries, a second, narrower
    gap the request did not name but the fix should close in the same pass; and a `MetricMap.CustomQueries()`
    change to prefer `binding.Query` over `Source`+`Kind` when set. **Extend
    `Tests/Anomalies/ConfigSurfaceCompletenessTests.cs`** with a case in the shape of
    `EveryNumericKnobOnACustomBindingCanBeSetFromTheConfigFile`/`OmittingTheGapChangeLeavesItZeroRatherThan
    InventingOne` — a `query` set on a `customMetrics` entry survives into `CustomMetricBinding.Query`, and a
    file missing `%selector%` in a custom entry's `query` is rejected the same way a built-in one already is —
    rather than a new test class, per the lead's instruction.

### 2. System context

**Touches**: one new field on `CustomMetricBinding` (`Query`, verbatim PromQL, mirroring `MetricBinding.Query`
— finding 9) and the matching change to `MetricMap.CustomQueries()` to prefer it over `Source`+`Kind`;
`AnomalyGuardConfigReader.ReadMap`'s `CustomMetrics` loop, to wire the **already-present**
`CustomEntry.Query` (inherited from `MetricEntry`) into the new binding field and validate `%selector%`
the same way the `Metrics` loop does (finding 10) — **no DTO/`AnomalyGuardConfigFile` change needed**, only
the reader; `Tests/Anomalies/ConfigSurfaceCompletenessTests.cs`, extended, not duplicated (finding 10); a new
`CustomMetricBinding` entry in `k8s/anomaly-guard/guard.lab-workload.json` (and the deployed ConfigMap)
binding `avg_over_time(up{%selector%}[<range>])`; a small, additive extension to `RunCustomPeer` for the
persistence gate (§3) — recommended shape: an opt-in `MinConsecutiveCycles` field on `CustomMetricBinding`,
default `1`, provably a no-op for the five channels already using it (§3, §5). **No `RunCoverage` method, no
`CreatedAt`-aware denominator, no new `MetricWindow` method** — all three were needed only by the superseded
design (findings 7-9).

**Answered by the lead, 2026-08-10 (round 1), reconciled against amendment 2**:
1. **Persistence gate**: required, not same-cycle. Redesigned in §3 as an opt-in extension to `RunCustomPeer`
   rather than a bespoke `_coverageBreach` dictionary in a bespoke `RunCoverage` — the underlying instruction
   (reuse `SilentPodCycles`'s value, don't add a second same-meaning knob) is unchanged.
2. **`GuardCycleResult` counter**: **answered — no coverage-specific field.** The lead checked the general
   case before answering: `AnomalyGuardService`'s blind-channel naming loop (`AnomalyGuardService.cs:482-498`)
   iterates `MetricIndex` only and contains zero references to custom channels, so **every one of the five
   deployed custom channels is already counted-but-never-named when blind, not just a hypothetical future
   coverage channel.** A coverage-specific field would have papered over one channel while leaving the other
   five silent. Opened as **`XC-11`** (general: name a blind custom channel the same way a blind built-in one
   already is). Coverage's own blindness is covered by `XC-11`, not by anything this task adds — no
   `GuardCycleResult` change here.
3. **Finding 1 (container channels can't be the source) confirmed** and the `docs/TASKS.md` `AN-D9` row was
   updated by the lead directly — no action needed from this role.
4. **Finding 4/6 (RunCustomPeer/IsWarmingUp)**: confirmed a live defect today for the five existing custom
   channels, independent of this task; recommend a separate `XC-` row. **After amendment 2, coverage itself no
   longer needs this fixed at all** — `avg_over_time` handles the young-pod case at the PromQL level.
5. **Finding 5 (FloorCalibrator/InertChannel exemption)**: confirmed required-before-ship, name-based on
   `AnomalyGuardOptions`, binary. Unaffected by amendment 2 — see the amendment-2 corrections folded into
   finding 5 above.

**Does not touch**: `MetricIndex` (stays at 13); the learned-state feature contract (coverage is a custom
channel, `MetricSnapshot` never sees it); `guard.json`'s existing sections (a coverage floor persists through
the same `FloorCalibrator`/`LearnedState` mechanism every other custom channel already uses, once the finding-5
exemption exists). **`PeerGroupOutlierDetector`/`PeerOutlierOptions`** (`Sources/Main/Statistics`, unchanged,
stateless, public API not touched) — the redesign works *with* their existing sample-gate and materiality-gate
semantics rather than trying to satisfy them with the wrong shape of input.

**Depends on**: nothing new — the `CreatedAt`/`IPodRoster` dependency the signed version needed is gone.

### 3. Boundaries and responsibilities — redesigned

**Where the coverage value comes from: PromQL, not C#.** Bind a custom channel (name e.g. `ScrapeCoverage`)
with the verbatim query `avg_over_time(up{%selector%}[<range>])`, `Kind: Ratio` (the resulting value is
already a `[0,1]` fraction; `Kind` still governs sampling semantics elsewhere even when `Query` wins the
query-building step, per `MetricBinding.Query`'s own precedent). Each grid point's value is then the fraction
of scrapes that succeeded over the trailing `<range>`, a continuous number, not a `0`/`1` sample — this is
what clears Defect 2's materiality gate: a pod sustaining 65% coverage reports medians around `0.65`, not
`1.0`, against healthy peers at `~1.0`, an absolute gap around `0.35` and a Cliff's delta that a real
degradation produces.

**`<range>` is an unmeasured parameter, not invented here — see §5.** Recommend defaulting it to
`AnomalyGuardOptions.RecentWindow` (already 15 minutes in the deployed lab config, already the number that
governs how far back the peer/rule families look) rather than a new duration, on the same "don't add a second
knob with the same meaning" instruction the lead gave for the persistence count. This needs confirming, not
assuming: the range has to be wide enough to give `MinimumSamplesPerPeer` (30) grid points of history *and*
narrow enough that a single missed scrape doesn't get so diluted it never clears `MinAbsoluteGap` once that
floor is measured (§5, item 2, carried over from the signed version).

**Why this needs no `CreatedAt` handling (finding 9).** `avg_over_time` over a range that predates a pod's
existence returns no sample for that instant — Prometheus excludes absent data from the average rather than
counting it as `0` — so a pod two minutes into its life with two minutes of unbroken `up=1` reads `1.0`, not
some fraction depressed by the rest of a 15-minute window it was never scheduled for. The `MetricWindow`
convention ("missing is NaN, never zero") and Prometheus's own semantics agree here, which is what makes the
signed version's bespoke denominator computation unnecessary rather than merely simplified.

**Persistence gate (lead's decision 1, redesigned shape): an opt-in extension to `RunCustomPeer`, not a
parallel `RunCoverage`.** `Classify`/`Demote` (`AnomalyGuard.cs:1017,1063`) already operate on a per-pod,
index-aligned array between `_peer.Detect(...)` and `pipeline.ObservePeerGroup(...)` — exactly the shape a
third gate needs. Recommended: a new `int MinConsecutiveCycles = 1` field on `CustomMetricBinding`, read only
when `> 1`; when active, `RunCustomPeer` tracks a small per-pod counter (keyed by pod name, one dictionary per
persistence-gated binding — a coverage-specific state, not `RunSilentPods`' `_silent`, because the predicates
differ: total silence and peer-outlier-on-a-continuous-value are conditions a pod can be in independently) and
only forwards a finding once the counter reaches the **threshold**, which reads `_options.SilentPodCycles`
directly rather than a new option — one number answers "how many cycles before I believe a coverage-related
degradation is real," per the lead's instruction against two knobs with the same meaning. **Constraint, not a
mandate on exact shape**: whatever the developer builds, it must be provably a no-op for the five channels
already on `RunCustomPeer` when `MinConsecutiveCycles` is left at its default — a regression test asserting
their existing behaviour is unchanged is the proof, not a review of the diff. A self-contained thin wrapper
(duplicating the `Detect`/`Classify`/`Demote`/`ObservePeerGroup` call sequence rather than extending the
shared method) is a legitimate alternative if the extension turns out awkward in practice; the requirement is
the outcome and the non-regression proof, not this exact shape.

**`GuardCycleResult` counter (lead's decision 2): closed — no coverage-specific field.** Its round-1
justification — `RunCoverage` sitting outside `RunCustom`'s counted loop — no longer applies, because coverage
now **is** an ordinary entry in that loop and inherits `blind`/`partial`/`unevaluable` tracking exactly as
`CpuPressure` and the other four already do; a broken `avg_over_time(up[...])` binding returning nothing for
every pod already moves the cycle's `blind` count and the summary log line. What none of the five existing
custom channels get, and what a coverage-specific field would only have fixed for one of six, is being
**named** individually — the `silentButBound` naming loop (`AnomalyGuardService.cs:482-498`) walks
`MetricIndex` only and has zero references to `window.CustomChannels`. The lead confirmed this is the general
gap, not a coverage-specific one, and opened **`XC-11`** for it. Nothing in `AN-D9` changes
`GuardCycleResult`.

**Where the "everyone degraded at once" case lands**: unaffected by amendment 2 — still the trend/level-shift
family on the common component (`RunCustomTrend`'s `CrossPeerBaseline`/`ObserveLevelShift` path,
`AnomalyGuard.cs:830-852`), which operates on `LevelShiftDetector.StepSize` over the cross-peer median series
rather than the per-peer-median-gap comparator Defects 1/2 broke. The continuous `avg_over_time` values help
this path too, though it was never structurally blind the way the peer path was.

**Rollout distinguishing, reconsidered under the new mechanism**: a terminating pod's brief coverage dip is
now damped twice over, not once. A single missed scrape inside a `<range>`-wide `avg_over_time` window moves
that grid point's fraction by roughly `1/N` (`N` = samples inside `<range>`) rather than reading as a hard
`0`, so it is far more likely to sit below `MinAbsoluteGap` once that floor is measured (§5); and the
persistence gate still requires the dip to recur across `SilentPodCycles` cycles before it is forwarded.
`StalePodsExcluded`/`StaleStepTolerance` (unchanged, `PrometheusMetricWindowSource.cs:242`) still removes a
pod entirely once it falls more than 2 steps behind the fleet's freshest sample, for the case that runs past
both of the above.

### 4. Quality requirements as parameters

| requirement | parameter | measured against |
|---|---|---|
| detect saturation | per-pod `avg_over_time(up[<range>])`, unit = fraction `[0,1]` of scrapes succeeded over the trailing range | `RS-6`'s 0/6 scrapes is the **total-silence extreme**, already `RunSilentPods`' case (finding 8) — **not sufficient alone as acceptance evidence for this signal** |
| **partial-degradation fixture, required** | a positive fixture with coverage sustained around **60-70%** on one pod, peers at ~100% | not yet produced — required per protocol step 2 and per the lead's instruction; this is what proves the signal is distinct from `RunSilentPods` rather than a slower version of it |
| distinguish outlier vs fleet-wide | Cliff's-delta materiality gate + `MinRelativeGap`/`MinAbsoluteGap`, unchanged, reused from `PeerGroupOutlierDetector` — now actually reachable, per findings 7-8 | no new number for the gates themselves; `MinAbsoluteGap` for this channel is still unmeasured, see below |
| distinguish rollout vs saturation | smoothing from `<range>` + a required persistence gate reusing `SilentPodCycles` + existing `StalePodsExcluded`/`StaleStepTolerance` | persistence gate is decided, not measured; smoothing's adequacy is part of the `<range>` spike (§5) |
| `<range>` for `avg_over_time` | **not set here** — recommended starting point is `RecentWindow` (15m in the lab), not a new number | see §5: needs to be checked against the partial-band fixture, not assumed |
| `MinAbsoluteGap`/`MinAbsoluteGapChange` for coverage | **not set here** — no measurement exists | see §5: propose the spike, do not invent a number |

No performance claim is made or needed — this is a per-cycle, minutes-scale computation, not a hot path; the
BenchmarkDotNet requirement in `CLAUDE.md` does not apply.

### 5. Technical risks, each with its spike

1. **`<range>` for `avg_over_time` is unmeasured.** Spike: bind the channel with `<range> = RecentWindow`
   (15m) against the required partial-band fixture (§4) and a healthy-fleet recording; confirm the fixture's
   sustained ~65% pod reads meaningfully below its peers' ~1.0 medians, and confirm a single transient scrape
   miss on an otherwise-healthy pod does not clear whatever `MinAbsoluteGap` gets measured (item 2). If 15m
   proves too wide (dilutes a real, shorter fault below detection) or too narrow (too few samples to clear
   `MinimumSamplesPerPeer = 30` at the deployed scrape interval), adjust and say why — don't guess a
   replacement without the same fixture.
2. **No threshold exists for `MinAbsoluteGap`/`MinAbsoluteGapChange` on coverage, and none should be invented
   here** (protocol step 5). Spike: bind the channel in the lab config, no rule armed, run it through a
   healthy cycle window, and measure how often a single transient scrape miss moves a pod's `avg_over_time`
   value on an otherwise-healthy pod — that number, not a guessed percentage, sets the floor above ordinary
   noise and below the partial-band fixture's ~35-percentage-point gap.
3. **Persistence-gate value**: reuses `SilentPodCycles` (default 2) per the lead's decision; confirm rather
   than assume by replaying a recorded rollout window (the shape `AN-A5` used) with the gate active and
   checking 2 cycles absorbs a terminating pod's dip now that smoothing also helps (§3). Raise only if it
   doesn't, and say why per the lead's instruction against a second, drifting knob.
4. **Required before ship, not a spike**: `FloorCalibrator.Propose` **and** `InertChannels()` both need the
   same per-custom-channel, name-based, binary exemption on `AnomalyGuardOptions` (finding 5), or a healthy
   coverage channel is deterministically reported as a confirmed defect after ~100 minutes of uptime on every
   deployment.
5. **The `up{namespace,pod}` selector's uniqueness on a customer cluster is unverified beyond this lab.** Here
   it was confirmed live to carry exactly one series per pod with no ambiguity, because the ServiceMonitor is
   the only one selecting these pods. A cluster running a service mesh sidecar or a second scrape target
   against the same pods could produce more than one `up` series and needs the `job` label added to the
   selector explicitly. Name this as a config precondition in the binding's own doc/comment.
6. **The `CustomMetricBinding.Query` addition (finding 9) and the config-reader wiring for it (finding 10) are
   new surface, however small, and need their own regression coverage**: existing custom channels bound via
   `Source`+`Kind` must be provably unaffected by `CustomQueries()` gaining a verbatim-preferred path, and the
   `%selector%` validation newly applied to the `CustomMetrics` loop must not reject any existing deployed
   binding — same "prove the default is a no-op" discipline as the persistence-gate extension. Concretely:
   extend `Tests/Anomalies/ConfigSurfaceCompletenessTests.cs` per finding 10, not a new test file.

### 6. Operability

Logged the same way `silentButBound`/`StalePodsExcluded` already are (`AnomalyGuardService.cs:482-525`). A
broken coverage binding surfaces through the existing `blind=N` cycle line and `PartialMetrics` (§3); it is
not individually named, same as `CpuPressure` and the other four custom channels today — closed by `XC-11`
generally, not by anything in this task. On restart:
coverage's `FloorCalibrator`/`PeerNoveltyTracker` state persists like every other custom channel's, once the
finding-5 exemption exists to stop it accumulating a harmful floor in the meantime; the persistence-gate
counter (§3) is **not** persisted across restarts, matching `RunSilentPods`' own `_silent` dictionary — a
restart re-earns the configured number of cycles, a known, accepted, pre-existing shape rather than a new one.

### 7. Decisions

**Retracted from the signed version**: "the one genuinely new binding (`up`) is config-only and additive" was
wrong — finding 9 requires a small code change (`CustomMetricBinding.Query`, `MetricMap.CustomQueries()`).

**Still no ADR.** The `Query` field addition is optional and additive to an existing record (same shape as
`MinAbsoluteGapChange`'s prior addition), not a breaking format change; nothing here changes what becomes
public API of consequence, crosses an assembly boundary, changes AOT reachability, or touches the
open/commercial boundary or a persisted on-disk format's existing shape.

### Architecture sign-off

Execution path: **neither** (monitoring/detection cycle, not inference or training). AOT-reachable: **no**
(`Sources/Anomalies` is outside the AOT smoketest's reachable set). Allocation policy: **load/cycle path**,
matching `RunCustomPeer`'s existing per-cycle allocation shape — not the zero-alloc hot path.

## BLOCKING QUESTIONS

### For the client

None. This is an internal detection-quality change with no user-facing contract change.

### For the analyst / team lead

**None open.** `GuardCycleResult` counter: answered, no coverage-specific field — `XC-11` covers the general
blind-custom-channel-naming gap. `CustomMetricBinding.Query`/config-reader wiring (finding 10): accepted, to
be implemented alongside finding 9 as one pass, `ConfigSurfaceCompletenessTests.cs` extended rather than
duplicated. Persistence-gate value (reuse `SilentPodCycles`) and the general `RunCustomPeer`/`IsWarmingUp`
gap (its own `XC-` row, not opened here) stand as answered in amendment 1. The lead is dispatching
implementation off this plan; nothing here awaits a further round unless implementation surfaces a new
disagreement.
