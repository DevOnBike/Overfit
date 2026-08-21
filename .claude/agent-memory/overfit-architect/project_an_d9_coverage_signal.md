---
name: an-d9-coverage-signal
description: AN-D9 per-pod scrape-coverage design (saturation signal) — SIGNED plan, key boundary findings
metadata:
  type: project
---

Plan: `docs/specs/an-d9-coverage-signal-plan.md`, SIGNED 2026-08-10, no analyst plan preceded it.

**Why the five container-level `MetricIndex` channels cannot be the coverage source**: `MetricNameCatalog.Container(...)`
(`CpuUsageRatio`, `CpuThrottleRatio`, `MemoryWorkingSetBytes`, `OomEventsRate`, `ContainerRestarts`) are
kubelet/cAdvisor-scraped, a different target from the app's own `/metrics` (Kestrel) endpoint that `RS-6`
found saturates. Coverage must come from `up{namespace,job,pod}` (Prometheus's own per-target scrape-success
gauge) — verified live 2026-08-10 (one read-only query against the lab's existing `:9090` port-forward,
`up{namespace="lab"}` → one series per pod, `job="lab-workload"`, all `1`). `up` never goes stale/missing on
scrape failure (Prometheus writes it explicitly every attempt), unlike a data series which keeps its last
value for up to `lookback_delta` (5 min default) — so `up` avoids the staleness-marker detection-latency trap
a data-series-based coverage measure would have.

**`RunCustomPeer` does not check `IsWarmingUp`** (verified by reading `AnomalyGuard.cs:762-810` end to end —
no call to the gate `RunTrend`'s per-pod loop uses at `AnomalyGuard.cs:1210`), and `PeerCohorts` groups by
declared `PeerGroup` not `ReplicaSet` (a ReplicaSet-keyed attempt was reverted for breaking canary
visibility — `PeerCohorts.cs:18-22`). So any pod-scalar fed through the generic custom-peer path with no
pod-age normalisation reads a fresh pod as a false outlier. Fix has to live in the caller (a new
`AnomalyGuard.RunCoverage`, CreatedAt-bounded denominator), not in `RunCustomPeer` itself — widening that
shared method would change behaviour for the five channels already flowing through it.

**`FloorCalibrator.Propose` has no per-custom-channel opt-out** from auto-fitting (`IsCountedEvent` only
covers built-in `MetricIndex` members, `FloorCalibrator.cs:565-569`). Coverage is healthy at a constant
ceiling (~1.0), the same shape that made `ContainerRestarts`/`OomEventsRate` need an exemption (`AN-C3`
lineage). Flagged as risk #3 with a spike, not fixed in this plan — worth checking if this recurs on the next
near-constant custom channel.

**"Everyone degraded at once" (fleet-wide Prometheus/scrape problem vs one saturated pod) needs no new
branching** — falls out of the existing peer-vs-trend split already documented in `PeerCohorts.cs:29-33`
("a cohort uniformly bad has no outlier within it... catching a whole cohort regressing together is the
trend family's job") plus `RunCustomTrend`'s existing `CrossPeerBaseline`/level-shift path. Reuse pattern,
not new mechanism.

See also [[project_an_d1_peer_novelty]] (same `RunCustomPeer`/`ObservePeerGroup` machinery, same
find-the-second-caller lesson) and [[reference_measured_baselines]].

**Amendment 2026-08-10, after the lead's answers**: `FloorCalibrator.InertChannels()` (`FloorCalibrator.cs:641-668`)
already iterates custom channels, and `InertChannel.IsConclusive => Value != 0.0` is unconditional — a
channel healthy at a constant **non-zero** ceiling (coverage at 1.0) is NOT the ambiguous case
(`OomEventsRate`'s constant-zero); it is squarely the case the type calls a confirmed defect. **Any future
custom channel whose healthy state is a non-zero constant will trip this the same way** — worth checking for
generally, not just for coverage. Also: persistence-gate decisions here reuse an *existing* option's
configured value (`SilentPodCycles`) rather than adding a same-meaning second knob, per explicit instruction
from the lead — a pattern worth applying whenever a new detector needs "how many cycles before I believe it."
Also confirmed as a role boundary in practice: asked to edit `docs/TASKS.md` by the team lead and declined —
outside the three write exceptions (own memory, plan's architecture sections, `docs/adr/`); named the exact
sentence for someone else to paste in instead.

**The full inert-channel chain, worth citing verbatim next time a custom channel is constant-when-healthy**:
`RunX` feeds `FloorCalibrator.Observe` → `ObservedRange.Fold` (`ObservedRange.cs:59-85`) → `Min==Max` when
every sample is identical → `ObservedRange.IsConstant` (`:56`) → `FloorCalibrator.IsInert`
(`FloorCalibrator.cs:670-673`, `count >= minimumObservations && observed.IsConstant`) → `InertChannels()`
(`:641-668`, already iterates custom channels via `_customChannels`) → `InertChannel.IsConclusive =>
Value != 0.0` (`InertChannel.cs:52`). A channel constant at any non-zero value trips the whole chain to
"confirmed defect," deterministically, after `minimumObservations` (default 240, ~20h). Only constant-*zero*
is treated as ambiguous. **Lesson on reviewer disagreement**: when a teammate says content is "missing" from
a file I just edited, re-read fresh from disk before disputing — in this case the content WAS already there
(likely a stale read on their side), but re-checking also surfaced a real gap (`ObservedRange` itself was
never named, only its downstream consequence) that was worth fixing anyway rather than just citing line
numbers back.

**Amendment 3, 2026-08-10 — the signed §3 mechanism was mathematically broken, caught by the developer, not
by me, and the lesson generalises beyond this task.** I designed `RunCoverage` to feed `PeerGroupOutlierDetector`
one aggregated scalar per pod per cycle. `PeerOutlierOptions.Balanced.MinimumSamplesPerPeer = 30`
(`PeerOutlierOptions.cs:74`) and `PeerGroupOutlierDetector.Exclude` counts `usable = bounds[i+1]-bounds[i]`
per peer (`PeerGroupOutlierDetector.cs:495-497`) — **one scalar means `usable==1` for every peer, so every
peer is excluded and the verdict is `InsufficientData` unconditionally.** I never checked the detector's own
sample-count contract against the shape of data I was proposing to feed it. **Lesson for next time an
architecture review proposes feeding an existing statistical primitive a NEW shape of input: read that
primitive's own minimum-sample/materiality gates FIRST and check the proposed input actually clears them,
before designing the orchestration around it** — I verified `PeerCohorts`/`RunCustomPeer`'s *structural*
behavior thoroughly but never traced a single value through `PeerGroupOutlierDetector.Detect`'s own arithmetic.

Second-order lesson: the "obvious fix" (feed the raw multi-sample `up` series instead of one scalar) clears
the sample-count gate but fails materiality — `MeasureGaps` compares **medians**, and the median of a binary
0/1 series only moves once the fraction crosses 50%, so it is blind to any partial degradation above that
line (`PeerGroupOutlierDetector.cs:186-193,561-587`). **A binary/indicator series is never the right shape to
feed a median-based peer comparator when the signal of interest is a rate below 50%** — needs a continuous
value (here: `avg_over_time` computed by Prometheus itself, not a raw sample).

Third-order finding, verified independently rather than accepted from the developer's proposal: their fix
(`avg_over_time(up[...])` as an ordinary `Ratio` custom channel) **cannot actually be expressed** —
`CustomMetricBinding` (unlike the built-in `MetricBinding`) has no `Query` override field
(`CustomMetricBinding.cs`, full read, no such field), so `Source`+`Kind` alone cannot render a function call
around a range vector. Worth generalising: **when accepting another agent's proposed fix, verify it against
the actual config/type surface it needs to ship through, not just its algorithmic claim** — a mathematically
correct fix can still be unshippable through the existing config surface, and that gap will not show up in an
argument about statistics.

**Amendment 4, 2026-08-10 — final closure, plan re-signed, lead dispatching implementation.**
`GuardCycleResult` counter: I reopened it (justification evaporated once `RunCoverage` was eliminated) rather
than silently carrying the prior decision forward — lead confirmed this was the right call and it surfaced a
**general** gap, not a coverage-specific one: `AnomalyGuardService`'s blind-channel naming loop
(`AnomalyGuardService.cs:482-498`) only walks `MetricIndex`, so all five deployed custom channels are already
counted-but-never-named when blind. Opened as `XC-11`, not fixed in `AN-D9`. Second config-surface gap found
when I verified the lead's exact wording rather than accepting it: `AnomalyGuardConfigFile.CustomEntry`
**already inherits `Query`** from its base `MetricEntry` (`AnomalyGuardConfigFile.cs:129,133`) — the DTO
field exists; what's actually missing is `AnomalyGuardConfigReader.ReadMap`'s `CustomMetrics` loop never
reading/validating/wiring it (no reference to `entry.Query` in that loop at all,
`AnomalyGuardConfigReader.cs:77-129`), same shape as the `minGapChange` precedent
(`Tests/Anomalies/ConfigSurfaceCompletenessTests.cs`) but one layer removed — DTO present, reader missing,
not both missing. **Lesson: when a teammate states a bug's location precisely, still verify which exact layer
(DTO vs reader vs runtime type) is missing before writing the finding** — the fix set was the same either way,
but citing the wrong layer would have sent the developer to add a field that already exists.
