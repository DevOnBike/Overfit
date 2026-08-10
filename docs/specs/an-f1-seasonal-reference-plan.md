# AN-F1 — restore the per-pod cross-peer reference

No prior analyst document exists for this task; the mechanism arrived pre-diagnosed and measured (team lead,
2026-08-10, `docs/TASKS.md` `AN-F1`). This file carries only the architecture review and design requested —
no separate problem statement is being asserted here beyond what was handed over.

## Settled mechanism (not re-diagnosed)

`AnomalyGuard.RunTrend` (`Sources\Anomalies\Incidents\AnomalyGuard.cs:1268-1368`) picks the per-pod trend
reference as `seasonal` when history exists and falls back to `common` only when it does not. Measured
2026-08-10 (`Tests\Anomalies\Diagnostics\SeasonalReferenceSubstitutionDiagnostics.cs`): on a 12-pod fleet
moving together, `common` reports 0 pods at every movement tested (flat, +5%, +15%, +30%); `seasonal` agrees
at flat and reports 12 of 12 at +15% and +30%. This is the day-one/rollout noise mechanism behind the
11→33 / 296→517 regression on the identical 298-cycle replay.

## Architecture review of the team lead's framing

1. **"Composed rather than substituted" is not quite the right shape, and literal composition is wrong.**
   Both `common` and `seasonal` are point estimates of the metric's own level (medians, not residuals) —
   `CrossPeerBaseline.TryBuild` writes the across-peer median at each sample
   (`Sources\Main\Statistics\CrossPeerBaseline.cs:115-117`); `MetricHistory.TryExpectation` writes the
   same-phase historical median (`Seasonal()`, `AnomalyGuard.cs:1389-1402`). Adding them would double the
   signal's own magnitude, not compose two corrections — there is nothing centred on zero to add. The
   detector only ever consumes **one** `expectation` span and tests `values − expectation`
   (`TrendDetector.Detect`, `Sources\Main\Statistics\TrendDetector.cs:94-144`), so "composition" has to mean
   **selecting** the better single reference, not summing two.

2. **The existing custom-channel twin already has the right shape, and it is the cheapest fix available.**
   `RunCustomTrend` (`AnomalyGuard.cs:926-1002`), which runs the identical decomposition for
   `CustomMetricBinding` channels, does **not** have this bug: it sets `expectation = common`
   unconditionally when decomposition succeeds and leaves it `Span<double>.Empty` otherwise — it never
   substitutes a seasonal reference in front of the peer one (it has none to offer). `RunTrend`'s bug is an
   inconsistency between two structurally parallel methods, not a missing capability. The fix is to make
   `RunTrend`'s per-pod branch match `RunCustomTrend`'s, with `seasonal` added back only as the fallback for
   when `common` is unavailable.

3. **The workload-level arm is unchanged and was already correct.** `_trend.Detect(common, times, options,
   double.NaN, seasonal)` (`AnomalyGuard.cs:1312`) tests the fleet's own trajectory against what it usually
   does at this hour — that is the one place seasonal correction belongs, and it stays exactly as written.

### Proposed fix

Reverse the priority at the per-pod level: `common` primary, `seasonal` fallback, `default` when neither is
available.

```csharp
var seasonal = Seasonal(metric, window, from);
double[]? common = null;

if (_options.DecomposeCommonMode && podCount >= CrossPeerBaseline.MinimumPeers)
{
    // ... build peers, as today ...
    common = new double[window.Length];

    if (CrossPeerBaseline.TryBuild(peers, common, new double[podCount]))
    {
        var verdict = _trend.Detect(common, times, options, double.NaN, seasonal);   // unchanged
        pipeline.Observe(WorkloadSubject(), metric.ToString(), verdict, from, to, common);

        ObserveLevelShift(
            pipeline, metric.ToString(), Adjust(common, seasonal), from, to,
            _floors.MinAbsoluteLevelShift(metric), null);
    }
    else
    {
        common = null;
    }
}

ReadOnlySpan<double> expectation = common is not null ? common : seasonal;
```

`TryBuild` returns `false` only when `peers.Count < CrossPeerBaseline.MinimumPeers`
(`CrossPeerBaseline.cs:78-81`), which the outer `if` already excludes — so the `else` is defensive, not
reachable today, and kept only so the fallback is provably correct rather than resting on that being true
forever. `expectation` is a local reassignment, not a signature change — no line downstream needs to change
beyond the `podCount` loop already reading `expectation`.

## What it costs (point 2)

**Nothing measured, against the number cited.** The 2551→376/day seasonal win
(`Tests\Statistics\SeasonalBaselineTests.cs:13-16`, `docs\aiops\aiops-detection-pipeline.md:268-272`) is a
`TrendDetector`-only ablation — raw series vs. residual-against-seasonal, driven directly at the detector
with no cross-peer component in the loop at all (same isolation style as the AN-F1 diagnostic itself: "driven
at the detector, not through the guard"). It is evidence that seasonal correction beats no correction. It is
**not** evidence that seasonal beats `common` — that comparison is exactly what the AN-F1 diagnostic ran, and
`common` won 0-to-12 on a moving fleet. The fix does not touch the regime the 2551→376 number was measured
in: per-pod seasonal correction stays exactly as effective wherever `common` is unavailable —
`podCount < CrossPeerBaseline.MinimumPeers` (a StatefulSet or canary with 1-2 replicas) or
`DecomposeCommonMode` off. Only the regime with 3+ peers and the option on — where a **strictly better**
reference already exists — stops using the weaker one.

**Why `common` is not merely different but better whenever it is available.** It is built from what the
peers are reporting *right now*, so it captures the daily curve (all peers ride it) **and** anything the
daily curve doesn't — a rollout, a load test, a real traffic spike outside the learned pattern. `seasonal`
only knows the historically usual pattern. Nothing `seasonal` removes at the per-pod level in a 3+-peer fleet
is left unremoved by `common`; `common` also removes movement seasonal cannot see.

## Does the workload-level arm change? (point 3)

No, and it should not — see finding 3 above. `RunTrend`'s per-pod branch should mirror `RunCustomTrend`'s,
which is what the fix above does. `RunCustomTrend` has no seasonal fallback at all (custom bindings carry no
`MetricHistory` entry — the history is keyed by `MetricIndex`, not by a channel name). That asymmetry is real
but is **out of AN-F1's scope**: bringing seasonal fallback to custom channels for the `podCount < 3` case is
a separate, smaller task, flagged below as an open question rather than folded in here.

## The `MinimumHistoryDays` interaction (point 4)

**The fix substantially de-silences this on its own, but does not eliminate it, and an explicit signal is
still owed.**

Before the fix: the per-pod reference flipped from `common` to `seasonal` the moment `MetricHistory` accumulated
`MinimumHistoryDays` of data — a change driven by elapsed calendar time, invisible to an operator and
uncorrelated with anything they did.

After the fix: for the per-pod path, reference selection is driven by `podCount >=
CrossPeerBaseline.MinimumPeers` (3) and `DecomposeCommonMode` — both structural, both already visible on
`overfit_guard_pods` and in the guard's own config. `MinimumHistoryDays` no longer silently steers *this*
choice. Two switches remain and neither is currently observable:

- Per-pod fallback to `seasonal` (or to nothing) when a deployment has fewer than 3 replicas, or the option
  is off.
- The pre-existing workload-level switch: `_trend.Detect(common, ..., seasonal)` runs raw when `seasonal` is
  empty and residual once history exists — this was true before AN-F1 and is unchanged by this fix, but it is
  the same class of silent switch the team lead is asking to close.

**Proposed visibility mechanism — additive telemetry, no signature or format change.** Add one counter to
`GuardTelemetry` (`Sources\Anomalies\Monitoring\GuardTelemetry.cs`), following the existing pattern exactly
(`Interlocked` field + `Catalog` entry, see `NoveltyHeld`/`_noveltyHeld` at lines 138-141 and 215-219):

```
overfit_guard_trend_seasonal_only_total
  "Per-pod or workload trend evaluations that used the seasonal reference (or none) because fewer than
  CrossPeerBaseline.MinimumPeers replicas were available, or DecomposeCommonMode is off. A deployment
  crossing that peer count changes which reference is judged against; this is how an operator sees it
  happen rather than inferring it from the incident rate."
```

Incremented once per metric per cycle in `RunTrend` (and, if the asymmetry above is later closed, in
`RunCustomTrend`) whenever `common is null`. This is a pure addition to an already-additive exposition
format (`GuardTelemetry.Render`) — no existing series changes shape, no `guard.json` contract changes, no
`TrendDetector`/`TrendResult` signature changes. **Not an ADR-worthy decision**: no public API in `Main`, no
assembly change, no AOT-reachability change (Anomalies is not reachable from `Tests/AotSmokeTest` —
confirmed by grep, zero hits), no wire-format restructuring, nothing crossing the open/commercial boundary.

## Irreversible decisions (point 5)

**None.** Confirmed by `find_references`/`find_callers`:

- `TrendDetector.Detect`'s signature is unchanged — 60 references across `Sources/Main`, `Sources/Anomalies`,
  `Sources/Benchmark` and the test suite, none of which need to change for this fix.
- `MetricHistory`'s stored shape is unchanged — `Seasonal()` still calls `TryExpectation` the same way.
- No `guard.json`/wire contract changes beyond the additive counter above.
- `RunTrend` is `private`, called only from `RunCycleCore` → `RunCycle` (`find_callers`, 81 call sites, all
  internal to `Sources/Anomalies` and the test suite) — this is not public API.

No ADR is needed.

## System context / boundaries

- **Execution path**: neither inference nor training. This is the periodic (default five-minute cadence)
  anomaly-detection cycle: `AnomalyGuardService.RunCycleAsync` → `AnomalyGuard.RunCycle` →
  `RunCycleCore` → `RunTrend`.
- **Assembly**: `Sources/Anomalies`, unchanged. Not reachable from `Tests/AotSmokeTest` (grepped, zero
  matches) — no AOT-reachability change.
- **Allocation policy**: neither hot-path nor load-path discipline applies. `RunTrend` already allocates
  per cycle per metric (`new double[window.Length]`, `new List<PeerSeries>(podCount)`,
  `new double[podCount]`) — a periodic control-plane batch job, not a per-call path. The fix adds no new
  allocation: `ReadOnlySpan<double> expectation = common is not null ? common : seasonal;` is a local
  reassignment.
- **Public surface**: no change. `RunTrend` and `Seasonal` are `private`; nothing here is public API.
- **Ownership/disposal**: not applicable — no `AutogradNode`, no pooled buffer introduced or removed.

## Verification the developer needs (protocol steps 2, 7, 8)

The existing `SeasonalExpectationTests` (`Tests\Anomalies\SeasonalExpectationTests.cs`) all use a 4-pod fleet
where every pod moves **identically** (same formula, independent noise only) — i.e. 100% common-mode
movement in every scenario. Traced through by hand: all four tests pass unchanged under this fix, because a
fleet moving in lockstep produces a near-zero residual against `common` exactly as it does against `seasonal`,
and the findings those tests assert on are the **workload**-level ones (`common` vs `seasonal`), which this
fix does not touch. **This means the existing suite would not have caught the AN-F1 regression, and will not
catch a re-regression either** — none of it exercises one pod diverging from its peers while history matches.
The developer needs a new fixture where the fleet's shared movement matches history (so the workload arm is
silent) **and** one pod diverges from the other three (so only the per-pod arm should fire) — that is the
scenario the current substitution swallows into "everyone gets a finding" or, worse, could still swallow if
the fix is implemented wrong. Promoting `SeasonalReferenceSubstitutionDiagnostics` from a `[LongFact]` report
into an asserting regression test (or adding an equivalent) is the mutation-tested artefact this task needs;
the diagnostic's own refuting arm (flat fleet, both references must agree) should be carried over.

## Open questions

**For the analyst:** none — this is a mechanical fix to a diagnosed regression, no business-rule input needed.

**For the client:** none.

**For the team lead / next round:**
1. Should the `RunCustomTrend` asymmetry (no seasonal fallback at all below 3 peers) be opened as its own
   follow-up task, or left as-is? I have not scoped it — it is a real gap but not part of AN-F1's measured
   regression, and custom bindings may not have a `MetricHistory` path to fall back to at all without further
   design. **Assumption if unanswered: leave it out of this task, flag as a candidate backlog row.**
2. Is the additive `overfit_guard_trend_seasonal_only_total` counter in scope for this task, or should
   visibility be satisfied more cheaply (e.g. a one-line comment plus reliance on `overfit_guard_pods`
   already being visible)? **Assumption if unanswered: implement the counter — it is a ~15-line, low-risk
   addition following an established pattern, and the team lead's ask #4 was explicit that the switch "must
   stop being silent."**

## Architecture review: this change carries no requirements beyond the general project rules in `CLAUDE.md`
beyond what is stated above. Reviewed on 2026-08-10.

## SUGGESTED IMPROVEMENTS TO MY ROLE

None this run. The navigator tools (`find_references`, `find_callers`) resolved the exact question that
mattered — whether `TrendDetector.Detect`'s signature or `RunTrend`'s caller graph made this harder to change
than it looks — cleanly and quickly, and reading `RunCustomTrend` alongside `RunTrend` (not named anywhere in
the team lead's message) was what turned "compose two references" into "restore the existing correct
pattern"; that came from following my own checklist item 2 ("is there a simpler way — is there already a type
that does most of this"), which is exactly what it is for.
