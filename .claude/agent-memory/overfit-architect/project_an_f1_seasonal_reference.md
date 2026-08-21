---
name: project-an-f1-seasonal-reference
description: AN-F1 fix design — RunCustomTrend already had the correct per-pod reference priority; RunTrend just needed to match it.
metadata:
  type: project
---

2026-08-10, plan at `docs/specs/an-f1-seasonal-reference-plan.md` (no prior analyst doc existed for AN-F1;
file is architecture-only).

**The bug and the fix.** `AnomalyGuard.RunTrend` (`Sources\Anomalies\Incidents\AnomalyGuard.cs:1268-1368`)
picked the per-pod trend reference as `seasonal` when history existed, falling back to `common`
(cross-peer) only when it did not — backwards. `AnomalyGuard.RunCustomTrend` (same file, lines 926-1002),
the structurally parallel method for custom-metric channels, already gets this right: `expectation = common`
unconditionally when decomposition succeeds, empty otherwise, no seasonal substitution at all. The fix is
priority reversal to match: per-pod expectation = `common` when `podCount >= CrossPeerBaseline.MinimumPeers
(3) && DecomposeCommonMode`, else `seasonal`, else empty. Workload-level arm
(`_trend.Detect(common, ..., seasonal)`) is unchanged — that one was already correct (tests the fleet's own
trajectory against its usual hour-of-day pattern).

**Why "compose, don't substitute" (as first framed) was the wrong shape.** Both `common`
(`CrossPeerBaseline.TryBuild`) and `seasonal` (`MetricHistory.TryExpectation`) are point estimates of the
metric's own level/units, not residuals — `TrendDetector.Detect` takes exactly one `expectation` span and
tests `values − expectation`. Adding the two would double the signal's magnitude. The real fix is selecting
the better single reference, and `common` is a strict superset of what `seasonal` can remove whenever ≥3
peers exist (it captures whatever peers are doing *right now*, daily curve included, plus anything the curve
doesn't explain — rollout, load test, real spike).

**Measured evidence.** `Tests\Anomalies\Diagnostics\SeasonalReferenceSubstitutionDiagnostics.cs`: 12-pod
fleet moving together, `common` reports 0 pods at every movement (flat/+5/+15/+30%), `seasonal` agrees at
flat but reports 12/12 at +15% and +30%. The oft-cited 2551→376/day seasonal win
(`SeasonalBaselineTests.cs`) is a detector-only ablation with **no cross-peer component in the loop** —
evidence seasonal beats nothing, not evidence it beats `common`. Don't conflate the two when someone cites
that number for a per-pod-with-peers scenario.

**Existing tests don't pin this.** `Tests\Anomalies\SeasonalExpectationTests.cs` uses a 4-pod fleet where
every pod moves identically in all 4 scenarios (pure common-mode) — traced by hand, all 4 pass unchanged
under either reference because the findings asserted on are the workload-level ones. None of it exercises one
pod diverging from peers while history matches, so it would not have caught the original regression and
won't catch a re-regression. Developer needs a new fixture for that.

**No ADR needed.** `find_references` on `TrendDetector.Detect` (60 hits) / `find_callers` on `RunTrend` (81
call sites) confirm no signature or public-surface change. `Sources/Anomalies` is not
AOT-smoketest-reachable (grepped `Tests/AotSmokeTest/Program.cs`, zero hits). Proposed additive
`GuardTelemetry` counter (`overfit_guard_trend_seasonal_only_total`) follows the existing
`Interlocked`+`Catalog` pattern (`NoveltyHeld` is the template) — pure exposition addition, not a wire-format
restructuring.
