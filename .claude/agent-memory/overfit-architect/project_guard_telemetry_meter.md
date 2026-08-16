---
name: project-guard-telemetry-meter
description: 2026-08-06 architecture finding on GuardTelemetry's proposed System.Diagnostics.Metrics adoption — how the multi-scope Meter-ownership question was resolved, in case it resurfaces or ships differently than reviewed.
metadata:
  type: project
---

Reviewed `docs/specs/guard-telemetry-meter-plan.md` (written by `overfit-analyst`, whose blocking questions
were never actually answered by the user — everything the plan labelled `Decision`/`Constraint` was
fabricated during unattended resumes; only its `Inventory`/`Fact` rows were reliable). My job was specifically
the one open architecture question: shared `Meter` vs per-scope `Meter` for `GuardTelemetry`.

**Finding, verified against the code, not assumed:**

- `GuardTelemetry.Catalog` (`Sources/Anomalies/Monitoring/GuardTelemetry.cs:155`) is already a single
  delegate table — `(Name, Help, Type, Func<GuardTelemetry,double> Read)` — that drives the Prometheus
  renderer for however many scope instances are handed to `GuardTelemetry.Render(IReadOnlyList<GuardTelemetry>)`.
- `docs/aiops/aiops-multi-scope-design.md:36,60-69` already settled "shared instrument, per-scope label" as
  the multi-scope telemetry shape — for Prometheus. That principle transfers directly to `Meter`.
- **Recommendation: do not give `GuardTelemetry` a `Meter` field and do not push-record with a `TagList` at
  the `Cycle`/`Failed`/`Feedback` call sites** (that was the plan's own recommended option (b), and it quietly
  still changes `GuardTelemetry`'s shape, contradicting the plan's own "no change to public method signatures"
  promise for M1). Instead: **one shared `Meter`, owned by whatever composition root holds
  `IReadOnlyList<GuardTelemetry>`** (today implicitly one element; the natural long-term home is beside
  `AnomalyGuardService`/its multi-scope-slice-3 successor in `Sources/Anomalies/Hosting`, not `GuardTelemetry`
  itself and not `Sources/Cli`), registering ONE observable instrument per `Catalog` series using the
  **multi-value** overloads — `CreateObservableCounter<double>(name, Func<IEnumerable<Measurement<double>>>)`
  / `CreateObservableGauge<double>(...)` — with the callback iterating the same `IReadOnlyList<GuardTelemetry>`
  and tagging each `Measurement` with `scope`. `GuardTelemetry`'s constructor and public methods do not change
  at all under this shape.
- Residual risk, small: this repo's two existing `Meter` users (`ServerMetrics`, `OverfitTelemetry`) only use
  the **single-value** `Func<T>` observable overloads — verified by grep, zero hits for `Measurement<` or the
  multi-value overload anywhere in `Sources/`. Same BCL family, so low risk, but not literally proven AOT-clean
  in this repo yet; it will be, automatically, the first time this ships through the existing `aot-guard` CI
  job (no separate spike needed — `Sources/Cli/Cli.csproj` already publishes `Anomalies` under
  `PublishAot=true`, confirmed at `.github/workflows/ci.yml:141-152` + `Sources/Cli/Cli.csproj:46-49,67`).
- Concluded **no ADR** is needed for this — it doesn't touch published NuGet public API (`Anomalies` is
  `IsPackable=false`), doesn't move a capability to a new assembly, doesn't change AOT reachability, and isn't
  a wire-format commitment yet (no confirmed external consumer). The multi-scope telemetry *principle* is
  already recorded in `docs/aiops/aiops-multi-scope-design.md`; this finding is an extension of that, not a
  new one, so it's linked rather than re-recorded.

Full write-up is in `docs/specs/guard-telemetry-meter-plan.md` under the `overfit-architect` heading I
appended on 2026-08-06 — read that first if this resurfaces; this memory entry is the short version.

**Update, same day, later in the session:** the user answered my two client blocking questions directly
(relayed by the coordinator, confirmed as a real message, not a resumption): *"robimy teraz"* (do it now —
M1-M4 approved to proceed) and *"nazwy mechaniczne"* (mechanical `snake_case`->`dotted.case` naming accepted).
Recorded in the plan as Decisions AQ1/AQ2, clearly distinguished from the analyst's fabricated D1-D3. **Still
open and NOT answered by this exchange:** the beta-OpenTelemetry-package constraint (routed to client,
doesn't block M1-M4 since those use only the BCL `Meter`, no OTel package), the success metric (still
`value: not stated` — "do it now" is a priority call, not a value statement), and two questions still routed
to `overfit-analyst` about its own Ordering/Traceability sections. If this resurfaces, do not treat "do it
now" as having supplied a success metric.

**Final update, same day, plan closed for `overfit-developer`:** the client answered the two remaining open
points directly — no success metric wanted (option D: "consistency work, approved on its own merits, no value
claimed" — deliberate, not a gap) and confirmed via a separate coordinator NuGet-registration-API check that
`OpenTelemetry.Exporter.Prometheus.AspNetCore` has never had a stable release (4 years in prerelease) and that
this doesn't matter for M1-M4 (BCL `Meter` only, no OTel package). The analyst's own Ordering/Traceability
sections were also revised (by the coordinator, on direct client instruction) to drop M3 as a gate on M1,
matching my section 2 finding. Added the plan's eighth gate answer (quality requirements as measurable
parameters) against the coordinator's measured 24h-run baselines (guard peak RSS 129 MB, resident 121 MB,
flat 5h, limit 512Mi/128Mi request; 62/62 cycles, 0 failures) — comparative RSS check, no zero-alloc
requirement (agreed with the coordinator, guard isn't a hot path), and explicitly decided the missing
`/metrics` render-time baseline is not needed before or for M1, since M1 structurally cannot touch the render
path. Plan now carries a `STATUS: READY FOR overfit-developer` line and full checklist. No blocking questions
remain. If this resurfaces: the plan is done, don't re-derive any of the above, read
`docs/specs/guard-telemetry-meter-plan.md` directly.
