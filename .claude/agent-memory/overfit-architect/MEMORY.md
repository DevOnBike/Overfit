# Memory index

- [Measured baselines — pointer](reference_measured_baselines.md) — canonical numbers now live in `docs/measured-baselines.md`; don't duplicate, cite + re-verify per row.
- [Assembly graph](project_assembly_graph.md) — verified 2026-08-06 project-reference graph for all of `Sources/`, `Tests/AotSmokeTest`, `Tests/Tests.csproj`, `Demo/*`.
- [Where design reasoning already lives](reference_design_reasoning_locations.md) — index of docs/comments to link instead of restating, for recurring boundary questions.
- [Guard telemetry Meter — architecture finding](project_guard_telemetry_meter.md) — 2026-08-06: multi-scope `Meter` ownership resolved (composition-root-owned observable instruments over `GuardTelemetry.Catalog`, not per-instance push-recording); no ADR needed.
- [Metric-window-source seam — signed APPROVED](project_metric_window_source_seam.md) — 2026-08-08: `IMetricWindowSource` extraction + `now`-parameterisation for replay. Key catch: "call twice, same instance" is NOT a determinism test when the class is stateful (`IncidentTracker` continuation flips Opened→Ongoing) — needs two cold instances, ≥2 cycles. No TimeProvider, no ADR. Perf-audit follow-up: cite whole-cycle `AnomalyGuardScaleBenchmark` (4.174ms@4pods–701ms@200pods) over component benchmarks; unqualified "N seconds" claims need a stated scale (168× spread here).
