---
name: capability-map
description: Subsystem -> capability -> file path index, so round one starts from a map instead of nothing
metadata:
  type: project
---

Seeded 2026-08-06 from CLAUDE.md's own architecture section (checked into repo, so treated as current) plus
direct reads this session. Re-verify a path before citing it in a plan — this drifts.

## Inference vs training (the split that matters most)
- Inference: `InferenceEngine` (caller-owned buffers, 0 B/call) -> `IInferenceBackend` ->
  `SequentialInferenceBackend` / `OnnxGraphInferenceBackend`. No `AutogradNode`.
- Training: `ComputationGraph` tape of `AutogradNode`, `graph.Backward(loss)`, `graph.Reset()` reclaims by
  `AutogradNodeOwnership` (`GraphTemporary`/`GraphAuxiliary`/`Parameter`/`ExternalBorrowed`/`View`).

## LM runtime (KV-cache, zero-alloc decode)
`Sources/Main/LanguageModels/Runtime/`: `CachedSlmInferenceEngine` -> `CachedSlmSession` (KV buffers + position)
-> `StackWeights` -> `BlockWeights` -> `SingleHeadWeights` (ReadOnlySpan into TensorStorage). Loaders: GGUF,
ONNX (`OnnxImporter` linear / `OnnxGraphImporter` DAG), safetensors, `.bin`. Architectures: GPT-2, Llama
(+RoPE llama3 scaling), Qwen2.5/3, Phi-3.5/4, Gemma-2, Mixtral, Qwen-MoE.

## Telemetry / metrics (three independent Meter instances, verified this session)
- `Sources/Main/Diagnostics/OverfitTelemetry.cs` — general `Meter`+`ActivitySource` for Main; 11 of its
  instruments are declared but never recorded (kernel/module timing = deliberately not wired, hot-path cost);
  ratcheted by `Tests/Diagnostics/TelemetryInstrumentWiringTests` (list can only shrink).
- `Sources/Server.AspNet/Services/ServerMetrics.cs` — `Meter` (AOT-safe, feeds dotnet-counters/OTel) DUAL-WRITES
  into interlocked counters that `Sources/Server.AspNet/Endpoints/MetricsEndpoints.cs` renders by hand as
  Prometheus text. Comment in the file explains why: OTel's Prometheus *exporter* package is reflection-heavy
  and breaks the AOT guard; the Meter API itself is fine.
- `Sources/Anomalies/Monitoring/GuardTelemetry.cs` — hand-rolled Prometheus renderer ONLY, no `Meter`. Uses a
  `Series[] Catalog` of delegates (not reflection, AOT-safe) so a metric's HELP/TYPE/read-fn live in one place.
  Multi-scope `Render(IReadOnlyList<GuardTelemetry>)` groups by series-then-instrument specifically to avoid
  duplicate HELP/TYPE when several scopes render into one document — tested
  (`Tests/Anomalies/Monitoring/GuardTelemetryScopeTests.SeveralScopesShareOneHeaderPerSeries`,
  `Tests/Anomalies/GuardTelemetryTests.EverySeriesCarriesHelpAndType`). Served by
  `Sources/Cli/GuardMetricsEndpoint.cs`, a bare `HttpListener` (not ASP.NET) on `--metrics-port`.
- `Demo/LocalAgentAspNetDemo` (non-packable demo, NOT AOT-published) is the only place the real
  `OpenTelemetry.Exporter.Prometheus.AspNetCore` package is used — pinned `1.15.3-beta.1` in
  `Directory.Packages.props`, never had a stable release, never AOT-published in this repo.

## AOT verification reality (verified 2026-08-06, matters for any AOT claim)
- `Tests/AotSmokeTest` references ONLY `Sources/Main` — does not touch Anomalies, Server.AspNet or Cli at all.
- `.github/workflows/ci.yml` `aot-guard` job ALSO publishes `Sources/Cli/Cli.csproj` under
  `PublishAot=true -r linux-x64` (Cli references Anomalies AND Server.AspNet), then smoke-runs `overfit list`.
  So the guard's `GuardTelemetry`/`GuardMetricsEndpoint` and the server's `ServerMetrics`/`MetricsEndpoints`
  ARE reached by a real Native-AOT publish in CI today — just not via AotSmokeTest specifically. A `Meter` in
  the guard, following the `ServerMetrics` pattern, is therefore already proven safe by an existing passing CI
  job, not an unverified novelty. What's genuinely unverified is only the OTel Prometheus **exporter** package
  (client's 1.17.0-beta.1 claim) — that would need its own real AOT publish, never run in this repo.

## Anomaly guard (see also project_aiops_detection.md)
`Sources/Anomalies/`: `Statistics/` (ranged) + `Anomalies/Rules` (absolute floors) + `Anomalies/Incidents`
(grouping) + `Anomalies/{Gpt,Baseline,Neuro}` (learned). `AnomalyGuard` (Incidents/) runs cycles;
`AnomalyGuardService` hosts it; `Sources/Cli/AnomalyGuardCommand.cs` wires `GuardMetricsEndpoint.TryStart`.
Multi-scope work in progress (slice 2 of 5 done — scope label on telemetry; slice 3, one host running N
scopes, is next).
