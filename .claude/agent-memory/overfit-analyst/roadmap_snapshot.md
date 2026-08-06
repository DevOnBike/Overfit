---
name: roadmap-snapshot
description: What ROADMAP-COMPLETED.md records as done, and what ROADMAP.md records as deliberately deferred with its reason — condensed, so round one starts from this instead of re-reading both files whole
metadata:
  type: project
---

Seeded 2026-08-06. Both files are large and move fast — treat this as a pointer to re-check, not a citation.

## ROADMAP-COMPLETED.md — done, relevant to telemetry/observability
- Guard self-monitoring shipped: `GuardTelemetry` (11 series originally, now 15 with scope+feedback),
  `--metrics-port`, k8s Service + ServiceMonitor. Multi-scope slice 2 (`scope` label, no-duplicate-HELP
  rendering) also shipped and tested.
- `PrometheusMetricSource`/`PrometheusHistoricalSource` parsers fixed (were silently returning empty lists —
  LINQ-banned code path had been commented out).
- Eleven dead telemetry instruments in `OverfitTelemetry` (Main) triaged: kernel/module timing left
  deliberately unwired (hot-path cost, needs its own benchmark justification), graph/module ones wireable,
  ratcheted by `TelemetryInstrumentWiringTests` so the dead-list can only shrink.
- `ServerMetrics` (Server.AspNet) — Meter + hand-rolled `/metrics`, explicitly NOT the OTel Prometheus
  exporter, with the AOT reasoning written into the file.

## ROADMAP.md — active / deferred, relevant here
- Active track: anomaly guard "from detects to a client can run it" — multi-scope slice 3 (one host, N
  scopes) is next; slices 4-5 (shared durable state, per-scope `--real` labels) after.
- Deferred: becoming a Kubernetes operator (buys declarative config already had via ConfigMap, costs RBAC/CRD/
  security review) — explicitly rejected, not merely postponed.
- `System.Diagnostics.Metrics + Activity telemetry (Telemetry/)` listed (line ~1004) as an idea feeding item
  #4 (production LocalAgent template's `/metrics`) — i.e. Meter-based telemetry for the *demo/LocalAgent* path
  was already on the roadmap as a "nice to have", separate from the guard.
- No line in either roadmap file proposes an OTLP exporter, a `dotnet-counters` story, or a `Meter` on the
  guard specifically — this request (guard Meter/OTel) is new ground, not a revisit of a scoped item.
