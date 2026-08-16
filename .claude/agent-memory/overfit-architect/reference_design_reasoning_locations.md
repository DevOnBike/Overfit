---
name: reference-design-reasoning-locations
description: Where recurring architectural reasoning already lives in this repo, so a review links instead of restating it and doesn't re-litigate a settled question as a fresh finding.
metadata:
  type: reference
---

- **Guard multi-scope design** (per-scope state ownership, what stays shared vs per-scope, telemetry
  labelling) — `docs/aiops/aiops-multi-scope-design.md`. Explicitly settles "shared instrument, per-scope
  label" for `GuardTelemetry` as of slice 2 (shipped). See [[project-guard-telemetry-meter]] for how this
  extends to the `System.Diagnostics.Metrics` side.
- **Guard rollout/backlog status and what's shipped vs pending** — `ROADMAP.md` (search "anomaly guard" /
  "multi-scope") and `docs/aiops/aiops-backlog.md`. Slice table for multi-scope work is at `ROADMAP.md:59-65`.
- **AOT/trim discipline and why the two-layer guard (BannedSymbols + AotSmokeTest) exists** — `CLAUDE.md`
  "Native-AOT discipline" section, and the `aot-guard` job comments in `.github/workflows/ci.yml:96-152`
  (the comments there are current and worth reading directly, not just the job name).
- **`Meter` vs Prometheus-exporter split (why the BCL Meter API is adopted but the OTel Prometheus exporter
  package is not)** — worked out once already in `Sources/Server.AspNet/Services/ServerMetrics.cs`'s own XML
  doc, and now a second time in `Sources/Main/Diagnostics/OverfitTelemetry.cs`. Any future "should X get a
  Meter" question should read these two files first — they are the precedent, not a blank design.
- **Instrument-declared-but-never-recorded failure mode** — `Tests/Diagnostics/TelemetryInstrumentWiringTests.cs`,
  a ratchet test (list can only shrink) written after finding 11/42 dead instruments in `OverfitTelemetry` and
  one in the guard on the same day (2026-08-02). Any new `Meter` gets one of these from day one, not
  retrofitted.
