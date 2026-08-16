---
name: project-assembly-graph
description: Verified project-reference graph across Sources/, Tests, and Demo/, so dependency-direction checks don't need re-deriving from scratch every run.
metadata:
  type: project
---

Verified 2026-08-06 by grepping every `.csproj`'s `<ProjectReference Include="...">` directly (not inferred
from `CLAUDE.md`, which can go stale). Re-verify before relying on this if it's been more than a few weeks —
this is exactly the kind of thing that drifts silently.

**Sources/ (shipped or internal-tooling assemblies):**

- `Main` — no project references. The library everything else builds on.
- `Analyzers` — no project references (by design; must depend on nothing in the tree).
- `AndroidBench`, `Anomalies`, `Extensions.AI`, `Mcp`, `Server` — reference `Main` only.
- `Server.AspNet` — references `Server`, `Main`, `Anomalies`.
- `Cli` — references `Main`, `Mcp`, `Server`, `Server.AspNet`, `Anomalies`. This is the widest-reaching
  shipped assembly and the one the `aot-guard` CI job publishes under `PublishAot=true` (see
  `project_guard_telemetry_meter.md` for why that matters).
- `Benchmark` — references `Main`, `Anomalies`.

**Tests:**

- `Tests/AotSmokeTest/AotSmokeTest.csproj` — references `Main` **only**. It does NOT reach `Anomalies` or
  `Server.AspNet` — confirmed by reading the csproj directly, not assumed. This is why the `aot-guard` CI
  job's second publish step (`Sources/Cli/Cli.csproj`) is the real AOT proof for anything in `Anomalies` or
  `Server.AspNet`; the smoketest alone would not catch a regression there.
- `Tests/Tests.csproj` — references `Main`, `Extensions.AI`, `Mcp`, `Server`, `Server.AspNet`, `Anomalies`,
  `Analyzers`.

**Demo/** — every demo references `Main` plus at most one other Sources project matching its purpose
(`AnomalyConsoleDemo` -> `Anomalies`, `AgentFrameworkDemo`/`EvaluationDemo` -> `Extensions.AI`,
`LocalAgentAspNetDemo` -> `Server`, `AndroidBenchApp` -> `AndroidBench`). `LabLoadDriver`/`LabWorkload` have
no project references at all (self-contained lab tooling).

Dependency direction holds everywhere checked: nothing shipped references `Tools/`, `Analyzers` references
nothing in-tree, and `Anomalies`/`Cli`/`Server`/`Server.AspNet` all point at `Main`, never the reverse.
