---
name: xc22-guard-metrics-endpoint
description: 2026-08-12 XC-22 boundary ruling — GuardMetricsEndpoint Cli→Anomalies/Hosting; the "stays internal" premise was false, and Anomalies/Hosting is all-public
metadata:
  type: project
---

**XC-22 SIGNED 2026-08-12.** Plan: `docs/specs/xc-22-guard-metrics-endpoint-assembly-plan.md`. Move
`GuardMetricsEndpoint` from `Sources/Cli` to `Sources/Anomalies/Hosting/` as **`public sealed class`**.

**Why:** the task row (and the dispatching agent) both claimed the type "stays `internal` while becoming
testable". **It cannot.** `Anomalies.csproj:53-54` grants `InternalsVisibleTo` to
`DevOnBike.Overfit.Tests` and `Benchmarks` only — **not** to Cli, whose assembly name is `overfit`
(`Cli.csproj:12`). An internal type there is invisible to its one caller `AnomalyGuardCommand.cs:182` and
the solution does not compile. The cited precedent `GuardAckAuthorization` is **`public`**, and so are all
five types in `Sources/Anomalies/Hosting/`. `Anomalies` is `IsPackable=false`, so `public` there is not a
published contract. Alternative kept on record: `internal` + `<InternalsVisibleTo Include="overfit" />`.

**How to apply:** when any Anomalies type must be reachable from Cli, the question is *public in a
non-packable assembly* vs *IVT to `overfit`* — never "just make it internal". Same for Server/Mcp.

**Measured while reviewing, so do not re-derive:**

- `Tests/AotSmokeTest` references **only** `Sources/Main` (`AotSmokeTest.csproj:44`). **Neither Cli nor
  Anomalies is behind the AOT gate.** Both are in the `overfit` Native-AOT *publish* graph.
- `HttpListener` produces **zero** trim/AOT diagnostics under `IsAotCompatible`+`IsTrimmable`+
  `TreatWarningsAsErrors` — probe with a live-analyzer negative control (`IL2075`, `IL2057` fired).
- Baseline `dotnet build -c Release` 2026-08-12: `Sources/Cli` **0** warnings, `Sources/Anomalies` **3**,
  all `OVERFIT006` in `Sources/Main/Kernels/Conv2DGemmKernels.cs`.
- **Cli↔Anomalies analyzer ladder differs in exactly two rules**: `OVERFIT033`/`OVERFIT034` are `error`
  for Anomalies and unset for Cli (`.editorconfig:509-511`). Everything else is shared
  (`.editorconfig:533-541`, `572-575`).
- **Anomalies bans 19 symbols, Cli 7** — moving a file *into* Anomalies imports `System.Linq`,
  `System.Reflection`, `Activator`, `Array.Copy`, `ArrayPool.Shared` bans plus `<Using Remove="System.Linq" />`.
- **`Anomalies` sets `GenerateDocumentationFile=true` and Cli does not** — any file moved in with partial
  `<param>` coverage arrives carrying CS1573.

**Stale-prose finding (for `overfit-reviewer`, not fixed here):** `GuardMetricsEndpoint.cs:185-193` says a
concurrent `/suppressions` scrape can throw `Collection was modified`. It cannot — `RunCycleAsync`,
`Acknowledge` and `ActiveSuppressions` all take `lock (_gate)` (`AnomalyGuard.cs:462/501/544`), present since
`e9a5642` **2026-08-05**, six days before the comment (`fc52686`, 2026-08-11). The backstop is still right;
its stated justification is not, so **a test cannot reproduce that defect** — pin the property instead, via
the `IClock` that `TryStart` already takes. See [[an-d9-coverage-signal]] for the same "verify the mechanism
before designing to it" shape.
