---
name: baselines
description: Measured baselines for the release gate — suite counts, packable set, how to run the AOT guard on this box, LongFact gate cost, known intermittent tests
metadata:
  type: reference
---

Verified **2026-08-15** on branch `gimli` (supersedes the 2026-08-13 pass; every corrected line is marked).

- **Fast suite** `dotnet test ./Tests/Tests.csproj -c Release`: **2643 passed / 0 failed / 276 skipped /
  2919 total**, 37 s wall (26 s in-test). Was 2618/0/274 on 2026-08-13.
- **`dotnet build -c Release Overfit.sln` exits 1 and it is NOT a code defect** — only `MSB3021/3026/3027`,
  the live navigator MCP server holding `Tools/SemanticNavigator/bin/.../overfit-navigator.dll`. Judge by
  diagnostics, not exit code. Each shipping project built alone exits **0**. Only compiler warnings in
  shipped code: 6× `OVERFIT006` in `Sources/Main/Kernels/Conv2DGemmKernels.cs:75-76` (pre-existing). Zero
  `CS1573/1574/0419/1734`.
- **THE AOT GUARD CAN BE RUN ON THIS BOX — do not report it as CANNOT TELL again.** No C++ toolchain on
  PATH (no `cl`/`clang`/`gcc`), but **Docker Desktop's Linux engine runs** (`docker info` exit 0, OSType
  linux; it was down on 2026-08-13). Recipe that worked, ~25 s after the context transfer:
  a throwaway Dockerfile `FROM mcr.microsoft.com/dotnet/sdk:10.0` + `apt-get install clang zlib1g-dev` +
  `dotnet publish ./Tests/AotSmokeTest/AotSmokeTest.csproj -c Release -r linux-x64 -p:PublishAot=true
  -p:TreatWarningsAsErrors=true`, built with context = `D:\Overfit`. Both it and the full CLI
  (`docker build -f Sources/Cli/Dockerfile .`) passed on 2026-08-15. `gh` is still unauthenticated, so CI
  job status remains unreachable — but the local container makes that irrelevant for AOT.
- **Docker build context is 1.17 GB** and takes ~74 s to transfer, because `.dockerignore`'s `TestResults/`
  is root-anchored and does not match `Tests/TestResults` (994 MB). `artifacts/` and
  `BenchmarkDotNet.Artifacts/` are not listed at all. All four are gitignored, so the tree is still clean.
- **Packable set — CORRECTED 2026-08-15.** `dotnet pack -c Release Overfit.sln` produces **5**:
  `DevOnBike.Overfit`, `.Cli`, `.Extensions.AI`, `.Mcp`, `.Server`. `Demo/LabLoadDriver` and
  `Sources/Server.AspNet` have since been set `IsPackable=false` (the 2026-08-13 note that they leak is
  fixed). `Templates/DevOnBike.Overfit.Templates.csproj` is **outside `Overfit.sln`** and is packed
  separately by `.github/workflows/publish-nuget.yml:72` — NOT a missed package, do not re-raise it.
- **Local pack ≠ what ships.** `Sources/Main/Main.csproj:12` hardcodes `<SourceRevisionId>N/A</SourceRevisionId>`,
  so a local pack yields nuspec `commit="N/A"` and assembly `10.1.0+N/A.N/A`. `publish-nuget.yml` passes
  `-p:SourceRevisionId=<sha>` and `-p:RepositoryCommit=<sha>` and `-p:ContinuousIntegrationBuild=true`, so
  the published package is correct. Check the workflow before calling this a defect.
- **`publish-nuget.yml` has no `needs:`** — a `workflow_dispatch` publish does not depend on CI, the
  `aot-guard`, the `analyzer-guard` or `Scripts/api_compat_check.py` being green on that commit.
- **Analyzer contract**: 47 rules; code and `AnalyzerReleases.Unshipped.md` agree exactly, both directions.
  `Shipped.md` still empty (2 lines) — long-standing convention, never once used, not a branch finding.
  `OVERFIT035/036/037/039` have no `.editorconfig` entry (035/036 default Error, 037/039 default Warning,
  neither fires anywhere today).
- **`[LongFact]` gate cost IS measured** — `docs/measured-baselines.md:255+`, 2026-08-13: **165.3 min for
  27 of 49 areas**, four areas carry 152 of it (`Anomalies` 63.3, `LanguageModels.Loading` 30.0, `.LoRA`
  27.8, `.Demo` 21.1), 22 areas still unmeasured. This role's definition still lists "nobody has timed it"
  as an unmet precondition — it is STALE.
- **Known intermittent** (`docs/TASKS.md`): `XC-10` `PromptCacheReuseTests`, `XC-13`/`XC-38`
  `XgboostParityTests.Prediction_IsZeroAllocation_AfterWarmup`, `XC-19` `Gpt1QLoRATests`, `TG-T12`
  `CircuitBreakerTests.Timeout_FiresWhenWallTimeExceeded` (load-dependent, NOT on the flaky list),
  `TG-T13` `DecodePoolIdleBurnTests.Pool_Parks_WhenIdle` (measures the whole process). None appeared in the
  2026-08-15 fast run.
- **Raised and consciously accepted, do not re-raise as defects**: the 58 API breaks vs 10.0.31 (`XC-34`/
  `XC-40`, itemised in the CHANGELOG, split intended); `MINOR` not `MAJOR` (repo policy: MAJOR tracks the
  .NET target, CHANGELOG:9); `Shipped.md` empty; `XC-27` (8 projects outside the solution).
