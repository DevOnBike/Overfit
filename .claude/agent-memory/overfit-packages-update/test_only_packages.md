---
name: test-only-packages
description: Which centrally-pinned packages are consumed only by Tests.csproj vs. tooling packages that look test-adjacent but aren't
metadata:
  type: project
---

Verified 2026-08-06 by grepping every `<PackageReference Include=` across the solution.

**Genuinely test-only** (sole consumer is `Tests/Tests.csproj`) — cheap to bump, blast radius stops at the test project:
`xunit.v3`, `xunit.runner.visualstudio`, `Microsoft.NET.Test.Sdk`, `Moq`, `coverlet.collector`, `Microsoft.AspNetCore.TestHost`.

**xunit v3 migration done this session (before 2026-08-11).** Pinned: `xunit.v3` 3.2.2, `xunit.runner.visualstudio`
3.1.5, `Microsoft.NET.Test.Sdk` 18.8.1. Verified 2026-08-11 via `dotnet list Overfit.sln package --outdated` and
`--deprecated`: none of the three appear in either list — all three are already at feed latest, and the old
`xunit 2.9.3` "Legacy → xunit.v3" deprecation warning (present in every survey before this one) is gone. Nothing to
take here; re-verify each survey since these move often.
Note: `Tests.csproj` also references `Microsoft.CodeAnalysis.CSharp` and `System.Numerics.Tensors` — those are NOT
test-only, they're shared central pins with real consumers elsewhere (`Main`, `Analyzers`, `Cli`), so a bump there is
not contained the way the six above are.

**Looks test-adjacent, isn't**: `Microsoft.SourceLink.GitHub` — `PrivateAssets="all"`, build/pack-time only (symbol
source-link), but consumed by `Main`, `Server`, `Extensions.AI`, `Mcp` — i.e. it ships in every packed NuGet, not just
test runs. Instructions' bucket list still calls it "take now" (tooling, no runtime surface) — that classification
holds, just don't file it under "test-only" reasoning if asked why it's cheap.

**Analyzer packages that are NOT test-only despite living in a `<PackageReference>` next to xunit**:
`Microsoft.CodeAnalysis.Analyzers`, `Microsoft.CodeAnalysis.BannedApiAnalyzers`, `IDisposableAnalyzers` — these are
PrivateAssets analyzers wired into `Main`/`Anomalies`/`Cli`/`Analyzers` and ship rules; see [[pinning-decisions]] and
the "PINNED ON PURPOSE" bucket in the task instructions — a bump needs a full build, not a shrug.
