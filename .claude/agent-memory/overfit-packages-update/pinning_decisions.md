---
name: pinning-decisions
description: The four deliberate below-latest pins in Directory.Packages.props, why, and how each was verified
metadata:
  type: project
---

All four verified directly against `Directory.Packages.props` comments + live checks, 2026-08-06 (SDK 10.0.110).

- **`Microsoft.CodeAnalysis.CSharp` @ 5.0.0** — must stay `<=` the SDK's own Roslyn, or the compiler emits CS9057
  and every OVERFIT analyzer rule silently stops firing while the build still "succeeds". Verified 2026-08-06:
  `csc.dll` shipped in SDK 10.0.110 reports ProductVersion `5.0.0-...` — the pin is exact, not just close. Repo's
  own comment says this package will show "outdated" forever; that is intentional. Re-check only after an SDK bump.
- **`Microsoft.CodeAnalysis.Workspaces.MSBuild` + `Microsoft.CodeAnalysis.CSharp.Workspaces` @ 5.0.0** (`Tools/SemanticNavigator`
  only, added 2026-08-06) — held at the *same* 5.0.0 as above, but for a different reason: MSBuildWorkspace loads the
  OVERFIT analyzers out of each project, so the navigator is an analyzer **host**; a host newer than the SDK's Roslyn
  diverges from what `dotnet build` actually ran. Not a compiler-load-failure risk like the CSharp pin, a host/target-drift risk.
- **`Microsoft.Build.Framework` @ 18.0.2** (`Tools/SemanticNavigator`, compile-time only via `ExcludeAssets="runtime"`) —
  must be `<=` the SDK's own MSBuild (else `MissingMethodException` at runtime), and must NOT be the 17.11.31 that
  `Workspaces.MSBuild` 5.0.0 resolves transitively (carries GHSA-w3q9-fxm7-j8fq, a Linux temp-dir DoS affecting the
  Microsoft.Build family across several 17.x preview ranges — confirmed via OSV 2026-08-06, 18.0.2 is outside every
  affected range). Verified 2026-08-06: `dotnet msbuild -version` on SDK 10.0.110 → **18.0.11**. NuGet has no published
  version between 18.0.2 and 18.0.11 (next after 18.0.2 is 18.3.3, already too high) — so 18.0.2 remains the *highest*
  version satisfying both constraints simultaneously. Re-check after every SDK bump; a newer SDK may open a window
  nothing currently fills.
- **`OpenTelemetry.Exporter.Prometheus.AspNetCore` @ 1.15.3-beta.1** — no stable release has ever been published for
  this package (confirmed via its own CHANGELOG.md, 2026-08-06); every version on NuGet, including the current latest
  (1.17.0-beta.1 as of 2026-08-06), is alpha/beta. This is [[opentelemetry-prometheus-beta-status]] — treat the version
  number as a hint to re-check, not the answer.

See also [[test-only-packages]] for the cheap-bump bucket.
