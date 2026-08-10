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
  (1.17.0-beta.1 as of 2026-08-06/07). This is [[opentelemetry-prometheus-beta-status]] — treat the version
  number as a hint to re-check, not the answer.

**This one is no longer a "decide deliberately, from scratch" item — it already has a deep, dated investigation.**
`docs/specs/guard-telemetry-meter-plan.md`, section "C0 resolved — no new library is needed" (coordinator,
2026-08-06): 33 published versions, 0 stable, prerelease since 2022-08-18 (1449 days at time of writing),
while the rest of the OTel suite shipped stable 1.17.0 the same day (verified independently 2026-08-07: `OpenTelemetry`,
`OpenTelemetry.Api`, `.Exporter.OpenTelemetryProtocol`, `.Exporter.Console/.InMemory/.Zipkin`, `.Extensions.Hosting/.Propagators`
all hit stable 1.17.0 on 2026-07-16, same day as Prometheus's 1.17.0-beta.1). The doc's conclusion: **the product does
not need this package at all** — `/metrics` scrape is already served by this repo's own hand-rolled, tested,
dependency-free Prometheus renderer (`Sources/Anomalies/Monitoring/GuardTelemetry.cs`), and if a customer ever wants
to push to their own OTel collector, the stable `OpenTelemetry.Exporter.OpenTelemetryProtocol` (stable since
2021-02-10) reads off the same `Meter` with no Prometheus-specific reflection.
**Sole current consumer is `Demo/LocalAgentAspNetDemo` (non-AOT demo, not the shipped `overfit` CLI)** — confirmed
by grep 2026-08-07, only file referencing the package. `Directory.Build.targets`'s `OVERFITPRERELEASE` guard is
now live (added since the 2026-08-06 investigation) and its accept-list is deliberately empty — **the guard
currently fires a warning on every `Sources/Main` build** (verified by building Main 2026-08-07), because nobody
has yet either accepted this pin with a reason/date or migrated the demo off it. This is not a stale warning to
silence; it is the guard doing exactly what it was built for. Three live choices for the user, unchanged since
2026-08-06: accept it explicitly (add an `OverfitAcceptedPrerelease` entry — hard to justify given the C0
finding), migrate `Demo/LocalAgentAspNetDemo` onto the stable OTLP exporter, or drop Prometheus-export from the
demo entirely. Don't re-run the C0 investigation from scratch — it's already done; only check whether anything
has changed (a stable release appearing, or the demo already migrated).

Also noticed 2026-08-07: **`TorchSharp-cpu` is centrally pinned in `Directory.Packages.props` (0.106.0) but has
zero `<PackageReference>` consumers anywhere in the solution** (grepped every `.csproj`, case-insensitive — only
hits are the pin itself and doc mentions). `dotnet list package --outdated` never surfaces it for any project,
which is consistent with "nothing restores it," not with "it's already at latest." This contradicts this agent's
own role definition, which lists `TorchSharp-cpu` alongside `MathNet.Numerics`/`Accord.Neuro`/`Microsoft.ML.OnnxRuntime`
as backing cross-checks/parity tests — those three ARE referenced (`Sources/Benchmark/Benchmarks.csproj`),
`TorchSharp-cpu` is not. Worth a one-line flag to the user each survey until it's either wired up or removed
from the central pin file — an orphaned pin isn't a security or build risk, but it's dead weight nobody will
notice going stale.

**RESOLVED 2026-08-10: the pin was removed.** Verified once more before deleting — `git grep -i torchsharp`
returns the pin itself, this note, the role definition and two doc mentions of the *library* as something
Overfit deliberately does not depend on; zero `PackageReference`. The role definition was corrected in the
same change, because it had listed `TorchSharp-cpu` among the parity-test backers and that was the reason
this looked like an oversight rather than dead weight. Nothing to flag on future surveys.

See also [[test-only-packages]] for the cheap-bump bucket.
