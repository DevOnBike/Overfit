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
- **`Microsoft.Build.Framework`** — bumped from 18.0.2 to **18.8.2 on 2026-08-11**, deliberately ABOVE the SDK's own
  MSBuild (still 18.0.11 as of that date — confirmed via `dotnet msbuild -version`), which looks like it violates the
  old "must be `<=` the SDK's MSBuild" rule. It was tested rather than reasoned about: the navigator calls no
  `Microsoft.Build.Framework` API directly (only `MSBuildLocator` and Roslyn's `MSBuildWorkspace`, pinned separately),
  so there is no call site a newer reference could resolve to missing API — `measure` (all 26 projects, 1667 docs) and
  `refs`/`impls` verified correct on 18.8.2. **Not exercised**: `callers`, `unused`, `serve` verbs. `Microsoft.NET.StringTools`
  must move in lockstep (same `ExcludeAssets="runtime" PrivateAssets="all"` shape) or MSBuildLocator's own MSBL001 gate
  fails the build — this is why the two pins are adjacent in `Directory.Packages.props`.
  **2026-08-11 re-check**: newest is now **18.9.6** (was 18.3.3 on 2026-08-06, grew fast). Still `<=`-vacuous by the
  same reasoning, but the specific verb battery above (and MSBL001) needs re-running before taking it — this is a
  "take with a build check", not a rubber stamp, precisely because the old easy proxy (version `<=` SDK) no longer
  applies and the real constraint (no API call resolves to something missing) has to be checked by hand each time.
- **`OpenTelemetry.Exporter.Prometheus.AspNetCore` — REMOVED, 2026-08-10 (task `XC-6`).** No longer in
  `Directory.Packages.props`; zero `OpenTelemetry.*` `PackageReference`/`PackageVersion` anywhere in the tree
  (verified 2026-08-11: grepped every `.csproj`, the only two hits are prose comments in `Tests.csproj` and
  `LocalAgent.AspNet.csproj` explaining what the demo does instead — no actual reference). This closes the
  1449-day-beta investigation below; nothing left to decide.

**Superseded by the removal above, kept for history.** `docs/specs/guard-telemetry-meter-plan.md`, section "C0
resolved — no new library is needed" (2026-08-06) found 33 published versions, 0 stable ever, prerelease since
2022-08-18, while the rest of the OTel suite shipped stable 1.17.0 the same day; sole consumer was
`Demo/LocalAgentAspNetDemo`; the product's own hand-rolled Prometheus renderer already covers `/metrics`. That
finding is presumably why `XC-6` removed the pin rather than accepting or migrating it.

**`OVERFITPRERELEASE` guard now fires zero times** — verified 2026-08-11 by building `Sources/Main/Main.csproj -c
Release` directly (not just grepping the accept-list): build succeeded, only pre-existing OVERFIT006/OVERFIT041
warnings, no `OVERFITPRERELEASE` line. **Zero prerelease pins remain in `Directory.Packages.props`** as of this
date — re-check this fresh each survey rather than assuming it stays true, since a prerelease pin is exactly the
kind of thing that creeps back in one dependency at a time.

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
