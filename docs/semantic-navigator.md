# Semantic navigator

`Tools/SemanticNavigator` answers questions about **symbols** instead of about text. It opens `Overfit.sln`
into a Roslyn workspace and exposes four queries, both as a CLI and as an MCP server.

It is **developer tooling and never ships.** `MSBuildWorkspace` resolves MSBuild, project SDKs and every
analyzer in the tree by reflection at runtime — the exact mechanism `Sources/Main` bans and the exact reason
this cannot be Native-AOT compiled. That is why it lives under `Tools/` rather than `Sources/`, and why it is
a separate MCP server from `Sources/Mcp` (which ships inside the AOT `overfit` CLI and is reflection-free by
construction). Merging them would drag an AOT-hostile dependency graph into the product to save a hundred
lines of JSON plumbing.

## Why it exists

Grep answers a different question than the one usually being asked. Searching for a method name finds the
comment that mentions it, the string literal, and the unrelated method with the same name on another type —
and **misses** the call made through an interface, the call made through a delegate, and the override in a
derived class. Every finding below came from the semantic answer differing from the textual one.

## Cost — measured, 2026-08-06

| phase | cost |
|---|---|
| open solution (MSBuild evaluation + parse), 25 projects | 3.2 s |
| build semantic model, 1587 documents | 3.9 s |
| **CLI, per invocation** (loads every time) | **~9 s** |
| **MCP server, startup** (once) | **7.7 s** |
| **MCP server, warm query** | **3–23 ms** |

This is the whole argument for running it as a server rather than as a CLI, and it is also a warning about
how to measure it. **Each query was run twice.** The first pass costs 140–1990 ms because per-document
semantic state is still being faulted in; the second costs 3–23 ms. A single sample would have reported a
number up to two orders of magnitude too high and made the tool look unusable.

The per-project breakdown printed by `measure` needs one caveat: **a project's compilation cost is charged to
whichever project pulled it in first**, so `Main` does not appear in the list — its cost sits inside `Tests`.
Only the total is meaningful.

## Queries

```powershell
dotnet build Tools/SemanticNavigator/SemanticNavigator.csproj -c Release

$nav = "Tools/SemanticNavigator/bin/Release/net10.0/overfit-navigator.exe"

& $nav measure                                  # time a load on this box
& $nav refs    InferenceEngine                  # every real reference, solution-wide
& $nav impls   IInferenceBackend                # implementations / derived classes
& $nav callers ComputationGraph.SoftmaxCrossEntropy --depth 2
& $nav unused  Anomalies                        # dead symbols, and test-only ones
& $nav serve                                    # MCP server over stdio
```

Names may be simple (`Run`), member-qualified (`InferenceEngine.Run`) or fully qualified. **An ambiguous name
is reported, never resolved by picking the first match** — an answer about the wrong overload is
indistinguishable from an answer about the right one once it reaches the reader.

## Use from Claude Code

`.mcp.json` in the repository root registers the server. It runs the built DLL through `dotnet`, so
**build first** — the server does not build the solution, it reads it.

## What `unused` does and does not mean

`unused` reports two distinct verdicts:

- **no references anywhere** — nothing in the solution mentions the symbol.
- **referenced only by tests** — production code kept alive solely by its own test. No compiler warning finds
  this, and it is the verdict worth having the query for.

Deliberately excluded, because a zero reference count does **not** mean unused for them:

- **overrides and interface implementations** — reached through the base, never by name;
- **anything carrying an attribute** — xUnit facts, JSON-serialized members and DI-registered types are all
  invoked by a framework that no call site mentions;
- **entry points** and implicitly declared symbols;
- **`public` symbols by default.** `DevOnBike.Overfit` is a published library: its public callers live outside
  this repository, so "nothing here calls it" is not evidence of anything. `--public` opts in.

Even so, treat output as **candidates**. The tool is exactly as blind as Roslyn is — a symbol reached only
through reflection, a source generator or a config string looks dead to it and is not.

## Findings from the first run

- **`Sources/Analyzers/Analyzers.csproj` registered `AnalyzerReleases.*.md` twice.**
  `Microsoft.CodeAnalysis.Analyzers` adds them as `AdditionalFiles` itself, so the explicit `<AdditionalFiles>`
  entries were redundant. `dotnet build` tolerated the duplicate silently; Roslyn's workspace rejected the
  whole project with "duplicate source file", meaning **every tooling query over the analyzer project returned
  nothing instead of an error**. Fixed by deleting the explicit includes.
- **Three dead `_jsonOptions` fields** in `Sources/Anomalies/Monitoring/` — `PrometheusMetricSource`,
  `PrometheusHistoricalSource`, `PrometheusTopologySource`. Verified not to be a latent bug: those files
  perform no deserialization at all, so the options are leftovers from when parsing moved elsewhere, not
  options that were meant to be passed and were forgotten.
- **`LiveMonitoringPipeline.CreateForTest`** is referenced only by tests, which is what it is named for.

## Traps hit while building it, so they are not hit again

- **`ExcludeAssets="runtime"` belongs on the `Microsoft.Build.*` references only.** Set as a project-wide
  property it also strips `Microsoft.CodeAnalysis.Workspaces` from the output, and the process dies at startup
  with a `FileNotFoundException` naming an assembly nobody removed on purpose. On the MSBuild references it is
  **required** — `Microsoft.Build.Locator` resolves the SDK's MSBuild at runtime and a second copy next to the
  exe wins assembly resolution, giving two MSBuild type identities in one process. Locator ships a build-time
  check (`MSBL001`) that names the exact incantation.
- **`Microsoft.CodeAnalysis.CSharp.Workspaces` is required and its absence is silent.** Without it
  `MSBuildWorkspace` opens the solution, reports **zero** projects, and raises "the language 'C#' is not
  supported" as a *non-fatal* diagnostic per project. Every query then answers "no results" and looks correct.
  This is why the loader surfaces `Failures` loudly on stderr and why the CLI prints them before any result.
- **`Microsoft.Build.Framework` must be at or below the SDK's own MSBuild.** Measured with
  `dotnet msbuild -version`: SDK 10.0.110 ships 18.0.11, so 18.0.2 is the highest usable published version.
  The 17.11.31 that `Workspaces.MSBuild` resolves transitively carries GHSA-w3q9-fxm7-j8fq and `NuGetAudit`
  fails the build on it, correctly.
- **`Microsoft.CodeAnalysis.Workspaces.MSBuild` is pinned to 5.0.0** to match the analyzers' own Roslyn. The
  navigator is an analyzer *host*: older than the analyzers cannot load them, newer than the SDK's Roslyn
  diverges from what `dotnet build` actually ran.
