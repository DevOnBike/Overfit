---
name: semantic-navigator-tool
description: Architecture and a self-measured doc-claim bug in Tools/SemanticNavigator, the Roslyn/MSBuildWorkspace dev tool added 2026-08-06.
metadata:
  type: project
---

`Tools/SemanticNavigator` (5 files, `overfit-navigator` exe) wraps `MSBuildWorkspace` + Roslyn `SymbolFinder`
to answer `refs` / `impls` / `callers` / `unused`, as a CLI and as an MCP server (`serve` verb,
`NavigatorMcpServer.cs`, newline-delimited JSON-RPC over stdio — correct for the MCP stdio transport, which is
newline-delimited, not LSP-style Content-Length framing). Registered in `.mcp.json` at repo root
(`dotnet Tools/SemanticNavigator/bin/Release/net10.0/overfit-navigator.dll serve` — DLL must be pre-built,
doc says so).

Deliberately outside `Sources/` and deliberately not AOT (`PublishAot=false`, `IsAotCompatible=false` in the
csproj, with a comment explaining why) — do not flag its reflection/LINQ use as an AOT violation, that
boundary is the whole design. It is a separate MCP server from `Sources/Mcp` on purpose (that one ships inside
the AOT `overfit` CLI and must stay reflection-free).

## Verified true on this pass (2026-08-06)

- `Sources/Analyzers/Analyzers.csproj`'s deleted `<AdditionalFiles Include="AnalyzerReleases.*.md">` — checked
  the actual nuget package targets file
  (`~/.nuget/packages/microsoft.codeanalysis.analyzers/{3.11.0,5.6.0}/buildTransitive/*.targets`): it really
  does `<ItemGroup Condition="Exists('$(MSBuildProjectDirectory)\AnalyzerReleases.Shipped.md')">` and adds it
  as `AdditionalFiles` itself. The explicit includes were a real duplicate. See
  [[reference-analyzer-releases-autofile]].
- `Directory.Packages.props`'s `Microsoft.Build.Framework` pin (18.0.2, "at or below the SDK's MSBuild") — ran
  `dotnet msbuild -version` on this box: SDK 10.0.110 → MSBuild 18.0.11. 18.0.2 ≤ 18.0.11, consistent with the
  claim.
- Adding the project to `Overfit.sln` does make it build on every `dotnet build -c Release`, including both
  CI matrix legs (`ci.yml` `build-and-test` runs `dotnet build -c Release --no-restore` with no project arg →
  picks up the whole solution on ubuntu-latest AND windows-latest). Bounded cost: compile-only (nothing
  publishes or runs it in CI), a few seconds + one extra NuGet restore set. Judged reasonable, not a defect.

## Bug found and measured myself (the top finding of the 2026-08-06 review)

`docs/semantic-navigator.md`'s cost table states **"MCP server, warm query: 3–23 ms"** as one blanket row
covering all four verbs. Built the tool, ran it live via a scripted JSON-RPC client against `serve`:

| call | measured |
|---|---|
| `find_references` (2nd call, same symbol) | 6 ms — matches the claim |
| `find_unused` Anomalies, no `--public`, warm steady-state | ~200-220 ms |
| `find_unused` Anomalies, `--public`, warm steady-state | ~5.7-5.8 s |
| `find_unused` Main, no `--public`, warm steady-state | ~1.3-3.0 s |
| `find_unused` Main, `--public`, warm steady-state | **~6.3-11.3 s** |

Root cause (visible by reading `NavigatorQueries.FindUnusedAsync`): it issues one solution-wide
`SymbolFinder.FindReferencesAsync` **per candidate symbol** in the project (N sequential searches, not one).
The *per-symbol* cost is actually within the claimed 3-23 ms range (694 Main candidates in 6.3 s ≈ 9 ms each)
— so the atomic query cost isn't the false part. What's false is presenting that figure as the **per-tool-call**
cost a caller experiences: for `find_unused` specifically it's the per-symbol cost times the candidate count,
which is 10x-500x higher depending on project size and `--public`. The doc's own "Findings from the first run"
section and "Traps hit" section never caveat `unused` differently. This is a live instance of the
"claims that outrun their evidence" class of finding — reproduce with the scripted JSON-RPC client (Python,
newline-delimited `{"jsonrpc":"2.0",...}` over the process's stdin/stdout) rather than trusting the table if
this doc gets touched again.

## Secondary finding (robustness, not measured, code-read only)

`NavigatorMcpServer.HandleAsync` only wraps `tools/call`'s `InvokeAsync` in try/catch. Field extraction earlier
in the method (`request["method"]?.GetValue<string>()`) is unguarded — a syntactically-valid-JSON but
wrong-shaped request (e.g. `"method": 123`) throws `InvalidOperationException` inside the `while` loop in
`RunAsync`, which has no surrounding try/catch of its own. That propagates up through `Program.Main`'s
top-level catch, which prints an error and returns 1 — killing the **whole server process**, not just failing
the one bad request, costing a ~7s reload to recover. Contrast with the file's own stated design goal (stdout
hygiene) — request-level fault isolation was not part of that goal but arguably should have been.

## Minor precision note in `find_unused` (not blocking)

`NavigatorQueries.IsCandidate` gates on `symbol.DeclaredAccessibility == Public` to decide "public, needs
`--public`" — but that's the symbol's *own* declared accessibility, not its *effective* one. A `public` method
on an `internal` class is not actually externally callable, yet gets excluded by default under the "public
callers live outside the repo" rationale, which doesn't apply to it. Biases the default-mode output toward
under-reporting real dead code. Small, disclosed-adjacent (the doc already says "treat output as candidates").
