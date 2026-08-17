---
name: navigator-index-stale-after-move
description: The overfit-navigator MCP server answers from a solution snapshot — right after a file move it still reports the pre-move declaration and namespace, so verify call-site churn with a compile, not with find_references.
metadata:
  type: reference
---

Observed 2026-08-12: minutes after `GuardMetricsEndpoint.cs` was moved from `Sources/Cli` to
`Sources/Anomalies/Hosting` and rebuilt, `find_references` still returned
`DevOnBike.Overfit.Cli.GuardMetricsEndpoint` declared at the deleted path.

**So after moving or renaming a type, the navigator's answer is about the old tree.** The evidence that call
sites still resolve is that the referencing project compiles — `dotnet build Sources/Cli/Cli.csproj -c Release`
clean is the check, not a reference count. The navigator is still the right tool for *pre-change* inventory.

Unrelated but same session: the server holds `overfit-navigator.dll` open, so every whole-solution build ends
in `MSB3021/3026/3027` on that copy step and `dotnet build` exits 1 with no compiler error. Judge that build
by its diagnostics, not its exit code.

**The recovery, used again 2026-08-15**: `dotnet sln list` and build each project except
`Tools/SemanticNavigator/SemanticNavigator.csproj` individually. Ten projects (Main, Anomalies, Cli, Mcp,
Server, Server.AspNet, Extensions.AI, Benchmarks, AotSmokeTest, Tests) all exit 0 that way, so
"`dotnet build -c Release` on the solution" as a completion check is satisfiable without killing the MCP
server. Related: [[doc-diagnostics-on-arrival]].
