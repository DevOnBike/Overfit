---
name: doc-diagnostics-on-arrival
description: Moving a .cs file INTO Sources/Anomalies (or any project with GenerateDocumentationFile) turns latent XML-doc mistakes into new warnings — CS1573 for partial <param> coverage and CS1574 for a stale <see cref>. Both measured 2026-08-12.
metadata:
  type: reference
---

`Sources/Anomalies/Anomalies.csproj` sets `GenerateDocumentationFile=true` with `NoWarn` covering only
`1591`. `Sources/Cli/Cli.csproj` sets neither, so doc comments there are never checked. A file moved from
Cli to Anomalies therefore arrives with every latent doc defect newly visible.

Measured on the `XC-22` move of `GuardMetricsEndpoint`, each by reverting the fix and rebuilding:

| defect | diagnostic |
|---|---|
| `TryStart` documented 2 of 5 parameters | `CS1573` at the signature line, once per undocumented parameter |
| `<see cref="Respond"/>` naming a method renamed to `RespondAsync` long ago | `CS1574` |

**Before moving a file into a doc-generating project, read its XML comments for undocumented parameters and
crefs that no longer resolve.** Both are one-line fixes at write time and both are new warnings otherwise.
Related: [[navigator-index-stale-after-move]].
