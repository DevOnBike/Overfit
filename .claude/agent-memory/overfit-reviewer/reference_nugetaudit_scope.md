---
name: nugetaudit-scope
description: Directory.Build.props' NuGetAudit + NU1901-1904-as-error setting is inherited by every project under D:\Overfit, including new ones anywhere in the tree — relevant when judging a new project's package version choices.
metadata:
  type: reference
---

`Directory.Build.props` (repo root) sets:

```xml
<NuGetAudit>true</NuGetAudit>
<NuGetAuditMode>all</NuGetAuditMode>
<WarningsAsErrors>$(WarningsAsErrors);NU1901;NU1902;NU1903;NU1904;CS4014</WarningsAsErrors>
```

MSBuild auto-imports `Directory.Build.props` into every project at or below its directory unless a project
opts out, so this applies solution-wide — a brand-new project dropped anywhere under `D:\Overfit` (e.g.
`Tools/SemanticNavigator`) inherits it automatically, with no per-project opt-in needed.

Consequence for review: when a new project pins a package version with a comment claiming "avoids known CVE
X", that claim is load-bearing for the *whole build*, not just that project — if the pin were wrong the build
would fail immediately (`NuGetAudit` promotes vulnerability warnings to errors), which is a cheap way to
partially verify the claim: a green `dotnet build -c Release` after adding the project is itself evidence the
audit passed, though it doesn't verify the *reasoning* in the comment (e.g. "why 18.0.2 specifically") — that
still needs an independent check (e.g. `dotnet msbuild -version` against the stated SDK-version claim).
