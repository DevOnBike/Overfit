---
name: analyzer-releases-autofile
description: Microsoft.CodeAnalysis.Analyzers auto-registers AnalyzerReleases.{Shipped,Unshipped}.md as AdditionalFiles if they exist in the project directory — verified in the actual nuget package, not just trusted from a comment.
metadata:
  type: reference
---

`~/.nuget/packages/microsoft.codeanalysis.analyzers/<version>/buildTransitive/Microsoft.CodeAnalysis.Analyzers.targets`
contains (verified at both 3.11.0 and 5.6.0, the latter being what `Directory.Packages.props` pins as of
2026-08-06):

```xml
<ItemGroup Condition="Exists('$(MSBuildProjectDirectory)\AnalyzerReleases.Shipped.md')" >
  <AdditionalFiles Include="AnalyzerReleases.Shipped.md" />
</ItemGroup>
<ItemGroup Condition="Exists('$(MSBuildProjectDirectory)\AnalyzerReleases.Unshipped.md')" >
  <AdditionalFiles Include="AnalyzerReleases.Unshipped.md" />
</ItemGroup>
```

So an explicit `<AdditionalFiles Include="AnalyzerReleases.Shipped.md" />` in `Sources/Analyzers/Analyzers.csproj`
(or any csproj referencing this package, with the files present next to it) really is a duplicate — MSBuild
tolerates it, Roslyn's own workspace (`MSBuildWorkspace`, used by `Tools/SemanticNavigator`) does not, and
rejects the whole project with "duplicate source file". Confirms the reasoning in the 2026-08-06 diff that
deleted the explicit includes from `Analyzers.csproj`. See [[project-semantic-navigator-tool]].

Method for re-checking: `find ~/.nuget/packages/microsoft.codeanalysis.analyzers/*/buildTransitive -iname
"*.targets"` then grep for `AdditionalFiles`.
