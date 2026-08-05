# overfit-reviewer memory index

- [Measured negative results (perf)](reference_measured_negative_results.md) — curated list of "obviously correct, measured worse" optimisations in this repo, with numbers and sources. Check before proposing ANY perf change.
- [SemanticNavigator tool (Tools/)](project_semantic_navigator.md) — Roslyn/MSBuildWorkspace dev tool, deliberately non-AOT, outside Sources/. Architecture + a doc-claim bug I measured myself.
- [Analyzer release-tracking auto-add](reference_analyzer_releases_autofile.md) — verified true: Microsoft.CodeAnalysis.Analyzers' own targets auto-register AnalyzerReleases.{Shipped,Unshipped}.md as AdditionalFiles if present; explicit `<AdditionalFiles>` for them is a real duplicate, not a false claim.
- [Directory.Build.props NuGetAudit is solution-wide](reference_nugetaudit_scope.md) — inherited by every project under D:\Overfit including new ones added anywhere in the tree; matters when judging a new project's dependency pins.
