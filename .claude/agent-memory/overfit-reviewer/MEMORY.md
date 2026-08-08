# overfit-reviewer memory index

- [Measured negative results (perf)](reference_measured_negative_results.md) — curated list of "obviously correct, measured worse" optimisations in this repo, with numbers and sources. Check before proposing ANY perf change.
- [SemanticNavigator tool (Tools/)](project_semantic_navigator.md) — Roslyn/MSBuildWorkspace dev tool, deliberately non-AOT, outside Sources/. Architecture + a doc-claim bug I measured myself.
- [Analyzer release-tracking auto-add](reference_analyzer_releases_autofile.md) — verified true: Microsoft.CodeAnalysis.Analyzers' own targets auto-register AnalyzerReleases.{Shipped,Unshipped}.md as AdditionalFiles if present; explicit `<AdditionalFiles>` for them is a real duplicate, not a false claim.
- [Directory.Build.props NuGetAudit is solution-wide](reference_nugetaudit_scope.md) — inherited by every project under D:\Overfit including new ones added anywhere in the tree; matters when judging a new project's dependency pins.
- [DI stray-registration blind spot](reference_di_stray_registration_blind_spot.md) — `Assert.Same` on two resolves of an interface CANNOT catch a stray extra registration of the concrete class under its own service type; verified empirically, don't trust a test docstring's claim otherwise.
- [Anomaly-guard seam review, Task 1](project_anomaly_guard_seam_review.md) — 2026-08-08, `IMetricWindowSource` extraction clean except one docstring overclaim (see DI blind-spot entry above).
