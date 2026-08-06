# Memory index — overfit-packages-update

- [Pinning decisions](pinning_decisions.md) — the 4 deliberate below-latest pins (Roslyn CSharp, SemanticNavigator Workspaces, Build.Framework, OTel Prometheus beta), each with how it was verified and on what date.
- [Test-only vs. tooling packages](test_only_packages.md) — which pins are genuinely contained to `Tests.csproj` vs. tooling that only looks test-adjacent (SourceLink ships in every packed NuGet).
- 2026-08-06 survey: solution has 0 vulnerable packages currently (`dotnet list package --vulnerable` clean); `xunit` 2.9.3 flagged **deprecated** (Legacy reason, alternative `xunit.v3`) by `dotnet list package --deprecated` — not a security issue, a family-migration notice.
- 2026-08-06: almost all 28+4 pinned packages are already at the latest version the feed offers — only `Microsoft.CodeAnalysis.CSharp`/`.Workspaces`/`.CSharp.Workspaces` (5.0.0, held below 5.6.0), `Microsoft.Build.Framework` (18.0.2, held below 18.8.2), and `OpenTelemetry.Exporter.Prometheus.AspNetCore` (beta, held below 1.17.0-beta.1) have real newer versions available. Re-verify via `dotnet list package --outdated` each run — do not assume this stays true.
