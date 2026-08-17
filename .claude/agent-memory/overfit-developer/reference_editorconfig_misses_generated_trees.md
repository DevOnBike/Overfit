---
name: editorconfig-misses-generated-trees
description: .editorconfig severities never reach a source-generated syntax tree, whatever the path looks like; the fix is a global AnalyzerConfig, and a bare `using System.Linq;` is an inert probe for RS0030
metadata:
  type: reference
---

**No `.editorconfig` section applies to a SOURCE-GENERATED syntax tree.** Severities are resolved per
tree from a map the compiler builds out of the csc command line; a generator adds its trees after that
map exists, so the tree matches nothing **by identity**, not by path. Its diagnostics fall through to
the analyzer's own default severity, which `TreatWarningsAsErrors` then makes fatal — so the failure is
**configuration-dependent** and an ordinary `dotnet build` never sees it.

Measured 2026-08-16 on `Sources/Anomalies`, aot-guard CI failure. `BindingExtensions.g.cs` (the
`Microsoft.Extensions.Configuration.Binder` generator, which runs only under `PublishAot`/`PublishTrimmed`,
emitting `section.GetChildren().Any()` for the one `section.Bind(file)` call) is reported at a path
**under `obj/`** and **ending `.g.cs`** — matching two sections that both set `RS0030 = none` — and still
reported `warning RS0030`.

**Read the severity word; it is the whole diagnosis.** Build with the AOT property but WITHOUT
`TreatWarningsAsErrors`:
- absent   → an editorconfig `none` section applied;
- `error`  → the `[*.cs]` section applied;
- `warning`→ **nothing applied**; that is the analyzer default showing through.

**The fix is a global AnalyzerConfig** (`is_global = true`), which is exactly the compiler's fallback for
trees the per-path map misses: `Sources/Anomalies/SourceGenerated.globalconfig`, wired with
`<EditorConfigFiles Include="..."/>`. Name it so it is NOT `.globalconfig` or it may be auto-discovered
and included twice. Real `.cs` files keep `error` from `[*.cs]` — .editorconfig wins over a global config
for a path it matches. NOT `<NoWarn>` and NOT `WarningsNotAsErrors` (`Directory.Build.props` forbids the
latter in writing); both are compilation-wide and disarm the ban on real code.

**`using System.Linq;` alone does NOT fire RS0030 — it is a vacuous probe.** Measured the same day: the
using directive produced no diagnostic at all, and the first "the ban is still armed" run was therefore a
false green. Use a symbol reference, e.g. `_ = global::System.Linq.Enumerable.Count(new int[] { 1 });`,
and run the capability arm (fix disabled) first. See [[self-consistency-mutation-blind]].

Cross-OS AOT publish cannot be verified on this box: `-r linux-x64` stops at *"Cross-OS native compilation
is not supported"*, and `-r win-x64` reaches ILCompiler (a ~100 MB `native/overfit.obj` proves it ran clean
under `TreatWarningsAsErrors`) then dies in the MSVC link because `vswhere.exe` is not on PATH.
