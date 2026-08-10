// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics.CodeAnalysis;

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────
// One suppression, and the file exists because the three cheaper mechanisms were tried and measured first.
//
// THE PROBLEM. `System.Text.Json`'s source generator emits `OverfitJsonContext.g.cs`, which uses
// `System.Reflection` — a namespace `BannedSymbols.txt` forbids, so RS0030 fires three times on code nobody
// wrote. Ordinarily that is a warning and harmless. Under the `aot-guard` CI job and the local
// `aot-publish.cmd`, which publish with `TreatWarningsAsErrors=true` so that IL2026 / IL3050 / IL31xx become
// hard errors, it is promoted to an error and FAILS THE GUARD — the one check standing behind this project's
// central claim.
//
// WHY THE SUPPRESSION IS CORRECT RATHER THAN CONVENIENT. The generated metadata exists precisely so that
// serialization does NOT reflect at runtime; it is the AOT-safe path, and `JsonSourceGenerationMode.Default`
// is required because both call sites deserialize. The reflection the analyzer objects to is
// `JsonSerializerContext` plumbing that ILCompiler itself then compiles and verifies — the guard proves the
// safety the ban was written to approximate.
//
// WHAT WAS TRIED AND MEASURED, 2026-08-10, so nobody repeats it:
//
//   1. `<WarningsNotAsErrors>$(WarningsNotAsErrors);RS0030;…</WarningsNotAsErrors>` — which the comment in
//      Directory.Build.props recommends, at length. It WORKS for the generated file and it also DISARMS
//      RS0030 on real code: a two-armed mutation adding `System.Array.Copy` to `Runtime/ContainedPath.cs`
//      produced 2 errors and a red build without it, and 2 WARNINGS and a green build with it. The comment's
//      claim that "a genuine error, not a promoted warning, cannot be downgraded" is false. Reverted.
//
//   2. `.editorconfig`. Two sections already set `RS0030 = none` for `[**/obj/**]` and `[*.g.cs]`, and
//      neither has any effect here. Adding `generated_code = true` changed nothing. Setting RS0030 to `none`
//      in the GLOBAL `[*.cs]` section changed nothing either — which is the measurement that settles it:
//      .editorconfig does not reach this document at all, because `EmitCompilerGeneratedFiles` is off and the
//      file is analysed under a synthetic path no glob matches. Those two sections are inert for this purpose.
//
//   3. Switching to `JsonSourceGenerationMode.Serialization`, which emits no metadata. It would break
//      `ScalerParams.Load` and `HmmParams.Load`, both of which deserialize.
//
// SCOPE IS ONE TYPE, deliberately. `Target` names the partial this repository declares, so the suppression
// covers the half the generator emits and nothing else: a `System.Reflection` use anywhere else in
// Sources/Main is still an error, which a mutation confirms.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────
[assembly: SuppressMessage(
    "ApiDesign",
    "RS0030:Do not use banned APIs",
    Justification = "System.Text.Json source-generated metadata: the AOT-safe path, verified by ILCompiler in the aot-guard job. See the file header for the three mechanisms measured and rejected first.",
    Scope = "type",
    Target = "~T:DevOnBike.Overfit.Data.Serialization.OverfitJsonContext")]
