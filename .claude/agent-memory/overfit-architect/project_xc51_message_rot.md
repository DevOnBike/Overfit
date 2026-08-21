---
name: xc51-message-rot
description: 2026-08-15 XC-51 sign-off — live advice vs historical record, the composed-identifier rule, and the analyzer descriptor census
metadata:
  type: project
---

`docs/specs/xc-51-analyzer-message-rot-plan.md`, signed APPROVED 2026-08-15. Analyzer diagnostic messages
naming deleted types. Nothing was compiled (machine-exclusion mutex held).

**The transferable ruling: live advice vs historical record.** A string telling a developer what to do NOW is
advice and must be corrected; a string recording what was done ON A DATE is a record and must not be, even
when it names something deleted. Correcting a record deletes the evidence a sweep happened and replaces it
with a claim false about the past. Found because `git log -S RentArray -- Sources/Main/Tensors/PooledBuffer.cs`
shows `RentArray`/`ReturnArray` **did exist** — added `2584375` 2026-05-30, removed `718d6f8` 2026-06-21 — so
`.editorconfig:252` (Runtime/ escalated 2026-06-12), `.editorconfig:259` (Whisper/ 2026-06-13) and
`Sources/Analyzers/README.md:121` are accurate records. The plan's acceptance grep would have forced all three
to be rewritten. **Before approving an edit to prose naming a deleted symbol, `git log -S` the symbol.**

**Composed identifiers in diagnostic messages — the rule is existence assertion, not composition.** 8 of 47
descriptors compose an identifier from a placeholder; 7 are safe (echo of the developer's code; a symbol the
analyzer resolved first — `SynchronousIslandAnalyzer.cs:175` `GetMembers(name + "Async")`; a name the fix will
create — `AsyncApiConventionAnalyzer`; a symbol the same generator emits). Exactly one asserts existence
without resolving: `CpuFeaturesGateAnalyzer.cs:30` `CpuFeatures.Has{0}`. A blanket "no composed member names"
rule would condemn six correct messages. Rule written into `Sources/Analyzers/README.md` "Authoring notes"
(the existing authoring contract, ~:142-149) — **not** an ADR.

**Analyzer census, verified by source scan 2026-08-15** (not by a build): **47** `new DiagnosticDescriptor(`
sites in `Sources/Analyzers`, every one a `static readonly` field; **47** ids in
`AnalyzerReleases.Unshipped.md`; **3** ids have no `DiagnosticId` const (`OVERFIT035`, `OVERFIT036` on the
generator, `OVERFIT900` internal on `OverfitPerfAnalysis`) — which is why enumeration must be reflection over
static fields, never `SupportedDiagnostics` or id consts.

**No pooled option exists for a per-call array of a REFERENCE type.** `PooledBuffer<T>` is `where T : struct`,
`TensorStorage<T>` `where T : unmanaged`, `stackalloc` unmanaged, raw `ArrayPool<T>.Shared` RS0030-banned in
Main, `OverfitResourcePool<T>` pools instances not arrays. `OVERFIT001`'s message now says hoist or
restructure; the rule's own one-time exemption (field/ctor) is the sanctioned escape.

**RS2000 family runs on `Sources/Analyzers` only** (`Analyzers.csproj` refs `Microsoft.CodeAnalysis.Analyzers`
5.6.0 + `EnforceExtendedAnalyzerRules=true`). **No `.editorconfig` entry and no `Directory.Build.props` rule
promotes RS2xxx to error**, so "the build succeeded" is NOT evidence that no RS2000 diagnostic fired — grep
the log for `RS2\d{3}`.

`.claude/pb12-old/` holds 686 files, a stale copy of `Sources/Main` incl. its `BannedSymbols.txt`. Exclude
`.claude/**` from every repo-wide acceptance grep. See [[amendment-sweep]].
