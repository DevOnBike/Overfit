STATUS: APPROVED
Author: overfit-analyst (§1–§14) + overfit-architect (§15 onwards)
Date: 2026-08-15 (analyst), 2026-08-15 (architect review and sign-off)
Slug: xc-51-analyzer-message-rot-plan

GATES:
  performance:        NOT_REQUIRED — no performance claim in this plan. Nothing here changes a code path;
                        the only executable artefact proposed is one xUnit test that runs reflection over two
                        already-loaded assemblies
  security:           NOT_REQUIRED — no parser, endpoint, gateway, audio/tokenizer decode or externally-fed
                        `unsafe` surface touched. Every change is a string literal in a diagnostic descriptor
                        or a markdown line
  leak-scan:          NOT_REQUIRED — no config, log, fixture, host name, path or token touched
  AOT:                NOT_REQUIRED — `Sources/Analyzers` targets `netstandard2.0`, is referenced by
                        `Main.csproj` with `OutputItemType="Analyzer"` and never ships (`Analyzers.csproj:6-8`
                        pins `PublishAot/PublishTrimmed/SelfContained` to false). The proposed test lives in
                        `Tests`, which is not AOT-published
  API-compatibility:  NOT_REQUIRED — no member of `DevOnBike.Overfit` is added, removed or changed. Diagnostic
                        *message text* is user-visible build output but is not API; nothing in this repository
                        matches on it (see §4, F6)

  verifier:           REQUIRED — T4 is a new test; its red run on unfixed source is the acceptance evidence
  reviewer:           REQUIRED — every edit is user-facing text, which is exactly the class a reviewer reads
  mutation-proof:     REQUIRED for T4 only. Its acceptance criterion **is** a mutation (§D6 orders it)
  release-readiness:  NOT_REQUIRED — no version, package, public API or CHANGELOG-visible change

`verifier`, `reviewer` and `mutation-proof` were left absent by the analyst on the reasoning that nothing has
been implemented yet; seeded here as `REQUIRED` so that a gate nobody ran and a gate nobody needed do not look
alike to `Scripts/plan_gate_check.py`.

**Nothing in this plan was compiled or executed.** A `[LongFact]` gate campaign held the machine-exclusion
mutex `Global\DevOnBike.Overfit.MachineMeasurement` for the whole of this investigation, so `dotnet build`,
`dotnet test` and `dotnet publish` were not run. Every claim below was established by reading files or by the
`overfit-navigator` semantic tools, and each claim says which. The two items that genuinely need a build to
settle are named in §7 as open questions for the developer, not asserted here.

---

# XC-51 — analyzer diagnostic messages that name things which no longer exist

Plan file for task `XC-51` (`docs/TASKS.md:183`). Written by `overfit-analyst`. No prior plan for this task.

## 1. What the client asked for, quoted

From `docs/TASKS.md:183`:

> "OVERFIT001's diagnostic message tells developers to use `PooledArray`, a type deleted on 2026-05-29 …
> **Every OVERFIT001 diagnostic in this repository therefore names a type nobody can use** … **While fixing
> it, check the whole rule set for the same rot**: this was found by accident, and nothing verifies that a
> diagnostic's suggested replacement still exists. A test that greps every `messageFormat` for type names and
> asserts each resolves would turn a recurring class of defect into a build failure"

And from the dispatching message, on priority:

> "**1. The sweep, which is the valuable half.** The known defect was found by accident. … `PooledArray` is
> one instance of a class."
> "**2. The durable guard, and be honest about whether it is worth building.** … If your honest read is that
> the guard costs more than the defect class it catches, **say that**."
> "**A standing constraint from the user: `Sources/Analyzers` must not depend on code in this repository.**"

There is no external client. The requester is this repository's own review process; the user whose need
matters is the developer who reads an OVERFIT diagnostic in their IDE and acts on its advice.

## 2. Inventory — what already exists

**Already exists**

| Thing | Where | Note |
|---|---|---|
| 47 `DiagnosticDescriptor`s across 44 files | `Sources/Analyzers/*.cs` | Counted; matches the 47 rule ids in `AnalyzerReleases.Unshipped.md:8-55` exactly, so the sweep below is complete rather than a sample |
| Analyzer test harness | `Tests/Analyzers/AnalyzerHarness.cs` | Compiles a snippet in memory, runs one analyzer, returns **ids only** (`:90-93`) |
| 11 analyzer test files | `Tests/Analyzers/` | None covers `HeapArrayAllocationAnalyzer`; none asserts on message text |
| Analyzers referenced by Tests **as a library** | `Tests/Tests.csproj:88-97` | Deliberate, with the reason in a comment: Tests is excluded from analyzer wiring, so the assembly can be referenced and its analyzers instantiated by hand |
| Release-tracking files | `Sources/Analyzers/AnalyzerReleases.{Shipped,Unshipped}.md` | `Shipped.md` is **empty** (header comments only) |

**Partially exists**

| Thing | What is there | What is missing |
|---|---|---|
| Any check that a diagnostic's advice resolves | Nothing at all | This is the whole of T4 |
| Coverage of `OVERFIT001` by a test | `PrefillAllocationTests.cs` mentions the id in a pragma; no analyzer test | A `HeapArrayAllocationAnalyzerTests` — **not proposed here**, it is a separate task |

**Does not exist**

- `PooledArray` / `PooledArray<T>` — `find_references` returns *"No source symbol named 'PooledArray'"*.
- `PooledBuffer<T>.RentArray` / `.ReturnArray` — `find_references` returns *"No source symbol named
  'PooledBuffer.RentArray'"*; a repo-wide text search finds the names only in prose (F4).
- The MSBuild source-file guards `BanJaggedFloatArrays` / `BanMultipleTopLevelTypes` — removed 2026-08-05,
  `Main.csproj:81-101` is now a comment recording the removal (F3).

## 3. Problem / user need / business goal / proposed solution

| | |
|---|---|
| **Problem** | A developer who follows an OVERFIT001 diagnostic looks for `PooledArray`, does not find it, and either guesses or asks. It has been wrong for three months (deleted 2026-05-29, found 2026-08-14) and was found by accident, while chasing an unrelated design question — which is the part that matters: nothing would have found it |
| **User need** | As the developer the analyzer is talking to, when a rule tells me what to use instead, that thing must exist and must work for my case |
| **Business goal** | Diagnostics stay trustworthy as the codebase moves. Secondarily: the class of defect stops being found by accident |
| **Proposed solution** | Fix the four named sites, and add a test that greps every `messageFormat` for type names and asserts each resolves |

**Success metric.** `not stated` by the client, and I will not invent one. The nearest honest formulation:
after this lands, **zero** identifiers named in any of the 47 descriptors fail to resolve, and a re-run of
the same sweep by hand in three months finds nothing new. §11 records what that is worth against its cost.

**Cost of not doing it:** small but recurring and asymmetric. The message is read at the moment a developer is
already interrupted; the actual incident on 2026-08-14 was that a stale name was cited as a *design
precedent* for a new wrapper, and the design was built on a type that does not exist until a `Glob` disproved
it (`docs/specs/machine-exclusion-symmetric-complete-plan.md:107-132`).

## 4. Investigation findings

All 47 descriptors had their `title`, `messageFormat` and `description` extracted and read. The named-thing
inventory across the whole rule set is small and closed: **nine** Overfit-owned identifiers plus BCL types,
keywords, rule ids, two doc paths and one editorconfig key. Full resolution table in §5.

### F1 — `PooledArray` is dead and named in five live places (the filed defect)

`find_references` on `PooledArray`: *"No source symbol named 'PooledArray'"*. `ROADMAP-COMPLETED.md:2007`
carries the deletion, amended 2026-05-29: *"`PooledArray<T>` was a duplicate of `PooledBuffer<T>` and has been
deleted."*

Live sites, all of which a developer can see:

| # | Site | Kind |
|---|---|---|
| 1 | `Sources/Analyzers/HeapArrayAllocationAnalyzer.cs:35` | `messageFormat` — the one in build output |
| 2 | `Sources/Analyzers/HeapArrayAllocationAnalyzer.cs:17` | XML doc |
| 3 | `Sources/Analyzers/AnalyzerReleases.Unshipped.md:8` | Notes column |
| 4 | `Sources/Analyzers/README.md:12` | Rule table |
| 5 | **`.editorconfig:59`** | Comment above the ratchet — **not named in the task row**, found by this sweep |

Three further copies are in `.claude/agents/overfit-architect.md:253`, `.claude/agents/overfit-developer.md:223`
and `.claude/skills/overfit-spec/SKILL.md:128,197`. **These are user-owned and out of scope** (§8) — reported,
not fixed.

### F2 — `OVERFIT015` composes a member name that resolves for only some inputs (new, found by the sweep)

`CpuFeaturesGateAnalyzer.cs:30`:

> `messageFormat: "Direct '{0}.{1}' — gate ISA and vector-width checks through CpuFeatures (CpuFeatures.Has{0}): …"`

`{0}` is `property.ContainingType.Name` (`:70`). `CpuFeatures` (`Sources/Main/Intrinsics/CpuFeatures.cs`,
read in full) declares exactly thirteen fields. Resolution of `CpuFeatures.Has{0}` by input:

| The developer wrote | Message tells them to use | Resolves? |
|---|---|---|
| `Avx2.IsSupported` | `CpuFeatures.HasAvx2` | yes |
| `Fma.IsSupported` | `CpuFeatures.HasFma` | yes |
| `Avx.IsSupported`, `Sse.IsSupported`, `Sse3.IsSupported`, `AvxVnni.IsSupported`, `Dp.IsSupported` | `HasAvx`, `HasSse`, `HasSse3`, `HasAvxVnni`, `HasDp` | yes |
| `Vector128/256/512.IsHardwareAccelerated` | `HasVector128/256/512` | yes |
| **`Avx512F.IsSupported`** | **`CpuFeatures.HasAvx512F`** | **no** — the field is `HasAvx512` (`:23`) |
| **`Avx512BW.IsSupported`** | **`CpuFeatures.HasAvx512BW`** | **no** — the field is `HasAvx512Bw` (`:27`) |
| **`Sse2` / `Ssse3` / `Sse41` / `AdvSimd` / any other intrinsics class** | `CpuFeatures.HasSse2` etc. | **no** — no such field; `CpuFeatures` covers only what the library uses |
| **`Avx2.X64.IsSupported`** (nested) | **`CpuFeatures.HasX64`** | **no** — `ContainingType.Name` of a nested type is `X64` |

This is the same defect class as F1, in a harder-to-see form: the dead name is *composed at report time*, so
no text search over the source finds it and — importantly — **the guard proposed in the task row would not
catch it either** (§6).

### F3 — `OVERFIT002` and `README.md` cite a build guard that was deleted on 2026-08-05

`JaggedArrayAllocationAnalyzer.cs:36` (description) ends:

> "float[][] is banned in `Sources/Main` entirely by the MSBuild guard."

`Main.csproj:81-101` is now a comment block opening *"THIS FILE NO LONGER CONTAINS CODE, and that is the point
of the last three changes"* and mapping `BanJaggedFloatArrays -> OVERFIT033` and
`BanMultipleTopLevelTypes -> OVERFIT034`. The mechanism named in the message no longer exists; the ban does,
under a different rule. `Sources/Analyzers/README.md:4` (*"the MSBuild structural guards (jagged arrays,
one-type-per-file)"*) and `:13` (*"a build ERROR via the MSBuild guard regardless"*) say the same.

`CLAUDE.md` carries a whole section, **"Source-file guards (MSBuild, build-time errors)"**, describing both
tasks as live. **`CLAUDE.md` is user-owned; reported here, not touched** (§8).

### F4 — the RS0030 message names two members that do not exist (new, found by the sweep)

`Sources/Main/BannedSymbols.txt:9` — the text a developer sees when RS0030 fires on `ArrayPool<T>.Shared`:

> "(2) class-lifetime `PooledBuffer<T>.RentArray(n)` / `PooledBuffer<T>.ReturnArray(arr)` for cases where the
> using scope doesn't fit (FastTensor, TensorStorage, loaders)."

`find_references` on `PooledBuffer.RentArray`: *"No source symbol named 'PooledBuffer.RentArray'"*.
`Sources/Main/Tensors/PooledBuffer.cs` read in full — the type has a constructor, `Span`, `Memory`, `Length`,
`IsAllocated` and `Dispose`, and nothing else. Its own doc comment (`:19-20`) states the current class-lifetime
pattern correctly: *"class field: rent in a constructor, `Dispose()` in the owner's Dispose"*, which is exactly
what `TensorStorage.cs:21,50` and `FastTensor.cs:19,95` do.

So the advice is not merely a dead name — it describes a mode of use the type has never had in its present
form. Same string in `Sources/Anomalies/BannedSymbols.txt:9`; also `CONTRIBUTING.md:32`,
`docs/code-patterns.md:27`, `Sources/Main/Tensors/README.md:17`, `Sources/Analyzers/README.md:121` and a code
comment at `Sources/Main/Onnx/OnnxGraphImporter.cs:156`.

This is **outside** the row's literal scope (`Sources/Analyzers`) but inside its defect class, and it is
arguably worse: RS0030 is an *error* in Main, so every developer who hits it reads this text. Scoped as a
**Should** in §9, with a blocking question (§13, Q1).

### F5 — `OVERFIT001`'s advice is impossible for a reference-element array

`PooledBuffer<T>` is declared `where T : struct` (`PooledBuffer.cs:39-40`). `OVERFIT001` fires on
`new {0}[]` for **any** element type (`HeapArrayAllocationAnalyzer.cs:55-60`; only jagged arrays are excluded,
`:87-90`). For `new SomeClass[n]` the message offers three options of which `PooledBuffer<T>` cannot compile
and `stackalloc` cannot either — leaving `TensorStorage<T>`, which is also for numeric data. The message is
not wrong about a name; it is wrong about applicability. Folded into T1 as a description edit rather than a
message edit, because the message has no room.

### F6 — nothing asserts on any diagnostic's message text

`Tests/Analyzers/AnalyzerHarness.cs:90-93` returns `results.OrderBy(...).Select(d => d.Id).ToArray()` — ids
only. A text search of `Tests/` for `GetMessage`, `MessageFormat` and the literal `Allocates 'new` returns
nothing. **No test breaks when a message changes.** (Established by reading + grep; not by running the suite —
see the note under GATES.)

### F7 — changing a message needs no release-file edit

Decisive evidence, extracted from the resource strings of
`~/.nuget/packages/microsoft.codeanalysis.analyzers/5.6.0/analyzers/dotnet/cs/Microsoft.CodeAnalysis.Analyzers.dll`
(the version pinned at `Directory.Packages.props:13`). The RS2001 message reads:

> "Rule '{0}' has a changed **'Category' or 'Severity'** from the last release. Either revert the update(s) in
> source or add a new up-to-date entry to unshipped release file."

The rest of the family checks presence (RS2000: *"Rule '{0}' is not part of any analyzer release"*),
duplicates, removed-but-still-reported, and header/entry syntax. **No rule in the family reads `title`,
`messageFormat`, `description` or the free-text Notes column.** On top of that, `AnalyzerReleases.Shipped.md`
is empty, so there is no "last release" for RS2001 to compare against at all.

Consequence for the task row's own framing: the row anticipated "the analyzer release-tracking files are
format-checked by the RS2000-family analyzers" as a reason for caution. That is true of *format*, but a
message change touches **neither the format nor anything those rules read**. The `Unshipped.md:8` Notes edit
in T1 is therefore a documentation improvement, not a compliance requirement — and it is safe: it changes the
Notes column only, leaving Rule ID, Category and Severity untouched.

## 5. The sweep — full resolution table

Only rules whose text names something checkable are listed; the remaining rules (`003`, `005`–`007`, `011`,
`012`, `016`–`019`, `021`, `027`–`032`, `034`, `037`, `039`–`046`) name only BCL types, C# keywords, external
rule ids (`CS4014`, `CA1068`, `CA1849`, `RS0030`) or prose, all of which resolve or are out of this
repository's control. Method column says how each was established.

| Rule | Named thing | Resolves? | Established by |
|---|---|---|---|
| `001` | `PooledBuffer<T>` | yes — `Sources/Main/Tensors/PooledBuffer.cs:39` | navigator + read |
| `001` | **`PooledArray`** | **NO** | `find_references` → no such symbol |
| `001` | `TensorStorage<T>` | yes — `Sources/Main/Tensors/Core/TensorStorage.cs` | read |
| `001` | `stackalloc` | keyword | — |
| `001` | *(implicit)* `PooledBuffer<T>` for reference element types | **not applicable** — `where T : struct` | read, F5 |
| `002` | `PooledBuffer<T>` / `TensorStorage<T>` | yes | as above |
| `002` | **"the MSBuild guard"** | **NO** — deleted 2026-08-05, now `OVERFIT033` | `Main.csproj:81-101` |
| `004` | `OverfitParallel` ("OverfitParallel-style") | yes — `Sources/Main/Runtime/OverfitParallel.cs` | navigator |
| `008` | `OverfitParallel.For` | yes — `OverfitParallel.cs:152` | grep of the declaration |
| `008` | `OverfitParallel.ForDecode` | yes — `OverfitParallel.cs:665` | `find_references` |
| `008` | `SuppressParallelismOnCurrentThread` | yes — `OverfitParallel.cs:482` | `find_references` |
| `008` (doc) | `docs/mnist-cnn-training-audit.md`, `docs/llamacpp-cpu-analysis.md` | yes, both exist | `ls` |
| `009` | `PooledBuffer<T>` | yes | navigator |
| `010` | `ValueStringBuilder` | yes — `DevOnBike.Overfit.Text.ValueStringBuilder`, `Sources/Main/Text/ValueStringBuilder.cs:55` | `find_references` |
| `013` | `IsEmpty`, `Interlocked` | BCL | — |
| `014` | `string.Equals`, `StringComparison.OrdinalIgnoreCase` | BCL | — |
| `015` | `CpuFeatures` (type) | yes — `DevOnBike.Overfit.Intrinsics.CpuFeatures` | `find_references` |
| `015` | **`CpuFeatures.Has{0}`** (composed) | **partially — fails for `Avx512F`, `Avx512BW`, `Sse2`, `AdvSimd`, nested `X64`, …** | read both files, F2 |
| `015` (desc) | namespace `DevOnBike.Overfit.Intrinsics` | yes, exact | navigator |
| `020` | `ReadOnlySpan<T>` / `Span<T>` | BCL | — |
| `022` | `Stack<T>` | BCL | — |
| `023` | "NASA Power of 10 rule 2" | external, correct | — |
| `024` | `OverfitEnvironment` | yes — `Sources/Main/Runtime/OverfitEnvironment.cs:14` | `find_references` |
| `024` (desc) | namespace `DevOnBike.Overfit.Runtime` | yes, exact | navigator |
| `024` (doc) | `overfit doctor` | yes — `Sources/Cli/Program.cs:328` | grep |
| `025` | `PooledBuffer<T>` | yes | navigator |
| `025`/`026` | `overfit_max_stackalloc_bytes` | yes — read at `StackallocSizeAnalyzer.cs:76,168-170`, documented at `.editorconfig:146` | read |
| `033` | `PooledBuffer<float>`, `TensorStorage<float>` | yes (`float` is a struct) | read |
| `033` (desc) | cross-ref to `OVERFIT002` | yes | `Unshipped.md:9` |
| `035`/`036` | `OverfitSchemas` (generated) | yes — emitted at `OverfitSchemasGenerator.cs:130` into `DevOnBike.Overfit.Schemas` | read |
| `038` | `BinaryReader`, `System.Text.Json`, `GetArrayLength` | BCL | — |
| `046` | "CS4014 … an error repo-wide" | yes — `Directory.Build.props:21` | read |
| `900` | `[OverfitHotPath]` | yes — `DevOnBike.Overfit.Diagnostics.OverfitHotPathAttribute`, `Sources/Main/Diagnostics/OverfitHotPathAttribute.cs:25` | `find_references` |
| *(README)* | `CpuFeatures.HasAvx2Fma` | yes — `CpuFeatures.cs:55` | read |
| *(README)* | `OverfitPerfAnalysis.LambdaCapturesEnclosingState` | **not verified** — the type exists (`OverfitPerfAnalysis.cs:14`); the member name was not confirmed against its declaration list. Low stakes, README only. See §7 |
| *(BannedSymbols)* | **`PooledBuffer<T>.RentArray` / `.ReturnArray`** | **NO** | `find_references` → no such symbol, F4 |

**Summary: four defects (F1–F4) across 47 descriptors plus the RS0030 banned-symbols text; one applicability
error (F5); everything else resolves.**

## 6. The guard — assessed, not accepted

The task row proposes: *"A test that greps every `messageFormat` for type names and asserts each resolves."*

### Where it can live, given the constraint

The user's constraint — **`Sources/Analyzers` must not depend on repository code** — is satisfied without
effort, because the dependency goes in the *test* project, which already references both sides:
`Tests/Tests.csproj:58` (Main) and `:95` (Analyzers as a **library**, with the reason spelled out in the
comment at `:88-93`). Nothing new is added to `Analyzers.csproj`. Direction of reference is
`Tests → {Main, Analyzers}`; `Analyzers → Main` stays absent.

**Enumeration must be by reflection over fields, not via `SupportedDiagnostics`.** Two descriptors —
`OVERFIT035`/`OVERFIT036` — live on `OverfitSchemasGenerator` (`:37,47`), an `IIncrementalGenerator` with no
`SupportedDiagnostics` at all, and `OverfitPerfAnalysis.HotPathRule` (`:20`) is `internal`. `Analyzers` has
**no `InternalsVisibleTo`** (checked: only `Main.csproj:45-53` declares any), so compile-time access is out —
but reflection with `BindingFlags.NonPublic | BindingFlags.Static` crosses that boundary at runtime and
reflection is permitted in tests. Walking every `static readonly DiagnosticDescriptor` field of every type in
the assembly yields all 47 and is the only enumeration that cannot silently miss a rule.

### How it avoids being a test that cannot fail

This is the real risk: a regex over prose that matches nothing passes forever, and this repository has been
burned by exactly that shape. Three controls, all cheap:

1. **Pin the extractor's yield.** Assert the extraction finds the known-present names — `PooledBuffer`,
   `TensorStorage`, `OverfitParallel`, `CpuFeatures`, `OverfitEnvironment`, `ValueStringBuilder`,
   `OverfitSchemas`, `OverfitHotPath` — from the real descriptors. If a refactor breaks the regex, this fails
   before the resolution assertion has a chance to pass vacuously.
2. **A negative control in the test itself.** A locally-constructed `DiagnosticDescriptor` whose message names
   `PooledArray` must be reported as unresolved by the same extractor+resolver pair the real assertion uses.
   This is the mutation, and it lives inside the test rather than being a manual step.
3. **Assert the resolver's own denominator.** `typeof(InferenceEngine).Assembly.GetTypes()` must return a
   non-trivial count before any lookup runs — an empty type list would make every name "resolve to nothing"
   or "not be checked", depending on polarity, and both read as green.

### False positives, and the scope that avoids them

The row's phrasing — "greps every `messageFormat` for type names" — is the version that does not work.
Messages legitimately contain `Span<T>`, `ReadOnlySpan<T>`, `IEquatable<T>`, `StringBuilder`,
`ConcurrentQueue`, `CancellationToken`, `Task`, `GC.SuppressFinalize`, `Array.Empty`, `Interlocked`,
`SynchronizationContext`, `MoveNext`/`Current`, the keywords `stackalloc`/`params`/`async void`, rule ids
(`OVERFIT046`, `CS4014`, `RS0030`, `CA1068`, `CA1849`), the editorconfig key `overfit_max_stackalloc_bytes`,
and format placeholders (`{0}`, `Has{0}`). A general CamelCase extractor has to carry a growing allow-list of
all of them, and every future message risks a false failure — the shape of guard nobody trusts and everybody
eventually suppresses.

**The tractable scope is the closed set of Overfit-owned names.** The sweep established, across all 47
descriptors, that this set has **nine** members and is stable: `PooledBuffer`, `PooledArray`, `TensorStorage`,
`FastTensor`, `CpuFeatures`, `OverfitParallel`, `OverfitEnvironment`, `OverfitSchemas`, `OverfitHotPath`, plus
`ValueStringBuilder`. Extract only identifiers matching `Overfit\w*`, `Pooled\w*`, or an explicit short list
(`TensorStorage`, `FastTensor`, `CpuFeatures`, `ValueStringBuilder`), resolve those against Main's type list,
and ignore everything else. False-positive rate: zero on today's text, and a new Overfit type named in a
message is exactly the case you want checked.

### Verdict

**Build it, in the narrow form. Do not build the general form.** Reasons, and the honest limits:

- It would have caught **F1**, three months earlier, at zero marginal cost per build.
- Extended to `BannedSymbols.txt` (T5) it would also catch **F4**, which is arguably the worse defect since
  RS0030 is an error in Main.
- It would **not** have caught **F2** (the name is composed at report time from a symbol the analyzer sees;
  no static text contains `HasAvx512F`) or **F3** ("the MSBuild guard" is prose about a mechanism, not an
  identifier). **That is two of the four findings — the majority of what the sweep actually produced.**
- So the guard covers one sub-class of the defect, not the class. It should be built and its limit written
  into its own doc comment, so nobody later reads a green suite as "diagnostic advice is verified".

The one thing that would catch F2's shape is structural rather than textual: **do not compose member names in
a message**. That is a fix (T2), not a guard, and it is cheaper than any check that would have found it.

## 7. What is not settled fact

| # | Type | Statement |
|---|---|---|
| 1 | Fact | `PooledArray` does not exist. `find_references` returns *"No source symbol named 'PooledArray'"*; `ROADMAP-COMPLETED.md:2007` records the 2026-05-29 deletion |
| 2 | Fact | Five live sites name it, one more than the task row lists — `.editorconfig:59` is the extra |
| 3 | Fact | `PooledBuffer<T>.RentArray`/`.ReturnArray` do not exist (`find_references`); `PooledBuffer.cs` read in full has only ctor/`Span`/`Memory`/`Length`/`IsAllocated`/`Dispose` |
| 4 | Fact | `PooledBuffer<T>` is `where T : struct` (`PooledBuffer.cs:39-40`) |
| 5 | Fact | The MSBuild guards were removed 2026-08-05 and replaced by `OVERFIT033`/`OVERFIT034` (`Main.csproj:81-101`) |
| 6 | Fact | No test asserts on message text — `AnalyzerHarness.cs:90-93` returns ids; grep of `Tests/` for `GetMessage`/`MessageFormat`/`Allocates 'new` finds nothing |
| 7 | Fact | The RS2000 family reads Category and Severity, never message text — RS2001's own string, extracted from the 5.6.0 package binary, names *"'Category' or 'Severity'"* and nothing else. `Shipped.md` is empty, so RS2001 has no baseline to compare against |
| 8 | Fact | `Tests` already references `Analyzers` as a library (`Tests.csproj:95`), so T4 needs no change to `Analyzers.csproj` and the no-dependency constraint holds |
| 9 | Fact | `Analyzers` declares no `InternalsVisibleTo`, so T4's enumeration must use reflection with `BindingFlags.NonPublic` |
| 10 | Assumption | The `OVERFIT015` message reaches `Avx512F`/`Sse2`/nested-`X64` inputs in practice. The *analyzer path* is certain (`{0}` is `ContainingType.Name`, `:70`); how often a developer trips it is not. **If unanswered: fixed anyway** — the fix is one string and does not depend on frequency |
| 11 | Assumption | Changing `Unshipped.md`'s Notes column alone triggers no RS2000-family diagnostic. Follows from finding 7, but was **not compiled** (machine busy). Cheapest retirement: T1's build |
| 12 | Assumption | The sweep is complete because 47 descriptors were extracted and 47 rule ids are listed in `Unshipped.md`. A descriptor built somewhere other than a `static readonly` field initialiser would have been missed by the extraction regex; none was seen, and the counts agreeing is the evidence |
| 13 | Open question (developer) | `OverfitPerfAnalysis.LambdaCapturesEnclosingState`, cited at `Sources/Analyzers/README.md:54`. The type exists; **I did not confirm the member name**. One `find_references` settles it |
| 14 | Open question (developer) | Do `Sources/Anomalies` and `Sources/Main` need two copies of `BannedSymbols.txt`, or can one be linked? Out of scope here; it is why F4 has two sites |
| 15 | Risk | A future message names a new Overfit type and the T4 allow-list does not cover its prefix, so the guard silently skips it. Cheapest retirement: control (1) in §6 asserts the extracted-name count, so adding a name without extending the list moves the number and fails |
| 16 | Risk | Someone reads a green T4 as "all diagnostic advice is verified". Cheapest retirement: the limit (F2/F3 uncovered) goes in the test's own doc comment, not only in this plan |
| 17 | Constraint | `Sources/Analyzers` must not reference repository code (user, standing) |
| 18 | Constraint | `.claude/**` and `CLAUDE.md` are user-owned; this plan reports their stale copies and does not change them |
| 19 | Constraint | Nothing in this plan was built or tested — the machine-exclusion mutex was held throughout |

## 8. Scope

**In scope**

- The five `PooledArray` sites (F1).
- `OVERFIT015`'s composed member name (F2).
- `OVERFIT002` + `Sources/Analyzers/README.md`'s MSBuild-guard prose (F3).
- `OVERFIT001`'s description, for the `where T : struct` applicability gap (F5).
- One narrow guard test (§6).
- `BannedSymbols.txt` × 2 and its four doc copies (F4) — **Should**, pending Q1.

**Won't (this time)**

| Not doing | Why |
|---|---|
| The general "grep every `messageFormat` for type names" guard | §6: false-positive story is unbounded, and it catches strictly less than the narrow form catches reliably |
| Editing `.claude/agents/*.md`, `.claude/skills/**` | User-owned. Three stale `PooledArray` copies reported in F1 |
| Editing `CLAUDE.md`'s "Source-file guards (MSBuild, build-time errors)" section | User-owned. Stale per F3; reported, and it is Q4 |
| Adding `HeapArrayAllocationAnalyzerTests` (rule behaviour) | A real gap — `OVERFIT001` has no analyzer test — but a different task. File it separately |
| Renaming `CpuFeatures` fields to match intrinsic class names | Option (b) under T2; touches `Sources/Main`, wider blast radius, and does not fix nested types. Recommended against, but it is the architect's call |
| Consolidating the two `BannedSymbols.txt` copies | Finding 14; a design question, not message rot |
| Any change to what a rule *detects* | This task is text only. A message edit must not change behaviour |

## 9. Priorities (MoSCoW)

| | Task | Rationale |
|---|---|---|
| **Must** | T1 — `PooledArray` in five sites + `OVERFIT001` description (F1, F5) | The filed defect. Every OVERFIT001 diagnostic names a dead type |
| **Must** | T2 — `OVERFIT015`'s composed member name (F2) | Same class, currently mis-advising on every AVX-512 and nested-type site |
| **Must** | T3 — MSBuild-guard prose in `OVERFIT002` + `README.md` (F3) | Names a mechanism deleted ten days ago; points a reader at `Main.csproj` for a task that is not there |
| **Should** | T4 — the narrow guard test (§6) | Turns F1's sub-class into a build failure. Explicitly does not cover F2/F3 |
| **Should** | T5 — `BannedSymbols.txt` `RentArray`/`ReturnArray` + four doc copies (F4) | Worse than F1 by exposure (RS0030 is an error in Main), but outside the row's stated scope — Q1 |
| **Won't** | see §8 | |

## 10. Tasks, as user stories with acceptance criteria

Every `Then` below is checkable by reading a file or running one command. None requires a benchmark.

---

### T1 — `OVERFIT001` stops naming a deleted type

> **As** a developer whose build just emitted OVERFIT001, **I want** the suggested replacement to be a type I
> can actually use, **so that** I can act on the diagnostic without checking whether it is true.

**Exact replacement text, all five sites:**

`Sources/Analyzers/HeapArrayAllocationAnalyzer.cs:35` —

```
            messageFormat: "Allocates 'new {0}[]' on the heap in per-call code — use PooledBuffer<T> (pooled scratch), TensorStorage<T> (tensor data), or stackalloc (small fixed-size)",
```

`Sources/Analyzers/HeapArrayAllocationAnalyzer.cs:16-18` (XML doc, the three lines as a block) —

```
    /// The Overfit hot-path contract is "no hidden allocations in inference": scratch memory comes
    /// from <c>PooledBuffer&lt;T&gt;</c> (pooled, value element types only — it is constrained
    /// <c>where T : struct</c>), long-lived tensor data from
    /// <c>TensorStorage&lt;T&gt;</c>, and small fixed-size scratch from <c>stackalloc</c>. A bare
```

`Sources/Analyzers/HeapArrayAllocationAnalyzer.cs:39` (description, F5) —

```
            description: "Per-call heap array allocations cause GC pressure on hot paths. Rent pooled memory or stackalloc instead; one-time allocations (field initializers, constructors, static constructors) are not flagged. Note that PooledBuffer<T> is constrained to value element types, so a per-call array of a reference type needs restructuring rather than pooling.");
```

`Sources/Analyzers/AnalyzerReleases.Unshipped.md:8` — Notes column only; Rule ID, Category and Severity
unchanged —

```
OVERFIT001 | Performance | Warning | Heap array allocation in per-call code — use PooledBuffer/TensorStorage/stackalloc
```

`Sources/Analyzers/README.md:12` —

```
| `OVERFIT001` | Heap array allocation (`new T[n]`, `new[]{…}`, `[…]` targeting an array) in **per-call** code — use `PooledBuffer<T>` (pooled scratch, value element types), `TensorStorage<T>` (tensor data) or `stackalloc` (small fixed-size) | ✅ shipped |
```

`.editorconfig:59` —

```
# OVERFIT001 (per-call heap array allocation → PooledBuffer<T>/TensorStorage<T>/
```

**Acceptance criteria**

- **Given** the working tree after T1, **When** `rg "PooledArray" Sources/ docs/ .editorconfig CLAUDE.md CONTRIBUTING.md` runs, **Then** the only matches are historical records that must not change — `ROADMAP-COMPLETED.md:2007`, `docs/TASKS.md:183` and `docs/specs/*.md` — and no match is in `Sources/Analyzers`, `.editorconfig` or any live guidance text.
- **Given** the edited analyzer, **When** `dotnet build -c Release` runs, **Then** it succeeds with **no RS2000-family diagnostic** (this is also the cheapest retirement of finding 11).
- **Given** the edited analyzer, **When** `dotnet test -c Release --filter "FullyQualifiedName~Analyzers"` runs, **Then** it is green — expected trivially, per F6, and it is the check that F6 was right.
- **Given** `HeapArrayAllocationAnalyzer.cs`, **When** its `Initialize`, `AnalyzeArrayCreation`, `AnalyzeCollectionExpression`, `ReportIfPerCall` and `IsOneTimeAllocationContext` are diffed, **Then** they are byte-identical to `HEAD` — the task changes text, never behaviour.

---

### T2 — `OVERFIT015` stops composing a member name that may not exist

> **As** a developer who wrote `Avx512F.IsSupported`, **I want** the diagnostic to point me at a facade that
> exists rather than at `CpuFeatures.HasAvx512F`, **so that** following the advice compiles.

**Two options. The architect decides; I recommend (a).**

**(a) Recommended — stop composing the name.** `CpuFeaturesGateAnalyzer.cs:30`:

```
            messageFormat: "Direct '{0}.{1}' — gate ISA and vector-width checks through CpuFeatures (see DevOnBike.Overfit.Intrinsics.CpuFeatures for the declared flags): one audit point, composed flags, same JIT constant-folding",
```

Cost: one string. Loses a little convenience for the eight inputs where the composed name was right; the
developer opens one file instead. Cannot go stale, because it names a type and not a member.

**(b) Rejected but recorded — rename the fields so composition holds.** Make `HasAvx512` → `HasAvx512F` and
`HasAvx512Bw` → `HasAvx512BW` in `Sources/Main/Intrinsics/CpuFeatures.cs`, updating callers. **Does not fix
the problem**: `Sse2`, `AdvSimd` and every nested `X64`/`V64` type still compose names for fields that will
never exist, since `CpuFeatures` deliberately declares only the ISAs the library uses (`CpuFeatures.cs:35-46`
records that intent). It also touches `Sources/Main` and every ISA call site — `find_references` on
`CpuFeatures` returns 43 references — for no complete gain.

**Acceptance criteria**

- **Given** option (a) applied, **When** `rg "Has\{0\}" Sources/Analyzers` runs, **Then** there is no match.
- **Given** the edited analyzer, **When** `dotnet build -c Release` runs, **Then** it succeeds and no OVERFIT015 site in the tree changes severity or location.
- **Given** `CpuFeaturesGateAnalyzer.cs`, **When** `AnalyzePropertyReference` and `IsHardwareIntrinsicsType` are diffed against `HEAD`, **Then** they are unchanged.

---

### T3 — `OVERFIT002` and the analyzer README stop citing a deleted MSBuild guard

> **As** a reader of OVERFIT002 who wants to know why `float[][]` is banned, **I want** to be sent to the rule
> that actually bans it, **so that** I do not open `Main.csproj` looking for a task that was deleted.

**Exact replacement text**

`Sources/Analyzers/JaggedArrayAllocationAnalyzer.cs:36` — final sentence of the description:

```
            description: "Jagged arrays cost one allocation per row plus the outer array and defeat cache locality. Per-call code should use a flat array sliced per row; one-time structures (field initializers, constructors) are not flagged. float[][] is banned in Sources/Main entirely by OVERFIT033.");
```

`Sources/Analyzers/README.md:3-6` — the intro, second clause:

```
In-repo **Roslyn performance analyzers** for `Sources/Main` — the second guard layer next to
`BannedSymbols.txt` (named APIs, RS0030). The structural guards that were MSBuild tasks until 2026-08-05
(jagged arrays, one-type-per-file) are now analyzer rules themselves, `OVERFIT033` and `OVERFIT034`.
Wired into `Main.csproj` as an analyzer reference (`OutputItemType="Analyzer"`,
nothing ships); tests and benchmarks are deliberately NOT covered.
```

`Sources/Analyzers/README.md:13` — parenthetical:

```
| `OVERFIT002` | Jagged array allocation (`new T[n][]` — N+1 heap objects, pointer-chase per row) in per-call code — flat `T[]` Span-sliced per row (`float[][]` is a build ERROR via `OVERFIT033` regardless) | ✅ shipped |
```

**Acceptance criteria**

- **Given** the working tree after T3, **When** `rg -i "msbuild guard" Sources/Analyzers` runs, **Then** there is no match.
- **Given** `Sources/Analyzers/README.md`, **When** a reader follows the `float[][]` claim, **Then** it names `OVERFIT033`, which `AnalyzerReleases.Unshipped.md:41` confirms exists.
- **Given** T3 is complete, **When** the developer reports, **Then** the report states that `CLAUDE.md`'s "Source-file guards (MSBuild, build-time errors)" section carries the same stale claim and was **not** edited because it is user-owned (Q4).

---

### T4 — a test that fails when a diagnostic names an Overfit type that does not exist

> **As** `Tests`, **I want** to resolve every Overfit-owned identifier named in every analyzer descriptor
> against the `DevOnBike.Overfit` assembly, **so that** deleting a type breaks the build of the advice that
> recommends it instead of going unnoticed for three months.

New file, `Tests/Analyzers/DiagnosticMessageNamesResolveTests.cs`. Shape (the developer owns the
implementation; these are the constraints, not the code):

1. Enumerate `typeof(HeapArrayAllocationAnalyzer).Assembly.GetTypes()`, and for each, every `static` field of
   type `DiagnosticDescriptor` with `BindingFlags.Public | NonPublic | Static`. **Not** via
   `SupportedDiagnostics` — that misses `OverfitSchemasGenerator`'s two (§6).
2. From each descriptor's `Title`, `MessageFormat` and `Description` (`.ToString()` on the
   `LocalizableString`), extract identifiers matching `Overfit\w*` or `Pooled\w*`, plus the literals
   `TensorStorage`, `FastTensor`, `CpuFeatures`, `ValueStringBuilder`. Strip a trailing `<…>`.
3. Exclude, by name, the things that are deliberately not types: rule ids `OVERFIT\d{3}`, the analyzer's own
   helper `OverfitPerfAnalysis`, the editorconfig key `overfit_max_stackalloc_bytes`, and the attribute
   spelling `OverfitHotPath` (resolve it as `OverfitHotPathAttribute`).
4. Resolve each against `typeof(DevOnBike.Overfit.Inference.InferenceEngine).Assembly.GetTypes()` by
   `Name`, tolerating the arity suffix (`PooledBuffer\`1`).

**Acceptance criteria**

- **Given** the tree as of this plan **minus** T1, **When** the test runs, **Then** it **fails**, naming `PooledArray` and the descriptor it came from. *(This is the mutation. If it passes on unfixed source, the test is worthless — run it before T1 or with T1 temporarily reverted.)*
- **Given** the tree after T1–T3, **When** the test runs, **Then** it passes.
- **Given** a locally-constructed `DiagnosticDescriptor` in the test whose `messageFormat` names `PooledArray`, **When** it is passed through the same extractor and resolver, **Then** it is reported unresolved — the negative control, so a broken regex cannot pass vacuously.
- **Given** the real descriptors, **When** the extractor runs, **Then** it yields **at least** `PooledBuffer`, `TensorStorage`, `OverfitParallel`, `CpuFeatures`, `OverfitEnvironment`, `ValueStringBuilder`, `OverfitSchemas`, `OverfitHotPath`, and **at least 40** descriptors were enumerated — pinning the denominator, so a regex or reflection change that stops matching fails loudly.
- **Given** the resolver, **When** it starts, **Then** it asserts the Main type list is non-empty before any lookup.
- **Given** the finished test, **When** it is read, **Then** its doc comment states plainly that it covers **only** statically-written Overfit-owned names, and covers **neither** names composed at report time (the `OVERFIT015` shape, F2) **nor** prose describing a mechanism (the `OVERFIT002` shape, F3).
- **Given** `Sources/Analyzers/Analyzers.csproj`, **When** diffed against `HEAD`, **Then** it is unchanged — the no-dependency constraint (finding 17) holds by construction.
- **Given** the test, **When** `dotnet test -c Release` runs, **Then** it is not `[LongFact]` — it must be well under a second (two `GetTypes()` calls and ~47 string scans).

---

### T5 — the RS0030 message stops naming `RentArray`/`ReturnArray` *(Should — see Q1)*

> **As** a developer whose build failed with RS0030 on `ArrayPool<T>.Shared`, **I want** the sanctioned
> alternative it names to exist, **so that** I can fix the error from the message alone.

**Exact replacement**, in **both** `Sources/Main/BannedSymbols.txt:9` and
`Sources/Anomalies/BannedSymbols.txt:9` (the line must stay one line — the file format is one banned symbol
per line):

```
P:System.Buffers.ArrayPool`1.Shared; Use PooledBuffer<T> (Tensors/PooledBuffer.cs) — the project's single audit point over ArrayPool<T>.Shared. Two lifetimes, one type: (1) scoped `using var buf = new PooledBuffer<T>(n, clearMemory: false);` for method-local scratch (including inside a Parallel.For lambda body); (2) class field — rent in the constructor, Dispose() in the owner's Dispose (FastTensor, TensorStorage). 2026-05-29 PoolComparisonBenchmark: .Shared is 3× faster single-thread AND ~3000× faster than ArrayPool.Create-based pools under multi-thread contention (TLS per-CPU caches vs lock-per-bucket). Wrapper overhead is ~0%.
```

The four doc copies take the same correction: `CONTRIBUTING.md:32`, `docs/code-patterns.md:27`,
`Sources/Main/Tensors/README.md:17`, `Sources/Analyzers/README.md:121`. The code comment at
`Sources/Main/Onnx/OnnxGraphImporter.cs:156` becomes
`// TensorStorage rents from ArrayPool<T>.Shared (via its PooledBuffer<T> field)`, which is what
`TensorStorage.cs:21,50` actually does.

**Acceptance criteria**

- **Given** the tree after T5, **When** `rg "RentArray|ReturnArray" --glob '!.claude/**' .` runs, **Then** matches remain only in `CHANGELOG.md` (historical) and `docs/specs/*.md`, and none is in a diagnostic message, a code comment or live guidance.
- **Given** both `BannedSymbols.txt` files, **When** diffed, **Then** line 9 is identical between them and the symbol part before the first `;` is unchanged.
- **Given** the edit, **When** `dotnet build -c Release` runs, **Then** it succeeds and RS0030 still fires where it fired before (the ban list itself is untouched).
- **NOT READY** if Q1 is unanswered — the scope box is unticked, not the criteria. Every other Definition-of-Ready box is ticked.

## 11. Value against cost

| Task | Value (client's metric) | Structural cost, and what I checked | Uncertain | Recommendation |
|---|---|---|---|---|
| T1 | `not stated`. Observable: one confirmed incident (a design built on the dead name, 2026-08-14) and an unknown number of unlogged developer detours | **Very low.** 5 string sites. No behaviour. No test asserts message text (`AnalyzerHarness.cs:90-93`). No release-file requirement (F7). No call sites move | Whether `Unshipped.md` Notes edits are truly free — finding 11, settled by T1's own build | **Do now.** Cheapest item in the plan and the one that was filed |
| T2 | `not stated`. Affects every AVX-512 and nested-intrinsics site that trips OVERFIT015 | **Very low** for option (a): 1 string. **Medium** for (b): `find_references` on `CpuFeatures` returns 43 references, and (b) still does not fix nested types | How often the failing inputs occur (finding 10) — does not change the fix | **Do now, option (a).** Frequency cannot make a wrong name right |
| T3 | `not stated`. A reader sent to a file where the mechanism no longer is | **Very low.** 3 prose sites | None | **Do now.** It rots further with every `Main.csproj` change |
| T4 | Prevents recurrence of F1's sub-class only. **Explicitly does not cover F2 or F3 — two of the four findings** | **Low.** One test file. No production change, no `Analyzers.csproj` change (`Tests.csproj:95` already has the reference). Needs no new oracle — the assembly's own type list is the oracle. Reflection needs `BindingFlags.NonPublic` because `Analyzers` has no `InternalsVisibleTo` | Whether the allow-list stays closed as messages are added — retired by the count assertion (risk 15) | **Do, after T1–T3**, so its first run is the mutation. Build the narrow form; the general form is a false-positive treadmill |
| T5 | `not stated`. Higher exposure than T1 — RS0030 is an **error** in Main, so this text is read at a hard stop | **Very low.** 2 message files + 4 doc copies + 1 comment. No code | Scope only (Q1) | **Do now if Q1 says yes.** Same class, worse exposure, same cost |
| *(not proposed)* General messageFormat guard | Would catch nothing the narrow form misses | Unbounded allow-list maintenance | — | **Probably not worth it.** §6 |
| *(not proposed)* `HeapArrayAllocationAnalyzerTests` | Real gap: `OVERFIT001` has no analyzer test | Low, but it is rule-behaviour work | — | **File separately.** Out of scope; do not fold rule-behaviour testing into a text fix |

## 12. Ordering

Highest uncertainty first, correctness before durability:

1. **T2** — the only finding whose *mechanism* the sweep had to derive rather than read. Fix it while §5's table is fresh.
2. **T1** — the filed defect. Its build also retires finding 11 (RS2000-family behaviour on a Notes edit).
3. **T3** — independent prose.
4. **T5** — if Q1 says yes; independent of 1–3.
5. **T4** — last, deliberately. **Run it once against unfixed source first** (or with T1 reverted) so its failure is observed. A guard whose red state nobody has seen is a guard nobody has tested.

No task depends on another's output; the ordering is about evidence, not compilation. Nothing here can be
partially applied in a way that leaves the tree broken.

## 13. Traceability

| Goal | User need | Task | Acceptance criterion | Verified by |
|---|---|---|---|---|
| Diagnostics stay trustworthy | Follow OVERFIT001's advice without checking it | T1 | `rg PooledArray` finds only historical records | grep + build + analyzer suite |
| Diagnostics stay trustworthy | Follow OVERFIT015's advice on an AVX-512 site | T2 | `rg "Has\{0\}"` finds nothing | grep + build |
| Diagnostics stay trustworthy | Find the rule that bans `float[][]` | T3 | `rg -i "msbuild guard"` in `Sources/Analyzers` finds nothing | grep |
| The class stops being found by accident | Deleting a type breaks the advice that recommends it | T4 | Fails on unfixed source naming `PooledArray`; passes after; negative control reported; ≥40 descriptors enumerated | the test itself, run twice |
| Diagnostics stay trustworthy | Fix an RS0030 error from its message | T5 | `rg RentArray` finds only history | grep + build |

## 14. Definition of Ready

| Task | AC written & testable | Oracle named | Execution path | Dependencies | Internally consistent | Independently verifiable | Ready? |
|---|---|---|---|---|---|---|---|
| T1 | yes | grep + build + `Tests/Analyzers` suite | n/a — no engine code | none | yes | yes | **READY** |
| T2 | yes | grep + build | n/a | none | yes | yes | **READY** — pending the architect's choice between (a) and (b), which is recorded as an option, not a gap |
| T3 | yes | grep | n/a | none | yes | yes | **READY** |
| T4 | yes | its own red run on unfixed source | n/a — test project, not AOT | run after T1–T3 | yes | yes | **READY** |
| T5 | yes | grep + build | n/a | none | yes | yes | **NOT READY** — scope box unticked (Q1) |

## BLOCKING QUESTIONS

**For the client (the user / the main session):**

1. **Is F4 in scope?** `Sources/Main/BannedSymbols.txt:9` and `Sources/Anomalies/BannedSymbols.txt:9` tell
   developers to use `PooledBuffer<T>.RentArray(n)` / `.ReturnArray(arr)`, which do not exist
   (`find_references`: no such symbol). Same defect class as XC-51, higher exposure — RS0030 is an *error* in
   Main. The task row scopes XC-51 to `Sources/Analyzers`.
   *If unanswered, I assume **yes** and T5 ships with T1–T3* — same cost, same class, and leaving a known-dead
   name in an error message to respect a scope boundary is the worse outcome.

2. **`OVERFIT015` — option (a) or (b)?** (a) drop the composed `CpuFeatures.Has{0}` from the message (one
   string; recommended). (b) rename `CpuFeatures.HasAvx512` → `HasAvx512F` and `HasAvx512Bw` → `HasAvx512BW`
   so composition holds for those two (43 references to `CpuFeatures`, and it still leaves `Sse2`, `AdvSimd`
   and nested `X64` composing dead names).
   *If unanswered, I assume **(a)**.*

3. **Build T4 at all, knowing it covers one sub-class?** It would have caught F1 three months earlier and
   costs one test file; it would **not** have caught F2 or F3 — two of the four findings. It is a Should, not
   a Must, and dropping it is a defensible call.
   *If unanswered, I assume **yes**, in the narrow form of §6.*

4. **Who fixes the two user-owned copies?** `CLAUDE.md`'s "Source-file guards (MSBuild, build-time errors)"
   section describes `BanJaggedFloatArrays` / `BanMultipleTopLevelTypes` as live MSBuild tasks; they were
   deleted 2026-08-05 (`Main.csproj:81-101`). `.claude/agents/overfit-architect.md:253`,
   `.claude/agents/overfit-developer.md:223` and `.claude/skills/overfit-spec/SKILL.md:128,197` still name
   `PooledArray`. Neither is mine or the developer's to edit.
   *If unanswered, I assume **the user does it**, and T3's report names them so they are not lost.*

**For the developer / architect, to challenge:**

- Finding 13 — I did **not** confirm `OverfitPerfAnalysis.LambdaCapturesEnclosingState`
  (`Sources/Analyzers/README.md:54`). One `find_references` settles it; if it is stale, fold the fix into T3.
- Finding 11 — I assert from the RS2001 resource string that a Notes-column edit triggers nothing. **Not
  compiled.** T1's build is the check; if RS2000-anything fires, stop and re-open this.
- Finding 12 — the sweep's completeness rests on 47 extracted descriptors matching 47 rule ids. A descriptor
  constructed anywhere other than a `static readonly` field initialiser would have been missed. T4's
  reflection-based enumeration is the durable answer to that and is a reason to prefer it over the regex.
- T4's exclusion list (§6 step 3) is my judgement from today's text. If any exclusion looks like it is hiding
  a real check rather than a false positive, say so — that is the failure mode this test has.

## SUGGESTED IMPROVEMENTS TO MY ROLE

Two, both with an incident in this run.

1. **My instructions send me to `find_implementations` and the ROADMAP for the capability map, but nothing
   tells me to check that a *message* names a live symbol.** This run found four instances of one defect class
   in an afternoon, three of them previously unknown, using only `find_references` on names read out of
   strings. That is a cheap, repeatable pass — "resolve every repo-owned identifier named in any
   user-facing string of the component you are analysing" — and it belongs in round one next to the
   inventory, because prose that names symbols is exactly where the navigator beats reading.
   *Smallest change:* one line in the round-one section.

2. **My "Run commands through your own `do-overfit-analyst.py`" rule cost nothing here but nearly hid
   something.** Extracting the RS2001 message from the packaged DLL (F7) settled a question the plan would
   otherwise have carried as an assumption, and it worked only because the script could do binary reads and
   regex without shell quoting. That is the rule paying off, and it is worth recording as a *positive*
   example in the rule's own text — every incident cited there today is a failure, which makes the rule read
   as defensive rather than as useful.
   *Smallest change:* add the one-line example to the rule's list of what it buys.

Nothing else this run. No missing tool, no boundary problem: the machine-busy constraint was real and is
recorded in the plan as two open items rather than papered over.

---

# 15. Architecture review — `overfit-architect`, 2026-08-15

**Verdict: APPROVED**, with six additions to the site lists (§A) and seven decisions (§D) the developer must
follow. The plan's structure is right, its sweep method is right, and its two hardest calls — the narrow guard
over the general one, and option (a) for `OVERFIT015` — are both correct. Everything below either extends a
list the sweep under-counted or settles a wording question the developer would otherwise have to invent.

**Short-form answers a developer cannot infer:**

> **Execution path:** neither — this is analyzer text plus one xUnit test; no inference and no training code
> is touched. **AOT-reachable:** no — `Sources/Analyzers` targets `netstandard2.0` and never ships
> (`Analyzers.csproj` pins `PublishAot/PublishTrimmed/SelfContained` false, `IsPackable=false`), and `Tests`
> is not AOT-published. **Allocation policy:** neither. **Assembly boundaries:** unchanged — no
> `ProjectReference` is added anywhere; `Analyzers -> Main` stays absent. **Public API surface:** unchanged.
> **Moat side:** open; nothing here touches the gateway.

**How everything below was established.** Files read, `git log -S`, and `mcp__overfit-navigator__*` for
symbols. **Nothing was compiled or executed** — the machine-exclusion mutex was held throughout this pass too.
Section §G lists exactly what a build must still settle.

## A. Review findings

### A1 — the `PooledArray` site list is five, not four, and the sixth site is the one that caused the incident

`Sources/Main/LanguageModels/Loading/README.md:29-30`, live text inside `Sources/Main`:

> "…do not stage a file through a scratch `byte[]` when it can be read into its destination. **`PooledArray`
> in `../../Runtime` is the `using`-shaped wrapper the `try/finally` rental sites here were swept onto.**"

Both halves are wrong: the type does not exist, and `PooledBuffer<T>` lives in `Sources/Main/Tensors/`, not
`Runtime/`. This is textually the *"`using`-shaped wrapper precedent"* that §3 says was cited on 2026-08-14 and
disproved only by a `Glob`. **Add it to T1.** Found by a repo-wide scan; the task row's own note
("three pure-documentation copies were corrected the same day") missed it.

### A2 — `PooledBuffer<T>.RentArray` / `.ReturnArray` were real API, removed 2026-06-21

`git log -S RentArray -- Sources/Main/Tensors/PooledBuffer.cs` returns two commits:

| commit | date | what it did |
|---|---|---|
| `2584375` | 2026-05-30 | **added** `public static T[] RentArray(int minimumLength) => ArrayPool<T>.Shared.Rent(minimumLength);` and `public static void ReturnArray(T[] array)` |
| `718d6f8` | 2026-06-21 | **removed** both, with their `<see cref>`s |

So F4's *"it describes a mode of use the type has never had in its present form"* is correct about today and
wrong about history. This is not pedantry — it decides which sites may be edited (§A3, §D2).

### A3 — T5's acceptance criterion, as written, forces the destruction of three historical records

Three `RentArray` sites are **records of work done while the method existed**, not advice:

| site | what it is | written |
|---|---|---|
| `.editorconfig:252` | *"Runtime/ ESCALATED 2026-06-12 … scratch pooled via `PooledBuffer.RentArray` with exact-length slices — measured 748 MB -> ~1 MB per 272-token prefill"* | 2026-06-12, method existed |
| `.editorconfig:259` | *"Whisper/ ESCALATED 2026-06-13 … pooled via `PooledBuffer.RentArray` with exact-length slices"* | 2026-06-13, method existed |
| `Sources/Analyzers/README.md:121` | the 2026-06-12 triage log — *"BatchNorm backward `sumDy`/`sumDyX` → `RentArray`/`ReturnArray` + try/finally"* | 2026-06-12, method existed |

T5's first acceptance criterion says matches must remain *"only in `CHANGELOG.md` (historical) and
`docs/specs/*.md`"*. Applied literally that forces an edit to all three, which would rewrite an accurate
account of what a sweep did into a false one. Ruling in **D2**; the criterion is amended there.

### A4 — F3's site list is short by two, both in files T3 already opens

- `docs/code-patterns.md:29` — the "use this instead" table still enforces jagged `float[][]` with
  *"MSBuild task `OVERFIT-JAGGED`"*. Same deleted mechanism as F3, live guidance, not in the plan's list.
- `Sources/Analyzers/README.md:57` — *"**This table stops at `OVERFIT020` and the family now runs to
  `OVERFIT038`.**"* The family runs to `OVERFIT046` plus `OVERFIT900`: 47 ids in
  `AnalyzerReleases.Unshipped.md`, 47 `new DiagnosticDescriptor(` sites in `Sources/Analyzers`. Same rot class,
  same file, one number.

Both fold into T3 at no extra cost.

### A5 — "a message must not compose a member name" is the wrong rule, and would condemn six correct messages

All 47 descriptors were scanned for a format placeholder glued to an identifier. **Eight** compose; **seven are
safe by construction** and one is not:

| descriptor | composed | why it is safe / not |
|---|---|---|
| `ConcurrentCollectionCountAnalyzer.cs:32` | `'{0}.Count'` | echo of the developer's own code |
| `NullPatternAnalyzer.cs:53` | `is {0}null` | echo |
| `RawParallelForAnalyzer.cs:32` | `Parallel.{0}` | echo |
| `ToArrayAnalyzer.cs:28` | `{0}.ToArray()` | echo |
| `SynchronousIslandAnalyzer.cs:62` | `'{1}Async'` | **resolved before reporting** — `:175` does `GetMembers(called.Name + "Async")` and reports only after `ReturnsAwaitable`, `ParametersMatch` and `ResultMatches` all pass |
| `AsyncApiConventionAnalyzer.cs:42` | `'{0}Async'` | a name the **fix will create**; its non-existence is the diagnostic |
| `OverfitSchemasGenerator.cs:40` | `OverfitSchemas.{0}` | the same generator emits the constant |
| **`CpuFeaturesGateAnalyzer.cs:30`** | **`CpuFeatures.Has{0}`** | **asserts that an unresolved symbol exists.** `{0}` is `property.ContainingType.Name` (`:70`), and nothing looks `Has{0}` up |

So the defect is **existence assertion without resolution**, not composition. The rule is written accordingly
in **D3**.

### A6 — T4's enumeration is provably complete today, and the id-based alternative provably is not

| | |
|---|---|
| `new DiagnosticDescriptor(` sites in `Sources/Analyzers` | **47**, and **every one** is a `static readonly` field — no descriptor is built in a method, a property or a collection initialiser |
| rule ids in `AnalyzerReleases.Unshipped.md` | **47**, and the two sets agree exactly |
| ids with **no** `public const string DiagnosticId` | **3** — `OVERFIT035`, `OVERFIT036` (on `OverfitSchemasGenerator`) and `OVERFIT900` (`OverfitPerfAnalysis.HotPathRule`, `internal`) |

The third row is the evidence for §6's choice: an enumeration keyed on `DiagnosticId` consts or on
`SupportedDiagnostics` misses three rules silently, reflection over `static` fields misses none. Confirmed by
scanning the source, not by running the test.

### A7 — the analyst's open question 13 is retired: the member exists

`Sources/Analyzers/OverfitPerfAnalysis.cs:160` declares
`internal static bool LambdaCapturesEnclosingState(IAnonymousFunctionOperation lambda)`. The
`Sources/Analyzers/README.md:54` citation resolves. Drop it from the open list; no edit needed.

### A8 — the plan's finding 8 holds; I checked the thing that would have made it false

`Tests.csproj:88-93` is a comment block and it **closes at `:93`**; `:95` is a live
`<ProjectReference Include="..\Sources\Analyzers\Analyzers.csproj" />` and `:96` adds
`Microsoft.CodeAnalysis.CSharp`. `Directory.Build.props:69-77` excludes `Tests` by name from the
`OutputItemType="Analyzer"` wiring, which is what makes the plain library reference legal. T4 needs no
`.csproj` change and the "`Sources/Analyzers` must not depend on repository code" constraint holds by
construction.

### A9 — F5 has no pooled answer, and the plan should say so rather than soften it

| candidate | constraint | usable for `new SomeClass[n]`? |
|---|---|---|
| `PooledBuffer<T>` | `where T : struct` (`PooledBuffer.cs:39-40`) | no |
| `TensorStorage<T>` | `where T : unmanaged` (`TensorStorage.cs:16-17`) | no |
| `stackalloc` | unmanaged element type | no |
| raw `ArrayPool<T>.Shared` | RS0030 **error** in `Main` | no |
| `OverfitResourcePool<T>` (`Sources/Main/Serving/OverfitResourcePool.cs:23`) | pools **instances**, not arrays | no |

There is no pooled option in this library for a per-call array of a reference type. Wording in **D1**.

## B. System context and boundaries

Nothing crosses an assembly boundary, so a diagram would carry no information the table does not.

| | |
|---|---|
| **What changes** | string literals in `Sources/Analyzers` (T1–T3), two `BannedSymbols.txt` files and their live doc copies (T5), one new file in `Tests/Analyzers/` (T4) |
| **Who consumes it** | the developer reading build output or an IDE squiggle. There is no programmatic consumer: `AnalyzerHarness.cs:90-93` returns ids, and no test in `Tests/` reads `GetMessage`/`MessageFormat` |
| **What depends on `Sources/Analyzers`** | every project except `Analyzers`, `Tests` and `AotSmokeTest`, via `Directory.Build.props:69-77` as `OutputItemType="Analyzer"`. `Tests` additionally references it as a **library** (`Tests.csproj:95`) |
| **Dependency direction after this change** | unchanged. `Tests -> {Main, Analyzers}`; `Analyzers -> {}` |
| **Source of truth for the rule set** | the 47 `static readonly DiagnosticDescriptor` fields. `AnalyzerReleases.Unshipped.md` is a parallel list kept in agreement by hand; `Shipped.md` is empty, so RS2001 has no baseline and RS2000-family tracking can never catch a **removed** rule (pre-existing, `XC-41`) |

## C. Quality requirements as parameters

| requirement | parameter | how measured | baseline |
|---|---|---|---|
| T4 stays in the fast suite | **< 1 s**, plain `[Fact]`, never `[LongFact]` | the suite's own per-test timing | two `GetTypes()` calls and 47 string scans; the fast-suite budget is `Tests/README.md`'s, not a new one |
| T4 cannot pass vacuously | **descriptor count >= 45** asserted (not 40) | the assertion itself | today's count is **47**, verified by source scan. A floor 7 below the truth survives a reflection change that silently drops six analyzers; a floor of 45 does not, and rules here are only ever added |
| T4 cannot pass vacuously (2) | the extracted-name set **contains** `PooledBuffer`, `TensorStorage`, `OverfitParallel`, `CpuFeatures`, `OverfitEnvironment`, `ValueStringBuilder`, `OverfitSchemas`, `OverfitHotPath` | the assertion itself | all eight verified present in §5's table |
| behaviour is unchanged | the named analyzer methods are **byte-identical to `HEAD`** | `git diff` | the analyst's ACs already say this for T1 and T2; extend the same clause to `JaggedArrayAllocationAnalyzer` in T3 |

No performance parameter exists in this plan and none should be invented for it.

## D. Decisions

### D1 — `OVERFIT001`'s message for a reference element type: say there is no pooled option

The honest answer to the team lead's question is that **there is none** (§A9). A message naming three types of
which none can compile is worse than one that names the restructure, so the message states the constraint and
the description carries the detail. **Replaces the analyst's T1 text for those two fields**; the other four T1
sites are unchanged from §10.

`Sources/Analyzers/HeapArrayAllocationAnalyzer.cs:35`:

```
            messageFormat: "Allocates 'new {0}[]' on the heap in per-call code — use PooledBuffer<T> (pooled scratch), TensorStorage<T> (tensor data) or stackalloc (small fixed-size); all three need a value element type, so a per-call array of a reference type must be hoisted or restructured away",
```

`Sources/Analyzers/HeapArrayAllocationAnalyzer.cs:39`:

```
            description: "Per-call heap array allocations cause GC pressure on hot paths. Rent pooled memory or stackalloc instead; one-time allocations (field initializers, constructors, static constructors) are not flagged. PooledBuffer<T> is constrained where T : struct, TensorStorage<T> where T : unmanaged, and stackalloc needs an unmanaged type — there is no pooled option for an array of a reference type, so the fix there is to allocate it once in a field or constructor (which this rule does not flag) or to remove the need for the array.");
```

The second sentence of the description matters more than it looks: the rule's **own one-time exemption is the
sanctioned escape**, so the advice is actionable and self-consistent rather than a dead end.

### D2 — live advice is corrected; a historical record is not

This is the general form of §A3 and it governs every site in T1 and T5.

> **A string that tells a developer what to do now is advice and must be corrected. A string that records what
> was done on a date is a record and must not be, even when it names something that no longer exists.**
> Correcting a record does not fix a defect; it deletes the evidence that a sweep happened and replaces it with
> a claim that is false about the past.

Classification, binding on the developer:

| site | class | action |
|---|---|---|
| `Sources/Main/BannedSymbols.txt:9`, `Sources/Anomalies/BannedSymbols.txt:9` | advice, at an RS0030 hard stop | **correct** (T5 text as written in §10 is accepted) |
| `CONTRIBUTING.md:32`, `docs/code-patterns.md:27`, `Sources/Main/Tensors/README.md:17` | advice | **correct** |
| `Sources/Main/Onnx/OnnxGraphImporter.cs:156` | a code comment describing current behaviour | **correct**, to the analyst's text |
| **`.editorconfig:252`, `.editorconfig:259`, `Sources/Analyzers/README.md:121`** | **record** (§A3) | **do not touch** |
| `.editorconfig:59` | advice — it is the policy header above the severity keys | **correct** (T1) |
| `Sources/Main/LanguageModels/Loading/README.md:29-30` | advice | **correct** (§A1) — name `PooledBuffer<T>` and its real location, `Sources/Main/Tensors/` |
| `ROADMAP-COMPLETED.md:2007`, `CHANGELOG.md:475`, `docs/TASKS.md`, `docs/specs/**` | record | do not touch (already the analyst's position) |

**T5's first acceptance criterion is amended to:**

> **Given** the tree after T5, **When** `rg "RentArray|ReturnArray" --glob '!.claude/**' .` runs, **Then** the
> remaining matches are exactly: `CHANGELOG.md`, `docs/specs/*.md`, **`.editorconfig:252`**,
> **`.editorconfig:259`** and **`Sources/Analyzers/README.md:121`** — the last three being dated records of the
> 2026-06-12/13 sweeps, which were accurate when written (§A2) — and none is in a diagnostic message, a live
> guidance table or a code comment describing current behaviour.

Note for whoever runs those greps: `.claude/pb12-old/` holds **686 files**, a stale copy of `Sources/Main`
including its `BannedSymbols.txt`. It is user-owned and gitignored; exclude `.claude/**` from every acceptance
grep, as T5 already does and T1 should.

### D3 — the composed-name rule becomes a standing repository rule, stated as existence assertion

**Yes, it becomes standing** — but not in the form "if a message names a member it must be a literal a test can
see". That form is unimplementable (`AsyncApiConventionAnalyzer` **must** compose the name it is asking for)
and would condemn six correct messages (§A5). The rule that fits the evidence:

> **A diagnostic message may compose an identifier from a format placeholder only when the composition is (a)
> echoing a symbol the developer wrote, (b) naming a symbol the analyzer resolved before reporting, or (c)
> naming a symbol the fix will create. A message must never assert that a symbol exists without the analyzer
> having resolved it — no test and no text search can catch that, because the name is never written down.**

**Where it is written:** `Sources/Analyzers/README.md`, in the existing **"Authoring notes"** list
(`:142-149`), as a fourth bullet beside `EnableConcurrentExecution` / `ConfigureGeneratedCodeAnalysis(None)` /
category and severity. That list is already the authoring contract for a new rule and is the only place a
future analyzer author is reading. **Adding this bullet is part of T3** (it edits the same file); do not create
a new document for one rule.

**Not an ADR** (see D5), and **not enforceable by T4** — that is D4.

### D4 — `CpuFeatures.Has{0}`: option (a), and option (c) recorded rather than taken

Option (a) as the team lead decided and the analyst recommended. A third option existed and is worth recording
so nobody re-derives it:

**(c) resolve the composed name.** `CpuFeaturesGateAnalyzer` could call
`compilation.GetTypeByMetadataName("DevOnBike.Overfit.Intrinsics.CpuFeatures")` in a compilation-start action
and emit `CpuFeatures.Has{0}` only when that member exists, falling back to naming the type otherwise. This is
**not** a dependency on repository code — it is a semantic lookup against the compilation being analysed, the
same mechanism `AsyncApiConventionAnalyzer.cs:132` already uses — so the standing constraint would hold.
**Not taken**, for two reasons: §8 scopes this task to text and forbids changing what a rule does, and (c)
needs its own analyzer test to be worth anything. It is the route back to the composed name if the convenience
is ever wanted; it is not this task.

**Do not rename the `CpuFeatures` fields.** Confirmed against the declaration list
(`Sources/Main/Intrinsics/CpuFeatures.cs`): the thirteen flags are `HasFma`, `HasDp`, `HasAvx`, `HasAvx2`,
`HasAvx512`, `HasAvx512Bw`, `HasAvxVnni`, `HasSse`, `HasSse3`, `HasVector128/256/512`, `HasAvx2Fma`. Renaming
two of them still leaves `Sse2`, `Ssse3`, `Sse41`, `AdvSimd` and every nested `X64` composing names that will
never exist, for 43 references of churn.

### D5 — no ADR, and the reason is on the record so it is not re-asked

`docs/adr/` holds one entry (`0001-peer-novelty-state-persistence-format.md`). This change is on none of the
irreversible list: it adds no public API, moves no capability between assemblies, changes no AOT reachability,
defines no on-disk or on-wire format, adds no dependency to `Main`, and does not touch the open/commercial
boundary. **D3's authoring rule is a repository convention, reversible by editing one bullet, and its home is
`Sources/Analyzers/README.md`.** Opening ADR 0002 for a diagnostic message would devalue the series.

### D6 — T4's red run is an ordered step with pasted evidence, not a hope

The analyst's §12 puts T4 fifth and says *"run it once against unfixed source first (or with T1 reverted)"*.
By step 5 there is nothing left to run it against, so the instruction depends on the developer remembering to
undo work. **That is not a gate.** Binding order:

1. **T4 first.** Write `Tests/Analyzers/DiagnosticMessageNamesResolveTests.cs` against the **unfixed** tree and
   run it. It must **fail**, and the failure must name `PooledArray` **and** the descriptor it came from.
   **The developer's report carries that output pasted verbatim.** A T4 whose red state was not observed is
   not accepted, and the negative control (a locally-constructed descriptor) is **not** a substitute — it
   exercises the extractor, not the enumeration, and would stay green if reflection returned nothing.
2. T2, T1, T3, T5 (the analyst's evidence-order within the fixes is fine and I am not reordering it).
3. Re-run T4: green.

Between (1) and (2) the suite is legitimately red. **Do not commit that state**, and do not "fix" T4 to make it
pass in step 1 — that is the whole experiment. Writing T4 first also removes the failure mode where a revert is
forgotten and a permanently-green guard ships.

### D7 — T4 does not grow to cover `BannedSymbols.txt`

§6 floats extending T4 to catch F4. **No, not in this task.** `BannedSymbols.txt` names *members*
(`PooledBuffer<T>.RentArray`), so it needs a member-level resolver, and its prose names most of the BCL — the
false-positive surface is the general form the plan already rejected for messages. F4's class stays uncovered
by any guard, deliberately, and **that fact goes in T4's doc comment** alongside the F2 and F3 limits, so
nobody reads a green suite as "the advice a developer sees has been checked". If it is ever worth automating,
it is its own task with its own extractor.

## E. Technical risks

| # | risk | cheapest experiment | order |
|---|---|---|---|
| R1 | An RS2000-family diagnostic fires on the `Unshipped.md` Notes edit, at **warning** severity, and is invisible because the build still succeeds | T1's build, with the log **grepped for `RS2\d{3}` at any severity** — not merely checked for exit 0. `.editorconfig` sets no severity for the family and `Directory.Build.props` does not promote it, so a warning would not fail the build | with T1 |
| R2 | `GetTypes()` on the `netstandard2.0` `Analyzers` assembly throws `ReflectionTypeLoadException` under `net10.0` | T4's first run. **If it throws, do not catch it and filter `ex.Types` for non-null** — that silently shrinks the denominator and is exactly the vacuous-pass shape §6 guards against. Fix the reference instead and say so | step 1 of D6 |
| R3 | An analyzer test asserts on message text after all and reddens | covered: F6 established none does, by reading `AnalyzerHarness.cs:90-93` and grepping `Tests/`. T1's `--filter "FullyQualifiedName~Analyzers"` run confirms it | with T1 |
| R4 | A future message names a new Overfit type outside the allow-list prefixes and T4 skips it | the count and contains assertions in §C. Retired by construction | — |

Nothing here needs a spike: every risk is retired by a command already in an acceptance criterion.

## F. Operability

Not applicable — nothing in this plan runs. The single operational property worth naming is that the
diagnostic message **is** the operator interface for these rules: it is read once, at a build failure, by
somebody already interrupted, and there is no second channel. That is the whole reason F4 outranks F1 on
exposure even though F1 is the filed defect.

## G. What I am NOT signing

Read this as part of the signature, not as a footnote. **Nothing in this plan, mine or the analyst's, has been
compiled or executed.**

| # | not verified | the exact check the developer must run |
|---|---|---|
| 1 | That a Notes-column edit in `AnalyzerReleases.Unshipped.md` triggers no RS2000-family diagnostic. The analyst's evidence is RS2001's resource string, which is good evidence about RS2001 and not about RS2000/2002-2008 | Build `Sources/Analyzers/Analyzers.csproj` (it carries `Microsoft.CodeAnalysis.Analyzers` 5.6.0 with `EnforceExtendedAnalyzerRules=true`, so the family runs there) and **grep the full log for `RS2` at any severity**. "The build succeeded" is not the check — no `.editorconfig` entry or `Directory.Build.props` rule promotes RS2xxx to error, so a real diagnostic would pass silently |
| 2 | That T4 compiles, that the `Analyzers` assembly loads reflectively under `net10.0`, or that `GetTypes()` returns 47 at runtime. The 47 is a **source scan**, not a runtime count | T4's first run (D6 step 1) |
| 3 | That the analyzer test suite is green today | `dotnet test -c Release --filter "FullyQualifiedName~Analyzers"` before any edit, so a pre-existing failure is not attributed to this task |
| 4 | That no test anywhere asserts on a message string | established by grep only. The full-suite run after T1–T3 is the real check |
| 5 | That the `.editorconfig:59` comment edit does not disturb section parsing | trivially safe as a comment, but not compiled; the T1 build covers it |
| 6 | Every line number in this plan, mine included | line numbers age faster than anything else here. Each of my citations names the **symbol** and quotes the deciding words; if a number does not match, trust the quoted text and re-locate |

## H. Definition of Ready — architecture

Problem understood; boundaries assigned (§B); execution path stated (neither); source of truth named
(the 47 descriptor fields); quality requirements measurable and checked (§C); risks carry their retiring
command (§E); no irreversible decision, so no ADR (D5); operability N/A (§F); nothing blocking open.
**Proportionate:** five string sites, one composed-name fix, two prose fixes, one 60-line test — against a
defect that ran three months and was found by accident.

**READY.** T5 is now unblocked: the team lead ruled F4 in scope (BLOCKING QUESTION 1 answered in the
dispatching message, quoted: *"**F4 … is IN SCOPE.** It is the same defect class with strictly higher
exposure"*), so §14's "NOT READY — scope box unticked" is retired.

## I. Open questions

**None blocking.** The analyst's Q1, Q2 and Q3 were answered by the main session before this review; Q4
(`CLAUDE.md` and `.claude/**`) stands as reported — those files are the user's, the analyst's assumption that
the user fixes them is the right one, and T3's report must name them so they are not lost. Add to that list
`CLAUDE.md`'s **"Source-file guards (MSBuild, build-time errors)"** section, which describes
`BanJaggedFloatArrays` and `BanMultipleTopLevelTypes` as live MSBuild tasks; `Sources/Main/Main.csproj:81-101`
records their removal on 2026-08-05 and their replacement by `OVERFIT033` / `OVERFIT034`.

## J. Handoff

`overfit-developer` implements in D6's order. `overfit-verifier` owns the T4 mutation evidence. `overfit-reviewer`
owns whether the replacement prose still describes the code — which is most of this task's risk, since every
edit is prose. **No performance claim is made anywhere in this plan**, so `overfit-perf-claim-auditor` has
nothing to rule on; the one measured figure that appears (the `PoolComparisonBenchmark` 3x / ~3000x in
`BannedSymbols.txt:9`) is carried forward **verbatim and unchanged** by T5 and is recorded in
`docs/measured-baselines.md:340`.

**Architecture review:** signed 2026-08-15 by `overfit-architect`. Execution path: **neither**. AOT-reachable:
**no**. Allocation policy: **neither**.

## SUGGESTED IMPROVEMENTS TO MY ROLE — `overfit-architect`

One, with an incident in this run.

**My instructions tell me to check that a plan's inputs exist, but not to check whether a string is advice or a
record.** §A3 found an acceptance criterion that, executed as written, would have rewritten three dated,
accurate accounts of a 2026-06 pooling sweep into false ones — and it took a `git log -S` to see it, because
from the working tree a stale record and stale advice are the same string. My own definition already says
"prose that outruns its evidence … a wrong one does not merely mislead, it destroys the record of an
experiment", which is the same idea pointed the other way; what is missing is the operational half.
*Smallest change:* one line under "Watch for these specifically" — **"before approving an edit to prose that
names a deleted symbol, `git log -S` the symbol: if the prose is dated and was true then, it is a record and
must not be corrected."**
