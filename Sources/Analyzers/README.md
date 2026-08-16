# DevOnBike.Overfit.Analyzers

In-repo **Roslyn performance analyzers** for `Sources/Main` — the second guard layer next to
`BannedSymbols.txt` (named APIs, RS0030). The structural guards that were MSBuild tasks until 2026-08-05
(jagged arrays, one-type-per-file) are now analyzer rules themselves, `OVERFIT033` and `OVERFIT034`.
Wired into `Main.csproj` as an analyzer reference (`OutputItemType="Analyzer"`,
nothing ships); tests and benchmarks are deliberately NOT covered.

## Rules

| Id | What it flags | Status |
|---|---|---|
| `OVERFIT001` | Heap array allocation (`new T[n]`, `new[]{…}`, `[…]` targeting an array) in **per-call** code — use `PooledBuffer<T>` (pooled scratch, value element types), `TensorStorage<T>` (tensor data) or `stackalloc` (small fixed-size) | ✅ shipped |
| `OVERFIT002` | Jagged array allocation (`new T[n][]` — N+1 heap objects, pointer-chase per row) in per-call code — flat `T[]` Span-sliced per row (`float[][]` is a build ERROR via `OVERFIT033` regardless) | ✅ shipped |
| `OVERFIT003` | Boxing conversions (value type → `object`/interface) in per-call code | ✅ shipped |
| `OVERFIT004` | Closure/delegate allocation in per-call code: **capturing** lambdas (non-capturing are compiler-cached → silent) and **instance** method groups (static are cached since C# 11 → silent) — `OverfitParallel` takes function pointers for a reason | ✅ shipped |
| `OVERFIT005` | `foreach` over an interface-typed collection (heap enumerator + interface dispatch per element; arrays/`List<T>`/`Span` are fine) | ✅ shipped |
| `OVERFIT006` | String interpolation/concatenation in per-call code (exception path + `ToString` overrides + constant folds exempt) | ✅ shipped |
| `OVERFIT007` | Expanded-form `params` calls (hidden array per call; empty `params` → cached `Array.Empty`, silent) | ✅ shipped |
| `OVERFIT008` | Raw `Parallel.For/ForEach/Invoke` — ignores `SuppressParallelismOnCurrentThread` (oversubscription in data-parallel replicas, measured 1.8× slower) and allocates per dispatch (measured ~925 KB vs 0 B on the OverfitParallel pool). **Warning repo-wide in Main** (no per-call exemption — a suppress leak in a ctor breaks replicas the same way); measured exceptions keep `#pragma` + the benchmark cited | ✅ shipped |
| `OVERFIT009` | `.ToArray()` in per-call code (explicit hot-path ban in `Sources/Main/README.md`) — slice the existing buffer or use pooled memory | ✅ shipped |
| `OVERFIT015` | Direct `Xxx.IsSupported` on a hardware-intrinsics class outside `CpuFeatures` — gate ISA paths through `CpuFeatures.HasXxx` (one audit point, composed flags like `HasAvx2Fma`, identical JIT constant-folding; the two strays found at introduction were migrated). **Warning repo-wide in Main**, like 008 — it's a convention rule, not an allocation-context rule | ✅ shipped |
| `OVERFIT010` | Per-call `new List<T>` / `Dictionary` / `HashSet` / `Queue` / `Stack` / `StringBuilder` / `MemoryStream` — pool it (`docs/performance-patterns.md` #1 ObjectPool, #26 ValueStringBuilder, #46p RecyclableMemoryStream) | ✅ shipped |
| `OVERFIT011` | Struct used as `Dictionary<K,V>`/`HashSet<K>` key without `IEquatable<K>` / record struct — the DEFAULT `Equals`/`GetHashCode` for structs is **reflection-based** (`#46g`); a passed `IEqualityComparer<K>` silences it. Design rule, no context exemption | ✅ shipped |
| `OVERFIT012` | Finalizer declared (`~T()`) — finalizable objects allocate slower and survive ≥2 GC generations; `IDisposable` + `GC.SuppressFinalize` instead (`#48a`) | ✅ shipped |
| `OVERFIT013` | `.Count` read on `ConcurrentQueue`/`ConcurrentBag` (segment-walking + sync) — use `IsEmpty` or an approximate `Interlocked` counter (`#80`) | ✅ shipped |
| `OVERFIT014` | `a.ToLower() == b.ToLower()` / `.ToUpper()` comparison — allocates a throwaway string per side; use `string.Equals(a, b, StringComparison.OrdinalIgnoreCase)` (`#46o`) | ✅ shipped |
| `OVERFIT016` | Large struct (est. > 64 B) passed by value — pass as `in` to avoid the per-call copy | ✅ shipped |
| `OVERFIT017` | Struct whose fields are all readonly is not declared `readonly struct` — every member access on a `readonly`/`in` reference makes a defensive copy | ✅ shipped |
| `OVERFIT018` | `readonly` field of a MUTABLE struct type — defensive copy per access, and mutations are silently lost | ✅ shipped |
| `OVERFIT019` | Non-capturing lambda not marked `static` — guards against a future edit accidentally introducing a capture (which would start allocating a closure per call) | ✅ shipped |
| `OVERFIT020` | Primitive-array parameter that provably never escapes — `ReadOnlySpan<T>`/`Span<T>` accepts slices and `stackalloc` without a copy | ✅ shipped |
| `OVERFIT021` | `else` / `else if` — invert into a guard clause + early return, `continue` in loops, a ternary, or a switch expression. **Readability rule, not perf** (measured: ternary/continue/inversion are free; only method EXTRACTION can cost, 2.25× when the JIT declines to inline). Ratchet: `suggestion` repo-wide, `error` per swept directory | 🟡 rollout |
| `OVERFIT022` | **Direct recursion** (NASA Power of 10 rule 1). A `StackOverflowException` CANNOT be caught in .NET — it kills the host process — so recursion over externally-authored input is an uncatchable crash. **`error` repo-wide**; every legitimate site carries `#pragma warning disable OVERFIT022` whose comment states the BOUND that makes it safe (a checked depth cap, or a structural log2(n) argument) | ✅ shipped |
| `OVERFIT023` | **Loop with no exit condition in its header** — `while (true)` / `for (;;)` / `do…while(true)` (NASA rule 2). An unbounded loop hangs the host with no exception, no stack trace and no log line. **`error` repo-wide**, same contract: suppress with a comment starting `BOUND:` naming what terminates it | ✅ shipped |
| `OVERFIT900` | A per-call rule fired inside a member/type marked `[OverfitHotPath]` — the per-member form of the per-directory severity ratchet. Always an error; the message names the underlying rule id | ✅ shipped |

## Extended catalog (grounded in `docs/performance-patterns.md`, 110 patterns)

**Tier A — SHIPPED 2026-06-13** as `OVERFIT010`–`OVERFIT014` (table above). All five are precise,
low-false-positive rules; 010 and 014 are allocation rules (one-time + exception exempt), 011/012/013
are convention rules. `OVERFIT015` (direct `Xxx.IsSupported` → `CpuFeatures.HasXxx`, source `#64`/`#67`)
shipped earlier as the broader form of the original "IsSupported inside a loop" idea.

**Tier B — heuristic, suggestion-only. SHIPPED 2026-06-13.** These read code shape rather than a hard
allocation, so they are **advisory: suggestion everywhere, and they do NOT escalate under
`[OverfitHotPath]`** (they don't route through `OverfitPerfAnalysis.Report`) — a heuristic must never
break a build.

| Id | Rule | Source pattern | Status |
|---|---|---|---|
| `OVERFIT016` | Large struct (estimated > 64 B = cache line) passed by value — pass as `in`. Size is summed from the field layout (no padding → under-counts, so only clearly-large structs trip); SIMD `Vector*` and generic type params are skipped | #56 | ✅ shipped |
| `OVERFIT017` | A plain (non-record) struct with ≥1 instance field where all instance fields are readonly and no settable property — declare it `readonly struct` to drop defensive copies | #2 | ✅ shipped |
| `OVERFIT018` | `readonly` field of a **mutable** struct (a non-readonly struct with ≥1 non-readonly instance field — catches enumerators, excludes immutable BCL values like `DateTime`/`Guid`): every access is a defensive copy, mutation is silently lost | #90 (the "Do NOT make this readonly" trap) | ✅ shipped |
| `OVERFIT019` | Non-capturing lambda without the `static` keyword — `static` makes the no-capture contract enforced (a future accidental capture becomes a compile error). Shares the capture analysis with `OVERFIT004` via `OverfitPerfAnalysis.LambdaCapturesEnclosingState` | #46h | ✅ shipped |
| `OVERFIT020` | A primitive-array parameter (`float[]`/`int[]`/`byte[]`/…) of a **private/internal**, non-async, non-iterator, non-ctor method that is only read/indexed and **provably does not escape** (no field/return/array-argument/lambda capture) → take a `ReadOnlySpan<T>` (or `Span<T>` if it writes elements) so callers pass arrays, slices or `stackalloc` without a copy. Intra-method escape analysis via `RegisterOperationBlockAction`; conservative (skips ctors/public surface so it never suggests breaking a stored-weights API) | span-friendly APIs | ✅ shipped |

**This table stops at `OVERFIT020` and the family now runs to `OVERFIT046`, plus `OVERFIT900`** (47 ids in
[`AnalyzerReleases.Unshipped.md`](AnalyzerReleases.Unshipped.md), 47 descriptors in the source). The rules added since — the
`else` ban, the two NASA reliability rules, the `stackalloc` pair, the async tier, the design guards and
`OVERFIT038` below — are described where they are enforced rather than here:
[`AnalyzerReleases.Unshipped.md`](AnalyzerReleases.Unshipped.md) carries one line each, `.editorconfig`
carries the severity *and the reason for it per directory*, and each analyzer's own doc comment carries the
incident that produced it. Backfilling this table is worth doing; duplicating those three is not.

| Id | What it flags | Status |
|---|---|---|
| `OVERFIT038` | A count read from `BinaryReader` / `System.Text.Json` that sizes an array, a `stackalloc`, a counted read or a loop bound with **no validator between the read and the use** — the ordering is the rule, because a guard a few lines below reads as done to a human and is not. Excludes reads that are not numbers (`while (reader.Read())`) and lengths the parser measured rather than the file declared (`GetArrayLength`) | ✅ shipped 2026-08-10, `error` in `Sources/Main` |

**Tier C — architectural (convention/attribute):**

- **`[OverfitHotPath]` — SHIPPED 2026-06-13** (Roslyn's `[PerformanceSensitive]`, `#35`).
  `DevOnBike.Overfit.Diagnostics.OverfitHotPathAttribute` (declared in `Sources/Main`, ships with the
  library) marks a method / constructor / property / class / struct as zero-allocation. Inside a marked
  member — or any member of a marked type, walking nested + outer types and accessor→property — every
  per-call rule (`001`–`007`, `009`–`011`, `013`, `014`) reports **`OVERFIT900` (error)** instead of its
  directory-configured severity: the per-member evolution of the per-directory editorconfig ratchet.
  `OVERFIT900`'s message names the underlying rule. Not escalated: `008`/`015` (already error repo-wide)
  and `012` (a type-level finalizer declaration, not a per-call op). Justify a site with
  `#pragma warning disable OVERFIT900` + a reason, or drop the attribute — never loosen `OVERFIT900` globally.
- `foreach` over `List<T>` in hot paths → indexed `for` / `CollectionsMarshal.AsSpan` (#46c) — too noisy globally, only under `[OverfitHotPath]`.

**Handled via `BannedSymbols.txt` instead of analyzer rules** (named symbols → cheaper; added 2026-06-12):
`Buffer.BlockCopy` (measured slowest copy in `CopyAndPoolBenchmark` — both Main sites migrated to `Span.CopyTo`),
`string.Intern` (#48b — global-table lock contention; Roslyn ditched it for `StringTable`),
`System.Threading.ReaderWriterLock` (#48d — obsolete kernel-mode lock, use `ReaderWriterLockSlim`).

**Deliberately NOT rules** (not statically decidable / measured-context-dependent): SOA vs AoS (#55),
false-sharing padding (#57), `[SkipLocalsInit]` adoption (#22 — project-level decision), anti-DRY
hot-method size (#93), `ManualResetEventSlim` spinCount (#95).

## Noise policy (OVERFIT001)

**One-time allocations are exempt**: field/property initializers, instance constructors (layer
weights live as long as the object), static constructors, and compiler-generated (implicit)
creations such as `params` arrays. Flagged is exactly the per-call surface: method bodies,
accessors, lambdas, local functions. Justified sites: `#pragma warning disable OVERFIT001` + a
comment saying why.

## Severity ladder (configured in the root `.editorconfig`)

Three levels with distinct meanings — **suggestion** = known debt, visible in the IDE;
**warning** = fix before you finish the feature; **error** = never merges.

- **Everywhere**: `suggestion`
- **Hot directories** (`Kernels/`, `Intrinsics/`, `Ops/`): **warning**
- **Errors** (criteria: zero existing sites + unconditional rule + trivial fix/pragma + high miss-cost):
  - `OVERFIT008` + `OVERFIT015` — **error repo-wide in Main** (suppress leak silently breaks DP
    training; CpuFeatures gating is a hard convention — same class as RS0030)
  - `OVERFIT001`/`002`/`009` — **error in `Kernels/` + `Intrinsics/`** (per-call allocation in a pure
    kernel is never right; a justified exception is an explicit `#pragma` with the reason)
- Clean a directory → escalate it in `.editorconfig`. Never loosen.

## Triage executed 2026-06-12 — build is at **0 warnings**

The 45 findings at introduction were resolved deliberately, not swept:

- **Fixed (17)**: 13× OVERFIT008 migrated to `OverfitParallel.For` (Adam ×3 — its explicit
  `Options` arg dropped, the wrapper supplies it; DataParallelTrainer ×3; TrainableLlamaModel ×3;
  OfflineTrainingJob ×2; DataAugmenter; FastRandomForest — semantics equal or better: they now
  inline gracefully under suppression). 4× OVERFIT001 pooled: PoolingKernels ×2 (the `inputW > 128`
  branch next to the existing stackalloc → conditional `PooledBuffer`, `default` disposes as a
  no-op) and BatchNorm backward `sumDy`/`sumDyX` → `RentArray`/`ReturnArray` + try/finally (every
  slot is assigned before read, so no Clear needed).
- **Pragma'd with the reason cited (4)**: Evolutionary ×2 (caller-configurable `ParallelOptions` /
  `Partitioner`+`ForEach` — APIs `OverfitParallel` doesn't have; never runs inside DP replicas),
  `TensorMath.Sequence` LSTM backward (stateful `localInit`/`localFinally` overload; the suppress
  case has its own inline branch) and its tape-recording `AutogradNode[]` (owned by the graph until
  `Reset()` — pooling would change the `Record` contract).
- **Downgraded to suggestion with rationale in `.editorconfig` (24)**: the CTC family (15 — OCR/
  training path, not serving-hot; beam search needs a redesign, re-escalate if OCR productises) and
  OVERFIT004 in `Ops/` (8 — training graph-ops dispatch through closures BY DESIGN; Spans cannot
  cross a delegate boundary; the zero-alloc CONTRACT is the inference path, training is
  allocation-quiet at ~4.6 KB/batch per the MNIST audit).

**`LanguageModels/Runtime` SWEPT & ESCALATED 2026-06-12** — all 85 sites resolved (batched-prefill
scratch pooled: measured **748 MB → 0 B** allocated per 272-token prefill at steady state, greedy
output bit-identical; load-time/by-contract sites carry justified `#pragma`s) and the directory is
now a **warning** in the ratchet. Lesson encoded in the code comments: rented arrays are longer than
requested — pass **exact-length slices** to any callee that reads `span.Length` (the MoE router bug
the suite caught). Remaining suggestion-level backlog (repo-wide): OVERFIT001 ~340 sites — `Loading`
60, `Audio/Tts` 38, `Whisper` 33, `DeepLearning` 30, …

## Authoring notes

- Target `netstandard2.0` (analyzers run inside the compiler/IDE host); `Microsoft.CodeAnalysis.CSharp`
  pinned ≤ the SDK compiler version.
- Use the `IOperation` API (`OperationKind.ArrayCreation`, `CollectionExpression`) — it sees through
  syntax variants; skip `IsImplicit` operations.
- Each rule: `EnableConcurrentExecution`, `ConfigureGeneratedCodeAnalysis(None)`, category
  `Performance`, default severity Warning (the editorconfig ratchet scopes it down/up per path).
- **A message may compose an identifier from a format placeholder only when the composition (a) echoes a
  symbol the developer wrote, (b) names a symbol the analyzer resolved before reporting, or (c) names a
  symbol the fix will create. Never assert that a symbol exists without having resolved it.** `OVERFIT015`
  did: it emitted `CpuFeatures.Has{0}` from the containing type's name, so `Avx512F.IsSupported` was
  answered with `CpuFeatures.HasAvx512F` against a field called `HasAvx512`, and `Sse2`/`AdvSimd` have no
  field at all. Nothing can catch that shape — the composed name is never written down, so neither a text
  search nor `Tests/Analyzers/DiagnosticMessageNamesResolveTests.cs` can see it. The seven other composing
  messages are safe by construction and are not the target of this rule.
