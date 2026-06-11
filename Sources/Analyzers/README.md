# DevOnBike.Overfit.Analyzers

In-repo **Roslyn performance analyzers** for `Sources/Main` — the third guard layer next to
`BannedSymbols.txt` (named APIs, RS0030) and the MSBuild structural guards (jagged arrays,
one-type-per-file). Wired into `Main.csproj` as an analyzer reference (`OutputItemType="Analyzer"`,
nothing ships); tests and benchmarks are deliberately NOT covered.

## Rules

| Id | What it flags | Status |
|---|---|---|
| `OVERFIT001` | Heap array allocation (`new T[n]`, `new[]{…}`, `[…]` targeting an array) in **per-call** code — use `PooledBuffer<T>` / `PooledArray` (pooled scratch), `TensorStorage<T>` (tensor data) or `stackalloc` (small fixed-size) | ✅ shipped |
| `OVERFIT002` | Jagged array allocation (`new T[n][]` — N+1 heap objects, pointer-chase per row) in per-call code — flat `T[]` Span-sliced per row (`float[][]` is a build ERROR via the MSBuild guard regardless) | ✅ shipped |
| `OVERFIT003` | Boxing conversions (value type → `object`/interface) in per-call code | ✅ shipped |
| `OVERFIT004` | Closure/delegate allocation in per-call code: **capturing** lambdas (non-capturing are compiler-cached → silent) and **instance** method groups (static are cached since C# 11 → silent) — `OverfitParallel` takes function pointers for a reason | ✅ shipped |
| `OVERFIT005` | `foreach` over an interface-typed collection (heap enumerator + interface dispatch per element; arrays/`List<T>`/`Span` are fine) | ✅ shipped |
| `OVERFIT006` | String interpolation/concatenation in per-call code (exception path + `ToString` overrides + constant folds exempt) | ✅ shipped |
| `OVERFIT007` | Expanded-form `params` calls (hidden array per call; empty `params` → cached `Array.Empty`, silent) | ✅ shipped |
| `OVERFIT008` | Raw `Parallel.For/ForEach/Invoke` — ignores `SuppressParallelismOnCurrentThread` (oversubscription in data-parallel replicas, measured 1.8× slower) and allocates per dispatch (measured ~925 KB vs 0 B on the OverfitParallel pool). **Warning repo-wide in Main** (no per-call exemption — a suppress leak in a ctor breaks replicas the same way); measured exceptions keep `#pragma` + the benchmark cited | ✅ shipped |
| `OVERFIT009` | `.ToArray()` in per-call code (explicit hot-path ban in `Sources/Main/README.md`) — slice the existing buffer or use pooled memory | ✅ shipped |
| `OVERFIT010` | Per-call `new List<T>` / `Dictionary` / `HashSet` | proposed |

## Noise policy (OVERFIT001)

**One-time allocations are exempt**: field/property initializers, instance constructors (layer
weights live as long as the object), static constructors, and compiler-generated (implicit)
creations such as `params` arrays. Flagged is exactly the per-call surface: method bodies,
accessors, lambdas, local functions. Justified sites: `#pragma warning disable OVERFIT001` + a
comment saying why.

## Ratchet rollout (configured in the root `.editorconfig`)

- **Everywhere**: `suggestion` (visible in the IDE, doesn't pollute the build)
- **Hot directories** (`Kernels/`, `Intrinsics/`, `Ops/`): build **warning**
- Clean a directory → flip it to `warning` in `.editorconfig`. Never loosen a flipped directory.

Exception to the ratchet: **OVERFIT008 is a warning repo-wide in Main** — a suppress leak is a
correctness-adjacent bug (it breaks data-parallel training throughput), not just allocation hygiene.

Backlog measured at introduction (2026-06-11/12), unique build warnings: OVERFIT001 **16** (hot dirs;
432 sites repo-wide as suggestions — `LanguageModels/Runtime` 85, mostly batched-prefill scratch,
actionable: pool it; `Loading` 60; `Audio/Tts` 38; `Whisper` 33), OVERFIT004 **8**, OVERFIT008 **16**
(Adam ×3, DataParallelTrainer ×3, TrainableLlamaModel ×3, …; `OverfitParallel.cs` itself carries the
one sanctioned `#pragma`), OVERFIT003/005/006/009 **1 each**, OVERFIT002/007 **0**.

## Authoring notes

- Target `netstandard2.0` (analyzers run inside the compiler/IDE host); `Microsoft.CodeAnalysis.CSharp`
  pinned ≤ the SDK compiler version.
- Use the `IOperation` API (`OperationKind.ArrayCreation`, `CollectionExpression`) — it sees through
  syntax variants; skip `IsImplicit` operations.
- Each rule: `EnableConcurrentExecution`, `ConfigureGeneratedCodeAnalysis(None)`, category
  `Performance`, default severity Warning (the editorconfig ratchet scopes it down/up per path).
