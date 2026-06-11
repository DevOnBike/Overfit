# DevOnBike.Overfit.Analyzers

In-repo **Roslyn performance analyzers** for `Sources/Main` — the third guard layer next to
`BannedSymbols.txt` (named APIs, RS0030) and the MSBuild structural guards (jagged arrays,
one-type-per-file). Wired into `Main.csproj` as an analyzer reference (`OutputItemType="Analyzer"`,
nothing ships); tests and benchmarks are deliberately NOT covered.

## Rules

| Id | What it flags | Status |
|---|---|---|
| `OVERFIT001` | Heap array allocation (`new T[n]`, `new[]{…}`, `[…]` targeting an array) in **per-call** code — use `PooledBuffer<T>` / `PooledArray` (pooled scratch), `TensorStorage<T>` (tensor data) or `stackalloc` (small fixed-size) | ✅ shipped |
| `OVERFIT002` | Per-call `new List<T>` / `Dictionary` / `HashSet` in hot paths | proposed |
| `OVERFIT003` | Boxing conversions (value type → `object`/interface) in hot paths | proposed |
| `OVERFIT004` | Delegate/closure allocation in hot paths (lambdas capturing locals — `OverfitParallel` takes function pointers for a reason) | proposed |
| `OVERFIT005` | `foreach` over an `IEnumerable<T>`-typed expression in hot paths (enumerator allocation; arrays/`List<T>`/`Span` are fine) | proposed |
| `OVERFIT006` | String interpolation/concatenation in hot paths (logging leftovers) | proposed |
| `OVERFIT007` | Calls to `params` methods in hot paths (hidden array per call) | proposed |

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

Backlog measured at introduction (2026-06-11): **432 unique sites** total; hot-dir warnings after
scoping: **17**. Largest pools: `LanguageModels/Runtime` 85 (mostly batched-prefill scratch —
actionable: pool it), `Loading` 60, `Audio/Tts` 38, `Whisper` 33.

## Authoring notes

- Target `netstandard2.0` (analyzers run inside the compiler/IDE host); `Microsoft.CodeAnalysis.CSharp`
  pinned ≤ the SDK compiler version.
- Use the `IOperation` API (`OperationKind.ArrayCreation`, `CollectionExpression`) — it sees through
  syntax variants; skip `IsImplicit` operations.
- Each rule: `EnableConcurrentExecution`, `ConfigureGeneratedCodeAnalysis(None)`, category
  `Performance`, default severity Warning (the editorconfig ratchet scopes it down/up per path).
