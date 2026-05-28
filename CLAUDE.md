# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Pure C# deep-learning / optimization engine targeting **.NET 10**, with a strong
"zero-allocation, Native-AOT-compatible CPU inference" identity. No native
binaries, no Python runtime, no ONNX Runtime dependency. Public NuGet ID is
`DevOnBike.Overfit`.

## Solution layout

`Overfit.sln` contains five projects:

```text
Sources/Main             DevOnBike.Overfit       library, AOT-compiled in CI
Sources/Benchmark        Benchmarks (exe)        BenchmarkDotNet harness
Tests                    DevOnBike.Overfit.Tests xUnit
UI                       DevOnBike.Overfit.UI    WPF MNIST demo (net10.0-windows)
Demo/Unity               UnitySwarmServer (exe)  swarm engine demo server
```

`Main` exposes `InternalsVisibleTo` to `DevOnBike.Overfit.Tests` and `Benchmarks`,
so tests can reach internals directly. Versions are pinned centrally in
`Directory.Packages.props` (CPM enabled); `Directory.Build.props` enables
`NuGetAudit` and promotes vulnerability warnings (`NU1901-1904`) plus `CS4014`
to errors.

## Common commands

```powershell
dotnet build -c Release                                         # whole solution
dotnet test -c Release                                          # all tests (fast ones only — see test discipline)
dotnet test -c Release --filter "FullyQualifiedName~Gpt2"       # subset
dotnet test ./Tests/Tests.csproj -c Release --collect:"XPlat Code Coverage" --results-directory ./coverage

dotnet run -c Release --project Sources/Benchmark -- --filter "*SingleInferenceBenchmark*"
.\Sources\Benchmark\run.cmd                                      # runs all benchmarks (--filter *)

dotnet publish ./Sources/AotSmokeTest/AotSmokeTest.csproj -c Release -r linux-x64 -p:PublishAot=true -p:TreatWarningsAsErrors=true  # real AOT guard (requires C++ toolchain locally)
.\update-code-headers.cmd                                        # applies file-header template (dotnet format / IDE0073)
.\cleanup.cmd                                                    # purge bin/obj/.vs caches
```

The `Benchmark` Program.cs has a single `BenchmarkRunner.Run<…>` line at the
bottom — most benchmarks are commented out; uncomment the one you want or use
the `--filter` CLI form.

Python conversion scripts (run from `Scripts/`) need a local Python with
`torch`, `transformers`, `huggingface_hub`, `numpy`:

```powershell
python Scripts/convert_gpt2.py --size small --out Tests/test_fixtures/
python Scripts/convert_gguf.py ...
```

## Native-AOT discipline (this is the trip-wire)

Two independent layers guard the library against trim/AOT regressions:

1. **`Sources/Main/BannedSymbols.txt`** (enforced by `Microsoft.CodeAnalysis.BannedApiAnalyzers` with `RS0030` set to **error** in `.editorconfig`) forbids the following at every `dotnet build`, not just at publish:
   - `System.Linq` (the namespace is also `<Using Remove="System.Linq" />` in `Main.csproj`)
   - `System.Reflection`
   - `System.Linq.Expressions.Expression`
   - `System.Activator`
   - `Array.Copy` (use `Span<T>.CopyTo`)
   - Raw `ArrayPool<T>.Shared` (use `PooledBuffer<T>` or `PooledArray`)
2. **`Sources/AotSmokeTest`** is a thin console exe that the `aot-guard` CI job publishes under `PublishAot=true` + `TreatWarningsAsErrors=true`. Libraries cannot be Native-AOT compiled directly (no entry point), so the smoketest is the real AOT consumer — ILCompiler actually runs, IL2026 / IL3050 / IL31xx warnings on reachable code are promoted to errors, and the resulting native binary is executed as a smoke check. Extend `Sources/AotSmokeTest/Program.cs` cautiously: each new touched type or method widens AOT verification scope but may surface latent trim warnings that block publish until the library is fixed.

Use explicit `for`/`foreach` over `Span<T>`, delegates over reflection, explicit
`new` over `Activator`. This rule applies to `Sources/Main` only — tests and
benchmarks may use LINQ.

`Sources/Main` also enforces hot-path conservatism (see `Sources/Main/README.md`):
no LINQ in runtime code, no hidden allocations in inference, no `.ToArray()`,
no `model.Forward(...)` in the inference hot path — go through
`InferenceEngine.Run(input, output)` with caller-owned buffers.

## Code style

`.editorconfig` enforces:

- block-scoped namespaces (warning)
- `csharp_prefer_braces = true`
- `dotnet_sort_system_directives_first`, no separate import groups
- `IDE0073` (file header) as a warning — the template is the AGPL/commercial
  notice in `.editorconfig`. `update-code-headers.cmd` applies it across the
  tree.

`IDisposableAnalyzers` is wired into `Main` — heed its diagnostics; the project
leans heavily on `using` + pooled `TensorStorage<T>` lifetimes.

## Architecture (the parts you need to read multiple files to see)

### Inference vs. training separation

There are **two distinct execution paths** with different allocation policies,
and mixing them is the single most common architectural mistake:

- **Inference**: `InferenceEngine` (caller-owned buffers, zero allocations per
  call) → `IInferenceBackend` → `SequentialInferenceBackend` /
  `OnnxGraphInferenceBackend`. No `AutogradNode`, no `ComputationGraph`.
- **Training**: `ComputationGraph` records a tape of `AutogradNode`s, then
  `graph.Backward(loss)` walks it; `graph.Reset()` reclaims temporaries by
  ownership. Operations that record tape live on the graph
  (`graph.Linear`, `graph.Conv2D`, `graph.Relu`,
  `graph.SoftmaxCrossEntropy`) — the older `TensorMath.*(graph, ...)` style is
  being migrated onto the graph facade (`docs/OverfitArchitectureRefactorPlan.md`).

### Autograd ownership model

Every `AutogradNode` carries an `AutogradNodeOwnership` tag set at creation
that determines who disposes it:

| Ownership | Disposed by |
|-----------|-------------|
| `GraphTemporary` | `graph.Reset()` |
| `GraphAuxiliary` | `graph.Reset()` (e.g. MaxPool index map, Softmax probs) |
| `Parameter` | the owning layer's `Dispose()` |
| `ExternalBorrowed` | the caller |
| `View` | never (no backing storage) |

`Parameter` is a first-class type; optimizers take `IEnumerable<Parameter>`
(`Adam(parms, lr)`, `SGD(...)`), and `layer.TrainableParameters()` is the
canonical way to enumerate them.

### GPT / SLM runtime (KV-cache, zero-alloc decode)

The language-model runtime in `Sources/Main/LanguageModels/Runtime/` is layered
so weights are **never copied** at session creation:

```text
CachedSlmInferenceEngine  ← public entry; FromGpt1(model) wires the adapter
  CachedSlmSession        ← per-session state: KV buffers + position counter
    StackWeights          ← BlockWeights[] + final norm + LM head
      BlockWeights        ← layer norms + per-head attention + FFN
        SingleHeadWeights ← ReadOnlySpan refs into TensorStorage for Q/K/V/O + biases
    KeyValueCache         ← pre-allocated K/V, O(N) decode
```

All weight handles are `ReadOnlySpan<float>` obtained from `TensorStorage` at
decode time, which is why session creation allocates the KV buffers
(~80 MB for GPT-2 Small) and **per-token decode allocates 0 B**.
`CachedGpt1ModelAdapter.RefreshWeightsFromModel()` is a deliberate no-op for
in-place weight updates (LoRA path).

### ONNX import — two importers

- `OnnxImporter.Load(path)` — linear topology → `Sequential`. Faster, simpler.
- `OnnxGraphImporter.Load(path, inputSize, outputSize)` → `OnnxGraphModel`
  (DAG) → wrap in `OnnxGraphInferenceBackend` then
  `InferenceEngine.FromBackend(backend)`. Required for ResNet/DenseNet-style
  skip connections.

External `.data` sidecar files (PyTorch ≥ 2.x default) are resolved
automatically. There is no `Google.Protobuf` dependency — protobuf parsing is
hand-rolled in `Sources/Main/Onnx/`. Unsupported operators throw a clear
`NotSupportedException` naming the operator.

## Test discipline

Read `Tests/README.md` and `Tests/LanguageModels/README.md` before adding
tests — the project is **strict** about test runtime.

- `dotnet test -c Release` must stay fast and contain only correctness checks.
- Anything long-running uses `[LongFact]` (defined in `Tests/LongFact.cs`, a
  `FactAttribute` subclass that auto-sets `Skip` so the test is skipped by
  default). To run a `[LongFact]` locally, temporarily flip it back to `[Fact]`.
  In scope for `[LongFact]`: integration tests that load real models from
  `c:\qwen3b\*` (GGUF / binary), training/checkpoint demos, profilers,
  PyTorch-parity diagnostics, RAM diagnostics, anything 10s+ on the dev box.
  **Out of scope** — keep these as `[Fact(Skip = "...")]` with the specific
  reason: bug-tracker skips ("pending optimization guard", "numerical
  instability"), flaky timing tests. The point of `[Fact(Skip=...)]` vs
  `[LongFact]` is preserving *why* it's skipped.
- Diagnostics / profilers live in `Tests/**/Diagnostics/` and are skipped by
  default via `[LongFact]`.
- Test layout is by domain first, purpose second:
  `Core/`, `DeepLearning/`, `LanguageModels/{GPT1,Runtime,Tokenization,Demo,Experimental,Diagnostics}`,
  `Data/`, `Evolutionary/`, `Forecasting/`, `Preprocessing/`, `Integrations/`,
  `Diagnostics/`, `Examples/`, `TestSupport/`. Keep one public class per file
  and name it after the subject under test.
- Test fixtures live under `Tests/test_fixtures/` and are copied to output via
  `<None Include="test_fixtures\**\*" CopyToOutputDirectory="PreserveNewest" />`.
  `Tests/Usings.cs` provides `global using Xunit;` — don't repeat it per file.
