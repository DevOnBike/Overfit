---
name: benchmark-class-map
description: Which benchmark class covers which decode/dispatch path, and the paths that have no benchmark at all. Verified 2026-08-14.
metadata:
  type: reference
---

# Benchmark class -> code path (verified 2026-08-14, HEAD e21e7c3)

- **`OverfitParallelBenchmark`** (`Sources/Benchmark/OverfitParallelBenchmark.cs`) covers
  `OverfitParallel.For` (the **parking** pool) vs `OverfitParallelLegacy.For` vs TPL `Parallel.For`.
  It uses `DispatcherBenchmarkConfig`, which deliberately drops the shared config's
  `InvocationCount=1 / UnrollFactor=1` pin — correct for a µs-scale dispatch.
  **It does NOT touch `ForDecode`.** Verified with Grep over `Sources/Benchmark` and
  `git log -S "ForDecode" -- Sources/Benchmark` (**zero commits, ever**).
- **`OverfitParallel.ForDecode` — the decode spin-pool — has NO benchmark class and never has.**
  8 production call sites (`find_references`): `Q4KDotKernel.ProjectParallel` / `ProjectGateUpParallel`,
  `Q4KGemvKernel.GemvParallel`, `Q6KDotKernel.ProjectParallel`, `Q6KGemvKernel.GemvParallel`,
  `Q8DotKernel.ProjectParallel`, `CachedMultiHeadAttention.Decode` / `TryDecodeWholeMatrix`.
- **The tok/s claims come from `[ModelFact]` diagnostics, not from `Sources/Benchmark`**:
  `Tests/LanguageModels/Loading/{Qwen3PerfTests,Phi3PerfTests,Phi4PerfTests,GemmaPerfTests}.cs`,
  `BielikSpeedTests.cs`, `QwenDecodeSpeedTests.cs`. `ModelFact : LongFact` sets `Skip` at discovery, so
  **`dotnet test` never runs them** and they cannot be forced on by a filter. Single-arm, one process,
  best-of-3, no canary. Any pool ON/OFF comparison through them is cross-process by construction, because
  `OVERFIT_DECODE_POOL` is read once into a `static readonly` field.
- **Stale artifact trap:** `Sources/Benchmark/BenchmarkDotNet.Artifacts/results/` holds a
  2026-05-15 report for a class named `OverfitParallelForBenchmark` that no longer exists, run under
  `InvocationCount=1` with `RatioSD` up to 11.83. It is not the source of any published figure and must not
  be cited.

## Model fixtures present on this box (2026-08-14)

`C:\qwen3-06b\Qwen3-0.6B-{Q8_0,Q4_K_M}.gguf`, `C:\qwen3b\*` (3B + qwen0.5b + `.repack`),
`C:\phi\{Phi-3.5-mini-instruct,phi-4}-Q4_K_M.gguf`, `C:\bielik\Bielik-4.5B-v3.0-Instruct-Q4_K_M.gguf`
(+1.5B, Q8_0, fp16), `C:\gemma\gemma-2-2b-it-Q4_K_M.gguf`, `C:\gpt2\`.
Note `Phi3PerfTests` points at `C:\phi\`, not `C:\phi35\`.
