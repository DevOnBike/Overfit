# Documentation update patch summary

This ZIP updates markdown documentation for the `evo-no-alloc` benchmark cleanup and architecture roadmap.

## Files included

```text
README.md
ROADMAP.md
docs/README.md
docs/InferenceBenchmarkSummary.md
docs/TrainingEngineFacade.md
Sources/Benchmark/README.md
Sources/Main/README.md
```

## Main changes

- Replaced stale benchmark numbers that still referenced old `model.Forward(...)` / autograd-path benchmark results.
- Documented current `InferenceEngine.Run(...)` zero-allocation benchmark baseline.
- Added honest ONNX naming guidance: `PreAllocated`, not `TrueZeroAlloc`, when BenchmarkDotNet reports allocations.
- Added current positioning vs ONNX Runtime and ML.NET.
- Added batch scaling conclusion and next performance target: `LinearKernels.ForwardBatched(...)`.
- Kept training benchmarks scoped as performance trend benchmarks, not zero-allocation gates.
- Added graph ownership / `graph.*` facade roadmap into `ROADMAP.md` and `Sources/Main/README.md`.
- Fixed scenario guide links in root `README.md` to match current files under `docs/scenarios/`.

## Not included

Scenario docs were not rewritten in this patch. Their links are corrected from the root README, and their benchmark guidance can be updated later if needed.
