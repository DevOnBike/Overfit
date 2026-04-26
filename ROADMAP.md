# Overfit Roadmap

Zero-allocation, pure C# deep-learning framework targeting high-performance CPU inference on .NET 10+.

**Philosophy:** minimal dependencies, predictable memory behavior, competitive small/medium CPU inference, and explicit separation between training, inference and kernels.

---

## Status snapshot

| Area | Status |
|------|--------|
| InferenceEngine prepared path | Stable; verified 0 B/op in current inference benchmarks |
| Linear / Conv / activation / pooling inference kernels | Stable for current small/medium CPU workloads |
| Autograd engine | Stable, pending ownership cleanup |
| TrainingEngine facade | Implemented; training allocations allowed |
| Adam / AdamW / SGD optimizers | Implemented |
| Evolutionary: GA + OpenAI-ES | Implemented; strategy Ask/AskThenTell measured 0 B/op in current benchmarks |
| Parallel fitness evaluation | Implemented |
| ONNX import/export | Not started, except benchmark model writers/helpers |
| GPU backend | Not started |

---

## Current benchmark position

Machine used for current public numbers:

```text
AMD Ryzen 9 9950X3D
Windows 11 25H2
.NET 10.0.7
BenchmarkDotNet 0.15.8
```

### Core inference claims

| Benchmark | Overfit | Comparison | Allocation |
|---|---:|---:|---:|
| Linear(784,10) single inference | ~250-300 ns | ONNX ~2.1-2.3 us | Overfit 0 B |
| Linear scaling 64/784/4096 inputs | ~80 ns / ~210 ns / ~1.08 us | ONNX ~1.39 / ~1.87 / ~3.74 us | Overfit 0 B |
| MLP 784->128->10 | ~3.7 us | ONNX ~5.2 us | Overfit 0 B |
| MLP 784->256->128->10 | ~10-12 us | roughly tied with ONNX | Overfit 0 B |
| Small CNN | ~5-6.5 us | roughly tied with ONNX | Overfit 0 B |
| Concurrent single-sample inference | ~516 ms | ONNX ~1811 ms | Overfit 0 B, ONNX ~117 MB |

### Known limitations

| Area | Current state | Next action |
|---|---|---|
| Large batches | ONNX wins at batch 64/256 | Implement `LinearKernels.ForwardBatched(...)` |
| Training allocations | allowed; current MLP train batch ~468 us / ~26.8 KB | profile after graph ownership cleanup |
| Thread scaling | useful directional signal | keep as profiling/trend benchmark |
| Legacy benchmarks | several were old `model.Forward(...)` / autograd path | keep them fixed on `InferenceEngine.Run(...)` or remove |

---

## Near-term priorities

### 1. Finish benchmark verification suite

- [ ] Keep central `BenchmarkConfig` for normal benchmark runs.
- [ ] Keep disassembly diagnostics in a separate config, not on every benchmark.
- [ ] Ensure inference benchmarks use `InferenceEngine.Run(...)` and preallocated buffers.
- [ ] Rename ONNX methods honestly: `PreAllocated`, not `TrueZeroAlloc`, when allocations remain.
- [ ] Treat `MinIterationTime` warnings as benchmark defects unless explicitly documented.
- [ ] Keep training benchmarks separate from zero-allocation inference claims.

### 2. Batched linear kernel

Current batch behavior is allocation-free but batch 64/256 favors ONNX Runtime because ONNX likely uses batched GEMM-style execution.

- [ ] Add `LinearKernels.ForwardBatched(...)` for `[B, InputSize] x [InputSize, OutputSize] -> [B, OutputSize]`.
- [ ] Add microbenchmarks for batch 16/64/256.
- [ ] Compare against current per-sample loop and ONNX Runtime.
- [ ] Keep batch path zero-allocation.

### 3. Autograd ownership cleanup

Architecture target:

```text
TrainingEngine = workflow facade
ComputationGraph = autograd brain / runtime
Parameter = long-lived trainable model state
AutogradNode = graph-visible value handle
Kernels = pure Span-based math
InferenceEngine = separate zero-allocation inference facade
```

Planned sequence:

1. Add graph operation facade: `graph.Linear`, `graph.Conv2D`, `graph.Relu`, `graph.SoftmaxCrossEntropy`.
2. Add `AutogradNodeOwnership` metadata.
3. Add graph factory methods for temporary/external/parameter-view nodes.
4. Introduce `Parameter` as separate model state.
5. Migrate `LinearLayer` first.
6. Migrate optimizers to `IEnumerable<Parameter>`.
7. Clean up graph reset/disposal by ownership.
8. Only then profile training allocations/performance.

---

## Medium-term priorities

### Performance

- [ ] Backward kernels cleanup: Linear/Conv backward should use pure span kernels where practical.
- [ ] Optimizer kernel profiling: Adam/AdamW state updates, zero-grad and allocation sources.
- [ ] Direct Conv2D training kernels: reduce im2col-style memory pressure where applicable.
- [ ] CPU SIMD audit: AVX2/AVX-512/AVX10 when available.
- [ ] Thread-scaling stabilization for large training workloads.

### Correctness

- [ ] Numerical equivalence tests across scalar/SIMD paths.
- [ ] Determinism policy for parallel training kernels.
- [ ] Optimizer formula audit against PyTorch references where applicable.
- [ ] Stable random seeds for flaky convergence tests.

### Features

- [ ] ONNX import for a minimal inference subset.
- [ ] ONNX export for interop and benchmark generation.
- [ ] Versioned checkpoint serialization.
- [ ] Standalone Softmax and CrossEntropy in addition to fused loss.
- [ ] Higher-level training API after graph ownership cleanup.

---

## Long-term ideas

- Graph compilation for fixed-shape training/inference graphs.
- Custom autograd operators with explicit forward/backward registration.
- Mixed precision and quantization for inference.
- Data loading and preprocessing pipeline improvements.
- Optional GPU backend investigation without compromising CPU-first design.
- Model/dataset packages outside the small core runtime.

---

## What Overfit is not trying to be

- Not a general-purpose replacement for PyTorch or TensorFlow.
- Not a Python shim.
- Not GPU-first.
- Not transformer-scale first.
- Not a model zoo in the core package.

The differentiator remains pure C#, predictable memory behavior, and small/medium CPU inference where the managed zero-allocation path matters.

---

## Contributing

Performance-sensitive PRs should include:

- correctness tests;
- before/after BenchmarkDotNet output;
- allocation measurements;
- documentation updates when public behavior changes.

License: GNU AGPLv3. For commercial licensing, contact devonbike@gmail.com.
