# Overfit Roadmap

Zero-allocation, pure C# deep learning framework targeting high-performance inference on .NET 10+.

**Philosophy:** minimal dependencies, predictable memory behavior, competitive with ONNX Runtime in edge/real-time scenarios.

---

## Status snapshot

| Area | Status |
|------|--------|
| Core tensor ops (FastTensor, TensorView) | ✅ Stable |
| Autograd engine | ✅ Stable |
| Linear / Conv2D / BatchNorm1D / ResidualBlock | ✅ Stable |
| LSTM layers | ✅ Implemented |
| Adam / AdamW optimizer (AVX-512 path) | ✅ Stable |
| MNIST training (full 60k, ResNet-style) | ✅ ~26 s/epoch, 99% accuracy |
| ONNX import/export | ❌ Not started |
| GPU backend | ❌ Not started |

---

## Benchmarks

### Linear 784→10 inference (AMD Ryzen 9 9950X3D, .NET 10)

| Batch | Overfit | ONNX Runtime | Winner | Overfit Alloc |
|------:|--------:|-------------:|:------:|--------------:|
| 1     | 188 ns  | 3,331 ns     | Overfit 17.7× | 0 B |
| 16    | 2,909 ns | 4,541 ns    | Overfit 1.56× | 0 B |
| 64    | 14,538 ns | 5,564 ns   | ONNX 2.61×    | 0 B |
| 256   | 18,852 ns | 12,242 ns  | ONNX 1.54×    | 6.8 KB |

**Edge case dominance:** single-sample inference is 17× faster than ONNX with zero allocations.

---

## Near-term priorities (next 1–3 months)

### Performance

- [ ] **Tiled MatMul with weight reuse across batch rows.** Current implementation does per-row dot products which underutilizes L1 cache on batches ≥ 64. Target: close the gap with MKL at batch 256 (from 1.54× to < 1.2×).
- [ ] **Evaluate `TensorPrimitives.MatrixMultiplyAdd`** (available in .NET 10). If it ships fused batched matmul+bias, a single call could beat our custom loop.
- [ ] **Conv2D im2col elimination.** Current path materializes im2col buffer (~10 MB/epoch). Research direct conv with SIMD tiling.
- [ ] **BatchNorm1D allocation cleanup.** Still allocates ~19 MB/epoch in training — identify and eliminate.
- [ ] **Adam.Step memory profiling.** 0.5 MB/step × 937 steps = 470 MB/epoch for MNIST. Find the source.
- [ ] **Document the ONNX Runtime overhead moat (~3.3 μs)** clearly in docs — this is the "why Overfit" story.

### Correctness

- [ ] **Stabilize `Sequential_SmallNetwork_Training_DecreasesLoss_WithoutBatchNorm` test.** Currently flaky (300 Adam steps on XOR with random init). Fix: set seed, raise steps to 500, or widen threshold.
- [ ] **AdamW weight decay formula review.** Current code applies decay to already-updated weight, not original. Either confirm this is intentional or fix to match reference PyTorch implementation.
- [ ] **Numerical precision audit** for AVX-512 vs AVX-2 vs scalar paths. Write property-based tests comparing outputs across backends on same input.
- [ ] **Determinism guarantee** documented per-layer. Currently `Parallel.For` in Linear forward may cause non-determinism under heavy load. Decide: always-deterministic mode, or document current behavior.

### Features

- [ ] **ONNX import** — minimum viable: load a trained ONNX model (Linear + Conv2D + BatchNorm + ReLU) and run inference.
- [ ] **ONNX export** — write models in ONNX format for interop with PyTorch/TensorFlow tooling.
- [ ] **Checkpoint serialization improvements.** Current `Sequential.Save/Load` is binary; add versioning and forward compatibility.
- [ ] **Softmax + CrossEntropy separation.** Currently fused in `SoftmaxCrossEntropy` only — allow standalone use.

### Unity Swarm Demo

- [ ] **Curriculum learning** — add predator only after gen 100, to stabilize orbit behavior learning first.
- [ ] **Multi-environment fitness evaluation** — evaluate each brain across 3–5 target/predator positions to reduce variance.
- [ ] **Model upgrade: 4→8→2 with hidden layer.** Current 10-parameter linear is expressive enough for orbit but limits emergent strategies. Benchmark convergence speed vs quality.
- [ ] **Elite averaging** for brain save (mean of top 10 instead of single best). More robust against noisy fitness.
- [ ] **Visualizer for genome evolution** — plot weight trajectories over generations.

---

## Medium-term (3–6 months)

### Framework breadth

- [ ] **Transformer building blocks** — MultiHeadAttention, LayerNorm, positional encodings.
- [ ] **Mixed precision support (FP16, BF16).** Start with inference only.
- [ ] **Quantization (INT8)** for deployed models. Post-training static quantization first.
- [ ] **Data loading pipeline** — async batching, prefetching, augmentation primitives.
- [ ] **Learning rate schedulers** — cosine, warmup, OneCycle.
- [ ] **Gradient clipping** (by norm and by value).

### Performance frontier

- [ ] **GPU backend investigation.** Options:
  - ComputeSharp / DirectML for Windows
  - ILGPU for cross-platform
  - Native interop with cuBLAS / ROCm
  - Evaluate complexity vs performance gain — Overfit's value prop is CPU-first, GPU should not compromise that.
- [ ] **CPU SIMD: AVX10 path** when hardware available.
- [ ] **Memory arena allocator** — per-generation allocator for training, reset between epochs.

### Developer experience

- [ ] **High-level API layer** — `model.Fit(trainData, epochs, batchSize)` style, hiding autograd mechanics.
- [ ] **Tensor broadcasting** — currently explicit, add implicit broadcast in elementwise ops.
- [ ] **Readable tensor print** — `.ToString()` that shows shape + sample values.
- [ ] **Training loop telemetry** — throughput (samples/sec), estimated time remaining, gradient norms.
- [ ] **Visual Studio debugger display** — `DebuggerDisplay` attributes on all public types.

### Documentation

- [ ] **Getting Started tutorial** — MNIST in under 50 lines.
- [ ] **Architecture deep-dive** — why ref structs, why TensorView, how the tape works.
- [ ] **Performance guide** — when to use Parallel.For, SIMD thresholds, memory layout tips.
- [ ] **Benchmark suite** — comparing against ONNX Runtime, TorchSharp, ML.NET on standardized tasks.
- [ ] **API reference** via DocFX.

---

## Long-term ideas

### Research / exploratory

- [ ] **Graph compilation.** JIT-compile the computation graph into a single optimized kernel (à la XLA). Could drop overhead dramatically for fixed-shape models.
- [ ] **Custom autograd operators** — let users define new ops with forward + backward, registered dynamically.
- [ ] **Distributed training** — data parallelism across multiple processes on a single machine first, then across nodes.
- [ ] **Structured pruning** built into the optimizer — automatic sparsification during training.
- [ ] **KV-cache** for sequence models (transformers, LSTMs) to accelerate autoregressive inference.

### Ecosystem

- [ ] **Overfit.Models** — zoo of pre-trained models (ResNet-18, small BERT, MNIST classifiers).
- [ ] **Overfit.Datasets** — MNIST, CIFAR-10, ImageNet subset loaders.
- [ ] **Overfit.Serving** — simple HTTP/gRPC server hosting a model with batching.
- [ ] **NuGet packaging** — split into `Overfit.Core`, `Overfit.Layers`, `Overfit.Optimizers`.
- [ ] **Source generators** — reduce boilerplate (layer registration, parameter collection).

### Demonstrations

- [ ] **Real-time audio classification** demo (edge device scenario).
- [ ] **Game AI integration** — Unity swarm was step 1, expand to MLAgents-style environment.
- [ ] **Anomaly detection service** building on the HMM work (ContinuousHMM, BaumWelchLearner already done).
- [ ] **.NET MAUI demo** — on-device ML on mobile.

---

## What Overfit is **not** trying to be

Clarity on scope prevents feature creep:

- **Not a general-purpose ML framework.** PyTorch and TensorFlow already exist and do that job well.
- **Not a research framework.** Dynamic graph manipulation, exotic optimizer variants, novel training techniques — out of scope.
- **Not a Python shim.** No TorchSharp-style wrapping of libtorch. The value is pure C#.
- **Not GPU-first.** CPU inference excellence is the differentiator. GPU support, if added, is a nice-to-have.
- **Not a model zoo.** Models may come later as a separate package, but the core stays small.

---

## Contributing

Issues and PRs welcome. Especially interested in:

- Performance regressions (benchmark-backed)
- Correctness bugs (minimal repro + expected behavior)
- ONNX import/export work
- New layer implementations with test coverage

Each PR should include a benchmark comparison (before/after) for performance-sensitive changes and a correctness test for new functionality.

---

## License

GNU AGPLv3. For commercial licensing, contact devonbike@gmail.com.