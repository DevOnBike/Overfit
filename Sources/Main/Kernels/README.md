# `Kernels` — the inner loops

The arithmetic that actually costs time: convolution, linear layers, activations, pooling, elementwise
work. Everything here is measured, not reasoned about.

## The rule for changing anything in this directory

**Write the benchmark first.** A `BenchmarkDotNet` class in `Sources/Benchmark` with the two shapes
side by side and `[Benchmark(Baseline = true)]` on the old one, before the change and before any claim
about it. A performance statement with no benchmark behind it is a guess however confident the
reasoning sounds, and this directory is where confident reasoning has been wrong most often.

Then, separately: **correctness first, speed second.** Write the clear version, pin it with a parity
test against a known-good reference, and only then make a second iteration for speed with the
validated version as the A/B baseline. Fusing the two steps produces a kernel that is unverifiable and
a change that cannot be isolated.

## Negative results that live here

These were implemented, measured, and reverted. They are listed because each looks like an obvious win
and re-proposing one costs a day:

- **Winograd F(2,3)** for 3×3 stride-1 convolutions (`Conv2DWinogradKernels`): parity-correct at cosine
  1.0, and **+79% slower** on the deep CNN, 119.7 → 214.4 ms. The 2.25× FLOP reduction was beaten by
  sequential scalar transforms, 16 small GEMMs, and a 16× blow-up in intermediate storage.
- **Register blocking** in the direct convolution, and **K-blocking + A-packing** in the im2col GEMM:
  both regressed.
- The simple register-blocked GEMM **beat** the cache-blocked one, and `TensorPrimitives` bulk SIMD
  **beat** a hand-written micro-kernel. In both cases the win was the opposite of the technique the
  literature recommends — the structure of the data around the technique decided, not the technique.

## Measurement traps this directory has actually hit

- **Wrong job for the workload.** The shared `BenchmarkConfig` pins `InvocationCount=1`, which suits
  multi-millisecond model runs and leaves a microbenchmark measuring timer noise: a ~15 µs operation
  produced `RatioSD` 0.44 and a phantom 1.61× regression that was 1.01 under `[SimpleJob]`.
- **The scaffolding outweighing the subject.** One benchmark reported a non-inlined call as *faster*
  than inlining it, because the synthetic body contained a saturating `float`→`long` cast whose cost
  dominated. If a result is backwards, suspect the benchmark before the runtime.
