---
name: measured-negative-results
description: Optimisations in Overfit that look obviously correct and were measured to be worse, then reverted — the standing list to check before proposing any perf change.
metadata:
  type: reference
---

Source for all of these unless noted: `CLAUDE.md` "Performance work" section, `Sources/Main/Intrinsics/Simd.cs`
comments, `ROADMAP.md` / `ROADMAP-COMPLETED.md`. Re-verify the source file/line still exists before citing it in
a review — this list is a pointer, not a substitute for reading the current comment.

## Reverted / regressed (do not propose again without new evidence)

- **Second FMA accumulator in `Simd.Dot`** (`Sources/Main/Intrinsics/Simd.cs` ~line 150). Mechanism was real —
  independent chain measured 2906→1626 ns at 65536 floats, 60.5→38.8 ns at 2048
  (`SimdDotAccumulatorBenchmark`, Ryzen 9 9950X3D). Reverted anyway: a path census showed `Dot` is never on
  a forward/inference path, only backward, and 100% of real training calls land at lengths (32, 68, 128, 512,
  784) — below where two accumulators start paying. The guard itself cost 9-11% at those lengths even when
  NOT taken (perturbed register allocation in a ~4 ns kernel). Worth revisiting only for dModel/dFF ≥ 2048.
- **`Avx512Threshold = 512` in the same file is UNMEASURED and probably wrong** (not reverted — never fixed).
  `Vector512WidthBenchmark` measured 512-bit vs 256-bit at 128 floats (a quarter of the threshold) and 512-bit
  won every op: Add 0.74x, MulAdd 0.80x, Dot 0.84x. Left in place deliberately because Add/MulAdd weren't
  path-censused in that session — moving the threshold on 3 microbenchmark points without knowing caller
  lengths is the same mistake the Dot-accumulator change made. Flag as "known stale, not yet fixed" — don't
  claim it's already handled, and don't silently propose a fix without a path census first.
- **Winograd F(2,3)** for 3x3 stride-1 convs — parity-correct (cos 1.0) but **+79% slower** on deepcnn
  (119.7→214.4 ms): sequential scalar transforms + 16 small GEMMs + 16x U/V/M blow-up beat the 2.25x FLOP cut.
- **Register-blocking** (direct convolution) — regressed, reverted.
- **K-blocking + A-packing** (im2col GEMM) — regressed, reverted (`ROADMAP.md` ~1279).
- **AVX-512 decode port** — regressed, reverted. Direct measurement is in `ROADMAP-COMPLETED.md` (~1642):
  decode is memory-bandwidth-bound after GQA K/V-once + fuse-quantize, so a faster dot kernel saves cycles
  already hidden behind weight-read latency.
- **Bias support in the Q4_K tiled prefill GEMM** (`GemmTiled`) — a path census showed `bias.IsEmpty` barred
  88% of prefill dispatches (all attention Q/K/V) from the tiled kernel; lifting it measured **0.999x, an
  exact tie**, because `ProjectBatchedWeightStationary` already amortises weight decode across the row tile —
  same thing the tiling does. The "~3x" figure in the kernel's own doc comment is against re-decode-per-row,
  not against the weight-stationary baseline — cite the comparison it's against, not the bare number.
- **`OverfitPool<T>`** — deleted. Measured 3x slower (typical) / ~3000x slower (pathological case) than
  `PooledBuffer<T>` / raw pool usage. (Also recorded in the user's own global memory.)
- **Q6_K weight-stationary** — built, measured **+13.5% slower** (`ffn_down`), reverted (`ROADMAP-COMPLETED.md`
  ~926). Canaries drifted only 1-2%, so the regression was real, not noise.
- **VNNI `vpdpbusd`** replacing AVX2 `vpmaddubsw`+`vpmaddwd` in Q4_K/Q6_K dot kernels — same-state A/B:
  AVX2 ≈ VNNI ≈ 19.1 tok/s, ~0 gain, reverted. Decode was already memory-bandwidth-bound at that point, not
  ALU-bound, so a faster dot instruction had nothing to save.
- **`ProjectParallel` for the LM head** — tested wiring into decode, reverted: `Parallel.For` allocates ~3 KB
  per call from task scheduling, breaking the 0 B/token decode contract, for only ~3% steady-state speedup at
  GPT-2 Small scale. The 10x figure in `LmHeadParallelBenchmark` is a *steady-state* number; per-token decode
  is dominated by the `Parallel.For` dispatch overhead itself.
- **Parallelizing `TensorMath.Add`** (residual add) — measured **regression**: +20% wall on GPT-1 batch=32,
  +55% on `Add` backward itself. `Add` is memory-bandwidth-bound (2 reads + 1 write = 3x data movement); on a
  typical desktop bus (~50 GB/s) 2-3 cores already saturate it, so `OverfitParallelFor`'s ~10 µs cold dispatch
  is pure cost.
- **FP16-resident weights** (Slot 2c) — attempted and reverted. Steady RAM **regressed** (14.36 GB → 15.85 GB);
  the F32→F16 load conversion churns multi-GB F32 buffers, and fusing the convert would need ~9 ops per 8
  elements vs. the 1-op hardware convert it was trying to avoid.
- **Unrolled fixed-tile GEMM specialisation** — inconclusive/reverted: failed outright at `cols: 8`, ran
  7-9x slower (70-83x single-threaded in one config).
- **Output-row banding** ("missing L2 blocking level") — measured **20% slower**, reverted.
- **Column-pairing the Q6_K port** — slower than the shipped kernel; reverting restored the baseline numbers
  exactly (2398.5/2399.6 ms, `ffn_down` 656/659 ms).
- **`Conv2D` migrated to `OverfitParallelFor`** — measured **+13% MNIST wall time**, reverted; Conv2D stays on
  `Parallel.For`. (Also in the user's global memory as `project_conv2d_regression.md`.)
- **Naive single-threaded cached decode** vs. already-parallel uncached recompute at demo lengths — cached
  measured **2700 ms/token vs. uncached 424 ms/token, 6x SLOWER** (`QwenGgufCachedDecodeSpeedTests`).

## Wins that were the *opposite* of the "obvious" move (context, not negatives)

- `TensorPrimitives` bulk-SIMD beat a hand-written micro-kernel.
- The simple register-blocked GEMM beat the cache-blocked one (structure of the data around the technique
  decided, not the technique itself).
- Sixteen independent FMA chains were **30% faster** at 256-bit, not slower, in one specific kernel — more
  independent chains cover latency better (contrast with the Simd.Dot case above, where the census made the
  same mechanism worthless because of call-site lengths, not because the mechanism was wrong).

## Measurement-environment traps already paid for (don't re-litigate)

- **Cross-process before/after does not work on this box.** A prefill change read as +5% while the untouched
  decode path in the *same run* moved +32% — interleave configurations ABAB in one process and time a canary
  path every sample.
- **`OVERFIT_TILED_PREFILL` is a dead flag whenever a `*.gguf.repack` sidecar sits next to the model** —
  `IsPrepacked` short-circuits it, so both arms silently run the same code path. Count paths taken (temporary
  dispatcher counter) before believing any kernel A/B.
