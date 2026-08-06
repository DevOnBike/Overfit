# Measured baselines and negative results

**The single place this repository keeps its numbers.** Agents, docs and comments should cite this file
rather than restating a figure, because a number copied into five places is a number that will be wrong in
four of them.

## How to use it

- **Every number here carries what it was measured on.** A figure without its model, quantisation, thread
  count, build and box is not evidence about anything — this repo has been burned by cross-process and
  cross-build comparisons and treats an uncited number as no number.
- **Re-verify before citing.** This file is a pointer to where the measurement lives, not a substitute for
  reading the current comment or re-running the benchmark. Code moves; numbers rot.
- **A negative result is the most valuable row here.** The list below is not a museum. It is the set of
  changes that look obviously correct and were measured to be worse — without it, each one gets proposed
  again, confidently, roughly once a quarter.
- **Do not extrapolate a technique between kernels.** The recurring lesson across every row: the structure of
  the data around a technique decides, not the technique.

Dev box for most figures: Ryzen 9 9950X3D, Windows, .NET 10, Release.

---

## Reverted or regressed — do not propose again without new evidence

| change | measured | why it lost |
|---|---|---|
| **Second FMA accumulator in `Simd.Dot`** | real 1.79× at 65536 floats (2906→1626 ns); 60.5→38.8 ns at 2048 | Path census: `Dot` is never on a forward/inference path, and 100% of real training calls land at lengths 32/68/128/512/784 — below where two chains pay. **The untaken guard branch alone cost 9–11%** at those lengths. Revisit only for dModel/dFF ≥ 2048. |
| **Winograd F(2,3)**, 3×3 stride-1 conv | parity-correct (cos 1.0), **+79% slower** on deepcnn, 119.7→214.4 ms | Sequential scalar transforms + 16 small GEMMs + 16× U/V/M blow-up beat the 2.25× FLOP cut. |
| **Register-blocking** (direct conv) | regressed | reverted |
| **K-blocking + A-packing** (im2col GEMM) | regressed | reverted |
| **AVX-512 decode port** | regressed | Decode is memory-bandwidth-bound after GQA K/V-once + fuse-quantize; a faster dot kernel saves cycles already hidden behind weight-read latency. |
| **Bias in the Q4_K tiled prefill GEMM** (`GemmTiled`) | **0.999× — an exact tie** | `bias.IsEmpty` barred 88% of prefill dispatches; lifting it changed nothing because `ProjectBatchedWeightStationary` already amortises weight decode across the row tile. The "~3×" in the kernel doc is against re-decode-per-row, **not** against weight-stationary. |
| **`OverfitPool<T>`** | 3× slower typical, ~3000× pathological vs `PooledBuffer<T>` | deleted |
| **Q6_K weight-stationary** | **+13.5% slower** on `ffn_down` | Canaries drifted 1–2%, so the regression was real, not noise. |
| **VNNI `vpdpbusd`** vs AVX2 `vpmaddubsw`+`vpmaddwd` | AVX2 ≈ VNNI ≈ 19.1 tok/s, ~0 gain | Decode was already bandwidth-bound, not ALU-bound. |
| **`ProjectParallel` for the LM head** | ~3% steady-state gain, **~3 KB allocated per call** | Breaks the 0 B/token decode contract. The 10× in `LmHeadParallelBenchmark` is steady-state; per-token decode is dominated by `Parallel.For` dispatch. |
| **Parallelising `TensorMath.Add`** (residual) | **+20% wall** on GPT-1 batch=32, +55% on `Add` backward | Memory-bandwidth-bound (2 reads + 1 write); 2–3 cores already saturate a ~50 GB/s bus, so a ~10 µs cold dispatch is pure cost. |
| **FP16-resident weights** | steady RAM **regressed** 14.36 → 15.85 GB | The F32→F16 load conversion churns multi-GB F32 buffers. |
| **Unrolled fixed-tile GEMM specialisation** | failed at `cols: 8`; 7–9× slower (70–83× single-threaded in one config) | reverted |
| **Output-row banding** ("missing L2 blocking level") | **20% slower** | reverted |
| **Column-pairing the Q6_K port** | slower; reverting restored 2398.5/2399.6 ms exactly | reverted |
| **`Conv2D` → `OverfitParallelFor`** | **+13% MNIST wall time** | Conv2D stays on `Parallel.For`. |
| **Naive single-threaded cached decode** | **2700 ms/token vs 424 ms/token uncached — 6× slower** | at demo lengths, against already-parallel recompute |

**Known stale and not yet fixed:** `Avx512Threshold = 512` in `Sources/Main/Intrinsics/Simd.cs` is
**unmeasured**. `Vector512WidthBenchmark` at 128 floats — a quarter of the threshold — had 512-bit winning
every op (Add 0.74×, MulAdd 0.80×, Dot 0.84×). Left alone deliberately: moving a threshold on three
microbenchmark points without a caller-length census is the same mistake the `Simd.Dot` change made. Do not
claim it is handled; do not fix it without the census.

## Wins that went the opposite way to the "obvious" move

- `TensorPrimitives` bulk-SIMD **beat** a hand-written micro-kernel.
- The **simple** register-blocked GEMM beat the cache-blocked one.
- Sixteen independent FMA chains were **30% faster at 256-bit**, not slower — more chains cover latency
  better. Contrast with `Simd.Dot` above, where the same mechanism was worthless because of call-site
  lengths, not because the mechanism was wrong.

## Shapes and call costs

| question | measured |
|---|---|
| `for` vs `foreach` over an array (.NET 10) | **not a lever** — ~2 ns, and the direction reverses with size |
| **declared type** on a hot path | **this is the lever**: an interface costs **2.4× iterating**, **4.6× indexing**, plus **32 B** for the enumerator. Do not "tidy" `T[]` into `IReadOnlyList<T>`. |
| monomorphism | does **not** guarantee zero allocation — this refuted an earlier explanation of my own |
| extracting a method when the JIT does not inline it | **2.25×** — which is why `else` removal prefers restructuring in place over extraction |
| ternary, `continue`, condition inversion | free |
| `OverfitParallelFor` vs `Parallel.For`, **decode** | **455 µs / 0 B** vs **2059 µs / 925 KB** |
| `OverfitParallelFor` vs `Parallel.For`, **Conv2D** | the opposite — see the regression table |

## Throughput and memory

| subject | measured |
|---|---|
| Bielik decode after the CPU sprint | 12.55 → **17 tok/s** (bit-identical output) |
| Qwen-3B with `OVERFIT_REPACK_GEMV` | **24.4 tok/s** (+30%) |
| gap to llama.cpp | **~1.13× uniform** — this is *not* parity; always best-of-N on both sides |
| QLoRA fine-tuning, 3B | ~3 GB RAM |
| Phi-4 14B Q4_K_M | ~3.8 tok/s |
| Android (Motorola Edge 50 Fusion), 0.5B Q4_K | ~3.8 tok/s |
| FFN and LM head | at the DRAM floor |

## Anomaly guard

| subject | measured |
|---|---|
| false positives, 2026-08-02, older build | **112/day** |
| false positives, 24 h to 2026-08-05 | **5/day** (Poisson 1–9), 292 cycles, 0 failures |
| of which `GcGen2HeapBytes` | 108/day → **0** — but partly an artefact of young heaps, **not** purely the floor repair |
| non-heap channels | 4/day → 5/day, i.e. **no change** — the floor repairs did not buy detection with noise |
| peer memory gap, generation `7f5fb9f88c` | median **18.8 MB**, clears the 9.52 MB floor in 52% of samples, top-2 set changed **0 of 64** times |
| peer memory gap, generation `7765564ff6` | median **5.9 MB**, clears the floor in **0 of 108** samples |
| peer visibility | **blind to a single OOMKill**, by construction |

## Semantic navigator (`Tools/SemanticNavigator`), 2026-08-06

| phase | cost |
|---|---|
| open solution (25 projects) | 3.2 s |
| build semantic model (1587 documents) | 3.9 s |
| CLI, per invocation | ~9 s |
| MCP server startup | 7.0–7.7 s once |
| warm **symbol** query (`refs`/`impls`/`callers`) | **3–23 ms** |
| warm `find_unused`, `Cli` / `Anomalies` / `Main --public` | 0.09 s / 2.75 s / **6.02 s** (694 candidates) — 9–45 ms per candidate |

**Measure twice.** Every warm query above was run in two passes: the first costs 100–1990 ms while
per-document state faults in, the second 3–23 ms. A single sample would have reported a number two orders of
magnitude too high.

## Measurement traps already paid for

- **Cross-process before/after does not work on this box.** A prefill change read +5% while the *untouched*
  decode path in the same run moved +32%. Interleave configurations **ABAB in one process** and time a canary
  path in every sample.
- **`OVERFIT_TILED_PREFILL` is dead whenever a `*.gguf.repack` sidecar sits next to the model** —
  `IsPrepacked` short-circuits it, so both arms run identical code. **Count the paths actually taken** before
  believing any kernel A/B.
- **Wrong job for the workload.** The shared `BenchmarkConfig` pins `InvocationCount=1`/`UnrollFactor=1`,
  which suits multi-millisecond model runs and leaves a microbenchmark measuring timer noise: a ~15 µs
  operation produced `RatioSD` 0.44 and a phantom 1.61× regression that was 1.01 under `[SimpleJob]`.
- **The scaffolding outweighing the subject.** A saturating `float`→`long` cast in a synthetic branch body
  made a non-inlined call measure *faster* than inlining it. **If a result is backwards, suspect the
  benchmark before the runtime.**
- **A thermally-throttled or loaded box invalidates an A/B.** Detect it with a canary — re-measure an
  unchanged path; if it moved, the box did, not your change.
