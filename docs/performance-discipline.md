# Performance discipline — how a perf claim is earned here

`CLAUDE.md` carries the rule; this file carries the incidents behind it. Everything below happened in this
repository, and every one of them produced a number that looked authoritative and was worthless.

## The order of work

**Correctness first, in a separate pass.** Write the clearest expression that gets the maths right and pin
it with a parity test — cosine against ONNX Runtime, a finite-difference gradient check, a known-good
output. Only once that is green do you make a **second** iteration for speed, keeping the validated version
as the A/B baseline and the parity test as the guard that the fast path still matches.

Do not fuse the two passes. A clever kernel written before its correctness is proven is unverifiable, and a
change that alters behaviour and timing at once cannot be A/B-isolated. Winograd (parity cos 1.0 first,
*then* measured, *then* reverted) and whole-matrix Q4_K attention (parity pinned, *then* a go/no-go
micro-bench before any refactor) were both done this way.

**Write the benchmark before the argument.** Put the question into `Sources/Benchmark` as a BenchmarkDotNet
class with both shapes side by side and `[Benchmark(Baseline = true)]` on the old one. This is cheap for
anything expressible as a small A/B — loop shapes, call shapes, allocation strategies, kernel variants — so
default to writing it rather than reasoning about it.

A worked failure: Theil–Sen shipped with 66 correctness tests and zero benchmarks, and hid 8.1 ms per call,
99% of it inside `Sort()`. The tests were right about everything they measured.

## Four ways a benchmark lies

### Wrong job for the workload

The shared `BenchmarkConfig` pins `InvocationCount=1` / `UnrollFactor=1`, which fits multi-millisecond model
runs and turns a microsecond routine into timer noise. A ~15 µs operation produced `RatioSD` 0.44 and a
phantom 1.61x regression that was 1.01 once re-run under `[SimpleJob]` with default invocation counts.

Check `RatioSD` and BenchmarkDotNet's own warnings before believing any ratio.

### The scaffolding outweighs the subject

`ElseRefactorBenchmark` reported a non-inlined call as *faster* than inlining it, because the synthetic
branch body contained a saturating `float`→`long` cast whose cost depends on where it lands. It was
measuring the cast.

**If a result is backwards, suspect the benchmark before the runtime.** Settle it with `--disasm` and a
variant with the suspect operation removed.

### The lever is not live

`OVERFIT_TILED_PREFILL` is a dead flag whenever a `*.gguf.repack` sidecar sits beside the model, because
`IsPrepacked` short-circuits it. Both arms ran an identical mix and the "measurement" was noise.

**Count the paths taken** — a temporary counter in the dispatcher — before believing any kernel A/B. A flat
1.00 is the signature of this, not evidence of "no difference".

### The box moved, not the code

Cross-process before/after does not work here: a prefill change read as +5% while the untouched decode path
in the same run moved +32%. Interleave the configurations run-by-run in ONE process (ABAB…, never
all-A-then-all-B) and time an unchanged **canary** path in every sample — if the canary shifted, the box
did.

The decode spin-pool assumes dedicated cores and is sensitive to background load.

## Negative results are the output

They are the most valuable thing this section produces, and this repository has reverted more measured
changes than it has kept:

| Change | Outcome |
|---|---|
| Register blocking (direct conv) | regressed, reverted |
| K-blocking + A-packing (im2col GEMM) | regressed, reverted |
| Winograd F(2,3), 3x3 stride-1 | parity cos 1.0, **+79%** slower on deepcnn (119.7 → 214.4 ms) — sequential scalar transforms, 16 small GEMMs and a 16x U/V/M blow-up beat the 2.25x FLOP cut |
| AVX-512 decode port | regressed, reverted |
| Bias in the Q4_K tiled prefill GEMM | **0.999x, an exact tie.** A path census showed `bias.IsEmpty` barred 88% of prefill dispatches from the tiled kernel; lifting it changed nothing, because `ProjectBatchedWeightStationary` already amortises weight decode across the row tile — the same thing the tiling does. The "~3x" in the kernel docs is against re-decode-per-row, not against weight-stationary |
| `OverfitPool<T>` | 3x to ~3000x slower, deleted |

The wins were the *opposite* of the obvious move: `TensorPrimitives` bulk-SIMD beat a hand micro-kernel, and
the simple register-blocked GEMM beat the cache-blocked one. **The structure of the data around a technique
decides, not the technique** — which is also why a technique that won in one kernel must not be extrapolated
to another without its own measurement.

## Eleven rules that survived a session of nine refuted hypotheses

From the 2026-07-22 prefill work (249 → ~283 tok/s). **Nine mechanism hypotheses were killed by
measurement, and every one of them sounded coherent beforehand.** These are what was left standing. They
are transcribed here from a private note because knowledge that lives outside the repository cannot be read
by anyone else.

1. **Two identical arms in the table are the cheapest canary there is.** A tile sweep showed `Tiled` and
   `Tiled_Cols8` — *the same configuration* — **21% apart**, with error bars of ±9%. Without that accidental
   duplicate the whole table would have been believed. Put an identical arm in deliberately.

2. **A one-armed measurement after a hot-path change is worthless.** A change in `attn_scores` "gave"
   280 → 292 tok/s — while `attn_kv`, untouched, also moved 141 → 121 ms. That was the box drifting. ABAB
   with untouched components as canaries gave an honest 1.01x.

3. **An impossible ordering means a broken benchmark, not a discovery.** A probe showed 512-bit *slower*
   than 256-bit. Cause: a helper with five vector parameters that the JIT declined to inline, so the
   measurement was of the calling convention. Inlined, the effect vanished entirely.

4. **`stackalloc` never reaches registers.** A roofline lost 2.8x to this — 0.79 instead of 2.19 TFLOP/s,
   below a real matmul, which is impossible for a loop that touches no memory. Use named locals with
   constant indices.

5. **Ablation inside the real kernel beats a microbenchmark.** Flags that disable individual fragments
   (`AblateF16Scales` and friends) give each part's share of an actual run; a microbenchmark measures the
   scaffolding around it.

6. **Never extrapolate a technique between kernels.** The same AVX-512 port gave **+13.8% in Q4_K and −20%
   in Q6_K**. The same scale hoisting gave 13% in Q4_K and 2.6% in Q6_K where 13% was predicted. A wider
   vector is not a property of the ISA — it is the ratio of broadcast cost to work done per broadcast, and
   that is per kernel.

7. **The ceiling has to match the instruction mix the code actually issues.** "78% of the float ceiling" came
   from dividing logical MACs by the floating-point ceiling, while the kernel issues one `vpmaddubsw` per 32
   MACs. Against the right ceiling it was 15%. See `PeakQ4KShape` in `MachineRooflineBenchmark`.

8. **Compute throughput in the repo, not in a script.** A matmul formula applied to a quantisation benchmark
   produced a fictitious 29.6 TFLOP/s. Hence `Sources/Main/Diagnostics/Throughput.cs` and a `WorkAmount`
   declared next to the benchmark.

9. **On this machine, cache blocking for streaming access does not work — three refutations in a row.** In
   the conv GEMM path: K-blocking regressed; N-panel grouping gave 0.8%, under the noise; and "fit the
   working set in 2 MB/core" showed **no cliff at all and a positive correlation** (A = 1152 KB → 658
   GFLOP/s, 9216 KB → 737). The reason is hardware: 128 MB of V-cache, so data that "should" stream from
   DRAM comes from L3. Goto/BLIS structure was designed for machines with sharp cache cliffs; this one has
   none. **Do not propose another blocking scheme without re-measuring the throughput curve first.**

10. **The cheapest measurement is the one you do not have to write.** The 2 MB/core hypothesis was settled by
    an **existing** profiler — `ConvGemmPartProfileTests`, a `[LongFact]` flipped to `[Fact]` for twenty
    minutes, zero blocking code written, unambiguous result. Before building a harness, check whether the
    repository already has one that answers the question.

11. **A controlled comparison inside one table beats a regression across the whole table.** conv9/10/11/12/13
    have identical M, K and A and differ only in N — 737 vs 287 GFLOP/s. That isolates N as the causal
    variable far harder than any fit across all layers. Look for pairs that differ in one parameter before
    you start modelling.

**What actually produced this session's gains**: one model that *predicted* — amortising fixed work per
block, which got the scaling with tile width right and the end-to-end result within 0.6% — and not the ideas
that sounded good.

## Two more measured results worth not re-discovering

- **`OverfitParallelFor` in decode, `Parallel.For` everywhere else.** A fair sustained benchmark:
  `ForDecode` **455 µs / 0 B** against `Parallel.For` at **2059 µs / 925 KB** — 4.5x and allocation-free.
  It does not generalise: migrating `Conv2D` to it measured **+13% MNIST wall time** and was reverted. The
  decode spin-pool assumes dedicated cores, which decode has and a training epoch does not.
- **Loop shape is not the lever; the declared type is.** `for` vs `foreach` over an array is ~2 ns and the
  direction reverses with size. An interface costs **2.4x** (`foreach`) to **4.6x** (indexing) plus 32 B for
  the enumerator. Monomorphism does **not** guarantee zero allocation — that refuted an earlier explanation
  of my own. Do not "tidy" a `T[]` into `IReadOnlyList<T>`.

## The bar

Never ship, claim or commit a perf win you have not measured on a stable box, best-of-N on **both** sides,
with one lever isolated. `overfit-perf-claim-auditor` owns the verdict on any performance claim and must not
be substituted for.

Related: [`measured-baselines.md`](measured-baselines.md), [`claim-to-test.md`](claim-to-test.md),
[`autoresearch-program.md`](autoresearch-program.md) for when the question is "what value" rather than
"what mechanism".
