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

## The bar

Never ship, claim or commit a perf win you have not measured on a stable box, best-of-N on **both** sides,
with one lever isolated. `overfit-perf-claim-auditor` owns the verdict on any performance claim and must not
be substituted for.

Related: [`measured-baselines.md`](measured-baselines.md), [`claim-to-test.md`](claim-to-test.md),
[`autoresearch-program.md`](autoresearch-program.md) for when the question is "what value" rather than
"what mechanism".
