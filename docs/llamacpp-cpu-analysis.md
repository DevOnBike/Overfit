# llama.cpp CPU inference — analysis & Overfit decode-kernel plan

**Status:** analysis complete (2026-05). §5 steps 1 + 1b **done & A/B-measured** —
Qwen-3B decode **2.71 → 4.01 tok/s (+48%)**, 0 B/token preserved; gap to llama.cpp
closed from 3.75× to **2.41×**. Steps 2–3 (quantization) not started.

**Origin — the benchmark that triggered this.** Same Qwen2.5-3B GGUF, single-stream CPU decode:

| | Throughput | RAM | Effective bandwidth |
|---|---:|---:|---|
| Overfit (F32 up-cast) | 2.58 tok/s | ~14 GB | 12 GB/token × 2.58 = ~31 GB/s |
| LLamaSharp (native llama.cpp) | 9.67 tok/s | ~6 GB | — |

~3.75× slower, ~2.3× more RAM.

---

## 1. Why this document

The FP16-resident experiment ("Slot 2c" in `ROADMAP.md`) tried to close this gap by halving weight bytes. It was **refuted** — measured −35% throughput; see the Slot 2c post-mortem in `ROADMAP.md`. To find the *real* levers, llama.cpp's CPU inference path was analysed directly (shallow clone, read-only).

**Headline:** the 3.75× gap is **two fixable things**, and — correcting an earlier pessimistic read — **neither needs a capability .NET 10 lacks.**

---

## 2. The gap, decomposed

The 3.75× is not one deficit. It is:

1. **Kernel quality.** Overfit's decode matmul is structurally inefficient (see §3.2 / §5 step 1). Its 31 GB/s effective rate is *under* the ~50–80 GB/s DRAM ceiling — the kernel is bottlenecked by its own access pattern, not by memory or compute.
2. **Bytes per weight.** Overfit streams F32 weights (32 bits). llama.cpp streams quantized weights (Q4_K ≈ 4.5 bits) — ~7× fewer bytes from RAM, and the matmul is integer-SIMD, not float.

**Correction to the earlier "decode is compute-bound" framing.** That was imprecise. Overfit's *current kernel* is bottlenecked by its access pattern (an outer-product GEMV that re-streams the output vector). A *properly written* GEMV becomes DRAM-bandwidth-bound on the weight stream — and only *then* does reducing bytes per weight (quantization) help. The order matters: **kernel first, then quantization.** FP16-resident was attempted out of order, and on the wrong precision — quantization beats FP16 on every axis.

---

## 3. How llama.cpp does it

### 3.1 Key files (paths under `ggml/src/` in the llama.cpp tree)

| File | Role |
|---|---|
| `ggml-cpu/ggml-cpu.c` | CPU backend core: `ggml_compute_forward_mul_mat`, threadpool, barriers, work planning. Per-type dispatch table `type_traits_cpu[]`. |
| `ggml-cpu/arch/x86/quants.c` | x86 AVX2/AVX512/VNNI per-type `vec_dot` kernels (Q4_0, Q8_0, Q4_K, Q6_K…) — the decode GEMV inner loop. |
| `ggml-cpu/quants.c` | Scalar/reference `vec_dot` + activation quantizers (`quantize_row_q8_0/q8_K`). |
| `ggml-cpu/vec.h` / `vec.cpp` | F32/F16/BF16 `vec_dot` and vector primitives. |
| `ggml-cpu/simd-mappings.h` | SIMD-width / FMA / F16↔F32 macro layer (the F16C path lives here). |
| `ggml-cpu/llamafile/sgemm.cpp` | tinyBLAS: register-blocked, cache-tiled GEMM for prefill. |
| `ggml-cpu/repack.cpp` | Block-interleaved weight repacking + `gemv`/`gemm` on interleaved weights. |
| `ggml-common.h` | Canonical quant block formats (`block_q4_0`, `block_q8_0`, `block_q4_K`, `block_q6_K`, `block_q8_K`). |

### 3.2 Decode — single-token GEMV (batch = 1)

Decode reads the entire weight set once per token; done right it is **memory-bandwidth-bound**. llama.cpp's speed comes from reading fewer bytes and doing the dot product in integer SIMD:

- **Activation quantization.** The F32 activation row is converted *once* into the weight's companion type — Q8_0 or Q8_K (one cheap pass over a ~2–3.5k-float vector).
- **Integer dot product.** Both operands are 8-bit ints. The MAC uses `vpmaddubsw` (`_mm256_maddubs_epi16`, unsigned×signed bytes → 16-bit) + `vpmaddwd` (`_mm256_madd_epi16`, 16-bit → 32-bit), or one-instruction `vpdpbusd` on VNNI hardware. **One 256-bit instruction = 32 INT8 MACs** (64 on AVX-512). Signed×signed uses the `vpsignb` (`_mm256_sign_epi8`) trick to fit the unsigned×signed form.
- **Scales applied per block, not per element.** Per-block FP16 delta `d` → one FP32 `fma` per 32- or 256-element block. INT32 accumulation within a super-block stays exact.
- **Memory layout.** Quant blocks are interleaved-by-design — quants and scales contiguous per block, one cache-line region per block. Weight rows are row-contiguous → linear, prefetcher-friendly.
- **Threading.** GEMV work is partitioned across **output rows** — each thread owns a disjoint contiguous band; no reduction, no false sharing.
- **F16C is barely used here.** For *quantized* decode the only F16 values are per-block scales (one per 32/256 weights), converted scalar/amortized. F16C matters only for a pure-F16-weight matmul, which is not the path you want.

### 3.3 Prefill — batched GEMM (batch > 1)

When the activation has many rows, work becomes compute-bound and llama.cpp switches strategy:

- **tinyBLAS** (`sgemm.cpp`) — register-blocked GEMM. An `RM×RN` tile of C is held entirely in vector registers (typically 4×6 or 4×3); the K loop loads `RM`+`RN` vectors and issues `RM×RN` FMAs, so each loaded operand is reused `RM` or `RN` times. Cache-blocked so the B working set fits L2.
- **repack path** — weights pre-permuted at load into interleaved super-blocks (`block_q4_Kx8` = 8 rows interleaved); one kernel produces 8 output columns per activation. `forward_mul_mat` picks `gemm` when rows > 3, else per-row `gemv`.

### 3.4 Threading model

- **Persistent threadpool** — threads created once, live for the process.
- **Hybrid spin-then-wait wakeup** — workers spin-poll an atomic graph-generation counter (`PAUSE`), then fall back to a condvar. Near-zero wakeup latency without burning idle cores — essential because a decode graph is hundreds of tiny ops.
- **Per-op sense-reversing atomic barrier** — no task graph; topological order + barriers.
- **Intra-matmul work-stealing** — threads claim chunks via an atomic `fetch_add` counter; load-balances uneven chunk cost.

Overfit already has the analogue: `OverfitParallelFor` (persistent threads, bulk-wake, 0-alloc dispatch).

---

## 4. Portability to pure-managed C# / .NET 10

| Technique | Portability | Notes |
|---|---|---|
| Quant block formats (Q4_K/Q6_K/Q8_0 structs) | **As-is** | Pure data layout. Biggest win, costs nothing structurally. |
| INT8×INT8 dot (`maddubs`+`madd`) | **With effort** | `Avx2.MultiplyAddAdjacent` = `vpmaddubsw`/`vpmaddwd`; `Avx2.Sign` = `vpsignb`. The whole quantized `vec_dot` is reproducible. |
| VNNI single-instruction dot | **With effort, gated** | `AvxVnni.MultiplyWideningAndAdd`; detect `AvxVnni.IsSupported`, fall back to `maddubs`. |
| F32 register-blocked GEMM (tinyBLAS) | **With effort** | `Vector256<float>` + `Fma.MultiplyAdd`. 4×3 tiles realistic; 4×6 may spill (16 YMM regs) — measure. |
| Block-interleaved weight repacking | **As-is** | Pure memory permutation at load. |
| Persistent pool + work-stealing chunk counter | **As-is** | `Interlocked.Add`/`CompareExchange`. Build on `OverfitParallelFor`. |
| Sense-reversing barrier, spin-then-wait | **As-is / with effort** | `Interlocked` + `Thread.SpinWait` + `ManualResetEventSlim`. |
| Per-vector F16C (`vcvtph2ps`) | **Not portable — but irrelevant** | .NET 10 has no per-`Vector` Half→float. In the quantized path F16 is only sparse per-block scales → amortized via `TensorPrimitives.ConvertToSingle`. **Does not block the quantized kernel.** |
| `_mm_prefetch`, Intel AMX | **Not portable** | No .NET intrinsic. Minor / niche; ignore. |

**Bottom line:** the entire quantized decode + prefill story ports to managed C#. The only lost pieces (per-vector F16C, software prefetch, AMX) are off the critical path.

---

## 5. Plan — what to do, in what order, when

| # | Step | Status | Measured / expected |
|---|------|--------|---------------------|
| 1 | Blocked GEMV + parallelize FFN/LM-head matmul | ✅ done | 2.71 → 3.63 tok/s (+34%), 0 B/token kept |
| 1b | Head-parallel attention | ✅ done | **3.63 → 4.01 tok/s (+10.5%)** — cumulative **2.71 → 4.01 (+48%)** |
| 2 | Q8_0 weight path + INT8 dot kernel | not started | ~3.5× fewer weight bytes; fixes RAM |
| 3 | Q4_K / Q6_K decode kernels | not started | ~7× fewer bytes; same-file benchmark vs llama.cpp |
| 4 | Tiled prefill GEMM (tinyBLAS-style) | separate track | prefill / batch>1 only |
| 5 | Intra-matmul work-stealing chunk counter | minor | fold in opportunistically |

### Step 1 — Parallelize the decode matmul — ✅ DONE & MEASURED

This doc originally proposed a "dot-product GEMV" here, on the theory that the
outer-product `Accumulate` re-streams the output vector through L2/L3. Two
sub-changes were implemented and **A/B-measured on Qwen-3B**; the theory was only
partly right.

- **Blocked-output `Accumulate`** — process the output in L1-resident tiles so it
  is not re-streamed. Bit-identical. **Measured +2.2%** (2.71 → 2.77 tok/s). The
  output re-streaming was a near-non-issue: the dev CPU (Ryzen 9 9950X3D) has a
  huge L2 + 3D V-cache, so the "re-streamed" output was always cache-resident.
  The earlier "~2–3×" estimate for this was simply wrong.
- **Parallelize FFN + LM-head matmul** — `SingleTokenProjectionKernel.ProjectParallel`
  rewritten on the zero-alloc `OverfitParallelFor` dispatcher (splits the output
  dimension into one band per worker), wired into `CachedFeedForwardBlock` and
  `CachedGptStack.ProjectLogits`. **Measured +31%** (2.77 → 3.63 tok/s). This was
  the real lever: the decode matmul was single-threaded, so ~33 GB/s ≈ a
  single-core DRAM ceiling; threading FFN/LM-head pulls weights from DRAM on
  multiple cores toward the aggregate ceiling.

The full output-major **layout flip was not done** — it has real blast radius
(the kernel is shared with the GPT-1/2 path, whose weights are the trainable
model's `TensorStorage`; flipping it breaks zero-copy + LoRA weight-visibility).
Blocked GEMV captured the cache angle for +2% without it; the structural win came
from threading, which needs no layout change.

**Result: 2.71 → 3.63 tok/s (+34%), 0 B/token preserved, 659 tests green.** Gap to
llama.cpp: 3.75× → 2.66×.

**Why not ~2×:** the attention Q/K/V/O projections are below the parallel
work threshold (each per-head matmul is too small to split per-projection) and
stay single-threaded — attention is ~⅓ of the per-token weight streaming.

### Step 1b — Head-parallel attention — ✅ DONE & MEASURED

`CachedMultiHeadAttention.Decode` parallelised across **KV groups** via
`OverfitParallelFor` — one worker per KV head, each owning a disjoint KV-cache
slot and a contiguous run of Q heads. Splitting by KV group (not by Q head)
keeps every cache write owned by exactly one worker → no write race, and needs
no split of `CachedSingleHeadAttention`. Each head writes its own band of
`_headOutputs`; a sequential reduction sums them (ascending → bit-identical to
the old per-head loop).

For GQA models (Qwen-3B: 2 KV heads) this is 2-way parallel; for MHA (GPT-2) it
is one worker per head. Attention decode is memory-bound, so even 2-way pulls
most of the available aggregate bandwidth.

**Measured: 3.63 → 4.01 tok/s (+10.5%).** 0 B/token preserved, 659 tests green.

**Cumulative (steps 1 + 1b): 2.71 → 4.01 tok/s (+48%); gap to llama.cpp 3.75× → 2.41×.**

### Step 2 — Q8_0 weight path + INT8 dot kernel

**What:** (a) `block_q8_0` format (32×int8 + 1×F16 scale); (b) an activation quantizer F32→Q8_0 (one cheap pass per matmul); (c) an INT8 `vec_dot` using `Avx2.MultiplyAddAdjacent` + `Avx2.Sign`, INT32 accumulation, one FP32 `Fma` per block for the scale; gate an `AvxVnni.MultiplyWideningAndAdd` fast path behind `AvxVnni.IsSupported`; (d) a GGUF loader path that keeps Q8_0 tensors quantized in RAM.

**Why:** weights drop 32→~9 bits → ~3.5× less RAM traffic on the now-bandwidth-bound kernel, and directly fixes the ~2.3× RAM gap. **The F16C limitation does not apply** — F16 survives only as per-block scales.

**Verify:** logit parity vs F32 (top-1 match for greedy, logits within Q8_0 noise); A/B throughput + RAM.

### Step 3 — Q4_K / Q6_K decode kernels

**What:** same INT8 machinery as step 2, plus nibble unpacking (`Vector256<byte>` and/shift) and 6-bit sub-scale decode. Port the structure of `ggml_vec_dot_q4_K_q8_K` / `q6_K_q8_K`: INT32 accumulation within the 256-element super-block, `bsums`-based min correction, one FP32 fma per super-block.

**Why:** Q4_K_M is what real Qwen-3B GGUFs ship as. Full ~7× byte reduction → best RAM, and lets Overfit benchmark the *same file* llama.cpp uses — removing the F32-vs-quant apples-to-oranges from the headline comparison.

**Verify:** top-1 logit match vs the F16 baseline on a canonical prompt; max abs logit diff within the Q4_K expected range.

### Step 4 — Tiled prefill GEMM *(separate track)*

A 4×3 register-tiled `Vector256<float>` GEMM with an L2 block on N, for `batch > 1`. Helps prompt processing (TTFT) and batched training — **not** single-stream decode. Belongs with the existing ROADMAP "Prefill: multi-token batched matmul" / "B > 1 batched training" work, not the decode track.

### Step 5 — Work-stealing chunk counter

An `Interlocked.Add`-based shared chunk counter so threads steal work when chunk costs differ. Marginal for uniform GEMV; fold in opportunistically while touching threading in steps 1–2.

### Sequencing & timing — "kiedy"

- **Steps 1 + 2 are one coherent sprint (~1.5–2 days).** Together they plausibly take decode from 2.58 → **~7–9 tok/s — near parity** — *before* touching K-quants. This is the launch-credibility milestone: external reviewers will benchmark Overfit against LLamaSharp, and "3.75× slower at 14 GB" is not a defensible launch position; "competitive within ~1.2–1.5× while pure-managed / zero-alloc / AOT-clean" is. **Recommend: do steps 1–2 before public launch.**
- **Step 3 (~2–3 days) soon after** — needed for an honest same-GGUF benchmark and the best RAM number. Can land just after launch if time is tight.
- **Step 4** rides with the already-planned prefill / batched-training track — not gated on the decode work.
- **Step 5** is opportunistic, no scheduling needed.

All estimates are rough; treat the throughput numbers as hypotheses to be A/B-verified, not promises.

---

## 6. Measurement discipline

Every step lands with: a parity test (output correctness vs the prior path), an A/B decode-throughput measurement on the real Qwen-3B model (best-of-N, steady-state), and a RAM measurement where relevant. Document unflattering results honestly — the Slot 2c post-mortem is the template. No step is "done" on an estimate.

---

## 7. Relationship to the ROADMAP

- **Steps 2–3 are the concrete kernel design for ROADMAP "Slot 2b"** (quantized weight storage at inference) — Slot 2b can now reference this document instead of being a sketch.
- **Step 1 is a new prerequisite** not previously in the ROADMAP — the single highest-leverage decode fix, and it requires no quantization.
- **FP16-resident ("Slot 2c") is superseded.** Quantization reduces bytes more than FP16 and its integer dot product is both faster and fully portable; FP16-resident is not worth revisiting. See the Slot 2c post-mortem.
- Honest ceiling: pure-managed C# cannot emit every intrinsic llama.cpp uses, but the decode-critical ones (INT8 SIMD) it *can*. A realistic target is closing 3.75× → ~1.2–1.5×, not exact parity.
