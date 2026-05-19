# llama.cpp CPU inference — analysis & Overfit decode-kernel plan

**Status:** analysis complete (2026-05). §5 steps 1 + 1b + the Q8 weight path
(2.3a LM-head, 2.3b FFN + attention) **done & A/B-measured** — Qwen-3B decode
**2.71 → 13.38 tok/s (4.9×)**, RAM **~14.4 GB → 5.90 GB (−59%)**, 0 B/token
preserved. Overfit now runs **1.38× faster than** the LLamaSharp reference
(9.67 tok/s / ~6 GB) at slightly less RAM. Q8 parity verified vs F32 (2.5).
Remaining: native Q8_0 GGUF load (2.4); step 3 (Q4_K/Q6_K) not started.

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
| 2 | Q8_0 weight path + INT8 dot kernel | ✅ done (FFN + LM-head + attention) | **4.01 → 13.38 tok/s (3.3×)**, RAM ~14.4 → 5.90 GB |
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

### Step 2 — Q8_0 weight path + INT8 dot kernel — ✅ DONE & MEASURED

**Design.** (a) a `block_q8_0`-style format — 32×int8 quants + 1×F32 scale per
block (`Q8DotKernel.BlockSize = 32`); (b) a symmetric activation quantizer
F32→Q8 (one pass, `scale = absmax/127`); (c) an INT8 `vec_dot` using
`Avx2.MultiplyAddAdjacent` (`vpmaddubsw`+`vpmaddwd`) + `Avx2.Sign` (`vpsignb`,
the signed×signed trick), INT32 accumulation, one FP32 multiply-add per block
for the scales.

**Layout.** Weights are stored **output-major** — row `o` is one output's full
contraction vector, in blocks of 32. The GGUF file already stores weight
matrices `[out, in]`; FFN/LM-head rows map straight across (no transpose), and
the per-head attention Q/K/V slices are contiguous file-row ranges (also no
transpose) — only the output projection needs a strided per-head gather. A
weight handle is a tagged union `DecodeWeight` { F32 `TensorStorage<float>` |
Q8 `Q8Weight` }; the decode kernels dispatch on the tag at the leaf, so the
transformer stack is precision-agnostic and the F32 GPT-1/2 path is untouched.

**Landed in three slices, each A/B-measured on Qwen-3B (best-of-3, steady-state):**

| Slice | What | Decode tok/s | Steady RAM |
|---|---|---:|---:|
| baseline | kernel track (steps 1+1b), all-F32 | 4.01–4.03 | ~14.4 GB |
| 2.3a | LM-head weight → Q8 | 4.33 | 13.5 GB |
| 2.3b-FFN | FFN gate/up/down → Q8 (~⅔ of weights) | 10.05 | 6.84 GB |
| 2.3b-attn | attention Q/K/V/O (per-head) → Q8 | **13.38** | **5.90 GB** |

**Result: 4.01 → 13.38 tok/s (3.3×), RAM ~14.4 → 5.90 GB (−59%), 0 B/token
preserved, 672 tests green.** The full decode matmul path (FFN, LM-head,
attention) is now Q8-resident; only embeddings, norms and biases stay F32.
Overfit runs **1.38× faster than** the LLamaSharp reference (9.67 tok/s / ~6 GB)
at slightly less RAM — past the launch-credibility milestone §5 called for.

**Honest caveats:**
- The 2.3b-FFN jump (+132%) is larger than the ~2× that quantizing ⅔ of the
  weight bytes alone predicts. Most likely the F32 13.5 GB footprint was under
  memory pressure on the dev box and 6.84 GB relieves it — i.e. part of the gain
  is paging relief, not pure kernel speedup. Not isolated; stated as the leading
  hypothesis.
- The 2.3b-attn jump (+33%) also beat the first estimate (attention is only ~12%
  of weight *bytes*). It is consistent with attention being ~⅓ of decode *time*:
  before this slice FFN/LM-head were already Q8-fast, so the unchanged F32
  attention dominated the remaining per-token cost (Amdahl) — quantizing it
  released a larger share than its byte fraction suggests.
- Q8_0 decode is **lossy** (8-bit weights *and* activations) — by construction,
  not a bug. Step 2.5 verified it: a teacher-forced top-1 parity test
  (`Q8DecodeParityTests`) loads the same qwen.gguf as F32 and as Q8 and compares
  32 decode steps — **28/32 greedy top-1 match, all 4 flips at genuine near-ties**
  (F32's own margin for its pick ≤ 0.47 logits; worst Q8/F32 ranking swing 1.39).
  Q8 never picks a token F32 strongly rejected. Verdict: the Q8 decode path is
  correct.

### Step 2.5 — parity verification — ✅ DONE

`Q8DecodeParityTests` (a `[LongFact]` — loads the real 3B model) teacher-forces
32 decode steps: it greedy-generates an F32 reference token sequence, then re-runs
the Q8 path feeding the F32 token at every step so a single argmax flip cannot
cascade — each step measures Q8 quantization error in isolation. The in-process
F32 reference comes from `GgufLlamaLoader.Load(path, quantize: false)`, a flag
added in 2.5 that loads every weight as F32 (the pre-quantization decode path).
Result: 28/32 greedy top-1 match, all mismatches at genuine near-ties — see the
caveat above.

**Remaining Q8 work:**
- **2.4** — GGUF loader reads `Q8_0` tensors straight from the file (today it
  up-casts to F32 then re-quantizes on load; reading native Q8_0 blocks removes
  the transient F32 buffer and the load-time quantize pass).

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
- Honest ceiling (original): pure-managed C# cannot emit every intrinsic llama.cpp uses, but the decode-critical ones (INT8 SIMD) it *can*. The original cautious target was closing 3.75× → ~1.2–1.5×, not exact parity. **Outcome (steps 1+1b + full Q8 weight path): measured 13.38 vs 9.67 tok/s — Overfit is 1.38× faster than the LLamaSharp reference at slightly less RAM**, well past the cautious estimate. Parity is now also verified — step 2.5's teacher-forced top-1 test confirms the Q8 decode path tracks F32 (28/32 greedy match, every flip at a genuine near-tie).
