# llama.cpp CPU inference — analysis & Overfit decode-kernel plan

**Status:** analysis complete (2026-05). §5 **steps 1, 2 and 3 all done &
A/B-measured.** Qwen-3B decode on this dev box improved **2.71 → 14.56 tok/s
(5.4×)** over Overfit's own starting point, RAM **~14.4 GB → 4.40 GB (−69 %)**,
0 B/token preserved. Parity verified vs F32 for both Q8 (2.5: 32/32) and Q4_K_M
(3.4: 29/32, worst swing 2.16). Native loads: Q8_0 (2.4/2.4b), Q4_K (3.2b),
Q6_K (3.3c).

⚠️ **Same-file A/B correction (2026-05-20).** Re-benchmarked LLamaSharp 0.27.0
on the *same* `qwen.q4km.gguf`: **27.5 tok/s @ 3.2 GB**. The earlier "Overfit
1.51× faster than LLamaSharp" line was wrong (it compared Overfit-Q4_K_M against
LLamaSharp's *FP16* number, 9.67 tok/s). Key diagnostic: FP16→Q4_K_M sped
llama.cpp up **2.85×** but Overfit only **~1.0×** → Overfit decode was *not*
bandwidth-bound (overhead-bound). **First lever acted on — GQA K/V-once**
(project each KV group's K/V once, not once per Q head; 8× redundant for Qwen):
Overfit decode **13.85 → 17.2 tok/s (+24 %)**, bit-identical, narrowing the
same-file gap from ~2.0× to **~1.6×** (llama.cpp still ahead + 27 % less RAM).
Overfit's defensible edge stays allocation (1 B vs 21 KB/token), pure-managed,
AOT-clean, no native dep. Full numbers in `overfit-bench/RESULTS.md`. Step 4
(tiled prefill GEMM) and step 5 (work-stealing) remain — separate tracks.

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
| 3 | Q4_K / Q6_K decode kernels | ✅ done (3.1 kernel, 3.2 dispatch+loader, 3.3 Q6_K, 3.4 parity+A/B) | **13.28 → 14.56 tok/s (+9.6 %)**, RAM 5.85 → **4.40 GB** (−25 %); 29/32 parity |
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
At Q8 Overfit ran 13.38 tok/s vs the LLamaSharp **FP16** reference (9.67 tok/s /
~6 GB). ⚠️ That FP16-vs-Q8 comparison flattered Overfit — the same-file A/B
(step 3 status block) shows LLamaSharp on Q4_K_M at 27.5 tok/s, ~2× faster than
Overfit's best. Decode-speed parity with llama.cpp is **not** reached; see the
step 3 "instructive finding".

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

**Load performance.** Model load was profiled (Qwen-3B, FP16 GGUF, 15.4 s):
**read 65%** (disk + FP16→F32 widen), **quantize 34%** (F32→Q8), layout 1%.
Three load optimisations landed:

- **Parallel quantize** — the F32→Q8 pass (`Q8Weight.QuantizeRows`) split over
  `OverfitParallelFor` (per-row independent, bit-identical). Load 15.4 s → 12.0 s.
- **2.4 — native Q8_0 read (FFN + LM-head)** — when a GGUF tensor is already
  `Q8_0`, its blocks are read straight in: `GgufReader.LoadTensorQ8_0Raw`
  de-interleaves the `block_q8_0` layout (int8 quants + F16→F32 scale) — no
  dequant, no re-quantize.
- **2.4b — native Q8_0 read (attention)** — extends it to the per-head Q/K/V/O
  projections (Q/K/V = contiguous block slices; the output projection = a
  strided per-head block gather).

The whole decode-weight set now loads straight from a Q8_0 GGUF with no F32
round-trip. Measured on Qwen-3B (`qwen.q8_0.gguf`, 3.37 GB; warm cache, same
file): **load 2.9 s (2.4) → 1.6 s (2.4b)** — the eliminated attention
Q8_0→F32→Q8 round-trip. Decode unchanged; **32/32** greedy top-1 parity vs the
FP16-quantized engine (`Q8DecodeParityTests`).

End-to-end on the dev box, loading the Q8_0 model vs the FP16 model:
**~12 s → ~1.6 s**. That ~7× is three effects stacked — the native read (no
dequant / quantize), the Q8_0 file being ~60% of the FP16 bytes, and
(box-specific) the 3.37 GB Q8_0 file fitting in the OS page cache where the
5.75 GB FP16 file does not on this box's RAM. Load time is genuinely
page-cache-sensitive — treat the absolute figures as dev-box measurements, not
portable constants.

**Remaining:**
- Sub-second *cold* load needs memory-mapping (zero-copy, lazy page-in) — a
  separate, larger change; the read is otherwise floored by disk I/O.

### Step 3 — Q4_K / Q6_K decode kernels — ✅ DONE & MEASURED

**3.1 + 3.1b — the Q4_K × Q8_K kernel.** `Q4KWeight` (Q4_K-resident,
output-major 144-byte super-blocks) + `Q4KDotKernel` — `QuantizeActivationQ8K`
(F32→Q8_K: per-block F32 scale + 256 int8 + 16 group sums), `Dot` (the
`ggml_vec_dot_q4_K_q8_K` identity `q8d·(d·Σ scale[s]·intdot[s] −
dmin·Σ min[s]·bsumpair[s])`), `Project` / `ProjectParallel`. The AVX2 path feeds
the unsigned 4-bit nibbles straight into `vpmaddubsw` (no `vpsignb` sign trick —
they *are* the unsigned operand, unlike signed Q8×Q8) + `vpmaddwd`; scalar
fallback bit-identical.

**3.2a/3.2b — per-weight dispatch + native Q4_K loader.** `DecodeWeight` became a
4-way tagged union `{F32 | Q8 | Q4_K | Q6_K}`. Each projection in
`CachedFeedForwardBlock` / `CachedSingleHeadAttention` /
`CachedGptStack.ProjectLogits` picks its kernel from the weight's resident
format — heterogeneous K-quant files are handled per-tensor, not all-or-nothing.
`GgufReader.LoadTensorQ4_KRaw` reads native Q4_K blocks straight through; the
loader peeks layer-0 type per tensor to decide whether scratch F32 needs to be
rented. Per-head `Wo` keeps the Q8 path — its headDim=128 contraction is below
the 256-element K-quant super-block, so per-head Wo can't be K-quant
(`attn_v`'s per-head contraction is dModel=2048, so it *is* K-quant-able like
Q/K — earlier scoping that lumped Wo and V together was wrong about V).

**3.3a/b/c — Q6_K kernel + AVX2 + wiring.** `Q6KWeight` (210 B/256 elements:
128 B ql + 64 B qh + 16 B signed int8 sub-block scales + 2 B FP16 d, no dmin) +
`Q6KDotKernel` — `Dot` rewrites `d·q8d·Σₛ scales[s]·Σ_{i∈s}(q[i]−32)·q8[i]` as
`unsignedDot − 32·bsumTerm` for AVX2 (`vpmaddubsw` again, on reassembled 6-bit
quants from `ql|qh<<4`). Q6_K's 16-element sub-blocks within 32-element AVX2
groups required `Vector256<int>.GetLower()/GetUpper() + Vector128.Sum` to apply
two different sub-block scales per AVX2 group. Loader adds
`LoadTensorQ6_KRaw` + `LoadQ6KNative` + per-head `LoadQkvHeadsQ6K`; LM-head
dispatch and the decode blocks gained Q6_K branches.

**3.4 — parity + A/B on a real `qwen.q4km.gguf`.** 32-step teacher-forced
top-1 parity test against an F32-from-the-same-Q4_K_M-file baseline: **29/32**
match, worst mismatch swing **2.16** (every mismatch is a near-tie). Test
ships as `[LongFact]`.

**Three-way A/B on the same Qwen2.5-3B-Instruct base** (best-of-3, 24 timed
tokens after 4 warm-up, single stream, dev box):

| Format    | Load   | Decode      | Steady RAM | Notes                                |
|-----------|--------|-------------|-----------:|--------------------------------------|
| FP16-src  |  7.1 s | 13.29 tok/s |   5 902 MB | baseline (dequant FP16 → F32 in RAM) |
| Q8_0      |  1.7 s | 13.28 tok/s |   5 847 MB | step 2 result                        |
| **Q4_K_M**| **1.4 s** | **14.56 tok/s** | **4 396 MB** | step 3 result — best Overfit format |

Within Overfit's own formats, Q4_K_M is marginally faster than the FP16-source
path *and* 25 % leaner on RAM than Q8 (4.40 vs 5.85 GB) — the win is mostly the
RAM and load time, not throughput (FP16-src 13.29 → Q4_K_M 14.56 is only +9.6 %).
That small decode delta is itself the tell: see below.

**vs LLamaSharp — same Q4_K_M file (2026-05-20, corrected).** Re-benchmarked
LLamaSharp 0.27.0 (`LLamaSharp.Backend.Cpu`, native llama.cpp, `GpuLayerCount=0`)
on the **same** `qwen.q4km.gguf`:

| Engine | Decode | Working set | alloc/token |
|--------|-------:|------------:|------------:|
| Overfit (Q4_K + Q6_K), pre-fix  | 13.85 tok/s | 4 389 MB (committed) | **1 B** |
| Overfit, K/V-once + fuse-quantize | **17.5 tok/s** | 4 389 MB (committed) | **1 B** |
| LLamaSharp (llama.cpp)          | **27.5 tok/s** (steady) | **3 205 MB** (peak, mmap) | 21 221 B |

**On equal footing llama.cpp decodes ~1.6× faster (was ~2.0× before the K/V-once
fix) and uses ~27 % less RAM.** The previous "Overfit 1.51× faster than
LLamaSharp" claim was an artifact of comparing Overfit-Q4_K_M against LLamaSharp's
*FP16* number (9.67 tok/s); corrected, llama.cpp wins decode and RAM on the same
file.

**The instructive finding — Overfit decode was not bandwidth-bound.**
FP16 → Q4_K_M sped llama.cpp up **2.85×** (9.67 → 27.5) but Overfit only
**~1.0×** (13.3 → 13.85). llama.cpp converts the ~3.5× byte reduction almost
linearly into speed — bandwidth-bound, as §3.2 predicted. Overfit gained almost
nothing from fewer bytes → per-token overhead dominated. **First lever acted on:
GQA K/V-once.** Under grouped-query attention every Q head in a KV group shares
one K/V weight set + cache slot, but the decode recomputed the K/V projection
once per Q head (8× for Qwen 16Q/2KV). Projecting it once per group reclaimed
**+24 % (13.85 → 17.2 tok/s)** — and cut the wasted K/V weight-read bandwidth 8×,
confirming the redundancy was partly bandwidth after all. Bit-identical (same
24-token greedy sequence before/after). **Remaining levers:** fuse
activation-quantize per-group, tighter Q4_K/Q6_K GEMV unroll + more accumulators,
core-utilisation profiling. Overfit's defensible edge stays allocation (1 B vs
21 KB/token), pure-managed, AOT-clean, no native binary. Numbers in
`overfit-bench/RESULTS.md`.

**Verified:** 680 / 0 / 68 (`-c Release`); zero allocations per decoded token
(`Gpt2GenerationDemoTests.Demo_Gpt2Small_KvCacheDecode_AllocatesZeroBytesPerToken`
contract held throughout); finite logits on the Q4_K_M decode end-to-end.

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
