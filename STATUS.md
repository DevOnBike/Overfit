# Overfit — work status & open decision

Date: 2026-05-19. Branch: `next`. Working tree has uncommitted Slot 2c work +
GTM artifacts. Nothing committed by the assistant — git is the user's job.

---

## ⬛ DECISION DUE TOMORROW — Slot 2c (FP16-resident weights)

### What it was

ROADMAP "Slot 2c" hypothesis: GGUF FP16 weights are up-cast to F32 at load, so
decode reads 2× the bytes; keeping weights FP16-resident and widening per L1-tile
in the matmul should give **~2× throughput + ~½ RAM**. The lever came out of the
LLamaSharp benchmark (Overfit F32 2.58 tok/s, 14 GB; LLamaSharp 9.67 tok/s, 6 GB).

### What the measurement actually says — hypothesis REFUTED

Rigorous A/B, best-of-3, same `qwen.gguf` (Qwen2.5-3B FP16), CPU decode:

| Metric | F32 (baseline) | FP16-resident | |
|--------|---------------:|--------------:|---|
| Throughput | **2.58 tok/s** (2.57/2.58/2.52) | **1.68 tok/s** (1.54/1.68/1.59) | **−35%** |
| Steady-state RAM | 14.36 GB | 15.85 GB | regressed |
| Managed alloc / token | 1 B | 1 B | unchanged |
| Logits finite / correctness | ✅ | ✅ | kernel parity bit-identical |

**Why throughput fell, not rose:** Overfit decode is **compute-bound, not
bandwidth-bound** — F32 decode reads only ~31 GB/s, far below the DRAM ceiling
(~50–80 GB/s). DRAM was never the bottleneck, so halving weight bytes unblocks
nothing, and the F16→F32 widen is pure added work. Widen tile 1024 vs 8192 made
no difference (1.68 vs 1.64) → it is the conversion work itself, not call
overhead. A fused single-pass kernel could trim −35% to maybe −10–20% but
**never positive** — you cannot do less work than F32 by adding a conversion.

**Why RAM regressed:** the `ToResident` loader loads F32, converts to F16, frees
F32 — but the multi-GB F32 `float[]` buffers are Large Object Heap allocations;
after the GC collects them it **retains the LOH segments** (does not return them
to the OS), so the working set stays high. The real RAM win (~8 GB vs ~14 GB)
needs the loader rewritten to load F16 **directly** (`LoadTensorAsF16` + transpose
in Half-space, no F32 intermediate).

### Tree state

- Default GGUF load path flipped **back to F32** (`fp16Resident` defaults off).
  No regression in the default path. The working tree is in a sane state.
- FP16-resident is an explicit opt-in: `CachedLlamaInferenceEngine.LoadGguf(path, fp16Resident: true)`.
- All Slot 2c infrastructure stays and is correct — nothing wasted.

### The two options

**A) Finish it as a RAM-only opt-in** — fused single-pass kernel (−35% → ~−15%)
   + loader rewrite (direct F16 load → real ~8 GB). Result: "run 3B in ~8 GB at
   ~15% lower throughput." ≈ 1 day. Worth it **only if RAM-constrained
   deployment is a product pillar.**

**B) Park it here** — infra stays, opt-in exists, default is F32. Correct the
   ROADMAP with the measured negative result and move on.

### Recommendation: **B (park).**

The actual goal — throughput, closing the gap to LLamaSharp — is dead by
measurement. RAM-only is a consolation prize that *widens* the LLamaSharp speed
comparison (5.8× vs the current 3.7×) and does not help the launch. Better to
invest the day in work that moves the strategy (see "If parked" below). Choose A
only if "fits in 8 GB" becomes a hard selling point for the regulated/on-prem
positioning.

---

## Slot 2c — what SUCCEEDED

- **Kernel** `SingleTokenProjectionKernel.ProjectHalf` / `AccumulateHalf` — FP16
  weight matmul, widens per L1 tile. Parity test **7/7, bit-identical** to the
  F32 path (`SingleTokenProjectionKernelHalfTests`).
- `GgufReader.LoadTensorAsF16` — raw F16 tensor load, no up-cast.
- `MatrixWeight` — F32 / FP16 / `float[]` union, threaded cleanly through
  `SingleHeadWeights` / `KvHeadWeights` / `BlockWeights` / `StackWeights` /
  `CachedLlamaInferenceEngine` and the decode kernels. Implicit conversions keep
  every existing F32 construction site (GPT-1/GPT-2, binary loader, tests)
  source-compatible.
- `GgufLlamaLoader` FP16-resident path + `fp16Resident` A/B toggle on `LoadGguf`.
- **Full test suite: 666 passed / 0 failed / 65 skipped.** The shared GPT-1 /
  GPT-2 / decode-block weight structs were refactored with zero regression.
- Implementation is correct on the real Qwen-3B: finite logits, sane tokens.

## Slot 2c — what FAILED

- **Throughput goal**: FP16-resident is 35% *slower*, not 2× faster. Decode is
  compute-bound — refuted by measurement.
- **RAM goal (as implemented)**: regressed (LOH segment retention). Achievable
  only with a loader rewrite, not delivered.
- Net: the ROADMAP "~2× decode throughput" estimate was a hypothesis; the
  benchmark killed it. The single most useful takeaway: **Overfit decode is
  compute-bound** — future decode-speed work must target compute (kernels, SIMD,
  parallelism), not memory bandwidth.

## Files touched by Slot 2c (uncommitted)

New: `Sources/Main/LanguageModels/Runtime/MatrixWeight.cs`,
`Tests/LanguageModels/Runtime/SingleTokenProjectionKernelHalfTests.cs`.
Edited: `SingleTokenProjectionKernel.cs`, `GgufReader.cs`, `SingleHeadWeights.cs`,
`KvHeadWeights.cs`, `BlockWeights.cs`, `StackWeights.cs`,
`CachedLlamaInferenceEngine.cs`, `CachedSingleHeadAttention.cs`,
`CachedMultiHeadAttention.cs`, `CachedFeedForwardBlock.cs`, `GgufLlamaLoader.cs`.
(`CachedGptStack.cs` / `CachedTransformerBlock.cs` needed no change — types flow
through.) Benchmark harness: `D:\overfit-bench\` (outside the repo).

## Cleanup owed regardless of A/B

- **`ROADMAP.md` "Slot 2c" section still claims "~2× decode throughput"** — this
  is now measured-false. Must be corrected with the negative result.
- Decide whether to keep the Slot 2c infra (recommend yes — correct + tested +
  opt-in) or revert (not recommended — it's clean and the toggle is harmless).

---

## The bigger picture — other open threads

**Go-to-market (in flight, the user's action to execute):**
- `GO-TO-MARKET.md` — full GTM plan (open-core dual-license + productized
  services).
- `launch-copy.md` — Phase 2 launch drafts: Show HN + r/dotnet (post on a
  Saturday; r/dotnet rules: weekend, "Promotion" flair, rewrite in own voice).
- Anchor blog post: `blog-anchor-zero-alloc-llm-csharp.md` +
  `zero-allocation-llm-inference-csharp.html` (styled standalone page).
- Landing `index.html` — extended with blog links; "Why not X?" table now
  includes an explicit **LLamaSharp** row (also added to `README.md`).
- `README.md` — "Why not just use…?" table + "Licensing" section.
- 3 LinkedIn articles (`linkedin-business/technical/marketing.md`); the business
  one was published.
- C# Digest: Jakub Chodounsky's welcome email — a reply draft was prepared
  (genuine intro, not a pitch); **not yet sent** by the user.

**Competitor analysis (done):** LlamaLib (niche, games/XR, not real competition);
LLamaSharp (the real incumbent — 3.7k★, native llama.cpp binding, MIT, Semantic
Kernel integration). Overfit's defensible ground = pure-managed / no native
dependency / AOT / no egress. Benchmark vs LLamaSharp recorded in
`D:\overfit-bench\RESULTS.md`.

## If Slot 2c is parked (Option B) — next-best work

Both serve the strategy in a way Slot 2c does not:
1. **Embedding-model support** (ROADMAP market-priority #1) → Overfit does RAG
   end-to-end → a second launch beat.
2. **Native-AOT single-file LLM demo** — `dotnet publish` an LLM-inference exe
   with zero native binaries. The one thing LLamaSharp structurally cannot show;
   currently only a claim, not an artifact.
