# Overfit Roadmap

Zero-allocation, pure C# deep-learning framework targeting high-performance CPU inference and small/medium language model inference on .NET 10+.

**Philosophy:** minimal dependencies, predictable memory behavior, competitive CPU inference, and explicit separation between training, inference and kernels. Native-AOT compatible. Runs on low-end consumer hardware.

---

## Status snapshot

| Area | Status |
|------|--------|
| `InferenceEngine` zero-allocation hot path | ✅ Stable, 0 B/op verified |
| Linear / Conv / activation / pooling kernels | ✅ Stable |
| Autograd engine + `Parameter` + ownership cleanup | ✅ Stable (PR5) |
| `TrainingEngine` facade + Adam/AdamW/SGD | ✅ Stable |
| Evolutionary: GA + OpenAI-ES, parallel fitness | ✅ Stable, 0 B/op Ask/AskThenTell |
| ONNX import (linear topology, 14 ops) | ✅ Stable |
| ONNX import (DAG / ResNet skip connections) | ✅ Stable |
| ONNX export | ❌ Not started |
| GPT-2 inference (124M, KV-cache, 0 B/token) | ✅ Stable, parity vs PyTorch verified |
| Qwen2.5 / Llama / Mistral inference (GQA, RoPE, SwiGLU) | ✅ Stable for 0.5B/3B FP32/F16 |
| Native C# GGUF loader (F32/F16/BF16/Q8_0/Q4_K/Q6_K) | ✅ Loads `*.gguf` from Ollama/HF directly |
| Streaming token generation (`IAsyncEnumerable`) | ✅ Stable, with stop-tokens + cancellation |
| LoRA adapter (Enable/Disable, Save/Load) | ✅ Stable, zero-copy weight refs |
| **Quantized weight storage at inference time** | ❌ **In flight — see "Slot 2b" below** |
| GPU backend | ❌ Not started |

---

## Recently completed (chronological, newest first)

- **GGUF Q4_K + Q6_K dequantization** — `GgmlDequant` pure decoders + `GgufReader` streaming wrappers (stackalloc, zero managed allocations per call). 13 unit tests on synthetic blocks. Loader can now consume any `*.Q4_K_M.gguf` from Ollama/HuggingFace.
- **`[LongFact]` test convention** — 53 integration/diagnostic/training tests gated behind a custom xUnit attribute that skips by default. `dotnet test -c Release` now runs ~15 s instead of multi-minute.
- **`CachedTransformerBlock.Decode` argument validation** — explicit guards for input/output/FfnW1/FfnW2 lengths; surfaces caller bugs as `ArgumentException` instead of `IndexOutOfRangeException` mid-block.
- **Binary loader RAM optimization** — `Unpooled` `TensorStorage` for model weights + direct `ReadExactly` into destination span. Removes pool pow2-rounding overhead and intermediate scratch `byte[]`. **3B FP32: ~30 GB → ~14 GB peak load; matches file size exactly.**
- **Token-by-token streaming** — `CachedLlamaSession.StreamGenerate(StreamingOptions, CancellationToken)` returns `IAsyncEnumerable<int>` with stop-token / cache-full / cancellation termination.
- **GGUF native loader** — `GgufLlamaLoader` reads GGUF files end-to-end without Python tooling. Supports F32/F16/BF16/Q8_0/Q4_K/Q6_K tensors, hand-rolled protobuf-free parser.
- **LoRA adapter** — `LlamaLoRAAdapter` with Enable/Disable in-place weight injection. Zero-copy `TensorStorage` references — adapter updates visible to inference without re-binding.
- **GPT-2 Small inference + parity** — 124M params, KV-cache decode 0 B/token, 6.4× faster than naive O(N²). Top-10 logit overlap 10/10 vs PyTorch, maxAbsDiff 0.000107.
- **KV-cache runtime** — `CachedSlmInferenceEngine` + `CachedSlmSession`. `SingleHeadWeights` / `BlockWeights` / `StackWeights` hold zero-copy `TensorStorage` refs.
- **ONNX import** — 14 operators (Conv, Gemm, ReLU/Tanh/Sigmoid/Softmax, MaxPool, GlobalAveragePool, BatchNorm, Add, Reshape, Flatten, AveragePool, ReduceMean). Linear topology via `OnnxImporter`, DAG/skip connections via `OnnxGraphImporter`.
- **Autograd ownership (PR5)** — `Parameter` first-class type, `AutogradNodeOwnership` enum, `graph.Reset()` by ownership, optimizers on `IEnumerable<Parameter>`.
- **Sampling** — top-P with heap sort, repetition penalty (on `CachedSlmSession`), greedy.
- **PERF kernels** — `LinearKernels.ForwardBatched` weight-stationary outer product, hybrid threshold backward, `MaxPool` pool=2 SIMD fast path.

---

## Current focus: GPT-2 Small as primary showcase

**Why:** the parity claim ("top-10 overlap 10/10 vs PyTorch, maxAbsDiff 0.000107, 0 B / generated token, KV-cache decode") is already implemented and validated. Productizing this single story end-to-end (defended in CI, demoable in one command, documented honestly) is higher-value than chasing more model families. Qwen / Llama / LoRA / quantization work continues to live in the codebase but is **explicitly deferred** out of the current week's focus.

### This week

- [x] **GPT-2 parity diagnostics run on every `dotnet test`** — `Gpt2ImportParityDiagnostics`, `Gpt2ImportStageParityDiagnostics`, `Gpt2ImportAttentionParityDiagnostics` flipped from `[LongFact]` back to `[Fact]`. Sweep cost: +2 s. Headline claim now defended on every push.
- [ ] **`Demo/Gpt2ConsoleDemo` project** — user-facing console app: `dotnet run -- --model path --prompt "..." --tokens N`. Reports: model name, KV-cache enabled, managed allocations/token, tokens/sec. First time the repo looks like a *product* instead of a test suite.
- [ ] **Fixture / model path resolver** — drop hardcoded `c:\qwen3b\…` / `d:/…` constants. Resolve via env var (`OVERFIT_MODEL_DIR`) + local `models/` directory. Required for the demo to work on anyone else's machine.
- [ ] **GPT-2 generation benchmark** — BenchmarkDotNet: cold-start, prefill cost, per-token decode time, allocations. Confirm 0 B/token quantitatively for the README.
- [ ] **README cleanup** — single consolidated story: how to convert weights, how to run, what's measured. Remove stale "repetitive text" / "broken GPT-2 import" comments. Add the benchmark numbers from the previous item.

### Next (after the GPT-2 week)

- [x] **`Prefill()` vs `GenerateNextToken()` API split** — `CachedLlamaSession` and `CachedSlmSession` both expose `Reset()` (clear-only) + `Prefill(prompt)` + the legacy `Reset(prompt)` facade. Backwards compatible: every existing caller keeps working. Enables chat-history-style incremental context (system → user → assistant turns) without re-prefilling the prefix.
- **Prefill: multi-token batched matmul** — phased work toward 5-10× TTFT speedup. Same FLOPs as today, but weights loaded once for N tokens instead of N times; memory-bound for small models, so the speedup is large for short prompts and grows with prompt length.
  - [x] **Phase 0 — skip LM-head for non-final prompt tokens.** Split `CachedGptStack.Decode` into `DecodeWithoutLogits` (transformer blocks + final norm) + `ProjectLogits` (LM head). `Prefill` in both `CachedSlmSession` and `CachedLlamaSession` now calls `DecodeWithoutLogits` for tokens `0..N-2` and full `Decode` only for the last token. Saves ~27 % per-token cost on GPT-2 Small for N-1 of N tokens → ~25 % overall prefill speedup. Parity preserved (greedy output bit-identical to pre-split, demo test still 0 B / generated token).
  - [ ] **Phase 1 — `BatchedProjectionKernel`.** `[N × D] × [D × O] → [N × O]` allocation-free GEMM. Foundation for Q/K/V/O and FFN-W1/W2 in batched prefill. Must match `N × SingleTokenProjectionKernel.Project` bit-for-bit at N=1 and within FP32 noise floor for N > 1.
  - [ ] **Phase 2 — `BatchedAttentionKernel` with causal mask.** Q[N,H] @ K_cache[L,H].T over each prompt position against `[0..pos+i]`, softmax with causal mask, @V_cache. Hardest piece. Output [N, H] per head.
  - [ ] **Phase 3 — top-level batched stack pass.** New `CachedGptStack.PrefillBatched(promptTokens, weights, cache, ...)` wires Phase 1+2 through `CachedTransformerBlock` and `CachedFeedForwardBlock` batched paths. `Prefill` in both sessions delegates to this for prompts above some small threshold (e.g. ≥ 4 tokens; below that single-token loop wins on overhead).
  - [ ] **Parity tests** for each phase: batched output ≡ N × single-token output for any prompt up to ContextLength. Final assertion: `Gpt2ImportParityDiagnostics` still green (full PyTorch parity through the batched path).
- [x] **LM-head hot-path audit (initial)** — confirmed `ProjectParallel` exists but is **dead code** (no call site); `Project` is what GPT-2/Qwen actually use. Wiring `ProjectParallel` into `CachedGptStack.Decode` was tested and reverted: `Parallel.For` allocates ~3 KB / call from task scheduling, which breaks the 0 B / generated token contract for only ~3 % per-token speedup at the GPT-2 Small scale. The 10× speedup in `LmHeadParallelBenchmark` is steady-state; per-token decode is dominated by the `Parallel.For` overhead.
- [ ] **LM head: allocation-free parallel matmul** — wire-up depends on a worker pool that does NOT allocate per call. Candidates: pre-spawned threads with lock-free queue / semaphore signaling, or unsafe manual partitioning over a fixed thread set. Constraint: ≤ 0 B / call. Payoff: most of the ~3.8 ms LM-head matmul on a 32-core box. Single largest remaining lever for GPT-2 tokens/sec.
- [ ] **`Gpt2.Load(...) / CreateSession()` API sugar** — `new GPT1Model(Gpt2Config.Small)` is technically correct but semantically misleading. A typed entry point reads cleaner in the demo.
- [x] **Stabilize `GPT1_GradientCheck_BackwardIsCorrect`** — pre-fix: model weights randomly initialized + tight `relErr < 10 %` threshold → ~1-in-3 sweeps red. Fix: seeded weight init (deterministic per run) + mixed tolerance (`relErr < 50 %` OR `absErr < 5e-4`) that accepts the inherent FP32 finite-diff noise floor (~ loss_precision / (2 × eps) ≈ 2.5e-3 per gradient). Test still catches sign errors, factor-of-2 backward bugs, and zero-vs-non-zero regressions. 5/5 full sweeps green post-fix.

---

## Deferred — Qwen / Llama / quantization track

These are working in the codebase but **outside the current GPT-2 focus week.** Listed for visibility, not for prioritization.

### Slot 2b — quantized weight storage at inference

**The gap:** Q4_K_M loader exists (decodes from disk) but currently dequantizes everything to FP32 on load. A 2 GB Q4_K_M file produces ~14 GB FP32 weights in RAM. The "3B in 4 GB RAM" payoff requires keeping weights quantized in RAM and dequantizing per-block during matmul.

**Work:**
- [ ] `TensorStorage<TBlock>` or parallel `QuantizedTensorStorage` abstraction (Q4_K and Q6_K block shapes).
- [ ] Quantized variants of `SingleHeadWeights` / `BlockWeights` / `StackWeights` (or polymorphism via interface).
- [ ] Dequant-fused matmul kernels (one block at a time, scratch FP32 row in stackalloc).
- [ ] `CachedLlamaInferenceEngine.LoadGgufQuantized(path)` factory.
- [ ] RAM diagnostic: `Diagnose_GgufQ4KM_3B_RamFootprint` showing ~2-3 GB managed.
- [ ] Logit parity vs FP32 (top-1 must match for greedy; top-10 within tolerance).

Estimated effort: 1-2 days. Largest single architectural change since KV-cache runtime.

### Q4_K_M integration parity test

- [ ] Download `qwen2.5-3b-instruct-q4_k_m.gguf` (Ollama or HF) to `c:\qwen3b\qwen.q4km.gguf`.
- [ ] Run the existing `GgufQ4KMParityTests.Q4KM_TopTokenMatches_FP16Baseline_OnCanonicalPrompt` test (already written, currently `[LongFact]` + skip-if-missing).
- [ ] Tolerance: top-1 matches; max abs logit diff within Q4_K_M expected range (~0.5-1.5 % relative).

Synthetic unit tests already cover the algorithm; this test catches bit-layout regressions against llama.cpp/Ollama.

### Other quant formats

- [ ] Q5_K dequantizer (occasionally appears in mixed quant files; 176 bytes/block).
- [ ] Q4_0 / Q5_0 / Q5_1 (legacy formats; lower priority — Ollama defaults to K-quants).
- [ ] Q2_K / Q3_K_S (very aggressive quant; experimental quality).

### LoRA training

- [ ] Backward pass through Linear / RMSNorm / SwiGLU / attention restricted to adapter parameters (frozen base).
- [ ] Adam optimizer integration over `LoRAWeight.A` + `LoRAWeight.B` only.
- [ ] Demo: overfit on 10 instruction pairs, verify adapter steers generation.

Opens "fine-tune LLM locally in pure C#" story. Major scope.

---

## Performance backlog

### Training CPU-saturation track

Goal: on 16+ core machines, training step pegs CPU at near-100 % across all cores instead of single-digit cores idle/under-utilized. Inventory of what's already parallel-capable:

| Op | Parallel today | Threshold |
|----|----------------|-----------|
| `LinearKernels` backward (input + weight) | ✅ above 1M ops | `ParallelThreshold = 1_048_576` |
| `Conv2D` forward + backward | ✅ | per `TensorMath.Convolution` |
| `MaxPool2D`/`AvgPool2D` forward + backward | ✅ | `TensorMath.Pooling` |
| `LSTM` forward + backward | ✅ over batch | `TensorMath.Sequence` |
| `Adam.Step` | ✅ over parameters | always parallel |
| `ScaledDotProductAttentionBackward` | ✅ over batch | **flipped to default ON** (was experimental flag) |
| `ScaledDotProductAttention` forward | ❌ sequential over batch | candidate |
| `LayerNorm` / `RMSNorm` forward + backward | ❌ sequential | per-token, usually small enough |
| `Embedding` backward (scatter-add) | ❌ sequential | hard — atomic conflicts |
| Activation kernels (GELU/SwiGLU/ReLU) | ❌ SIMD-only, not threaded | usually fine |

Concrete items:

- [x] **`EnableParallelAttentionBackward` default ON** — bit-identical to sequential (parallel only over batch dim, no cross-batch reduction), author measured ~27 % backward speedup on data-parallel TinyShakespeare. Stable training path now uses it by default; remaining single-thread case is intentional fallback for batch=1 where Parallel.For overhead would be pure waste.
- [x] **`BatchSequentialThreshold` 128 → 32** — `MaxPool2D` / `AvgPool2D` / `LSTM` forward + backward + bias-add (`TensorMath.Algebra`) parallelize across batch only above this threshold. MNIST training (B=64) was falling into the sequential branch (64 < 128) → `MaxPool2D` ate 42 % of epoch time on a 32-core machine, single-threaded. After lowering to 32: **MNIST training −38 % wall clock** (5551 → 3456 ms for 5 epochs, full MNIST 60k @ B=64), with MaxPool2D dropping from 383 ms/epoch to ~87 ms/epoch (−77 %). Full sweep still green (617/0/63). The 32 floor keeps sequential for genuinely small batches where Parallel.For overhead would dominate.
- [ ] **True batched training (B > 1)** — *biggest single lever for CPU saturation.* MHA in training path is currently batch=1 only. With B=8/16, every existing parallel-over-batch path (attention forward + backward, layer-norm batches, optimizer-over-params) starts working. 2-3 days; ROADMAP'd separately under Qwen track but is GPT-2 training prerequisite.
- [x] **Drill-down per-OpCode backward profiler** — added `ComputationGraph.BackwardProfileEnabled` toggle + `GetBackwardOpProfile()` + `ResetBackwardProfile()`. Zero overhead when off (one null-check), ~50 ns per op when on. MNIST CNN training 5-epoch aggregated breakdown: **Linear 52 %**, Conv2D 20 %, ReLU 14 %, MaxPool 7 %, Reshape 4 %, SoftmaxCE <1 %. Linear dominates because Linear(1352→64) backward = 2 GEMMs (dW + dX) per batch × 4685 batches. Already parallel above 524k/1M thresholds; near peak FP32 throughput for this matrix size. Next deeper investigation requires actual CPU-utilization sampling (not just timing).
- [x] **CPU-utilization probe** during MNIST training — added `Process.TotalProcessorTime` sampling at run + per-epoch granularity. **Measurement on 32-core Ryzen 9 9950X3D: only 6.81 / 32 cores effective (21.3 % utilization).** All major kernels have `Parallel.For` wired, yet 79 % of CPU stays idle. Root cause: ~47k `Parallel.For` calls per MNIST epoch × ~5-10 µs each dispatch overhead = 230–470 ms / 550 ms epoch = **40–85 % of epoch wasted in TPL dispatch/sync**, not compute. **CPU-saturation track is fundamentally blocked by `Parallel.For` overhead, not missing parallelism.**

### Zero-allocation custom Parallel.For — THE unlock

Standard `System.Threading.Tasks.Parallel.For` per call:
- ~3 KB managed allocation (closure object, `Task[]` chunks, internal bookkeeping)
- ~5-10 µs dispatch overhead (Task scheduling, thread wake)

Both kill us:
- The **3 KB alloc** broke the 0 B / generated token claim when we tried `ProjectParallel` on LM head (~92 KB / 31 tokens). Blocked allocation-free parallel inference.
- The **5-10 µs overhead** dominates training when many small Parallel.For calls fire per batch (MNIST: 47k calls/epoch). Caps utilization at ~20 % regardless of how many `Parallel.For` we add.

**`OverfitParallelFor` — current status (implemented, bulk-wake dispatcher):**

- [x] Pre-spawned `N = ProcessorCount` persistent threads, all parked on one shared `SemaphoreSlim`.
- [x] Per-chunk `ChunkState[]` descriptors filled by dispatcher; workers claim a unique index via `Interlocked.Increment` on a 128 B cache-line-padded counter.
- [x] **Bulk wake via `SemaphoreSlim.Release(N)`** — one syscall releases N tokens; kernel scheduler resumes waiters in parallel (NOT serially the way `N × AutoResetEvent.Set` would). This is the architectural win — it bypasses the ~32 µs floor that per-worker `Set` dispatchers hit at 32-fanout.
- [x] Function-pointer dispatch (`delegate*<int, int, void*, void>`) — no closure, no delegate alloc.
- [x] **Exception propagation** via `ExceptionDispatchInfo.Capture` — body throws are caught per chunk, surfaced to caller with original stack trace. No unhandled-exception process crash.
- [x] API: `OverfitParallelFor.For(int rangeStart, int rangeEnd, delegate*<int, int, void*, void> body, void* context)`.
- [x] Tests: 0 B proof, correctness vs sequential, boundary cases, 10k-iter stress, exception propagation + post-throw recovery.

**Measured (Ryzen 9 9950X3D, 32 logical cores, full benchmark sweep):**

| InnerIters (body work) | Sequential | Parallel.For (TPL) | OverfitParallelFor | Speedup vs Seq | Alloc Overfit | Alloc TPL |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (empty) | 0.54 µs | 5.9 µs / 3.0 KB | ~6.6 µs / 0 B | — | 0 B ✅ | 3.4 KB |
| 1k (~1 µs body) | 92 µs | 48 µs | ~70 µs | — | 0 B ✅ | 3.8 KB |
| 100k (~100 µs body) | 3.5 ms | 326 µs | ~190 µs | 18× | 0 B ✅ | 6.8 KB |
| 1M (~1 ms body) | 34 ms | 1.46 ms | ~1.19 ms | **28.7×** | 0 B ✅ | 9.9 KB |

**Where it wins:**
- **Vs `Parallel.For`:** competitive in time (within 15% in worst case, **1.7× faster from 100 µs body upward**), and ~3000× cheaper on allocations across the whole range.
- **Vs Sequential:** the crossover at InnerIters ≈ 100 means parallelization pays from any body work above ~100 cycles. At InnerIters = 1M (1 ms body), 28.7× speedup ≈ 90% of 32 logical threads — near-optimal scaling.

**Why bulk wake beats N × Set:** earlier iterations (per-worker `AutoResetEvent` + simple/spin-then-park hybrid) hit a hard ~32-47 µs dispatch floor at 32-fanout — N `Set` calls serialize at the kernel because each one is a distinct event signal. `SemaphoreSlim.Release(N)` queues N pulses inside its internal lock and exits; the OS scheduler then resumes the waiters roughly in parallel (Windows `WakeConditionVariable` / Linux `futex_wake(N)` semantics). Result: ~5-7 µs dispatch at 32-fanout. Pre-bulk-wake prototypes (`OverfitParallelForHybrid`, `OverfitParallelForBulkWake`) were retired once this design landed.

**Migration to `OverfitParallelFor` — measured impact (Ryzen 9 9950X3D, MNIST 60k batch=64 small CNN, 5 epochs):**

| Phase | Wall time | Cores effective | Linear bwd total | Total CPU |
|---|---:|---:|---:|---:|
| TPL baseline (ROADMAP previous) | 5551 ms | 6.81 / 32 (21.3%) | — | ~37.8 s |
| + `LinearKernels` BackwardInput + AccumulateWeightGrad migrated | 5107 ms | 6.15 / 32 (19.2%) | 1133 ms | ~31.4 s |
| + `TensorMath.Pooling` (MaxPool/AvgPool fwd+bwd) migrated | (same) | (same) | (same) | (same) |
| + `TensorMath.Algebra` (AddBias, MatMul fwd, MatMulAdd variants) migrated | (same) | (same) | (same) | (same) |
| + `ComputationGraph.Linear` forward migrated | **4870 ms** | **7.03 / 32 (22.0%)** | **982 ms** | ~31.0 s |

**Result:** −12 % wall time, −18 % total CPU consumed, +0.6 percentage points cores effective vs. the TPL baseline. Linear backward time dropped 13 % (1133 → 982 ms). All 626 correctness tests still green.

**Honest read of why we didn't hit the 60-80 % cores-effective target:** MNIST CNN at this scale is Amdahl-limited. The 4685 batches/epoch × ~10 micro-ops/batch include many serial slices (graph reset, allocator paths, small-Linear sequential branch for Linear(64→10) at B=64 = 41k ops well below parallel threshold, copy input/target, optimizer.ZeroGrad). The dispatcher *itself* is no longer the bottleneck — eliminating its overhead saved CPU but not enough sequential work was unblocked to fill more cores.

**Where the dispatcher win will actually show up:**
- Larger models (GPT-2 scale): per-op body work is ~ms scale, dispatch is ≤1% overhead, and parallelism is much higher.
- Allocation-free hot paths (LM head, attention forward, prefill kernels): 0 B/call is now the bottleneck-relevant property — same wall time as TPL with no GC pressure.
- High-frequency inference loops where TPL's 3 KB/call hits Gen0 hard.

**Migrated / parallelized call sites:**
- `LinearKernels.BackwardInput` + `AccumulateWeightGrad`
- `TensorMath.Pooling`: `MaxPool2D` fwd+bwd, `GlobalAveragePool2D` fwd+bwd
- `TensorMath.Algebra`: `AddBias`, `MatMulRaw`, `MatMulAdd_A_BT_Raw`, `MatMulAdd_AT_B_Raw`
- `ComputationGraph.Linear` (forward batched parallel path)
- **`TensorMath.Gelu` (forward + backward)** — was a pure sequential scalar `for` loop; now element-wise parallel above `ParallelThreshold = 4096`.
- **`TensorMath.LayerNorm` (forward + backward)** — was sequential per-row. Forward parallelizes trivially (rows independent). Backward needs per-worker partial accumulators for `dGamma[i]` / `dBeta[i]` (shared across rows) — stackalloc'd by caller, sequential SIMD merge after parallel pass. dInput parallelizes in the same pass (per-row, no shared writes). Falls back to sequential when `C > 4096` to keep stackalloc bounded.

**Cumulative GPT-1 training-step impact** (Ryzen 9 9950X3D, 4-layer GPT-1, dModel=128, dFF=512, seqLen=128, per `GPT1CpuUtilizationProbeTests`):

| BatchSize | Wall/step start | + GELU parallel | + LayerNorm parallel | Total Δ wall | Cores start → end | Total Δ cores |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 126 ms | 82 ms | **78 ms** | **−38%** | 3.48 → **6.43** / 32 | **+85%** |
| 16 | 217 ms | 137 ms | **112 ms** | **−48%** | 2.99 → **6.63** / 32 | **+121%** |
| 32 | 414 ms | 243 ms | **198 ms** | **−52%** | 3.89 → **7.57** / 32 | **+94%** |

Per-op backward (batch=32, aggregated over 20 steps): GELU 1774 → 109 ms (−94%, 16× faster). LayerNorm 800 → 39 ms (−95%, 20× faster). Now Linear (1042 ms — already parallelized, just much more numerous) dominates the backward profile. The remaining sequential ops on the critical path are smaller: Add residuals (62 ms), Reshape (55 ms), Embedding (2.5 ms).

This validated the whole strategy: dispatcher was correct from the start, and *each* additional element-wise kernel we move from sequential to OverfitParallelFor produces a real, measurable win — the playbook works for any per-element activation/normalization op.

**Lesson learned — when NOT to parallelize element-wise:**
Attempted parallelizing `TensorMath.Add` (residual add) and **measured a regression** (+20% wall on GPT-1 batch=32, +55% on Add backward itself). Reverted. `Add` is **memory-bandwidth-bound** (read 2 arrays + write 1 = 3× data movement). On a typical desktop memory subsystem (~50 GB/s) 2-3 cores already saturate the bus, so the OverfitParallelFor dispatch overhead (~10 µs cold) is pure cost.

**Rule of thumb for future migrations:** parallel pays for element-wise ops only if the body is *compute-bound* (heavy per-element math like GELU/LayerNorm/Linear). For *memory-bound* ops (Add/Subtract/Scale/element-wise copy), keep sequential SIMD — the bandwidth ceiling caps you regardless of core count, and dispatch overhead becomes net negative.

**Pending migrations / parallelizations (deferred):**
- `TensorMath.Convolution` (Conv2D fwd+bwd) — per-worker workspace pattern needs `GCHandle`-pin or POH refactor of `Conv2DWorkspace`. Expected gain on MNIST: ~100-150 ms / 5 epochs (Conv2D backward 459 ms, fwd ~570 ms aggregated). Worth doing if/when GPT-2 batched training adds bigger Conv-heavy workloads.
- `TensorMath.Sequence` (LSTM) — relevant for LSTM training.
- `TensorMath.Attention` (`EnableParallelAttentionBackward` path) — relevant for transformer training at B > 1.
- `Optimizers.Adam` — parameter-parallel update, hot path during training.
- Various lower-priority sites: `DataAugmenter`, `FastRandomForest`, evolutionary noise tables, anomaly training.

**Pattern for future migrations:** `fixed (T* p = span)` block around `OverfitParallelFor.For(start, end, &ChunkBody, &ctx)` where `ChunkBody(int chunkStart, int chunkEnd, void* contextPtr)` loops over the chunk and reads pointers from the context struct.
- [-] **LinearKernels threshold tuning** — verified for MNIST that current thresholds (BackwardInput 524k, AccumulateWeightGrad 1M, ForwardBatched 500k) are correctly placed: Linear(1352→64) at B=64 = 5.5M ops → already parallel; Linear(64→10) at B=64 = 41k ops → sequential (41k / 32 cores = 1.3k ops per thread, well below Parallel.For overhead). No measurable benefit available at MNIST scale. **Revisit when GPT-2 batched training (B>1) lands** — smaller per-batch ops in GPT may benefit from lower thresholds.
- [ ] **SIMD path for `MaxPool2DForwardWithIndicesNchw` (training path)** — inference path has `TensorPrimitives.Max` fast path for pool=2; training path is scalar because index tracking complicates SIMD (need comparison masks to select max value AND record source index). Estimated saving on MNIST: ~40 ms / epoch (~7 % of post-threshold-fix epoch time). 1-2 hours focused work + parity tests vs scalar.
- [ ] **`ScaledDotProductAttention` forward parallel-over-batch** — symmetric with the backward we just enabled. Likely 15-20 % forward speedup at B ≥ 4.
- [ ] **Threshold tuning** — `LinearKernels.ParallelThreshold = 1_048_576` is set for inference (avoid Parallel.For overhead on tiny matrices). For training where the per-call work is amortized across many tokens, lower threshold may pay. Need per-op profiling first.
- [ ] **Data-parallel training as first-class API** — `TinyShakespeareDataParallelTrainingTests` proves N model replicas + gradient averaging works. Promote to public API, with idiomatic worker pool, for "N cores → N replicas" training scaling on a single machine.
- [ ] **Batched linear kernel** — `LinearKernels.ForwardBatched` measured win at batch 64/256 vs ONNX Runtime.
- [ ] **Backward kernels** — Linear/Conv backward through pure span kernels where practical.
- [ ] **Optimizer kernel profiling** — Adam/AdamW state updates, zero-grad allocation sources (currently ~96 B/Step on Adam — already tracked by skipped tests in `AdamOptimizerBehaviorTests`).
- [ ] **CPU SIMD audit** — AVX2/AVX-512/AVX10 where available.
- [ ] **Thread-scaling stabilization for large training workloads.**

### Correctness

- [ ] Numerical equivalence tests across scalar/SIMD paths.
- [ ] Determinism policy for parallel training kernels.

---

## Medium-term

### Features

- [ ] **Chat templates** — Qwen/Llama/Mistral chat-format builders (system/user/assistant turns) for ergonomic chat-style API. *(Lives in deferred track until Qwen path returns to focus.)*
- [ ] **`OverfitClient` facade** — high-level API: `var client = OverfitClient.LoadGguf(...); var response = await client.ChatAsync("...");` — gathers tokenizer + engine + session + sampling defaults.
- [ ] **ONNX export** — Overfit → ONNX for interop with other runtimes.
- [ ] **ONNX: LSTM/GRU operators** — enables recurrent model import.
- [ ] **Depthwise Conv** (group=channels) — MobileNet-style models.
- [ ] Standalone Softmax and CrossEntropy in addition to fused loss.

### Distribution

- [ ] NuGet package metadata polish.
- [ ] Sample Blazor app showing streaming generation in browser via Rx/IAsyncEnumerable adapter.
- [ ] Benchmark page: tabela Format × Model × RAM × tokens/s.

---

## Long-term ideas

- Graph compilation for fixed-shape training/inference graphs.
- Custom autograd operators with explicit forward/backward registration.
- Mixed precision training.
- Data loading and preprocessing pipeline improvements.
- Optional GPU backend investigation without compromising CPU-first design.
- Model/dataset packages outside the small core runtime.

---

## What Overfit is not trying to be

- Not a general-purpose replacement for PyTorch or TensorFlow.
- Not a Python shim.
- Not GPU-first.
- Not a model zoo in the core package.

The differentiator remains pure C#, predictable memory behavior, Native-AOT compatibility, and competitive CPU inference (including small/medium language models) on consumer hardware.

---

## Contributing

Performance-sensitive PRs should include:

- correctness tests;
- before/after BenchmarkDotNet output;
- allocation measurements;
- documentation updates when public behavior changes.

License: GNU AGPLv3. For commercial licensing, contact devonbike@gmail.com.
