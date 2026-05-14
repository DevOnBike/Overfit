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

## Near-term

### Slot 2b — quantized weight storage at inference

**The gap:** Q4_K_M loader exists (above) but currently dequantizes everything to FP32 on load. A 2 GB Q4_K_M file produces ~14 GB FP32 weights in RAM. The "3B in 4 GB RAM" payoff requires keeping weights quantized in RAM and dequantizing per-block during matmul.

**Work:**
- [ ] `TensorStorage<TBlock>` or parallel `QuantizedTensorStorage` abstraction (Q4_K and Q6_K block shapes).
- [ ] Quantized variants of `SingleHeadWeights` / `BlockWeights` / `StackWeights` (or polymorphism via interface).
- [ ] Dequant-fused matmul kernels (one block at a time, scratch FP32 row in stackalloc).
- [ ] `CachedLlamaInferenceEngine.LoadGgufQuantized(path)` factory.
- [ ] RAM diagnostic: `Diagnose_GgufQ4KM_3B_RamFootprint` showing ~2-3 GB managed.
- [ ] Logit parity vs FP32 (top-1 must match for greedy; top-10 within tolerance).

Estimated effort: 1-2 days. Largest single architectural change since KV-cache runtime.

### Slot 3 — Q4_K_M integration parity test

- [ ] Download `qwen2.5-3b-instruct-q4_k_m.gguf` (Ollama or HF) to `c:\qwen3b\qwen.q4km.gguf`.
- [ ] Integration test: load via `GgufLlamaLoader`, run greedy on canonical 3-token prompt, compare top-1 token + top-10 overlap vs FP16 GGUF baseline.
- [ ] Tolerance: top-1 matches; max abs logit diff within Q4_K_M expected range (~0.5-1.5 % relative).

Synthetic unit tests (already in repo) cover the algorithm; this test catches bit-layout regressions against llama.cpp/Ollama.

### Other quant formats

- [ ] Q5_K dequantizer (occasionally appears in mixed quant files; 176 bytes/block).
- [ ] Q4_0 / Q5_0 / Q5_1 (legacy formats; lower priority — Ollama defaults to K-quants).
- [ ] Q2_K / Q3_K_S (very aggressive quant; experimental quality).

### Performance

- [ ] Batched linear kernel — `LinearKernels.ForwardBatched` measured win at batch 64/256 vs ONNX Runtime.
- [ ] Backward kernels — Linear/Conv backward through pure span kernels where practical.
- [ ] Optimizer kernel profiling — Adam/AdamW state updates, zero-grad allocation sources (currently ~96 B/Step on Adam — already tracked by skipped tests in `AdamOptimizerBehaviorTests`).
- [ ] CPU SIMD audit — AVX2/AVX-512/AVX10 where available.
- [ ] Thread-scaling stabilization for large training workloads.

### Correctness

- [ ] Stabilize `GPT1_GradientCheck_BackwardIsCorrect` LMHead numerics (currently flaky around threshold).
- [ ] Numerical equivalence tests across scalar/SIMD paths.
- [ ] Determinism policy for parallel training kernels.
- [ ] Q4_K_M / Q6_K bit-exact parity vs llama.cpp output (Slot 3 above).

---

## Medium-term

### Features

- [ ] **LoRA training** — backward pass through adapter only (frozen base). Requires backward through Linear/RMSNorm/SwiGLU/attention. Major scope; opens "fine-tune LLM locally in pure C#" story.
- [ ] **Chat templates** — Qwen/Llama/Mistral chat-format builders (system/user/assistant turns) for ergonomic chat-style API.
- [ ] **`OverfitClient` facade** — high-level API: `var client = OverfitClient.LoadGguf(...); var response = await client.ChatAsync("...");` — gathers tokenizer + engine + session + sampling defaults.
- [ ] **ONNX export** — Overfit → ONNX for interop with other runtimes.
- [ ] **ONNX: LSTM/GRU operators** — enables recurrent model import.
- [ ] **Depthwise Conv** (group=channels) — MobileNet-style models.
- [ ] Standalone Softmax and CrossEntropy in addition to fused loss.

### Distribution

- [ ] NuGet package metadata polish, sample console app (50-line Qwen chat).
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
