# Teaching C# to Do Deep Learning — Engineering Notes from the Latest Overfit Releases

Most teams reach for Python the moment "machine learning" enters the conversation. Across the last few release cycles, we have been chipping away at the assumption underneath that habit — with **Overfit**, a deep-learning and inference engine written in **pure C#**.

No native binaries. No Python runtime. No ONNX Runtime dependency. Just managed code, `Span<T>`, and strict Native-AOT discipline — with an inference hot path that allocates **zero bytes**.

Here is what has landed.

## A parallel-runtime performance sprint

Too much of the training path was sequential scalar code. We built `OverfitParallelFor` — a zero-allocation, bulk-wake work dispatcher (~5 µs warm dispatch, 0 B per call) — and rewrote the hot kernels on top of it: GELU, LayerNorm, scaled-dot-product attention, and the Linear backward kernels.

On a 4-layer GPT-1 training step (batch size 32):

- Per-step wall time: **414 ms → 114 ms (−72%)**
- GELU backward: **1774 ms → 42 ms (42×)**
- LayerNorm backward: **800 ms → 37 ms (22×)**
- Real workload — TinyShakespeare, 300 training steps: **~60–120 s → ~2 s**, with zero numerical regression (the gradient check still passes).

Honest context: PyTorch on CPU (Intel MKL / oneDNN) is still ~2.2–3.6× faster on raw matrix multiply — decades of hand-tuned AVX-512 assembly is hard to out-run from managed code. What the sprint did was close the gap from ~7–8× down to ~2.2–3.6×. Overfit's axis was never raw GEMM throughput — it is predictable, allocation-free, AOT-compatible execution with no native dependency.

## Real language models — loaded, run, and fine-tuned

- **GPT-2 Small (124M parameters)** in pure C#. KV-cache decode allocates **0 bytes per generated token** and scales O(N) instead of O(N²). Verified against a PyTorch reference: top-10 logit overlap 10/10, max absolute difference ≈ 0.0001 — the float32 noise floor.
- **Qwen / Llama / Mistral** inference — grouped-query attention, RoPE, SwiGLU — tested against Qwen2.5-3B.
- **Native GGUF loader** — reads Ollama and HuggingFace `.gguf` files directly, including Q4_K and Q6_K quantization. Hand-rolled parser, no protobuf dependency.
- **Streaming token API** — `IAsyncEnumerable<int>` with stop-tokens and cancellation.

## LoRA fine-tuning — the newest addition

Overfit now LoRA-fine-tunes a GPT model end to end, in pure C#: low-rank adapters train on a frozen base model (the language-model head and the per-block feed-forward matrices), then merge in place — visible to the KV-cached decode path with zero kernel changes. Adapter-only Adam, a compact multi-entry `.bin` format. Training and inference, with no Python in the loop.

## Engineering discipline

- Binary model loader RAM optimization — peak load for a 3B FP32 model: **30 GB → 14 GB**.
- Versus ONNX Runtime: a single linear-model inference runs **7.6× faster at 0 bytes allocated**; under 8-thread concurrency, **3.6× faster** while ONNX Runtime allocates 117 MB of managed memory.
- Native-AOT compilation is guarded in CI — LINQ, reflection, and expression trees are banned outright in the runtime library, analyzer-enforced.

## Where it stands

Overfit is not replacing PyTorch, and it is not GPU-first. It targets one thing: predictable, allocation-free CPU inference — and now training and fine-tuning — for small and medium models, including language models, in pure managed .NET. For the deployments where zero allocations and zero native dependencies genuinely matter.

Open source under AGPLv3 (commercial licensing available) — **DevOnBike/Overfit** on GitHub.

#dotnet #csharp #MachineLearning #DeepLearning #LLM #Performance #NativeAOT #OpenSource

---

<!-- ─────────────────────────────────────────────────────────────────────────
  NEW DRAFT (2026-05-25) — the in-process agent stack. Raw material: edit to your
  own voice before posting. Numbers are measured (Qwen2.5-3B Q4_K_M on the dev box).
  ───────────────────────────────────────────────────────────────────────── -->

# From "runs LLMs" to "runs agents" — in pure C#, in one process

The last Overfit cycle was about *running* language models in managed .NET. This one is about
building **agentic features** on top of them — RAG, tool calling, structured output — with no
Python, no model server, no vector database, and nothing leaving the process.

## What landed

- **Memory-mapped GGUF.** Weights (and the quantized token-embedding table) are now file-mapped
  instead of copied onto the managed heap. A 3B `Q4_K_M` model loads with a **~220 MB live managed
  heap** — the weight pages are shared/clean and paged in on demand. Bit-identical decode vs the
  copy path (max logit diff 0).
- **Embeddings + a built-in vector store.** The same model produces pooled, L2-normalised sentence
  embeddings; `VectorStore` is an in-process cosine index (flat dot-product scan + top-K, no external
  DB). RAG retrieval becomes a function call.
- **Constrained decoding.** This is the part I care about. Instead of *asking* the model for JSON in
  the prompt and repairing the result, an `ITokenConstraint` masks the logits **before every token**:
  anything that would break the grammar is set to −∞, so the model physically cannot sample it.
  - `JsonGrammarConstraint` → guaranteed well-formed JSON (a value-type character DFA: bit-stack for
    nesting, sub-DFAs for numbers/strings/escapes).
  - `ToolCallConstraint` → a guaranteed-valid `{"name": "<one of your tools>", "arguments": <json>}`
    envelope; the name is constrained to the registered tool set, the arguments to valid JSON. Parse,
    then dispatch to a C# delegate.

On a real Qwen-3B, "What is the weather in Paris?" returns
`{"name": "get_weather", "arguments": {"city":"Paris"}}` — parsed and dispatched to a C# function.
The whole loop (load → RAG → tool call → JSON) is one runnable console demo.

## Honest scope

Constrained decoding fixes **structure, not reasoning** — a 3B model's *content* is still a 3B
model's content; what's guaranteed is that the tool name is always valid and the JSON always parses,
so there's no retry/repair code. And it's still CPU-first: llama.cpp decodes ~1.5–2× faster. The
axis here is in-process, zero-dependency, AOT-compatible agent infrastructure for .NET — the model,
the embeddings, the vector store and the grammar enforcement all inside your app.

Open source under AGPLv3 (commercial licensing available) — **DevOnBike/Overfit** on GitHub.

#dotnet #csharp #LLM #RAG #AItools #StructuredOutput #NativeAOT #OpenSource
