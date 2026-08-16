# `LanguageModels` — the LLM subsystem

Everything needed to load, run, constrain, fine-tune and build on top of a transformer, in pure C# with
no Python, no ONNX Runtime and no native library. `OverfitClient` is the convenience entry point.

## Map

| Directory | Role |
|---|---|
| `Loading` | GGUF, safetensors, `.bin` → weights. Import only; there are no exporters. |
| `Tokenizers` | Text ↔ token ids, one per vocabulary format. |
| `Runtime` | The decode engine. Zero allocations per token. Start here for performance. |
| `Rope`, `Quantization` | Rotary embeddings; quantisation kinds and options. |
| `Sampling` | `SamplingPipeline`, logit processors, Mirostat. |
| `Constraints` | Grammar/schema/regex-constrained decoding. |
| `Chat`, `Memory` | Multi-turn sessions; history compaction when the window fills. |
| `Tools`, `Agents` | Tool definitions and the ReAct / critic loops. |
| `Embeddings`, `Retrieval` | Sentence encoders and RAG, including retrieval-quality evaluation. |
| `LoRA` | Adapters, including QLoRA against a frozen quantised base. |
| `Whisper` | Speech to text. |
| `Skills` | Skill and prompt evaluation and optimisation. |
| `Contracts` | The interfaces everything above agrees on. |

## Verified on real models, not on fixtures

Every loader has been run against real weights and checked for coherent output, and several
architectures needed a specific correction that only wrong-looking text revealed: GPT-2 (byte-parity),
Qwen-2.5/3, Llama, Phi-3.5 and Phi-4, Gemma-2, Qwen1.5-MoE and Mixtral-8x7B. The recurring lesson is
that these models fail *fluently* — a wrong RoPE pair layout, a missing QK-norm or the wrong
`norm_topk_prob` convention produces grammatical, confident, meaningless text and never throws.

## Scope of the public engine

The open engine is the offline and batch story, plus correctness. Real-time and GPU work is not
promised here. Decode is roughly **1.13× behind llama.cpp, uniformly across models** — measured
best-of-N on both sides, and stated as a gap rather than as parity.
