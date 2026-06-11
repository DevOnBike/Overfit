# Overfit demos

> **Overfit lets .NET teams embed private LLMs, RAG, C# tool calling and guaranteed JSON
> directly into their own process — no Python, no native binary, no model server, no data egress.**

Every demo here is pure C# / .NET 10. Start with #1.

---

## 1. LocalAgentAspNetDemo — a private LLM agent inside an ASP.NET service ⭐

The flagship: the whole Overfit stack exposed over HTTP, ready to drop into an existing .NET
service. Minimal API + Swagger, single-tenant, zero-allocation decode. No UI — just endpoints + curl.

| Endpoint | What it does |
|---|---|
| `POST /chat` | Multi-turn chat |
| `POST /documents/index` | Index the bundled sample docs for RAG |
| `POST /rag/query` | Retrieve + answer, grounded, with cited sources |
| `POST /agent` | Forced C# tool call (constrained), dispatched + result |
| `POST /chat/json` | Guaranteed-valid JSON output |
| `GET /health` | Model fingerprint, load time, mmap flag |
| `GET /metrics` | Prometheus scrape endpoint (OpenTelemetry) |
| `/swagger` | Interactive UI, request bodies pre-filled with examples |

Everything stays in one .NET process: **no Python, no native binary, no model server, no data egress.**

```powershell
# Point ModelPath / OVERFIT_MODEL_DIR at a Qwen2.5 GGUF (or a safetensors dir); RAG needs a MiniLM
# embedder (EmbeddingModelPath / OVERFIT_EMBEDDING_DIR).
dotnet run -c Release --project Demo/LocalAgentAspNetDemo
```

→ [LocalAgentAspNetDemo/README.md](LocalAgentAspNetDemo/README.md)

---

## 2. AgentDemo — the same value proposition in one console run

The fastest way to see the stack end to end with zero HTTP setup. One process loads a Qwen2.5 GGUF
and runs, in sequence:

- **memory-mapped load** — weights are file-mapped, so the live managed heap stays tiny
- **in-process RAG** — embeddings + a built-in cosine vector store (no external DB)
- **C# tool calling** — constrained decoding forces a valid call, dispatched to your C# delegate
- **guaranteed JSON** — well-formed by construction (the grammar is enforced at the logit level)

```powershell
# Point OVERFIT_MODEL_DIR at a Qwen2.5 GGUF directory (defaults to C:\qwen3b):
dotnet run -c Release --project Demo/AgentDemo
```

It shows: **GGUF load → RAG → tool calling → guaranteed JSON.**
→ [AgentDemo/README.md](AgentDemo/README.md)

---

## 3. Gpt2ConsoleDemo — minimal GPT-2 text generation

The smallest possible "load a model and generate" example — GPT-2 from a converted `.bin`, pure C#.

```powershell
dotnet run -c Release --project Demo/Gpt2ConsoleDemo
```

## 4. AnomalyConsoleDemo — train a small model on your own signal

Trains a small GPT on a numeric series and flags anomalies — the "run your own model, not just
someone else's" story (LoRA-adaptable per deployment).

```powershell
dotnet run -c Release --project Demo/AnomalyConsoleDemo
```

## 5. EvaluationDemo — Microsoft.Extensions.AI.Evaluation, fully local

Microsoft's official LLM-as-judge evaluation framework (Coherence / Fluency / Groundedness) scoring a good
and a bad RAG answer — with an **Overfit in-process model as the judge**. No Azure, no key, no egress.
See [`docs/meai-evaluation.md`](../docs/meai-evaluation.md) for the calibration notes (small judges catch
bad answers reliably; calibrated absolute scores want a ≥7B judge).

```powershell
dotnet run -c Release --project Demo/EvaluationDemo -- C:\path\to\judge.gguf   # default: C:\qwen3b\qwen.q4km.gguf
```

---

## Speech & audio (pure C#)

No GPU, no Python, no native binary. Each demo has its own README plus a `download-materials.cmd`
(fetches the model / sample via `curl`, into the shared `Demo\materials\` folder) and a `run.cmd`.

### WhisperDemo — speech-to-text

Transcribe a WAV or MP3 with a whisper.cpp ggml model, CPU-only. Any sample rate (resampled to
16 kHz), mono or stereo. ~60× real-time on tiny; validated English and Polish.

```powershell
Demo\WhisperDemo\download-materials.cmd   # model (~77 MB) + jfk.wav -> Demo\materials
Demo\WhisperDemo\run.cmd                  # transcribe (English);  run.cmd pl  for Polish
```

→ [WhisperDemo/README.md](WhisperDemo/README.md)

### MicDemo — live microphone speech-to-text (Windows)

Record N seconds from the mic and transcribe, in a loop. Mic capture uses the built-in Windows
`winmm` API via P/Invoke (no NuGet); the core engine stays platform-neutral.

```powershell
Demo\MicDemo\download-materials.cmd       # model -> Demo\materials
Demo\MicDemo\run.cmd pl 5                  # Polish, 5 s rounds (default: en, 5 s)
```

→ [MicDemo/README.md](MicDemo/README.md)

### Mp3Demo — the from-scratch MP3 decoder

Decodes an MPEG-1/2/2.5 Layer III file to a 16 kHz WAV — no native binaries, no external libraries,
zero per-frame allocation, ~160× real-time.

```powershell
Demo\Mp3Demo\download-materials.cmd       # sample.mp3 -> Demo\materials
Demo\Mp3Demo\run.cmd                      # or:  run.cmd C:\music\song.mp3
```

→ [Mp3Demo/README.md](Mp3Demo/README.md) · [docs/mp3-decoding.md](../docs/mp3-decoding.md)

---

## Experimental

### Unity Swarm Engine — 100k-agent game-dev showcase

A separate experiment (not the core pitch): a .NET swarm server streaming steering forces to a
Unity client at 100k agents. Impressive, but unrelated to the LLM/RAG/agent story above.
→ [Unity/README.md](Unity/README.md)
