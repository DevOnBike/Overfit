# Overfit demos

> **Overfit lets .NET teams embed private LLMs, RAG, C# tool calling and guaranteed JSON
> directly into their own process — no Python, no native binary, no model server, no data egress.**

Every demo here is pure C# / .NET 10. Start with #1.

---

## 1. AgentDemo — the whole value proposition in one console run ⭐

The fastest way to see what Overfit is for. One process loads a Qwen2.5 GGUF and runs, end to end:

- **memory-mapped load** — weights are file-mapped, so the live managed heap stays tiny
- **in-process RAG** — embeddings + a built-in cosine vector store (no external DB)
- **C# tool calling** — constrained decoding forces a valid call, dispatched to your C# delegate
- **guaranteed JSON** — well-formed by construction (the grammar is enforced at the logit level)
- **no Python, no native binary, no model server**

```powershell
# Point OVERFIT_MODEL_DIR at a Qwen2.5 GGUF directory (defaults to C:\qwen3b):
dotnet run -c Release --project Demo/AgentDemo
```

It shows: **GGUF load → RAG → tool calling → guaranteed JSON.**
→ [AgentDemo/README.md](AgentDemo/README.md)

---

## 2. LocalAgentAspNetDemo — the same stack as an ASP.NET service 🌐

The CTO/architect view: the agent exposed over HTTP, ready to drop into an existing .NET service.
Minimal API + Swagger, single-tenant, zero-allocation decode. No UI — just endpoints + curl.

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

```powershell
# Same model resolution as AgentDemo; needs a MiniLM embedder for RAG (EmbeddingModelPath / OVERFIT_EMBEDDING_DIR).
dotnet run -c Release --project Demo/LocalAgentAspNetDemo
```

→ [LocalAgentAspNetDemo/README.md](LocalAgentAspNetDemo/README.md)

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

---

## Experimental

### Unity Swarm Engine — 100k-agent game-dev showcase

A separate experiment (not the core pitch): a .NET swarm server streaming steering forces to a
Unity client at 100k agents. Impressive, but unrelated to the LLM/RAG/agent story above.
→ [Unity/README.md](Unity/README.md)
