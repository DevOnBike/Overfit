# Overfit Local Agent Starter

**Private RAG, tool calling and JSON inside ASP.NET — no Python, no Ollama, no model server, no data leaving the process.**

A reference template a .NET developer can drop into an existing service to add a private agent. Loads a local GGUF model (Qwen / Llama / Mistral family) via `OverfitClient.LoadGguf`, exposes Minimal-API endpoints, and runs entirely in your .NET process.

> **Phase 1 status (walking skeleton).** This commit ships `/health`, `/chat`, `/reset` over a singleton `OverfitClient`. Phases 2–4 (RAG over your local documents, C# tool calling, guaranteed JSON output, `/metrics` with tokens/sec + allocations per token, Dockerfile + compose) layer on top of this skeleton without changing the wire-up.

---

## 5-minute quickstart

### 1. Get a GGUF model

Easiest: download Qwen2.5-3B-Instruct Q4_K_M (~1.9 GB) from HuggingFace or any Ollama mirror. The demo accepts any GGUF in the Qwen / Llama / Mistral / Mixtral / Qwen-MoE family.

### 2. Point the demo at it

Pick **one** of the two paths:

**(a) appsettings.Development.json** — set the absolute path:

```jsonc
{
  "ModelPath": "C:\\qwen3b\\qwen.q4km.gguf"
}
```

**(b) Environment variable** — point at a directory containing exactly one `*.gguf`:

```powershell
$env:OVERFIT_MODEL_DIR = "C:\qwen3b"
```

### 3. Run

```powershell
dotnet run -c Release --project Demo/Overfit.LocalAgent.AspNet
```

First request loads the model into mmap-backed K-quant weights (~1–2 seconds, sub-300 MB live managed heap). Subsequent calls are zero-allocation on the decode hot path.

### 4. Try it

```powershell
# Health — confirms the model loaded and the host is listening.
curl http://localhost:5234/health

# Chat — first turn.
curl -X POST http://localhost:5234/chat `
  -H "Content-Type: application/json" `
  -d '{ "message": "Explain in one sentence what zero-allocation decode means." }'

# Reset conversation between sessions.
curl -X POST http://localhost:5234/reset
```

Expected `/chat` response shape:

```json
{
  "reply": "Zero-allocation decode means …",
  "stats": {
    "promptTokens": 21,
    "generatedTokens": 28,
    "tokensPerSecond": 18.6,
    "allocatedBytes": 28,
    "usedKeyValueCache": true
  }
}
```

`allocatedBytes` is measured per-call — Overfit targets **1 B per token** on the decode hot path. If you see > a few hundred bytes per generated token, something on the host side (e.g. logging, JSON serialisation overhead) is dominating, not the engine.

---

## Endpoints (Phase 1)

| Method | Path | What it does |
|---|---|---|
| `GET` | `/` | Redirects to `/health` |
| `GET` | `/health` | `200 OK` with model filename, runtime, process privacy flag |
| `POST` | `/chat` | `{ message: "…" }` → `{ reply, stats }`. Conversation persists in the singleton client |
| `POST` | `/reset` | Clears conversation history; re-applies the system message |

System message comes from `SystemMessage` in `appsettings.json` (override per environment via `appsettings.Development.json`).

---

## What this is **not** yet (Phase 2–4 roadmap)

- ❌ RAG over your documents (`/documents/index`, `/rag/query`) — Phase 2 wires `VectorStore` + `SentenceEmbedder.ForMiniLm` over a `Data/` folder.
- ❌ C# tool calling (`Tools/CreateTicketTool.cs`, `Tools/LookupCustomerTool.cs`) — Phase 3 passes a `ToolCallConstraint` to `client.Send(..., constraint: …)` so the logit mask forces a valid call dispatched to your delegate.
- ❌ Guaranteed-JSON endpoint (`/chat/json`) — Phase 3 uses the JSON-mode constraint so the response is schema-valid by construction, not by retry.
- ❌ `/metrics` (tokens/sec, allocations/token, model hash, tool-call log) — Phase 4.
- ❌ `Dockerfile` + `compose.yaml` — Phase 4.
- ❌ Microsoft.Extensions.AI adapter (`OverfitChatClient : IChatClient`) — optional Phase 5; lets Overfit slot in as a drop-in local backend wherever `IChatClient` is expected.
- ❌ .NET Aspire dashboard variant — optional Phase 6.

---

## Architecture notes

- **Singleton `OverfitClient`** — chat session state is shared across HTTP requests. Single-tenant demo only. For multi-tenant, swap to per-tenant client pool or session-per-request.
- **No authentication / no rate limiting** — `[Authorize]`, an API-key middleware, or `RateLimiter` are normal additions; left out so the wire-up stays readable.
- **Conversation persistence is in-memory** — restart loses history. Sliding-window eviction (`new ChatSession(slidingWindow: true)`) keeps long sessions cheap; surfaced via `OverfitClient` for future phases.
- **Model lifecycle** — `Lifetime.ApplicationStopping` disposes the client so mmap-backed weights release cleanly before the process exits.

---

## Why this beats the alternatives

The selling point of this template is what it **doesn't** require:

| Approach | What you'd otherwise need |
|---|---|
| OpenAI / Anthropic API | External network call, API key, data egress, rate limits |
| Python sidecar (FastAPI + transformers) | Second container, second runtime, IPC hop, dependency management |
| Ollama server | Separate process, separate health checks, separate deploy step |
| ONNX Runtime + GGUF-converted model | 100+ MB native binary, P/Invoke overhead, no GGUF mmap |

Overfit ships as a NuGet package. Your model is a file on disk. Your agent is a singleton inside your existing ASP.NET process. That's the entire story.

---

## Build

```powershell
dotnet build Demo/Overfit.LocalAgent.AspNet -c Release
```

Or as part of the solution: `dotnet build Overfit.sln -c Release`.

## Feedback / commercial

Production deployment, custom tools, regulated-industry setup, or multi-tenant variants — see [`../../COMMERCIAL.md`](../../COMMERCIAL.md). Contact: `devonbike@gmail.com`.
