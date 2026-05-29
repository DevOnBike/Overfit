# Overfit Local Agent Starter

**Private RAG, tool calling and JSON inside ASP.NET — no Python, no Ollama, no model server, no data leaving the process.**

A reference template a .NET developer can drop into an existing service to add a private agent. Loads a local GGUF model (Qwen / Llama / Mistral family) via `OverfitClient.LoadGguf`, exposes Minimal-API endpoints, and runs entirely in your .NET process.

> **Status.** All four phases shipped — Phase 1 (chat: `/health`, `/chat`, `/reset`), Phase 2 (RAG: `/documents/index`, `/rag/query`), Phase 3 (agent: `/agent` forced C# tool call, `/chat/json` guaranteed JSON), Phase 4 (observability: `/metrics` in Prometheus format, `Dockerfile` + `compose.yaml`). Optional Phase 5 (Microsoft.Extensions.AI adapter) and Phase 6 (Aspire dashboard) are not built yet.

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

**For RAG** (optional — `/chat` works without it): also point at a MiniLM sentence-embedding directory (`config.json` + `vocab.txt` + `model.safetensors`), via `EmbeddingModelPath` in appsettings or the `OVERFIT_EMBEDDING_DIR` env var:

```jsonc
{ "EmbeddingModelPath": "C:\\minilm" }
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

### 5. Try RAG over the sample documents

Two markdown files ship in `Data/` — a customer policy handbook and a support FAQ for a fictional "Acme Software". Index them, then ask grounded questions:

```powershell
# Index the documents (chunks + embeds into the in-process vector store).
curl -X POST http://localhost:5234/documents/index

# Ask a question answered from the documents.
curl -X POST http://localhost:5234/rag/query `
  -H "Content-Type: application/json" `
  -d '{ "question": "Can an EU customer get a refund after 10 days, and how?", "topK": 4 }'
```

Expected `/rag/query` response shape:

```json
{
  "reply": "Yes. EU customers have a statutory 14-day right of withdrawal [1] …",
  "sources": [
    { "index": 1, "id": "company-policy.md#2", "similarity": 0.72, "snippet": "Right of withdrawal (EU customers) …" },
    { "index": 2, "id": "support-faq.md#5", "similarity": 0.61, "snippet": "How do I cancel? …" }
  ],
  "promptTokens": 240,
  "generatedTokens": 58,
  "tokensPerSecond": 18.4
}
```

The model is told to answer **only** from the retrieved context and to cite the bracketed source numbers. The retrieval step — embedding + cosine search — runs entirely in-process against the `VectorStore`; no external vector database, no embedding API.

### 6. Make the agent take an action (tool calling)

`/agent` forces the model to choose and call exactly one registered C# tool, then dispatches that call to the C# handler and returns the result. The demo ships two tools backed by an in-memory store: `lookup_customer` and `create_ticket`.

```powershell
# The model picks lookup_customer and supplies { "email": ... }.
curl -X POST http://localhost:5234/agent `
  -H "Content-Type: application/json" `
  -d '{ "message": "What plan is ada@northwind.example on?" }'

# The model picks create_ticket and supplies subject + priority + customerEmail.
curl -X POST http://localhost:5234/agent `
  -H "Content-Type: application/json" `
  -d '{ "message": "Open a high-priority ticket for sam@brightlabs.example about a failed SSO login." }'
```

Expected `/agent` response shape:

```json
{
  "toolName": "create_ticket",
  "arguments": { "customerEmail": "sam@brightlabs.example", "subject": "Failed SSO login", "priority": "high" },
  "result": { "created": true, "id": "TICK-1042", "customerEmail": "sam@brightlabs.example", "subject": "Failed SSO login", "priority": "high", "status": "open" }
}
```

The tool name and the JSON arguments envelope are forced at the logit level by `ToolCallConstraint` — the model **cannot** emit an unregistered tool name or malformed arguments. No regex parsing of free text, no retry-on-bad-JSON loop. `ToolCall.TryParse` on the constrained reply always succeeds.

### 7. Get guaranteed-valid JSON

`/chat/json` constrains the output to well-formed JSON by construction (`JsonGrammarConstraint`). Describe the shape you want in the message:

```powershell
curl -X POST http://localhost:5234/chat/json `
  -H "Content-Type: application/json" `
  -d '{ "message": "Extract intent and email as JSON from: I want to cancel, reach me at jo@tinkergarden.example" }'
```

The response is returned with `Content-Type: application/json` and is guaranteed to parse. (Field-level schema typing — required keys, enums — is the JSON-Schema follow-on; today the guarantee is well-formedness.)

### 8. Watch the metrics

`GET /metrics` exposes Prometheus text exposition format — no exporter dependency:

```powershell
curl http://localhost:5234/metrics
```

```text
# HELP overfit_build_info Static info about the loaded model (value is always 1).
# TYPE overfit_build_info gauge
overfit_build_info{model="qwen.q4km.gguf",fingerprint="9f3a1c0b7d2e4a51",mmap="true"} 1
# HELP overfit_model_load_seconds Time to load the model at startup.
# TYPE overfit_model_load_seconds gauge
overfit_model_load_seconds 1.84
# HELP overfit_requests_total Requests handled, by endpoint.
# TYPE overfit_requests_total counter
overfit_requests_total{endpoint="chat"} 3
overfit_requests_total{endpoint="agent"} 2
# HELP overfit_allocated_bytes_total Total bytes allocated during generation (Overfit targets ~0 B/token).
# TYPE overfit_allocated_bytes_total counter
overfit_allocated_bytes_total 96
# HELP overfit_decode_tokens_per_second Decode throughput of the most recent generation.
# TYPE overfit_decode_tokens_per_second gauge
overfit_decode_tokens_per_second 18.6
# HELP overfit_tool_calls_total Tool calls dispatched, by tool name.
# TYPE overfit_tool_calls_total counter
overfit_tool_calls_total{tool="create_ticket"} 1
overfit_tool_calls_total{tool="lookup_customer"} 1
```

`overfit_allocated_bytes_total / overfit_generated_tokens_total` is the per-token allocation — it should sit near zero, which is the whole point. The model fingerprint is a fast partial hash (size + head + tail), enough to pin which model build is running without hashing the whole multi-GB file. Full metric reference: [`Observability/otel.md`](Observability/otel.md).

## Run the full stack with Docker

`compose.yaml` builds the agent image and runs Prometheus scraping its `/metrics`. The model file stays on your machine — it's mounted into the container, never baked into the image, never sent anywhere.

```powershell
# 1. Edit compose.yaml: set the volume source to your local GGUF directory (default C:/qwen3b).
# 2. Bring up the agent + Prometheus.
docker compose -f Demo/Overfit.LocalAgent.AspNet/compose.yaml up --build
```

- App: `http://localhost:5234` (e.g. `curl http://localhost:5234/health`)
- Prometheus: `http://localhost:9090` — graph `overfit_decode_tokens_per_second` or the per-token allocation

The image is a framework-dependent publish for portability; switch to Native AOT (`-p:PublishAot=true` on a build image with clang + zlib, final stage on `runtime-deps`) for a smaller, faster-starting container.

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

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/` | Redirects to `/health` |
| `GET` | `/health` | `200 OK` with model filename, runtime, process privacy flag, RAG index status |
| `POST` | `/chat` | `{ message: "…" }` → `{ reply, stats }`. Conversation persists in the singleton client |
| `POST` | `/reset` | Clears conversation history; re-applies the system message |
| `POST` | `/documents/index` | Chunks + embeds every `*.md` in the data directory into the in-process vector store. Returns per-file chunk counts |
| `POST` | `/rag/query` | `{ question: "…", topK?: 4 }` → `{ reply, sources[], stats }`. Retrieves top-K chunks and answers from them |
| `POST` | `/agent` | `{ message: "…" }` → `{ toolName, arguments, result }`. Forces one registered C# tool call and dispatches it |
| `POST` | `/chat/json` | `{ message: "…" }` → guaranteed well-formed JSON (`application/json`) |
| `GET` | `/metrics` | Prometheus text exposition — requests, tokens, allocations/token, tok/s, tool calls, RAG latency |

System message comes from `SystemMessage` in `appsettings.json` (override per environment via `appsettings.Development.json`).

RAG is optional: `/chat` works with no embedding model configured. `/documents/index` and `/rag/query` return a clear `400` with setup instructions until `EmbeddingModelPath` / `OVERFIT_EMBEDDING_DIR` is set.

---

## Roadmap

- ✅ **Phase 1** — `/health`, `/chat`, `/reset` over a singleton `OverfitClient`.
- ✅ **Phase 2** — RAG over your documents (`/documents/index`, `/rag/query`) via `VectorStore` + `SentenceEmbedder.ForMiniLm` over the `Data/` folder.
- ✅ **Phase 3** — C# tool calling (`Tools/ToolRegistry.cs`: `lookup_customer`, `create_ticket`) via `ToolCallConstraint`, dispatched to C# handlers, plus a guaranteed-JSON endpoint (`/chat/json`) via `JsonGrammarConstraint`.
- ✅ **Phase 4** — `/metrics` (Prometheus: requests, tokens, allocations/token, tok/s, tool calls, RAG latency, model fingerprint) + `Dockerfile` + `compose.yaml` (agent + Prometheus) + `Observability/`.
- ❌ **Phase 5** (optional) — Microsoft.Extensions.AI adapter (`OverfitChatClient : IChatClient`); lets Overfit slot in as a drop-in local backend wherever `IChatClient` is expected.
- ❌ **Phase 6** (optional) — .NET Aspire dashboard variant.

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
