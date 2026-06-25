# Overfit Local Agent Starter

**Private RAG, tool calling and JSON inside ASP.NET — no Python, no Ollama, no model server, no data leaving the process.**

A reference template a .NET developer can drop into an existing service to add a private agent. Loads a local GGUF model (Qwen / Llama / Mistral family) via `OverfitClient.LoadGguf`, exposes Minimal-API endpoints, and runs entirely in your .NET process.

> **Status.** Phases 1–5 shipped — Phase 1 (chat: `/health`, `/chat`, `/reset`), Phase 2 (RAG: `/documents/index`, `/rag/query`), Phase 3 (agent: `/agent` forced C# tool call, `/chat/json` guaranteed JSON — well-formed, or schema-conforming with an optional `schema`), Phase 4 (observability: `/metrics` in Prometheus format, `Dockerfile` + `compose.yaml`), **Phase 5 (OpenAI-compatible API: `/v1/chat/completions` with SSE streaming, `/v1/embeddings`, `/v1/models`).** The Microsoft.Extensions.AI adapter (`Overfit.Extensions.AI` — `OverfitChatClient : IChatClient`, `OverfitEmbeddingGenerator : IEmbeddingGenerator`) ships as a separate package; a .NET Aspire dashboard variant is the remaining optional phase.
>
> **OpenAI compatibility.** The `/v1/*` surface follows the **de-facto OpenAI Chat Completions / Embeddings / Models shapes** (the same ones ollama / vLLM / llama.cpp-server expose) — point any OpenAI client/SDK at the base URL. `response_format` is honoured: `{"type":"json_object"}` guarantees well-formed JSON and `{"type":"json_schema","json_schema":{"schema":{…}}}` constrains the output to **conform** to that JSON-Schema (typed / required / enum fields) — by construction, no retries. It is a pragmatic subset otherwise: text content only (no multimodal arrays), no `tools` / `n>1` / `logprobs` / `logit_bias`; chat completions are stateless per request and serialized through a single-flight gate (the demo shares one model session — single-tenant).
>
> The same JSON-Schema constraint backs the demo's own `POST /chat/json` — pass an optional `"schema"` (JSON-Schema text) to get a schema-conforming object instead of merely well-formed JSON.
>
> ```bash
> curl -X POST http://localhost:5234/v1/chat/completions -H "Content-Type: application/json" \
>   -d '{"messages":[{"role":"user","content":"Capital of France? One word."}],"max_tokens":16,"temperature":0}'
> # → {"object":"chat.completion","choices":[{"message":{"role":"assistant","content":"Paris"},"finish_reason":"stop"}],"usage":{...}}
> # add "stream":true for Server-Sent-Events token streaming; /v1/embeddings needs an EmbeddingModelPath (MiniLM).
> ```

---

## Bielik — Polish local agent (preset) 🇵🇱

A ready-made **Polish** preset runs the demo on [Bielik](https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct-GGUF) (a Polish LLM) over Polish documents — chat, RAG, tool calling and guaranteed JSON, all in pure .NET, no Python / Ollama / model server / data egress.

```powershell
.\download-bielik.cmd                                              # the LLM (~4.8 GB GGUF → C:\bielik)
.\download-embedder.cmd                                            # the RAG embedder (~90 MB → C:\minilm)
dotnet run -c Release --project Demo/LocalAgentAspNetDemo --launch-profile bielik
```

The `bielik` profile sets `ASPNETCORE_ENVIRONMENT=Bielik`, which loads [`appsettings.Bielik.json`](appsettings.Bielik.json): Bielik GGUF (loaded with the tokenizer read from the file — no sibling `tokenizer.json` needed), a Polish system prompt, and `DataPath: Data-pl/` (regulamin, polityka reklamacji, RODO, FAQ). Everything else (endpoints, metrics) is identical to the default.

Open **[http://localhost:5234/swagger](http://localhost:5234/swagger)** and click **Try it out → Execute** — the request bodies are pre-filled (the `/decision/refund` example is already in Polish). Or use curl:

```powershell
# RAG over the Polish documents
curl -X POST http://localhost:5234/rag/query -H "Content-Type: application/json" -d '{ "question": "Ile dni ma klient z UE na odstąpienie od umowy?", "topK": 4 }'
# → "Klient z UE ma 14 dni na odstąpienie od umowy, liczone od daty zakupu."

# A C# tool call from a Polish request
curl -X POST http://localhost:5234/agent -H "Content-Type: application/json" -d '{ "message": "Załóż zgłoszenie reklamacyjne o wysokim priorytecie dla klienta anna@firma.pl." }'

# A structured business decision as guaranteed JSON (field names English, reason in Polish)
curl -X POST http://localhost:5234/decision/refund -H "Content-Type: application/json" -d '{ "message": "Klient z UE kupił produkt 10 dni temu, nie był używany, chce odstąpić od umowy." }'
# → { "eligible": true, "reason": "Klient ma prawo do odstąpienia w ciągu 14 dni...", "requiredAction": "accept_refund", "confidence": 0.95 }
```

> Bielik-4.5B Q8_0 decodes at ~8 tok/s on CPU. Retrieval ranking uses MiniLM (English-centric) — fine for the demo, but a Polish/multilingual embedder would rank Polish passages better. Borderline temporal questions ("refund *after* 10 days") are at the edge of a 4.5B's reasoning and can flip — lead with clear factual questions.

**Full step-by-step setup & run guide: [`BIELIK.md`](BIELIK.md).**

## 5-minute quickstart

### 1. Get a model

Easiest: download Qwen2.5-3B-Instruct Q4_K_M (~1.9 GB) from HuggingFace or any Ollama mirror. The demo accepts any GGUF in the Qwen / Llama / Mistral / Mixtral / Qwen-MoE family — **or** an unpacked HuggingFace directory (`model.safetensors` + `config.json` + tokenizer), loaded natively via `OverfitClient.LoadPretrained` (no GGUF conversion step).

> **Model choice — read this before going smaller.** Tool calling here is constrained-decoded, so *any* model emits valid structure, args and JSON. But choosing the *right* tool is a reasoning task: a **3B routes the demo's tool suite 6/6**, while a **0.5B is unreliable (~4/8 — it over-picks the first tool for create-ticket requests)** even though its chat/RAG/JSON are fine and ~2× faster. **Use 1.5B+ for the agent endpoint.** The demo defaults to 3B for this reason.

### 2. Point the demo at it

Pick **one** of the two paths:

**(a) appsettings.Development.json** — a `*.gguf` file **or** a directory with `model.safetensors`:

```jsonc
{
  "ModelPath": "C:\\qwen3b\\qwen.q4km.gguf"   // GGUF file …
  // "ModelPath": "C:\\qwen3b"                // … or a HuggingFace safetensors directory
}
```

**(b) Environment variable** — point at a directory containing a `*.gguf` (preferred) or `model.safetensors`:

```powershell
$env:OVERFIT_MODEL_DIR = "C:\qwen3b"
```

**For RAG** (optional — `/chat` works without it): also point at a MiniLM sentence-embedding directory (`config.json` + `vocab.txt` + `model.safetensors`), via `EmbeddingModelPath` in appsettings or the `OVERFIT_EMBEDDING_DIR` env var:

```jsonc
{ "EmbeddingModelPath": "C:\\minilm" }
```

### 3. Run

```powershell
dotnet run -c Release --project Demo/LocalAgentAspNetDemo
```

First request loads the model into mmap-backed K-quant weights (~1–2 seconds, sub-300 MB live managed heap). Subsequent calls are zero-allocation on the decode hot path.

### 4. Try it — Swagger (easiest)

Open **[http://localhost:5234/swagger](http://localhost:5234/swagger)** (`GET /` redirects there). Every endpoint is listed with its request body **pre-filled with a ready-to-run example** — expand one, click **Try it out → Execute**, no typing. This is the fastest way to explore the whole agent.

Prefer the terminal? The same calls as curl (the curl examples below are exactly what Swagger sends):

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

> The sections below show each endpoint as curl for reference — but you can run all of them from Swagger with the pre-filled examples instead.

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

`/chat/json` constrains the output to **well-formed JSON by construction** (`JsonGrammarConstraint`). Pass an optional **`schema`** (JSON-Schema text) and it constrains the output to **conform to that schema** instead (`JsonSchemaConstraint` — value types, all `required` properties present, only declared keys under `additionalProperties:false`, string `enum` values, nested objects, simple arrays). Describe the shape you want in the message:

```powershell
# well-formed JSON (no schema)
curl -X POST http://localhost:5234/chat/json `
  -H "Content-Type: application/json" `
  -d '{ "message": "Extract intent and email as JSON from: I want to cancel, reach me at jo@tinkergarden.example" }'

# schema-conforming JSON (typed / required / enum) — enforced at the logit level, no retries
curl -X POST http://localhost:5234/chat/json `
  -H "Content-Type: application/json" `
  -d '{ "message": "Classify the sentiment of: the support was fantastic.", "schema": "{\"type\":\"object\",\"properties\":{\"sentiment\":{\"type\":\"string\",\"enum\":[\"positive\",\"negative\",\"neutral\"]}},\"required\":[\"sentiment\"],\"additionalProperties\":false}" }'
```

The response is returned with `Content-Type: application/json` and is guaranteed to parse; with a `schema` it is also guaranteed to satisfy that schema (the same constraint backs `/v1` `response_format: json_schema`). The schema subset is documented at `JsonSchemaCompiler`; the open follow-ons are a per-state mask cache (throughput) and token healing (rare dead-end repair on weak models with multiple free-form string keys).

### 8. Watch the metrics

`GET /metrics` exposes Prometheus text exposition format — no exporter dependency:

```powershell
curl http://localhost:5234/metrics
```

The instruments come from `System.Diagnostics.Metrics` (`Meter`) and are exported by OpenTelemetry, so the exposition is standard OpenMetrics (`_total` counters, `_bucket`/`_sum`/`_count` histograms, an `otel_scope_name` label):

```text
# TYPE overfit_build_info gauge
overfit_build_info{otel_scope_name="Overfit.LocalAgent",model="qwen.q4km.gguf",fingerprint="9f3a1c0b7d2e4a51",mmap="true"} 1
# TYPE overfit_model_load_seconds gauge
# UNIT overfit_model_load_seconds seconds
overfit_model_load_seconds{otel_scope_name="Overfit.LocalAgent"} 0.84
# TYPE overfit_requests_total counter
overfit_requests_total{otel_scope_name="Overfit.LocalAgent",endpoint="chat"} 3
overfit_requests_total{otel_scope_name="Overfit.LocalAgent",endpoint="agent"} 2
# TYPE overfit_generated_tokens_total counter
overfit_generated_tokens_total{otel_scope_name="Overfit.LocalAgent"} 45
# TYPE overfit_allocated_bytes_total counter
overfit_allocated_bytes_total{otel_scope_name="Overfit.LocalAgent"} 0
# TYPE overfit_tool_calls_total counter
overfit_tool_calls_total{otel_scope_name="Overfit.LocalAgent",tool="create_ticket"} 1
# TYPE overfit_decode_rate_per_second histogram   (per-generation throughput; _bucket lines elided)
overfit_decode_rate_per_second_sum{otel_scope_name="Overfit.LocalAgent",endpoint="chat"} 46.2
overfit_decode_rate_per_second_count{otel_scope_name="Overfit.LocalAgent",endpoint="chat"} 3
# TYPE overfit_rag_search_seconds histogram       (retrieval latency; _bucket lines elided)
overfit_rag_search_seconds_sum{otel_scope_name="Overfit.LocalAgent"} 0.016
overfit_rag_search_seconds_count{otel_scope_name="Overfit.LocalAgent"} 1
```

`overfit_allocated_bytes_total / overfit_generated_tokens_total` is the per-token allocation — it should sit near zero, which is the whole point. The model fingerprint is a fast partial hash (size + head + tail), enough to pin which model build is running without hashing the whole multi-GB file. Full metric reference: [`Observability/otel.md`](Observability/otel.md).

## Run the full stack with Docker

`compose.yaml` builds the agent image and runs Prometheus scraping its `/metrics`. The model file stays on your machine — it's mounted into the container, never baked into the image, never sent anywhere.

```powershell
# 1. Edit compose.yaml: set the volume source to your local GGUF directory (default C:/qwen3b).
# 2. Bring up the agent + Prometheus.
docker compose -f Demo/LocalAgentAspNetDemo/compose.yaml up --build
```

- App: `http://localhost:5234` (e.g. `curl http://localhost:5234/health`)
- Prometheus: `http://localhost:9090` — graph `overfit_decode_rate_per_second` (histogram) or the per-token allocation

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

## Production hardening (auth · audit · probes)

Three config switches turn the demo into a deployable PoC — all **off by default** so it still runs out of the box.

- **API-key auth.** Set `ApiKey` (config or the `ApiKey` env var) and every call must present it as `X-API-Key: <key>`
  or `Authorization: Bearer <key>`; missing/wrong → `401` **before the model is touched**. The key is stored only as a
  SHA-256 hash and compared in constant time (never logged). Probe + Swagger paths stay open.
- **Audit trail.** Every handled request appends one JSON line — `{ ts, id, method, path, status, ms, client, model,
  sources, tool }` — **metadata only, never the prompt or the answer**, so it honours "data never leaves the process".
  `client` is a pseudonymous key fingerprint; `model` is the loaded model's fingerprint; `sources` / `tool` record
  which documents a RAG query retrieved and which C# tool the agent invoked. Streams to the structured log by default;
  set `AuditLogPath` for an append-only JSONL file you can retain and tail.
- **Kubernetes-style probes.** `/healthz` (liveness — never touches the model, so a long generation can't fail it) and
  `/readyz` (readiness — `200` once the model is loaded). The container `HEALTHCHECK` uses `/readyz`; the rich
  `/health` stays for humans. The Deployment + Service + model-PVC pattern in
  [`docs/overfit.k8s.yaml`](../../docs/overfit.k8s.yaml) applies — point it at this image and probe `/readyz`.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/` | Redirects to `/health` |
| `GET` | `/health` | `200 OK` with model filename, fingerprint, runtime, process privacy flag, RAG index status |
| `GET` | `/healthz` | Liveness probe — `200` while the process is up (never touches the model) |
| `GET` | `/readyz` | Readiness probe — `200` once the model is loaded, `503` while loading |
| `POST` | `/chat` | `{ message: "…" }` → `{ reply, stats }`. Conversation persists in the singleton client |
| `POST` | `/reset` | Clears conversation history; re-applies the system message |
| `POST` | `/documents/index` | Chunks + embeds every `*.md` in the data directory into the in-process vector store. Returns per-file chunk counts |
| `POST` | `/rag/query` | `{ question: "…", topK?: 4 }` → `{ reply, sources[], stats }`. Retrieves top-K chunks and answers from them |
| `POST` | `/agent` | `{ message: "…" }` → `{ toolName, arguments, result }`. Forces one registered C# tool call and dispatches it |
| `POST` | `/chat/json` | `{ message: "…", schema?: "…" }` → guaranteed JSON (`application/json`): well-formed, or schema-conforming when `schema` is supplied |
| `GET` | `/metrics` | Prometheus text exposition — requests, tokens, allocations/token, tok/s, tool calls, RAG latency |

System message comes from `SystemMessage` in `appsettings.json` (override per environment via `appsettings.Development.json`).

RAG is optional: `/chat` works with no embedding model configured. `/documents/index` and `/rag/query` return a clear `400` with setup instructions until `EmbeddingModelPath` / `OVERFIT_EMBEDDING_DIR` is set.

---

## Roadmap

- ✅ **Phase 1** — `/health`, `/chat`, `/reset` over a singleton `OverfitClient`.
- ✅ **Phase 2** — RAG over your documents (`/documents/index`, `/rag/query`) via `VectorStore` + `SentenceEmbedder.ForMiniLm` over the `Data/` folder.
- ✅ **Phase 3** — C# tool calling (`Tools/ToolRegistry.cs`: `lookup_customer`, `create_ticket`) via `ToolCallConstraint`, dispatched to C# handlers, plus a guaranteed-JSON endpoint (`/chat/json`): well-formed via `JsonGrammarConstraint`, or schema-conforming via `JsonSchemaConstraint` when an optional `schema` is passed.
- ✅ **Phase 4** — `/metrics` (Prometheus: requests, tokens, allocations/token, tok/s, tool calls, RAG latency, model fingerprint) + `Dockerfile` + `compose.yaml` (agent + Prometheus) + `Observability/`.
- ✅ **Phase 5** — OpenAI-compatible API (`/v1/chat/completions` + SSE streaming, `/v1/embeddings`, `/v1/models`); `response_format` honoured (`json_object` → well-formed, `json_schema` → schema-conforming).
- ✅ **Phase 6** — Microsoft.Extensions.AI adapter, shipped as the separate `Overfit.Extensions.AI` package (`OverfitChatClient : IChatClient`, `OverfitEmbeddingGenerator : IEmbeddingGenerator`) — slot Overfit in as a drop-in local backend wherever `IChatClient` / `IEmbeddingGenerator` is expected. (The demo's `/v1` surface uses the engine directly; the adapter is for in-process `IChatClient` consumers.)
- ✅ **Phase 7 (production hardening)** — API-key auth (SHA-256 + constant-time, `401` before the model), append-only metadata-only audit trail (`AuditLogPath`), Kubernetes-style `/healthz` + `/readyz` probes, non-root container + `HEALTHCHECK`, single-file k8s manifest. Turns the starter into a deployable, auditable PoC (maps to the "Private .NET RAG/Agent PoC" in [`COMMERCIAL.md`](../../COMMERCIAL.md)).
- ❌ **Phase 7** (optional) — .NET Aspire dashboard variant.

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
dotnet build Demo/LocalAgentAspNetDemo -c Release
```

Or as part of the solution: `dotnet build Overfit.sln -c Release`.

## Feedback / commercial

Production deployment, custom tools, regulated-industry setup, or multi-tenant variants — see [`../../COMMERCIAL.md`](../../COMMERCIAL.md). Contact: `devonbike@gmail.com`.
