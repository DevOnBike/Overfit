# DevOnBike.Overfit.Server

An **OpenAI-compatible HTTP server** for the [Overfit](https://github.com/DevOnBike/Overfit) in-process
LLM runtime — dependency-free (`System.Net.HttpListener` + `System.Text.Json` source-gen), **no ASP.NET Core**,
so it drops cleanly into a Native-AOT single binary. Point any OpenAI client at the base URL and only change the
model name.

## What you get

- `OverfitOpenAiServer.Serve(…)` — a blocking server exposing:
  - `POST /v1/chat/completions` — streaming (SSE) **and** non-streaming, with `response_format` →
    JSON / JSON-Schema constrained decoding; sampling via `temperature` / `top_p` / `min_p`
    (llama.cpp-server extension; takes precedence over `top_p`)
  - `POST /v1/embeddings` (when started with a sentence embedder), `POST /v1/audio/speech` (when started with a TTS engine)
  - `GET /v1/models`, `GET /health` (reports pool load), `GET /openapi.yaml` (the contract), `GET /docs` (Swagger UI)
- Two overloads control concurrency:
  - `Serve(client, …)` — one session; requests are **serialized** through it (like a local llama.cpp server).
  - `Serve(pool, …)` — a `OverfitResourcePool<OverfitClient>` of N sessions; up to N chat completions decode
    **concurrently**, and excess load is shed with **HTTP 503** rather than queued unboundedly.
- `DevOnBike.Overfit.Server.OpenAi` — the OpenAI request/response DTOs + a source-generated
  `JsonSerializerContext` (`OpenAiJsonContext`) and host-agnostic `OpenAiChatMapping` (sampling /
  `response_format` → constraint / history replay) you can reuse from ASP.NET or your own host.

## Usage

```csharp
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.Server;

using var client = OverfitClient.LoadGguf(@"C:\models\qwen2.5-3b-instruct-q4_k_m.gguf");
client.AddSystem("You are a concise assistant running locally in pure .NET.");

// Serves until the token is cancelled. One session → requests are serialized.
OverfitOpenAiServer.Serve(client, "qwen2.5-3b", host: "127.0.0.1", port: 11434,
    systemMessage: "You are a concise assistant running locally in pure .NET.");
```

For concurrent sessions, load N clients (they share the model weights via mmap — the OS page cache
de-duplicates them — so the extra cost is N× the KV cache, not N× the weights) and serve a pool:

```csharp
using DevOnBike.Overfit.Serving;

var clients = new List<OverfitClient>();
for (var i = 0; i < 4; i++)
{
    var c = OverfitClient.LoadGguf(@"C:\models\qwen2.5-3b-instruct-q4_k_m.gguf");
    c.AddSystem("You are a concise assistant running locally in pure .NET.");
    clients.Add(c);
}
using var pool = new OverfitResourcePool<OverfitClient>(clients);   // owns + disposes the clients

OverfitOpenAiServer.Serve(pool, "qwen2.5-3b", host: "127.0.0.1", port: 11434,
    systemMessage: "You are a concise assistant running locally in pure .NET.");
```

This is exactly what the [`overfit serve`](https://github.com/DevOnBike/Overfit) CLI command uses —
`overfit serve <model>` is one session, `overfit serve <model> --sessions 4` builds the pool.

## Notes

- **Concurrency** is bounded by the session pool. One session (`Serve(client, …)` / `--sessions 1`, the default)
  serializes requests. A pool of N (`Serve(pool, …)` / `--sessions N`) decodes up to N chat completions at once;
  each request rents a session for its decode, waits up to 30 s for a free one, then is shed with **503**.
  Embeddings and TTS use single shared engines and are serialized. `GET /health` reports pool load
  (`size`, `active`, `available`, `rented`, `rejected`, `peak`).
- **Throughput caveat (measured, honest):** each decode already parallelizes across all CPU cores, so on a
  single box N concurrent decodes *share* the cores — the pool buys **concurrency / fairness / load-shedding**
  (no head-of-line blocking, bounded queueing), **not** N× throughput. To trade latency for throughput, cap the
  per-session worker count (`OVERFIT_DECODE_WORKERS`) so N sessions × workers ≈ core count.
- Binding to `127.0.0.1`/`localhost` needs no elevation; `0.0.0.0`/`*` may need a URL ACL / admin on Windows.
- The decode worker pool **parks when idle** (spin-then-park) — a server waiting for requests sits at ~0% CPU.
- Not included (add at your edge / reverse proxy): authentication, rate limiting, TLS. The server is the
  inference surface, not a hardened public gateway.

## License

Dual-licensed: **AGPL-3.0-or-later** for open source; a **commercial license** is available for
closed-source / SaaS / regulated use — see `COMMERCIAL.md`.
