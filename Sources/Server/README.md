# DevOnBike.Overfit.Server

An **OpenAI-compatible HTTP server** for the [Overfit](https://github.com/DevOnBike/Overfit) in-process
LLM runtime — dependency-free (`System.Net.HttpListener` + `System.Text.Json` source-gen), **no ASP.NET Core**,
so it drops cleanly into a Native-AOT single binary. Point any OpenAI client at the base URL and only change the
model name.

## What you get

- `OverfitOpenAiServer.Serve(client, modelName, host, port, systemMessage, …)` — a blocking, single-flight
  server exposing:
  - `POST /v1/chat/completions` — streaming (SSE) **and** non-streaming, with `response_format` →
    JSON / JSON-Schema constrained decoding
  - `GET /v1/models`, `GET /health`
- `DevOnBike.Overfit.Server.OpenAi` — the OpenAI request/response DTOs + a source-generated
  `JsonSerializerContext` (`OpenAiJsonContext`) and host-agnostic `OpenAiChatMapping` (sampling /
  `response_format` → constraint / history replay) you can reuse from ASP.NET or your own host.

## Usage

```csharp
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.Server;

using var client = OverfitClient.LoadGguf(@"C:\models\qwen2.5-3b-instruct-q4_k_m.gguf");
client.AddSystem("You are a concise assistant running locally in pure .NET.");

// Serves until the token is cancelled. One request at a time (single model session).
OverfitOpenAiServer.Serve(client, "qwen2.5-3b", host: "127.0.0.1", port: 11434,
    systemMessage: "You are a concise assistant running locally in pure .NET.");
```

This is exactly what the [`overfit serve`](https://github.com/DevOnBike/Overfit) CLI command uses.

## Notes

- Requests are served **strictly one at a time** (single-tenant model session / one KV cache), like a local
  llama.cpp server. For multi-tenant use a session-per-request pool.
- Binding to `127.0.0.1`/`localhost` needs no elevation; `0.0.0.0`/`*` may need a URL ACL / admin on Windows.

## License

Dual-licensed: **AGPL-3.0-or-later** for open source; a **commercial license** is available for
closed-source / SaaS / regulated use — see `COMMERCIAL.md`.
