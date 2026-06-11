# DevOnBike.Overfit.Mcp

An **MCP (Model Context Protocol) server** for the [Overfit](https://github.com/DevOnBike/Overfit) in-process
LLM runtime — dependency-free JSON-RPC 2.0 over **stdio**, with typed wire contracts (`Protocol/`) serialized
through a **source-generated** `JsonSerializerContext` (no SDK, no reflection, no DI), so it drops cleanly into
a Native-AOT single binary (AOT-verified end-to-end). Plug local, zero-egress AI tools into any MCP host:
Claude Code, Claude Desktop, IDEs.

## What you get

- `McpServer` — a minimal stdio MCP server (`initialize` / `ping` / `tools/list` / `tools/call`,
  newline-delimited JSON-RPC, logs on stderr). Requests are served one at a time (single-tenant model
  session underneath — same stance as the OpenAI server).
- `McpTool` — name + description + raw JSON Schema + a `Func<JsonElement?, McpToolResult>` handler.
  Tool-execution failures flow back as `isError` results (so the host model can self-correct), never as
  protocol errors.
- `Protocol/` — the typed JSON-RPC/MCP wire contracts (`JsonRpcRequest`, `JsonRpcResponse<T>`,
  `McpInitializeResult`, `McpToolsListResult`, `McpCallToolResult`, …) with a **source-generated**
  `McpJsonContext : JsonSerializerContext`: compile-time serializers, no reflection, every closed
  `JsonRpcResponse<T>` registered explicitly. `JsonElement` is used only where the JSON is genuinely
  dynamic (the echoed `id`, `params`/`arguments`, `inputSchema`). The same contracts will serve the
  future MCP *client* (host role) — the framing is symmetric.
- `OverfitMcpTools` — factories for the built-in tools backed by the Overfit runtime:
  - **`ask`** — prompt → a local GGUF chat model (Qwen / Llama / Phi / Gemma / Mistral / Bielik …)
  - **`rag_query`** — question → answer with citations over a private, locally-indexed document folder
  - **`transcribe`** — WAV/MP3 file → text via Whisper, pure C# on the CPU

## Usage

```csharp
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.Mcp;

using var client = OverfitClient.LoadGguf(@"C:\models\qwen2.5-3b-instruct-q4_k_m.gguf");

var tools = new[] { OverfitMcpTools.CreateAsk(client) };
var server = new McpServer("overfit", "10.0.23", tools, log: Console.Error);
server.Run(Console.In, Console.Out);   // blocks until the host closes stdin
```

This is exactly what the `overfit mcp` CLI command runs. Register it with an MCP host, e.g. Claude Code:

```bash
claude mcp add overfit -- overfit mcp C:\models\model.gguf
```

Or use the helper scripts next to this README: `mcp-register.cmd` / `mcp-unregister.cmd` /
`mcp-status.cmd` / `mcp-run.cmd` (foreground debug) / `mcp-smoke.cmd` (no-Claude handshake test).

## Notes

- stdio transport only (local process = on-brand zero egress). The protocol layer is host-agnostic —
  an HTTP transport can reuse it.
- Everything stays on your machine: the model, the documents, the audio. No keys, no egress.
- Validated three ways: 12 model-free protocol tests, end-to-end over stdio against the real CLI
  binary, and inside real Claude Code (`✔ Connected` + a live session driving all three tools —
  including a Polish RAG question, confirming the chat-model embeddings are multilingual). Details
  and the PowerShell `--` gotcha: [`docs/mcp.md`](../../docs/mcp.md).

## License

Dual-licensed: **AGPL-3.0-or-later** for open source; a **commercial license** is available for
closed-source / SaaS / regulated use — see `COMMERCIAL.md`.
