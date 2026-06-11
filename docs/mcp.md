# MCP server — plug local Overfit AI into Claude Code / Desktop / IDEs

`overfit mcp` runs an **MCP (Model Context Protocol) stdio server**: any MCP host (Claude Code,
Claude Desktop, Cursor, VS Code extensions…) gets local, **zero-egress** AI tools backed by the
Overfit runtime — the model, your documents and your audio never leave the machine.

The protocol layer (`DevOnBike.Overfit.Mcp` package) is JSON-RPC 2.0 with **typed wire contracts**
(`Protocol/` DTOs) serialized through a **source-generated** `JsonSerializerContext`
(`McpJsonContext`, same pattern as the OpenAI server's `OpenAiJsonContext`) — **no SDK dependency,
no reflection, no DI** — so it ships inside the same Native-AOT binary / ~34 MB Docker image as the
rest of the CLI. AOT-verified: zero IL/trim analyzer warnings, and the published native `overfit.exe`
(ILCompiler) serves the full handshake.

## Quick start

```bash
# register with Claude Code (model path or local-store name; pull one first if needed)
claude mcp add overfit -- overfit mcp C:\models\qwen2.5-3b-instruct-q4_k_m.gguf

# everything on: private-docs RAG + Whisper transcription
claude mcp add overfit -- overfit mcp C:\models\model.gguf --rag-dir C:\docs --whisper-model C:\whisper\ggml-tiny.bin
```

Then just ask Claude things like *"use overfit to ask the local model …"*, *"query my private docs
for the refund policy"*, or *"transcribe C:\audio\meeting.mp3"*.

> **PowerShell gotcha:** pwsh swallows the bare `--` separator (it's PowerShell's own end-of-parameters
> token), so `claude mcp add` mis-parses the server's options. Run the command from **cmd / git-bash**,
> or quote it in pwsh: `claude mcp add overfit '--' overfit mcp <model.gguf> ...`.

## Tools

| Tool | Enabled by | What it does |
|---|---|---|
| `ask` | always | prompt → the locally-loaded GGUF chat model (Qwen / Llama / Phi / Gemma / Mistral / Bielik …), stateless per call |
| `rag_query` | `--rag-dir <folder>` | question → grounded answer **with source citations** over your `.txt`/`.md` files. Indexed once at startup; embeddings come from the chat model's own hidden states (multilingual — works for Polish corpora, no second model) |
| `transcribe` | `--whisper-model <ggml>` | WAV/MP3 file → text via Whisper, pure C# on the CPU. The Whisper model loads lazily on first call |

Tool-execution failures (missing file, empty folder…) come back as MCP `isError` results, so the
host model can read them and self-correct.

## What you can build with it

- **Private-docs Q&A inside your AI tooling** — index a contracts/wiki/compliance folder once, then ask
  Claude "what does our refund policy say?" mid-task; answers come back grounded with `[n]` citations.
- **Cheap local delegation** — let the host offload mechanical bulk work (summarise these 50 files,
  classify this text, draft variants) to the local model via `ask` instead of burning host tokens.
- **Audio in the loop** — "transcribe the meeting recording and act on it": local Whisper feeds the
  transcript straight into the host's workflow, EN + PL.
- **Regulated / air-gapped dev boxes** — the corpus, the model and the audio are files on the machine;
  Overfit makes **no** network calls. (Honest scope: whatever a tool *returns* enters the host's
  conversation — with a cloud host like Claude that answer text does go to the host's API. Keep
  secrets out of tool results, or use a local host.)
- **Your own tools, any MCP host** — `McpServer` + a custom `McpTool` turns any C# function into a
  tool for Claude Code / Desktop / Cursor: internal APIs, databases, domain calculators, without
  writing a protocol layer.

## Spec conformance (honest)

Implements the **tools profile** of the official MCP spec — the capability-negotiation design makes a
tools-only server fully conformant (capabilities advertise exactly what we support):

| Spec area | Status |
|---|---|
| JSON-RPC 2.0 framing, stdio transport (newline-delimited, logs → stderr) | ✅ per spec |
| `initialize` lifecycle + version negotiation (echo a supported revision, else offer our latest: 2025-06-18 / 2025-03-26 / 2024-11-05) | ✅ per spec |
| `notifications/initialized`, notifications never answered | ✅ per spec |
| `ping` → empty result | ✅ per spec |
| `tools/list` (name / description / JSON-Schema `inputSchema`), `tools/call` (text content + `isError`) | ✅ per spec |
| Protocol vs execution errors (`-32601`/`-32602`/`-32700` vs `isError`) | ✅ per spec |
| Shutdown by closing stdin | ✅ per spec |
| `resources` / `prompts` / `completions` / `logging` capabilities | ➖ not offered (conformantly absent from `capabilities`) |
| `tools/list` pagination (`cursor`), `listChanged` notifications | ➖ static tool set, full list always returned |
| Structured tool output (`structuredContent` / `outputSchema`, 2025-06-18 optional) | ➖ text content only |
| `notifications/cancelled` / progress | ➖ accepted and ignored (requests run to completion) |
| Streamable HTTP transport | ➖ stdio only (the protocol layer is transport-agnostic) |

Interop proof: registered in **real Claude Code** (the reference host) — `✔ Connected` health check and
a live session driving all three tools.

## Protocol notes

- **stdio transport** (newline-delimited JSON-RPC; one request at a time — single-tenant model
  session, same stance as `overfit serve`). All Overfit logs go to **stderr**; stdout carries only
  protocol frames.
- Implements `initialize` (spec revisions `2025-06-18` / `2025-03-26` / `2024-11-05`), `ping`,
  `tools/list`, `tools/call`. The server stops when the host closes its stdin.
- Embedding in your own host: `new McpServer(name, version, tools).Run(Console.In, Console.Out)`
  with tools from `OverfitMcpTools` or your own `McpTool` (name + description + raw JSON Schema +
  handler). See [`Sources/Mcp/README.md`](../Sources/Mcp/README.md).

## Docker

The MCP server ships as a Docker image variant — same ~34 MB chiselled Native-AOT binary as the
serve image, with `ENTRYPOINT ["overfit", "mcp"]`. The MCP host pipes stdio straight into the
container (`docker run -i` — no port, no HTTP), so no .NET and no local build are needed:

```bash
claude mcp add overfit -- docker run -i --rm -v /host/models:/models devonbikeit/overfit:mcp \
    /models/model.gguf --rag-dir /docs --whisper-model /models/ggml-tiny.bin
```

Tags on Docker Hub: `:mcp` (latest) and `:<version>-mcp`, published by the same "Docker Hub"
workflow as the serve image (shared layers — the second target is tag-only cost). Build locally with
`Sources/Cli/docker-build-mcp.cmd`; register/smoke the dockerized server with
`Sources/Mcp/mcp-register-docker.cmd` / `mcp-smoke-docker.cmd` (they handle the volume mounts:
model folder → `/models`, rag folder → `/docs`, whisper folder → `/whisper`).

Validated: the containerized server (Linux native binary) answered the full `initialize` +
`tools/list` handshake over `docker run -i` with a host-mounted GGUF.

## Helper scripts (`Sources/Mcp/*.cmd`)

| Script | What it does |
|---|---|
| `mcp-register.cmd <model.gguf> [rag-dir] [whisper-ggml]` | registers the server in Claude Code (re-register-safe; sidesteps the PowerShell `--` gotcha) |
| `mcp-unregister.cmd` | removes the registration |
| `mcp-status.cmd` | shows the registration + spawns a health check (`✔ Connected`) |
| `mcp-run.cmd <model.gguf> [options]` | runs the server in the foreground for debugging (type JSON-RPC lines into the console) |
| `mcp-smoke.cmd <model.gguf> [options]` | no-Claude smoke test: pipes a real `initialize` + `tools/list` handshake in and prints the raw responses |
| `mcp-register-docker.cmd <model.gguf> [rag-dir] [whisper-ggml]` | registers the **dockerized** server (`devonbikeit/overfit:mcp`; override via `OVERFIT_MCP_IMAGE`) — no .NET on the machine needed |
| `mcp-smoke-docker.cmd <model.gguf> [image]` | the same handshake smoke against the Docker image |

All of them prefer the Native-AOT publish output, then the plain Release build, then a global
`overfit` on PATH. Registration scope is the **current directory's** Claude project — run them from
the repo (or your project) root.

## Validated

Three layers of proof, strongest last:

1. **Protocol** — 12 fast model-free tests (`Tests/Mcp/McpServerProtocolTests.cs`): handshake +
   version negotiation, `tools/list` shape, dispatch, `isError` semantics, JSON-RPC error codes,
   notifications-get-no-response.
2. **End-to-end over stdio** against a real `overfit.exe` process (Qwen3-0.6B + whisper-tiny):
   `ask` → "Paris", `rag_query` → grounded answer with `[n]` citations + Sources list,
   `transcribe` → the JFK clip verbatim.
3. **Inside real Claude Code** — registered with `claude mcp add`, health check `✔ Connected`,
   and a live Claude session drove all three tools, including `rag_query` answering a **Polish**
   question over a Polish document correctly with citations — the chat-model embeddings are
   multilingual in practice, not just on paper.
