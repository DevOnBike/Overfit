# `overfit` CLI

The command-line front-end for Overfit: an **OpenAI-compatible local LLM server** plus model `pull` / `chat` /
`embed` / `tts`, all pure .NET — no Python, no native engine. It links `Sources/Main` (the runtime) and
`Sources/Server` (the HttpListener OpenAI server).

```text
overfit pull  <hf-owner/repo> [--file x.gguf]   download a GGUF into the local store
overfit serve <model> [--host --port ...]       OpenAI-compatible HTTP server
overfit chat  <model>                            interactive chat in the terminal
overfit tts   --text ... --out x.wav             text-to-speech (watermarked)
overfit list                                     list locally-cached models
```

`overfit serve` exposes `/v1/chat/completions` (streaming + non-streaming), `/v1/embeddings`, `/v1/audio/speech`,
`/v1/models` and `/health`. See [`../Server/README.md`](../Server/README.md) for the endpoint contracts.

## Three ways to ship it

| Channel | What you get | How |
|---|---|---|
| **Native-AOT binary** | one self-contained native exe, no .NET runtime | `dotnet publish Cli.csproj -c Release -r <rid> -p:PublishAot=true` |
| **Docker image** | tiny chiselled image (~34 MB), model mounted at runtime | `docker-build.cmd` → `docker-run.cmd` (below) |
| **.NET global tool** | `dotnet tool install -g DevOnBike.Overfit.Cli` → run `overfit` (IL, needs the .NET runtime) | published by `.github/workflows/publish-nuget.yml` |

> The native binary / Docker image are the **AOT** story (no runtime). The global tool is the cross-platform
> **convenience** story (IL, one `dotnet tool install`). Same code, two trade-offs — pick per audience.

## Docker — quick local build & run

The scripts live here, next to the `Dockerfile`. The build context is the **repo root** (the CLI references
`Sources/Main` + `Sources/Server`); both scripts handle that for you.

```bat
REM build the image (tag defaults to overfit:local)
Sources\Cli\docker-build.cmd

REM run it with a model mounted from the host (model is a POSITIONAL arg; default port 8080)
Sources\Cli\docker-run.cmd C:\qwen3-06b\Qwen3-0.6B-Q8_0.gguf 8080
```

The model is **never baked into the image** — `docker-run.cmd` bind-mounts the model's folder to `/models` and
serves it. Full hosting guide (model-on-boot, free tiers like HF Spaces / Oracle): [`../../docs/docker.md`](../../docs/docker.md).

## Endpoints — curl & PowerShell

Examples assume the server on `http://localhost:8080` (the `docker-run.cmd` default; the bare binary defaults to
`11434`). Responses are printed **raw, pretty-formatted**. PowerShell needs no extra tools (`ConvertTo-Json` is
built in); the bash examples pipe to [`jq`](https://jqlang.github.io/jq/) (`winget install jqlang.jq`).

> PowerShell gotchas: keep the JSON body on **one line** in **single quotes**, and use **`curl.exe`** (plain `curl`
> is an alias for `Invoke-WebRequest`). `Invoke-RestMethod` deserializes and the console hides nested fields — pipe
> it back through `ConvertTo-Json -Depth 10` to see everything.

### `GET /health` — readiness  ·  `GET /v1/models` — loaded model id

```bash
curl -s http://localhost:8080/health | jq
curl -s http://localhost:8080/v1/models | jq
```

```powershell
Invoke-RestMethod http://localhost:8080/health | ConvertTo-Json
Invoke-RestMethod http://localhost:8080/v1/models | ConvertTo-Json -Depth 6
```

### `POST /v1/chat/completions` — chat

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"overfit","messages":[{"role":"system","content":"You are concise."},{"role":"user","content":"Capital of France?"}],"max_tokens":200,"temperature":0.7,"top_p":0.9}' | jq
```

```powershell
curl.exe -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" `
  -d '{"model":"overfit","messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":200}' |
  ConvertFrom-Json | ConvertTo-Json -Depth 10
```

Just the text: bash `... | jq -r '.choices[0].message.content'` · PowerShell `(Invoke-RestMethod ...).choices[0].message.content`.

Sampling fields: `temperature` (≈0 → greedy), `top_p`, `min_p` (llama.cpp-style — keep tokens with
P ≥ `min_p` × P(top); takes precedence over `top_p`), `max_tokens`.

### `POST /v1/chat/completions` with `"stream": true` — SSE streaming

```bash
curl -N -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"overfit","messages":[{"role":"user","content":"Tell a one-line joke."}],"stream":true}'
```

```powershell
# stream raw SSE chunks to the console
curl.exe -N -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" `
  -d '{"model":"overfit","messages":[{"role":"user","content":"Tell a one-line joke."}],"stream":true}'
```

### `POST /v1/embeddings` — embeddings  *(requires `serve --embed-model <bert-dir>`, else `501`)*

```bash
curl -s http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"overfit","input":["the cat sat","a feline rested"]}' | jq '.data[].embedding | length'
```

```powershell
curl.exe -s http://localhost:8080/v1/embeddings -H "Content-Type: application/json" `
  -d '{"model":"overfit","input":"the cat sat on the mat"}' |
  ConvertFrom-Json | ConvertTo-Json -Depth 6
```

### `POST /v1/audio/speech` — text-to-speech  *(requires `serve --tts-model <orpheus.gguf> --tts-snac <dir>`, else `501`; returns binary audio → save to a file)*

```bash
curl -s http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"overfit","input":"Hello from Overfit.","voice":"tara","response_format":"wav"}' \
  -o speech.wav
```

```powershell
Invoke-WebRequest http://localhost:8080/v1/audio/speech -Method Post -ContentType 'application/json' `
  -Body '{"model":"overfit","input":"Hello from Overfit.","voice":"tara","response_format":"wav"}' `
  -OutFile speech.wav
```

## Tuning (environment variables)

The defaults are right for almost everyone — these are escape hatches:

| Variable | Default | What it does |
|---|---|---|
| `OVERFIT_REPACK_GEMV` | off | `1` repacks Q4_K weights into the 8×8 GEMV layout at load → **~+30% decode** on a 3B (costs extra RAM for the repacked copy; pairs well with `OVERFIT_DECODE_WORKERS=16` on a 16-core box) |
| `OVERFIT_DECODE_WORKERS` | `min(cores, 10)` | caps the decode dispatch width (per-token matmuls are dispatch-bound, not core-hungry) |
| `OVERFIT_DECODE_POOL` | on | `0` disables the decode spin-pool (≈ −28% decode on small models). The pool **spins-then-parks**: it burns no CPU between requests, so there is no idle-cost reason to turn it off |

## Versioning

The version is centralised in [`../../Directory.Build.props`](../../Directory.Build.props) (all projects inherit
it). Bump it with [`../../bump-version.ps1`](../../bump-version.ps1) (`patch` / `minor` / `major` / `x.y.z`); a
`v<version>` git tag triggers both the Docker Hub and NuGet publish workflows.
