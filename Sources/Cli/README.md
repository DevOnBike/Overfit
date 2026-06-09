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

## Hitting the server with curl

Print the **raw response, pretty-formatted**:

```bash
# bash / Git Bash / WSL — pipe to jq
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"overfit","messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":200}' | jq
```

```powershell
# PowerShell — keep the JSON body on ONE line in SINGLE quotes, then format the response.
curl.exe -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" `
  -d '{"model":"overfit","messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":200}' |
  ConvertFrom-Json | ConvertTo-Json -Depth 10

# or Invoke-RestMethod (which deserializes — alone it HIDES nested fields, so pipe it back to ConvertTo-Json):
Invoke-RestMethod http://localhost:8080/v1/chat/completions -Method Post -ContentType 'application/json' `
  -Body '{"model":"overfit","messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":200}' |
  ConvertTo-Json -Depth 10
```

Just the answer text: bash `... | jq -r '.choices[0].message.content'`, PowerShell `(Invoke-RestMethod ...).choices[0].message.content`.

```bash
curl http://localhost:8080/health           # readiness
curl http://localhost:8080/v1/models        # model id
```

## Versioning

The version is centralised in [`../../Directory.Build.props`](../../Directory.Build.props) (all projects inherit
it). Bump it with [`../../bump-version.ps1`](../../bump-version.ps1) (`patch` / `minor` / `major` / `x.y.z`); a
`v<version>` git tag triggers both the Docker Hub and NuGet publish workflows.
