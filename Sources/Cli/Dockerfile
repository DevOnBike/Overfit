# Native-AOT build of the `overfit` CLI, exposing the OpenAI-compatible HTTP server (`overfit serve`).
#
# Identity: a single self-contained NATIVE binary (no .NET runtime, no ASP.NET) on a chiselled base —
# tiny image, fast cold start. The model is NOT baked in (GGUF files are large + licensed separately):
# provide it at runtime via a mounted volume, e.g.
#
#   docker build -f Sources/Cli/Dockerfile -t overfit .            # build context = REPO ROOT
#   docker run -p 8080:8080 -v /host/models:/models overfit \
#       /models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
#
# The model is a POSITIONAL arg of `serve` — append the in-container path after the image name. Extra
# server options follow it (they pass through to the entrypoint):
#   /models/chat.gguf  --embed-model /models/bge.gguf  --tts-model /models/orpheus.gguf  --tts-snac /models/snac
#
# To pre-populate a volume from a HuggingFace GGUF using the same binary (no curl needed):
#   docker run -v /host/models:/models --entrypoint /app/overfit overfit \
#       pull <hf-owner/repo> --file <file.gguf>

# ---- build stage: SDK + Native-AOT toolchain (mirrors .github/workflows/ci.yml aot-guard) ----
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
RUN apt-get update && apt-get install -y --no-install-recommends clang zlib1g-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# InvariantGlobalization + PublishAot are already set in Cli.csproj; -r selects the native RID.
RUN dotnet publish ./Sources/Cli/Cli.csproj -c Release -r linux-x64 -p:PublishAot=true -p:GenerateDocumentationFile=false -o /app
# Result: /app/overfit — a self-contained native ELF.

# ---- runtime stage (MCP, build with --target mcp): same native binary, stdio MCP server ----
# An MCP host runs this with an INTERACTIVE pipe (stdio transport - no port, no HTTP):
#   claude mcp add overfit -- docker run -i --rm -v /host/models:/models devonbikeit/overfit:mcp #       /models/model.gguf [--rag-dir /docs] [--whisper-model /models/ggml-tiny.bin]
FROM mcr.microsoft.com/dotnet/runtime-deps:10.0-noble-chiseled AS mcp
WORKDIR /app
COPY --from=build /app/overfit /app/overfit
ENTRYPOINT ["/app/overfit", "mcp"]

# ---- runtime stage (default): chiselled runtime-deps (glibc + zlib only; no shell, no .NET) ----
FROM mcr.microsoft.com/dotnet/runtime-deps:10.0-noble-chiseled AS runtime
WORKDIR /app
COPY --from=build /app/overfit /app/overfit

EXPOSE 8080
# exec-form entrypoint: `docker run <img> --model /models/x.gguf` appends to these args.
ENTRYPOINT ["/app/overfit", "serve", "--host", "0.0.0.0", "--port", "8080"]
