# Running Overfit as a Docker image

Overfit ships an **OpenAI-compatible HTTP server** (`overfit serve`) built as a **Native-AOT** binary — a single
self-contained native ELF on a chiselled base. No .NET runtime, no ASP.NET, no Python: a tiny image with a fast
cold start, which is exactly what scale-to-zero / free-tier hosting rewards.

The **model is never baked into the image** (GGUF files are large and licensed separately). You provide it at
runtime — by mounting a volume, or by letting the container download it on first boot.

- Image build: [`Sources/Cli/Dockerfile`](../Sources/Cli/Dockerfile)
- Publish pipeline: [`.github/workflows/docker-publish.yml`](../.github/workflows/docker-publish.yml)
- Server endpoints: [`Sources/Server/README.md`](../Sources/Server/README.md) (`/v1/chat/completions`,
  `/v1/embeddings`, `/v1/audio/speech`)

## Build

The build context is the **repository root** (the CLI references `Sources/Main` + `Sources/Server`):

```bash
docker build -f Sources/Cli/Dockerfile -t overfit:local .
```

## Run

The entrypoint is `overfit serve --host 0.0.0.0 --port 8080`; anything you append after the image name passes
through to it. The model is a **positional argument** of `serve` — mount a directory holding a `.gguf` and append
its in-container path:

```bash
docker run --rm -p 8080:8080 \
  -v /host/models:/models \
  overfit:local \
  /models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf

curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"overfit","messages":[{"role":"user","content":"Say hi in one word."}]}' | jq
```

(PowerShell: pipe to `ConvertFrom-Json | ConvertTo-Json -Depth 10` for the same pretty raw output — see the
[CLI README](../Sources/Cli/README.md#endpoints--curl--powershell).)

A container at rest costs ~nothing: the decode worker pool **spins-then-parks**, so between requests the server
sits at ~0% CPU (no need to set `OVERFIT_DECODE_POOL=0` — that was the workaround for the old pure-spin pool and
costs ~28% decode throughput). Optional tuning via `-e` (see the [CLI README](../Sources/Cli/README.md#tuning-environment-variables)):
`OVERFIT_REPACK_GEMV=1` (~+30% decode on a 3B, costs RAM), `OVERFIT_DECODE_WORKERS=<n>`.

Extra server options follow the positional model path:

```bash
docker run --rm -p 8080:8080 -v /host/models:/models overfit:local \
  /models/chat.gguf \
  --embed-model /models/bge-small-en.gguf \
  --tts-model /models/orpheus.gguf --tts-snac /models/snac
```

### MCP variant (`:mcp` tags)

The same image is published with a second entrypoint — `overfit mcp` (an **MCP stdio server** for
Claude Code / Desktop / IDEs; no port, the host pipes stdio via `docker run -i`). Tags `:mcp` and
`:<version>-mcp`, built from the `mcp` Dockerfile target. See [`mcp.md`](mcp.md#docker).

### Populating a volume without a model file on hand

The same binary can fetch a HuggingFace GGUF into the volume (no `curl` needed in the image):

```bash
docker run --rm -v /host/models:/models --entrypoint /app/overfit overfit:local \
  pull <hf-owner/repo> --file <file.gguf>
```

## Kubernetes

A single-file manifest stands the same server up on a cluster — Deployment + Service + a model
PersistentVolumeClaim (an initContainer downloads the GGUF into it on first start, so `kubectl apply` works
end-to-end):

```bash
kubectl apply -f docs/overfit.k8s.yaml
kubectl port-forward svc/overfit 8080:8080
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"overfit","messages":[{"role":"user","content":"Say hi in one word."}]}' | jq
```

Edit the four marked spots in [`overfit.k8s.yaml`](overfit.k8s.yaml) — image, HuggingFace model repo/file, and
resources (a 0.5B-Q4 needs ~1 GB RAM, a 3B ~4 GB). Liveness/readiness use the server's `/health`; the chiselled
image runs non-root so `fsGroup` makes the volume writable. For external traffic switch the Service to
`type: LoadBalancer` (cloud) or `NodePort`. To skip the download, pre-populate the claim and drop the
initContainer (notes in the file).

## Choosing a model

The server loads any architecture Overfit supports (see [`supported-models.md`](supported-models.md)). For a
**public demo image** pick a small, permissive, validated GGUF so it starts fast and runs on free hardware:

| Model | Q4_K_M size | License | Notes |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | ~0.4 GB | Apache-2.0 | smallest sane default; runs on ~1 GB RAM |
| Qwen3-0.6B | ~0.5 GB | Apache-2.0 | newest, validated by us, ~60–70 tok/s |
| Qwen2.5-1.5B-Instruct | ~1.1 GB | Apache-2.0 | noticeably smarter, still light |
| Gemma-2-2B-it | ~1.7 GB | Gemma | best quality here; ~2.5 GB RAM |

> **GitHub Releases as model host:** each release asset can be up to **2 GB**, so any of the above fits as a raw
> `.gguf`. Do **not** bother 7-zipping it — quantized GGUF is high-entropy: measured ultra (`-mx=9`) compression is
> only **~1.2 %** on `Q4_K_M` and **~16 %** on `Q8_0`. To ship a model larger than 2 GB, split it into multiple
> release assets and concatenate on download — compression won't get you under the limit.

## Free / cheap hosting

Native-AOT shrinks the **binary**, not the model's **RAM**. A 0.5B-Q4 model needs ~1 GB RAM, a 3B needs ~4 GB —
that, not image size, is the real hosting constraint.

| Host | Free allowance | Verdict |
|---|---|---|
| **Hugging Face Spaces** (Docker SDK) | 2 vCPU + **16 GB RAM** | ✅ Best fit, on-theme (HF hosts the models). Runs up to ~3–7B. Sleeps when idle; public; slow CPU. |
| **Oracle Cloud Always-Free** (Ampere ARM) | up to 4 OCPU + **24 GB RAM** | ✅ Most capable free box. Needs a VM + `linux/arm64` image (see the arm64 note in the workflow). |
| **Google Cloud Run** | scale-to-zero, up to 8 GB | 🟡 AOT cold-start shines, but you pay past the free vCPU-second / egress quota. |
| **Fly.io / Railway / Render** free tiers | 256–512 MB RAM | ❌ Too small for a real LLM (only a ~135 M toy). |
| **GHCR / Docker Hub** | image registry | ✅ Free for *publishing* the image (not running). |

### Hugging Face Spaces (download-on-boot)

Spaces has no volume mount, so have the container fetch the model on first boot. The smallest robust way is a
derived image whose entrypoint pulls then serves — e.g. a `Dockerfile` in your Space:

```dockerfile
FROM <your-dockerhub-user>/overfit:latest
# Spaces sets $HOME to a writable dir; pull into it, then serve from there.
# `serve`'s model is positional, so it comes right after the sub-command.
ENTRYPOINT ["/app/overfit"]
CMD ["serve", "/data/model.gguf", "--host", "0.0.0.0", "--port", "7860"]
```

(Pre-populate `/data` with a one-off `overfit pull`, or bake a tiny model in a build step — your call on the
size/licence trade-off.) Spaces expects the app on port **7860**.

## Publishing to Docker Hub (CI)

[`.github/workflows/docker-publish.yml`](../.github/workflows/docker-publish.yml) builds and pushes on a `v*.*.*`
tag or a manual run. The **image version is read from `Directory.Build.props`** (the same `<Version>` the NuGet
packages + global tool ship under), so the image tag is always consistent with the source version — each run
publishes `:<version>` + `:latest` + `:sha-<short>`. One-time setup:

1. Docker Hub → Account Settings → Security → create an **Access Token** (Read/Write).
2. GitHub repo → Settings → Secrets and variables → Actions → add `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`.
3. `./bump-version.ps1` to set the version, then `git tag v$(version) && git push --tags` → publishes e.g.
   `<user>/overfit:10.0.22`, `:latest`, and `:sha-abc1234`.

The image is **amd64** by default. For arm64 (Oracle Ampere), add a native `ubuntu-24.04-arm` matrix job and a
`docker manifest` merge — AOT cross-compilation from amd64 needs the arm64 clang toolchain, so a native arm runner
is the clean path.
