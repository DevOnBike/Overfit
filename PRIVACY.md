# Privacy Policy — Overfit

_Last updated: 2026-07-20_

Overfit is a pure-C#/.NET library and command-line tool that runs large language models
**inside your own process, on your own hardware**. Its entire reason to exist is that your
data never has to leave the machine. This policy states plainly what that means, and it is
verifiable — the source is open (AGPL-3.0-or-later) and there is nothing here you cannot
confirm by reading it.

## The short version

- **We collect nothing.** Overfit has no analytics, no telemetry endpoint, no usage
  reporting, no licence-activation call, no "phone home" of any kind.
- **Your prompts, model weights, documents and outputs stay on your machine.** Inference,
  embeddings, RAG retrieval and evaluation all run locally, in-process, on the CPU.
- **The only network traffic is traffic you explicitly ask for** (see below), and it never
  goes to us.

## What runs locally

All of the core functionality — loading a GGUF / safetensors / ONNX model, chat, streaming
generation, embeddings, vector search, the skill/prompt eval harness — executes entirely
in your process. No Python runtime, no model server, no sidecar, no external API. Nothing
you feed the model, and nothing the model produces, is transmitted anywhere.

## The only network calls Overfit makes

Overfit opens a network connection only when **you** initiate one of these actions, and in
every case the destination is a third party you have chosen — never DevOnBike:

- **Downloading a model** (`overfit pull …`) fetches the model file from the model host you
  name (e.g. Hugging Face). Standard file download; no data of yours is sent beyond the
  request for that file, plus an `HF_TOKEN` if you set one to access a gated repo.
- **Running the built-in server** (`overfit serve`, the OpenAI-compatible endpoint, the
  Redaction Gateway) opens a listening socket that **you** start and control. It serves
  requests on your network to clients you point at it; it does not originate outbound calls.
- **Optional metric sources** (the Prometheus integration) connect to a Prometheus endpoint
  **you** configure. Off unless you wire it up.

If you never pull a model over the network (e.g. you copy the file in yourself) and never
start a server, Overfit makes no network connections at all.

## Telemetry / observability

Overfit exposes standard .NET `Meter` and `ActivitySource` instrumentation (counters,
histograms, traces) under the meter name `DevOnBike.Overfit`, so that **you** can observe
your own workloads. This instrumentation has **no exporter of its own** — the data goes
nowhere unless you attach an OpenTelemetry (or similar) exporter and choose a destination.
Out of the box it is inert beyond your own process.

## The plugin / skills

The Overfit plugin (skills `overfit-add-llm` and `overfit-skill-eval`) generates code and
configuration in your project and runs local commands (`dotnet …`) that you review. The
skills add Overfit to your codebase and drive it locally; they transmit nothing to
DevOnBike.

## Data we hold about you

None. Because Overfit collects nothing and reports nothing, DevOnBike holds no data about
your use of it. There is no account, no usage record, and nothing to request, export, or
delete.

## Changes to this policy

If this policy changes, the updated version will be published in this file in the
repository, with the date above updated.

## Contact

Questions about privacy, licensing, or a commercial license:
**devonbike@gmail.com** — see [`COMMERCIAL.md`](COMMERCIAL.md) and [`LICENSE.md`](LICENSE.md).
