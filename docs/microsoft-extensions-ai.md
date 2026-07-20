# Use Overfit as your `IChatClient` (swap Ollama, one line, no Docker)

Microsoft's [`Microsoft.Extensions.AI`](https://learn.microsoft.com/dotnet/ai/microsoft-extensions-ai)
is now the standard abstraction for AI in .NET — `IChatClient` and `IEmbeddingGenerator` are the
interfaces every provider (OpenAI, Azure, Ollama, …) and Semantic Kernel speak. Overfit ships a
first-class adapter, so a **local GGUF model running in your own process** is a drop-in for any of them.

```bash
dotnet add package DevOnBike.Overfit
dotnet add package DevOnBike.Overfit.Extensions.AI
```

## The one line

```csharp
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.Extensions.AI;
using Microsoft.Extensions.AI;

var overfit = OverfitClient.LoadGguf("model.gguf", mmap: true);
IChatClient chat = overfit.AsChatClient();

Console.WriteLine(await chat.GetResponseAsync("What is the capital of France?"));
```

That `IChatClient` is the real thing — everything downstream in the Microsoft.Extensions.AI ecosystem
(function invocation, response caching, OpenTelemetry, Semantic Kernel) works unchanged. The only
difference from a cloud client: **the model runs in this process, on the CPU — no Python, no model
server, no Docker, no cloud key, no data egress.**

## Swap it into the official .NET AI template

The [.NET AI Chat template](https://learn.microsoft.com/dotnet/ai/quickstarts/ai-templates)
registers a local model via an Ollama container:

```csharp
// .NET AI template default — needs a running Ollama + Docker:
builder.Services.AddChatClient(
    new OllamaApiClient(new Uri("http://localhost:11434"), "llama3.2"));
```

Replace that one registration with Overfit — no container, no sidecar, no separate process:

```csharp
var overfit = OverfitClient.LoadGguf("model.gguf", mmap: true);
builder.Services.AddChatClient(overfit.AsChatClient());
```

Everything else in the template — the RAG pipeline, the chat UI, the DI — is untouched.

> **Even faster:** `dotnet new overfit-chat` scaffolds a complete, ready-to-run version of this
> (see [`Templates/`](../Templates/) / `DevOnBike.Overfit.Templates`).

## Streaming

```csharp
await foreach (var update in chat.GetStreamingResponseAsync("Write a haiku about .NET"))
{
    Console.Write(update.Text);
}
```

## Microsoft Agent Framework — a local model behind Microsoft's agent SDK

[Microsoft Agent Framework](https://learn.microsoft.com/agent-framework/overview/) (`Microsoft.Agents.AI`) is the
successor that merged **Semantic Kernel and AutoGen** into one SDK. It builds agents on top of any
`IChatClient` — so Overfit is a model provider for it with **no adapter code**:

```csharp
IChatClient chat = overfit.AsChatClient();
AIAgent agent = new ChatClientAgent(chat, instructions: "You are a concise assistant.");

var response = await agent.RunAsync("What is the capital of France?");
```

That's the whole integration: **a Microsoft Agent Framework agent whose model runs in your process, on the CPU,
offline.** No Ollama, no Docker, no cloud key.

> **Verified, not asserted.** [`Demo/AgentFrameworkDemo`](../Demo/AgentFrameworkDemo) runs exactly this against
> `Microsoft.Agents.AI` **1.13.0** on a Qwen2.5-0.5B GGUF, and is built by CI so the claim can't rot:
> `dotnet run --project Demo/AgentFrameworkDemo -- model.gguf`
>
> Scope of the check: agent construction + `RunAsync`. Tool calling and multi-agent workflows route through the
> same `IChatClient` and should follow, but they have not been measured here — so they are not claimed.

## Semantic Kernel

Semantic Kernel also consumes `IChatClient`, so Overfit plugs straight in — but note SK is now **superseded by
Agent Framework** (above), which is where new work should go:

```csharp
var kernelBuilder = Kernel.CreateBuilder();
kernelBuilder.Services.AddSingleton<IChatClient>(overfit.AsChatClient());
var kernel = kernelBuilder.Build();
```

## Embeddings for RAG

The same adapter exposes Overfit's bit-parity BERT embeddings (MiniLM / BGE / E5) as a standard
`IEmbeddingGenerator` — so any Microsoft.Extensions.AI vector-store / RAG pipeline can use them:

```csharp
using var embedder = SentenceEmbedder.ForMiniLm(@"C:\minilm");
IEmbeddingGenerator<string, Embedding<float>> gen = embedder.AsEmbeddingGenerator();

var v = (await gen.GenerateAsync(["hello world"])).First().Vector;
```

## Why this matters

You get the entire Microsoft.Extensions.AI / Semantic Kernel ecosystem — the tooling .NET teams are
standardizing on in 2026 — while keeping inference **in-process, offline, Native-AOT-friendly, and free
of any Python or native runtime**. It's the on-device / air-gapped counterpart to Foundry Local and
Ollama, expressed through the exact same interface your code already targets.
