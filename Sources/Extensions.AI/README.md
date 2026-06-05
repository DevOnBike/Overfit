# DevOnBike.Overfit.Extensions.AI

[Microsoft.Extensions.AI](https://learn.microsoft.com/dotnet/ai/microsoft-extensions-ai) adapter for
[Overfit](https://github.com/DevOnBike/Overfit) — exposes the pure-.NET, in-process Overfit LLM runtime as a
standard `IChatClient` and `IEmbeddingGenerator`, so it drops into Semantic Kernel and any
`Microsoft.Extensions.AI` pipeline (caching, telemetry, function-invocation middleware, DI) by changing one line.

No Python, no model server, no native binary, no data leaving the process.

## Chat

```csharp
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.Extensions.AI;
using Microsoft.Extensions.AI;

using var overfit = OverfitClient.LoadGguf("qwen.q4km.gguf");
IChatClient chat = overfit.AsChatClient();

// Non-streaming
Console.WriteLine(await chat.GetResponseAsync("What is the capital of France?"));

// Streaming
await foreach (var update in chat.GetStreamingResponseAsync("Count to five."))
{
    Console.Write(update);
}
```

`ChatOptions.Temperature` (0 → greedy), `MaxOutputTokens`, and `TopP` are honoured; usage and a
`length`/`stop` finish reason are reported. Calls are stateless (the full message list is replayed) and
serialized through a single-flight gate — the wrapped session is single-tenant.

## Embeddings

```csharp
using var embedder = SentenceEmbedder.ForMiniLm(@"C:\minilm");
IEmbeddingGenerator<string, Embedding<float>> gen = embedder.AsEmbeddingGenerator(dimensions: 384);

var vectors = await gen.GenerateAsync(["hello world", "second text"]);
```

## DI / pipelines

```csharp
builder.Services.AddChatClient(overfit.AsChatClient())
    .UseDistributedCache()
    .UseOpenTelemetry();
```

Dual-licensed: AGPL-3.0-or-later for open source, commercial license available — see `COMMERCIAL.md`.
