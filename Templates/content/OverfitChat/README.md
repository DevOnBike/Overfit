# OverfitChat — a local LLM chat app in .NET

A Minimal API streaming-chat app that runs a real language model **in-process** with
[Overfit](https://github.com/DevOnBike/Overfit), exposed as a standard
`Microsoft.Extensions.AI` **`IChatClient`**. No Python, no Ollama, no Docker, no cloud key.

## Run it (60 seconds)

1. Get a small GGUF model (~400 MB):
   ```bash
   curl -L -o model.gguf https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
   ```
   (or set `"ModelPath"` in `appsettings.json` to any `.gguf` you already have.)

2. Run:
   ```bash
   dotnet run
   ```

3. Open the printed URL and chat. Tokens stream from the model running on your CPU.

## What to look at

`Program.cs` is the whole story — two lines wire a local model in as your `IChatClient`:

```csharp
var overfit = OverfitClient.LoadGguf(modelPath, mmap: true);
builder.Services.AddChatClient(overfit.AsChatClient());   // Overfit IS your IChatClient
```

Because it's a standard `IChatClient`, the rest of the Microsoft.Extensions.AI ecosystem —
function calling, response caching, telemetry middleware, Semantic Kernel — works unchanged.
This is the same registration the official .NET AI template uses for Ollama/OpenAI; here the
model just runs **inside this process**, offline.

## Notes

- **Single shared session** for simplicity — fine for a demo/single user. For multi-user
  throughput, load N `OverfitClient`s or use a session pool.
- **Any GGUF** works: Qwen, Llama, Phi, Gemma, Mistral, … A 0.5B–1B Q4_K model is a good balance.
- Overfit is dual-licensed **AGPL-3.0-or-later / commercial**.
