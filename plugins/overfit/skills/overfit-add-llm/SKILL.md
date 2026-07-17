---
name: overfit-add-llm
description: Add local, in-process LLM inference to an existing .NET project with Overfit — no Python, no Ollama, no cloud. Use when the user wants to run a private/local LLM inside their .NET app, load a GGUF model in C#, add chat/RAG/embeddings on the CPU, replace an OpenAI/Ollama/Azure call with an on-device model, or expose a local model as a Microsoft.Extensions.AI IChatClient. For a brand-new app, prefer the `dotnet new overfit-chat` template instead.
---

# Add a local LLM to a .NET app with Overfit

Wires [Overfit](https://github.com/DevOnBike/Overfit) — a pure-C#/.NET, in-process LLM runtime — into an
existing project so it loads a GGUF model and runs on the CPU. No Python, no model server, no data egress.

**New project?** Use the template instead: `dotnet new install DevOnBike.Overfit.Templates` →
`dotnet new overfit-chat`. This skill is for adding Overfit to an **existing** codebase.

## Instructions

1. **Identify the target.** Find the `.csproj` to add inference to and classify it: **web / DI host**
   (ASP.NET, Minimal API — has `WebApplication.CreateBuilder`) vs **console / library** (direct calls).
   Overfit targets **`net10.0`**; if the project is older, say so and confirm before bumping the TFM.

2. **Add the packages.**
   ```bash
   dotnet add package DevOnBike.Overfit                 # the runtime
   dotnet add package DevOnBike.Overfit.Extensions.AI   # only if you want IChatClient / DI / Semantic Kernel
   ```
   Add `Microsoft.Extensions.AI` too if the app will call `builder.Services.AddChatClient(...)`.

3. **Get a model (GGUF).** Check for an existing one first: a `*.gguf` in the repo, an `OVERFIT_MODEL_DIR`
   env var, or ask the user for a path. If there is none, download a small one to test with:
   ```bash
   curl -L -o model.gguf https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
   ```
   Then **gitignore it** — add `*.gguf` to `.gitignore` (models are large; never commit them). Size guidance:
   a 0.5B–1B Q4_K is a good default; only go to 3B+ if the box has the RAM (mmap keeps the managed heap tiny,
   but the OS still pages the weights).

4. **Wire it.** Pick the path that matches the project:

   - **Web / DI — as a standard `IChatClient`** (drops into the .NET AI template & Semantic Kernel):
     ```csharp
     using DevOnBike.Overfit.LanguageModels;
     using DevOnBike.Overfit.Extensions.AI;
     using Microsoft.Extensions.AI;

     var overfit = OverfitClient.LoadGguf(builder.Configuration["ModelPath"] ?? "model.gguf", mmap: true);
     builder.Services.AddSingleton(overfit);
     builder.Services.AddChatClient(overfit.AsChatClient());   // inject IChatClient anywhere
     ```
     Stream in an endpoint via `IChatClient.GetStreamingResponseAsync(prompt)` (`update.Text` per token).

   - **Console / library — direct:**
     ```csharp
     using DevOnBike.Overfit.LanguageModels;

     using var client = OverfitClient.LoadGguf("model.gguf", mmap: true);
     Console.WriteLine(client.Send("Say hello in one sentence."));
     ```

5. **Verify end-to-end.** `dotnet build`, then actually run it and send one prompt — confirm tokens come
   back (don't stop at "it compiles"). For a web app, hit the endpoint; for a console app, run it.

6. **Report the wiring + the model path**, and point at the follow-ons the user is likely to want next
   (below). Leave the tree clean/staged — do not commit (that's the user's call).

## Follow-ons (mention, don't build unless asked)

- **RAG:** `SentenceEmbedder.ForMiniLm(dir).AsEmbeddingGenerator()` → a standard `IEmbeddingGenerator`;
  pair with Overfit's in-process `VectorStore`.
- **OpenAI-compatible server (no code):** `dotnet tool install -g DevOnBike.Overfit.Cli` then
  `overfit serve model.gguf` → point any OpenAI client at `http://localhost:8080/v1`.
- **Streaming, tool calling, guaranteed JSON, samplers (top-nσ / DRY):** via `ChatSession` / `SamplingOptions`.

## Rules

- **Never commit model files** (`*.gguf`). Add them to `.gitignore`.
- **Prefer `mmap: true`** for GGUF — low, reclaimable working set instead of an F32 blow-up.
- **License:** Overfit is **AGPL-3.0-or-later / commercial** — flag this if the target is a closed-source or
  distributed product (they may need the commercial license: `devonbike@gmail.com`).
- **Don't fabricate the model path** — use an existing model, an env var, or download one; ask if unsure.
- Do the git work as the user's action: stop at a built, verified, staged state and report.
