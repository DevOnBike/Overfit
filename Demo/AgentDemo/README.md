# Agent demo — the whole in-process stack in one run

One console walkthrough of **agentic LLM work entirely inside a .NET process** — no Python,
no native binary, no model server, no network. It loads a single Qwen2.5 GGUF and runs, in
sequence:

1. **Memory-mapped load** — weights are file-mapped, so the live managed heap stays tiny
   (the model pages are shared/clean, not committed private RAM).
2. **In-process RAG** — embed documents and a query with the same model, rank by cosine.
3. **Tool calling** — constrained decoding forces a valid `{"name": …, "arguments": …}`
   call, which is parsed and **dispatched to a C# delegate**.
4. **Structured output** — constrained decoding guarantees the reply parses as JSON.

Steps 3–4 can't emit malformed output: the grammar is enforced at the logit level, so the
model *physically cannot* pick an invalid token — no prompt-engineering, no retry/repair.

## Run it

```powershell
# Point at a Qwen2.5 GGUF directory (the .gguf + tokenizer.json). Defaults to C:\qwen3b.
$env:OVERFIT_MODEL_DIR = "C:\qwen3b"
dotnet run -c Release --project Demo/AgentDemo
```

Get a model the usual way (e.g. `ollama pull qwen2.5:3b-instruct-q4_K_M`, or a HuggingFace
`*-q4_k_m.gguf`) and drop it in that directory alongside `tokenizer.json`.

## What it looks like (Qwen2.5-3B Q4_K_M)

```
Loaded in 674 ms.
Live managed heap with the model loaded: 222 MB (weights are file-mapped …)

1. In-process RAG (embeddings + cosine search)
Query: "Which landmark is in France?"
  cos=0.861  The Eiffel Tower is a wrought-iron lattice tower in Paris, France.
  cos=0.791  Photosynthesis converts light energy into chemical energy in plants.
  cos=0.779  The TCP three-way handshake establishes a reliable network connection.
  cos=0.853  Mount Everest is the highest mountain above sea level, in the Himalayas.
Top match → The Eiffel Tower …

2. Tool calling (guaranteed-valid call → dispatch to C#)
User:        What is the weather in Paris?
Model emits: {"name": "get_weather", "arguments": {"city":"Paris"}}
Dispatch:    get_weather(...) → 18°C, light rain  (args: {"city":"Paris"})

3. Structured output (guaranteed well-formed JSON)
Model emits: [ { "name": "Alice", "age": 3 } ]
Parsed OK → root is Array.
```

(A 3B model's *content* is its own business — the value Overfit guarantees is the
**structure**: the tool name is always a registered tool and the arguments / JSON always parse.)

## The APIs it uses

| Step | Type |
|------|------|
| load | `CachedLlamaInferenceEngine.LoadGguf(path)` (mmap default) |
| RAG | `CachedLlamaSession.Embed(tokens, pooling, normalize)` |
| tool calling | `ToolCallConstraint` + `ToolCall.TryParse` → your `Func<JsonElement,string>` |
| JSON mode | `JsonGrammarConstraint` passed to `ChatSession.Send(…, constraint)` |

All of it is `DevOnBike.Overfit` — one NuGet package, AOT-compatible, zero native dependencies.
