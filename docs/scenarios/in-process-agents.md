# Overfit for In-Process Agents

**For .NET engineers building agentic features — RAG, tool calling, structured output — who don't want a Python service, a model server, or data leaving the process.**

---

## The Problem

You want an LLM feature inside a .NET app: answer from your own documents (RAG), call your own functions (tool calling), or return a typed object the rest of your code can consume (structured output).

The usual stacks all add moving parts:

- **A hosted API** (OpenAI / Anthropic) — best quality, but your data leaves the boundary, and you depend on a network and a vendor.
- **A local model server** (Ollama / llama.cpp server / vLLM) — a separate process to deploy, monitor, and talk to over HTTP, plus a vector DB next to it.
- **"Prompt and pray" JSON** — you ask the model for JSON in the prompt and hope. Small models drift: prose before the brace, a missing quote, a truncated object. You write retry-and-repair code.

**Overfit collapses all of that into your process.** One NuGet package, pure managed C#, no native binary, no Python, no server, no network. The model, the embeddings, the vector store, and the grammar enforcement all live inside your app — and structured output is *guaranteed*, because it's enforced at the token level, not requested in a prompt.

---

## The whole loop, in one process

The runnable [`Demo/AgentDemo`](../../Demo/AgentDemo/README.md) does exactly this end to end on a single GGUF model. The pieces:

### 1. Load (memory-mapped — tiny managed footprint)

```csharp
using DevOnBike.Overfit.LanguageModels.Runtime;

using var engine = CachedLlamaInferenceEngine.LoadGguf(@"C:\models\qwen.q4km.gguf");  // mmap default
using var session = engine.CreateSession(1024);
```

Weights are memory-mapped: a 3B `Q4_K_M` model loads with a **~220 MB live managed heap** — the weight pages are file-backed (shared/clean), not committed private RAM. No Python, no native dependency.

### 2. RAG — embeddings + a built-in vector store

```csharp
using DevOnBike.Overfit.LanguageModels.Retrieval;

var store = new VectorStore(session.EmbeddingDimension);
foreach (var (id, text) in documents)
{
    store.Add(id, session.Embed(tokenizer.Encode(text)), payload: text);   // L2-normalised
}

var hits = store.Search(session.Embed(tokenizer.Encode(query)), topK: 5);   // cosine top-K
var context = hits[0].Payload;
```

The same model produces the embeddings; the `VectorStore` is an in-process cosine index (no external vector DB). Retrieval is a function call, not a network round-trip.

### 3. Tool calling — a guaranteed-valid call dispatched to C#

```csharp
using DevOnBike.Overfit.LanguageModels.Tools;
using DevOnBike.Overfit.LanguageModels.Chat;

var tools = new[]
{
    new ToolDefinition("get_weather", "Get the current weather for a city."),
    new ToolDefinition("get_time", "Get the current time in a timezone."),
};
var dispatch = new Dictionary<string, Func<JsonElement, string>>
{
    ["get_weather"] = args => Weather.Lookup(args.GetProperty("city").GetString()),
    ["get_time"]    = args => Clock.Now(args),
};

var chat = new ChatSession(session, tokenizer, template);
var reply = chat.Send("What is the weather in Paris?", in options,
                      constraint: new ToolCallConstraint(tools, tokenizer));

if (ToolCall.TryParse(reply, out var call))           // always parses — enforced at decode
{
    using var args = JsonDocument.Parse(call.Arguments);
    var result = dispatch[call.Name](args.RootElement); // dispatch to your C# function
}
```

Constrained decoding forces the canonical envelope `{"name": "<one of your tools>", "arguments": <json>}`. The tool name is **always** one you registered; the arguments **always** parse. The model only supplies the choice and the values — never the structure.

### 4. Structured output — guaranteed well-formed JSON

```csharp
using DevOnBike.Overfit.LanguageModels.Constraints;

var reply = chat.Send("Return a person object with name and age.", in options,
                      constraint: new JsonGrammarConstraint(tokenizer));

using var doc = JsonDocument.Parse(reply);            // never throws — JSON is enforced
```

---

## Why "guaranteed" is the point

Most structured-output stacks *ask* the model for JSON and then validate/repair. Overfit masks the logits before every token: any token that would break the grammar is set to `-∞`, so the model **physically cannot** sample it.

- No "Here is the JSON:" preamble. No trailing prose. No missing brace.
- No retry loop, no repair pass, no second model call to fix the first.
- A small model (3B, even quantized) becomes reliable for structured tasks, because the structure can't be violated — only the content is up to the model.

This is the same idea as llama.cpp's GBNF grammars, implemented in pure managed C# and wired straight into the sampler (`ITokenConstraint`).

---

## What you get

- **One deployment, one runtime.** Your .NET app. No sidecar, no model server, no vector DB, no Python environment.
- **Data never leaves the process.** No network hop during inference — by construction. (See [regulated industries](regulated-industries.md).)
- **AOT-compatible, no native binary.** `dotnet publish /p:PublishAot=true` gives a self-contained executable; nothing to P/Invoke.
- **Small footprint.** Memory-mapped weights keep the managed heap in the low hundreds of MB even for a 3B model.
- **Bring your own model.** Any Qwen / Llama / Mistral GGUF (incl. `Q4_K_M` straight from Ollama) — loaded directly, no conversion step.

---

## When Overfit is the wrong choice

- **You need frontier-model quality.** A 3B local model won't match GPT-4-class output; constrained decoding fixes *structure*, not *reasoning*. Use a hosted API when quality dominates and data egress is acceptable.
- **You need maximum decode throughput on one box.** llama.cpp/LLamaSharp decode ~1.5–2× faster (hand-tuned native AVX-512/VNNI). Overfit's axis is in-process, zero-dependency, allocation-controlled execution — not raw matmul speed.
- **You need a typed argument schema enforced at decode.** Today tool arguments are guaranteed-valid JSON and your handler validates types; per-tool JSON-Schema enforcement is on the [roadmap](../../ROADMAP.md).

---

## Further reading

- [`Demo/AgentDemo`](../../Demo/AgentDemo/README.md) — this whole loop, runnable in one command.
- [Main README](../../README.md) — project overview and benchmarks.
- [Regulated industries](regulated-industries.md) — when data egress is not an option.
- [ASP.NET Core](aspnet-microservice.md) — serving inference from a .NET service.
- [TECHNICAL.md](../TECHNICAL.md) — full API quick-start and architecture.
