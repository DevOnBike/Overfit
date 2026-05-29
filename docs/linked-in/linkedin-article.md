<!-- Two versions below: (1) a short LinkedIn FEED POST, (2) the full ARTICLE. Use one. -->

## ▶ Short version — LinkedIn feed post

> **Pure C#. Real agents. Zero Python.**
>
> Every "let's add AI" ends with a new vendor, a new server, and your data leaving the building. It doesn't have to.
>
> Overfit is an AI engine written entirely in C# — no Python, no native binary, no model server. Load **GPT-2 / Qwen / Llama / Mistral** straight from Ollama or HuggingFace (GGUF, incl. `Q4_K_M`) and run real agentic features **in-process**:
>
> 🔎 **RAG** — answer from your own docs (built-in vector store, no external DB)
> 🛠 **Tool calling** — the model calls your C# functions, valid by construction
> 🧱 **Structured output** — guaranteed well-formed JSON, every time
>
> It all runs on the CPU servers you already have — **zero bytes allocated per token**, data that never leaves your process, Native-AOT single binary.
>
> Honest: a small local model won't out-think a frontier cloud model. But for these jobs, **privacy + predictable cost + ownership** win.
>
> Open source 👉 **DevOnBike/Overfit**
>
> What would you build if a real AI agent were just another C# library?
>
> #dotnet #AI #LLM #RAG #OpenSource

---

## ▶ Full version — LinkedIn article

<!-- Pick one title below, delete the rest. -->

# Pure C#. Real Agents. Zero Python.

<!-- Alt titles (pick one):
  • "Your AI Agent Can Live Inside Your App — Not in Someone Else's Cloud"
  • "Ship AI Agents in .NET — No Python, No Server, No Data Leaving the Building"
  • "The AI Agent With No Second Stack"
-->

Every "let's add AI" conversation tends to end the same way: a new vendor billed per request, a new
service to run, a new place your customers' data has to travel to, and a new technology stack your
team has to learn. The feature is small. The footprint never is.

It doesn't have to be that way. **Overfit** is an AI engine written entirely in C# — the same
language, runtime, and tooling a .NET team already uses every day. No Python. No native binary. No
model server. And now it does more than *run* language models — it lets you build **agentic
features** that live inside your own application, with data that never leaves it.

Here's what that unlocks — first in plain terms, then briefly under the hood.

---

## Start with the models you already know

Overfit loads popular open models **directly — no conversion step, no Python tooling**:

- **GPT-2**, **Qwen 2.5**, **Llama 3.x**, and **Mistral** — from GGUF files (including quantized
  `Q4_K_M` / `Q6_K` / `Q8_0`) pulled **straight from Ollama or HuggingFace**.
- **PyTorch-exported networks** via ONNX import (CNNs, ResNet-style architectures).

Then you query them **in-process** — prompt in, text out, token-by-token streaming. Inference is a
function call inside your app, not a network hop to a server.

And because it's **pure C#, zero-allocation, CPU-only**, that simple fact unlocks places most AI
stacks can't go:

- **No GPU, no Python on the box** → it runs on the commodity CPU servers you already operate, and
  scales down to edge and embedded devices.
- **Zero bytes allocated per token** → no garbage-collection pauses, no tail-latency spikes — so it
  fits latency-sensitive paths (fraud scoring, trading, real-time decisioning) and high-throughput
  services that can't afford GC jitter.
- **Pure managed + Native-AOT** → ships as a single self-contained binary: smaller containers,
  faster cold starts, and a clean bill of health for supply-chain and security audits (no native
  library to vet).
- **In-process** → embed it in an ASP.NET service, a desktop app, a background worker, even a game —
  anywhere .NET already runs.

That foundation — known models, loaded and queried in managed code — is what the agentic features
below are built on.

---

## What you can build

### An assistant that actually knows your product
Your app finds the relevant passages in *your own* documentation and answers from them — retrieval
done entirely in-process, with a built-in search index that lives in your application's memory. Your
documents are never uploaded to a third-party service.
**Outcome:** grounded answers, fewer escalations, and not one internal page sent outside your boundary.

### An assistant that *does* things — safely
Useful assistants don't just talk; they act: look up an order, check stock, open a ticket. The risk
everyone hits is reliability — the model "decides" to call a function and returns something
malformed, and suddenly you're writing defensive code for every way the AI could misbehave.
Overfit removes that failure mode at the source: when the model requests one of your functions, the
request is **valid by construction** — it can only ever name a function you registered, with
correctly-formed arguments. Your code just runs it.
**Outcome:** real automation, without a brittle "what if the AI returns garbage" layer.

### Reliable structured data out of messy text
A lot of value is simply turning free text into structured data your existing systems can use — a
form, a record, a typed result. Normally the model returns *mostly* clean data, so you ship
validation, retries, and the occasional 2 a.m. incident. With Overfit the output is **guaranteed**
well-formed every time — not "usually, if the prompt is good."
**Outcome:** AI output you can feed straight into existing workflows.

### And beyond inference: adapt and own the model
Overfit also **fine-tunes** models to your own data in-house (no data leaves), and can **learn what
'normal' looks like** for your systems to flag anomalies — adapting per deployment. These are things
a service that merely *runs* someone else's model can't do.

---

## Why this is different

Every scenario above runs as part of your .NET application, on ordinary CPU servers you already
operate — no GPU required:

- **Your data stays in-house.** Nothing is sent to an external AI provider during use.
- **Predictable cost.** No per-request meter that grows with adoption.
- **Your team, your stack.** The engineers you have, in the language they already use.
- **A smaller risk surface.** No Python runtime, no extra service — less to secure, patch, and audit.

An honest note: a small in-house model won't out-think a frontier cloud model on the hardest
reasoning. But for these scenarios — answering from your content, triggering your actions, producing
reliable structured output — **privacy, predictable cost, and ownership** matter more than the last
few points of raw intelligence. That's the trade a growing number of regulated, on-prem, and
cost-conscious teams want to make — and a .NET organisation can now make it without rebuilding
anything.

---

## Briefly, under the hood

For the engineers: it's pure managed C# on .NET 10 — Native-AOT compatible, with an inference hot
path that allocates **zero bytes per generated token**.

- **Bring your own model.** Any Qwen / Llama / Mistral GGUF (including `Q4_K_M` straight from Ollama)
  loads directly — no conversion step.
- **Small footprint.** Weights are memory-mapped, so a 3-billion-parameter model loads with a
  **~220 MB managed heap** — the model pages are shared and file-backed, not committed private RAM.
- **Guaranteed structure.** The "valid by construction" guarantee isn't a prompt trick: before every
  token, the engine masks out anything that would break the required format (valid JSON, a real tool
  call), so an invalid token literally can't be produced — no retry, no repair.
- **In-process RAG.** Embeddings plus a built-in vector store — no external vector database.

The whole loop — load a model → answer from your docs → call a C# function → return structured data
— runs in a single process and ships as one NuGet package. Honest positioning: on raw decode speed,
hand-tuned native engines like llama.cpp are still faster; Overfit's axis is in-process,
zero-dependency, allocation-controlled execution where your data, cost, and stack stay yours.

---

It's open source (AGPLv3), with commercial licensing available for closed-source products.

👉 **DevOnBike/Overfit** on GitHub.

*What would you build if a real AI agent were just another C# library?*

#AI #Agents #RAG #LLM #dotnet #csharp #DataPrivacy #OpenSource #TechStrategy
