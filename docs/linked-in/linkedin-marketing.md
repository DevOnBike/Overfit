# Pure C#. Real LLMs. Zero Python.

Here is a belief worth challenging: **"If you want AI, you need Python."**

We just spent several release cycles proving it wrong.

Meet **Overfit** — a deep-learning and inference engine written in **100% C#**. No Python runtime. No native binaries. No ONNX Runtime. Nothing but managed .NET code.

And it does not just run toy models. Here is what it does **today**:

🧠 **Runs real language models.** GPT-2, Qwen, Llama-family — loaded and generating text in pure C#.

⚡ **Zero bytes allocated per token.** Not "low allocation." *Zero.* On every generated token — measured on every single test run.

🎯 **Fine-tunes models to your data.** LoRA fine-tuning means you can train an AI model on what *your* business knows — in-house, with data that never leaves your servers.

🚀 **It got fast.** A recent performance sprint cut a core training workload by **72%** — and turned a 1–2 minute real-world training run into roughly **2 seconds**.

📦 **Reads the formats you already have.** Ollama and HuggingFace `.gguf` models — including quantized ones — load directly. No conversion tooling required.

✅ **And it is honest about it.** Outputs verified against PyTorch: top-10 prediction overlap, 10 out of 10. We benchmark in public — caveats included.

The result? AI you can build, run, and *own* — on the .NET stack you already have, on hardware you already own, with data that never leaves the building.

No Python tax. No GPU requirement. No vendor lock-in.

It is open source. Go look:

👉 **DevOnBike/Overfit** on GitHub.

What would you build if AI were just another C# library?

#AI #LLM #dotnet #csharp #MachineLearning #OpenSource #Innovation #TechThatShips

---

<!-- ─────────────────────────────────────────────────────────────────────────
  NEW DRAFT (2026-05-25) — agentic angle, punchy. Raw material: edit to your own
  voice before posting.
  ───────────────────────────────────────────────────────────────────────── -->

# A whole AI agent. One .NET process. Zero Python.

Last time: real LLMs in pure C#. This time: a real **agent** — and still no Python, no server, no
network, no data leaving your process.

One `dotnet run` now does the full loop on a single model file:

📚 **RAG** — embed your documents and search them with a built-in in-process vector store. No
external vector database.

🛠 **Tool calling** — the model calls *your* C# functions. The call is **guaranteed valid by
construction** — the tool name is always one you registered, the arguments always parse. No
"what if the model returns garbage" code.

🧱 **Structured output** — ask for JSON, get JSON. Every time. Not "usually." It's enforced at the
token level — an invalid token literally can't be sampled, so there's no retry-and-repair.

🪶 **Tiny footprint** — a 3B model loads with a ~220 MB .NET heap; the weights are memory-mapped.

The trick behind the guarantees: constrained decoding. Before every token, Overfit masks out
anything that would break the grammar. The model doesn't *try* to produce valid output — it
*can't* produce invalid output.

No model server. No vector DB. No API key. No egress. Just a C# library inside your app.

It's open source:

👉 **DevOnBike/Overfit** on GitHub.

If "AI agent" in your stack means five extra services — what would you cut if it were one C# library?

#AI #Agents #RAG #LLM #dotnet #csharp #OpenSource #TechThatShips
