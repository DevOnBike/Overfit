# Overfit — launch copy (Phase 2)

Post all of these in one coordinated window. Every link points back to the blog post.
After posting: respond to **every** comment, fast — for a new project, that response speed is the marketing.

URLs:
- Blog (build log): `https://overfit-ml.com/zero-allocation-llm-inference-csharp.html`
- Repo: `https://github.com/DevOnBike/Overfit`
- Site: `https://overfit-ml.com`

---

## 1. Hacker News — Show HN

**Submit as a link post.**
**URL to submit:** `https://github.com/DevOnBike/Overfit`
**Title:**

```
Show HN: Overfit – LLM inference in pure C#, 0 bytes allocated per token
```

**First comment** (post immediately after submitting):

> I've been building Overfit — a deep-learning and inference engine written entirely in C#, on .NET 10. No Python, no native binaries, no ONNX Runtime dependency.
>
> The thing I set out to prove: you can run real language models — GPT-2, and the Qwen / Llama / Mistral families — in-process in managed code, with zero bytes allocated per generated token on the KV-cache decode path. There's a CI test that fails the build if decode allocates anything, so the claim stays honest.
>
> The build log goes into the parts that were actually hard — KV-cache decode at 0 B/token, and the bit I'm most pleased with: `Parallel.For` allocates ~3 KB per call, which is incompatible with a zero-allocation contract, so I wrote a bulk-wake work dispatcher (the trick is `SemaphoreSlim.Release(N)` instead of N × `AutoResetEvent.Set`): https://overfit-ml.com/zero-allocation-llm-inference-csharp.html
>
> Honest scope: it's CPU-only, not GPU. PyTorch on CPU is still 2.2–3.6× faster on the raw training GEMM — Intel MKL is decades of hand-tuned assembly I'm not going to out-code. Overfit's axis is predictable, allocation-free, Native-AOT-compatible execution with no native or Python dependency — for .NET shops, regulated / on-prem environments where data can't leave the process, and edge.
>
> It also does LoRA fine-tuning and ONNX import. AGPLv3 + commercial. Happy to answer anything — and genuinely curious whether the zero-allocation obsession reads as worth it or as overkill.

---

## 2. r/dotnet

> **⚠ r/dotnet self-promotion rules — break them and the post gets removed:**
> - **Weekend only** — post Saturday or Sunday.
> - **"Promotion" flair** — set it when posting.
> - **Tied to a release version** (e.g. v1.0). Tag a clean release; frame the post as that release announcement.
> - **Not AI-written** — r/dotnet removes AI-generated posts. **The draft below is raw material — rewrite it in your own voice, do not paste it.** A post that reads as generated gets pulled and dents your name in that community. Tells to kill: "I'm excited to share", balanced em-dash sentences, the generic "What works today:" listicle voice. Write like a message to a colleague: opinionated, specific, slightly rough.

**Title:**

```
Overfit: running and fine-tuning LLMs in pure C# — no Python, no native deps, 0 alloc/token (build log)
```

**Body — raw material; rewrite in your own words (see rules above), do not paste verbatim:**

> I wanted to use language models from .NET without standing up a parallel Python stack — a Python runtime, a model server, a separate deploy pipeline, a separate thing for security to audit. So I built **Overfit**: a deep-learning and inference engine written entirely in C#, targeting .NET 10. No Python, no native binaries, no ONNX Runtime dependency.
>
> What works today:
>
> - GPT-2, and the Qwen / Llama / Mistral families — loaded from GGUF (incl. Q4_K / Q6_K) straight from Ollama / HuggingFace.
> - KV-cache decode at **0 bytes allocated per generated token** — verified by a CI test that fails the build on regression.
> - LoRA fine-tuning, in-process.
> - ONNX import (14 operators, branching DAG topology).
> - Native-AOT compatible — single-file deploy.
>
> I wrote a build log on the genuinely hard parts — KV-cache at 0 B/token, and writing an allocation-free parallel dispatcher because `Parallel.For` allocates ~3 KB per call: https://overfit-ml.com/zero-allocation-llm-inference-csharp.html
>
> Honest about the limits: CPU-only, not GPU; PyTorch CPU is still 2.2–3.6× faster on the raw training GEMM. The point of Overfit isn't beating MKL — it's predictable, allocation-free, no-native-dependency execution for .NET shops, regulated / on-prem environments, and edge.
>
> Repo: https://github.com/DevOnBike/Overfit · AGPLv3 + commercial licence.
>
> Feedback welcome — especially from anyone who's tried to do ML in .NET and bounced off the tooling.

---

## 3. LinkedIn — short post (NOT another article)

A business article went out yesterday. Today: a short post, technical/launch angle — complementary, not a repeat.

> Yesterday I wrote about *why* a .NET team shouldn't need a Python stack to use AI. Today, the *how*.
>
> I've published a build log on what it actually took to run LLMs in pure C# — GPT-2 and Llama-family models, in-process, with zero bytes allocated per generated token. The hard parts: KV-cache decode with no allocations, and writing a parallel work dispatcher from scratch, because the standard one allocates 3 KB per call.
>
> It's also up on Hacker News today — discussion here: [paste the HN link once it's live]
>
> Build log → https://overfit-ml.com/zero-allocation-llm-inference-csharp.html
>
> Overfit is open source. If you build in .NET and have wanted AI without the Python tax — this one's for you.
>
> #dotnet #csharp #MachineLearning #LLM #OpenSource

---

## Coordination & timing

- **Launch on a Saturday** — r/dotnet's rules force the weekend, and Show HN + LinkedIn both work fine on a weekend. Show HN best mid-morning US Eastern.
- HN link = the repo (people can use it); the blog deep-dive goes in the first comment. r/dotnet + LinkedIn link the blog directly.
- Post HN first, grab the HN item URL, paste it into the LinkedIn post before publishing.
- Then: clear your calendar for ~a day. Reply to every comment and issue, fast and straight. That is the launch.
