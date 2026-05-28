# Overfit — Commercial Licensing & Services

> **Overfit embeds local LLMs, RAG, tool calling and guaranteed JSON directly into your .NET process — no Python, no model server, no native binary, no data egress.**

Overfit is dual-licensed (see [`LICENSE.md`](LICENSE.md) for the full text of both options — AGPLv3 and the commercial alternative). Most users will be fine on the open-source path. This page is for the cases where you can't be, or where you'd rather have the author ship the integration than learn the framework cold.

---

## Who this is for

You probably need a commercial engagement if **any** of the following describes you:

- You have a **.NET system** and need a local LLM / RAG / agent **inside** your process — not behind a Python sidecar or an external API.
- You **cannot send data to OpenAI / Anthropic / external endpoints** — regulated industry, customer NDAs, GDPR / HIPAA / SOC2 scope, government, defence, on-prem-only.
- You ship a **closed-source** product (.NET service, desktop app, SaaS backend, embedded system) and need a commercial license to lift the AGPLv3 source-disclosure obligation.
- You already run a **Python / Ollama / llama.cpp / Triton sidecar** and want to collapse it into your .NET process.
- Your existing .NET inference path has **GC pauses or AOT-publish problems** eating tail latency, and you want it audited.
- You want **performance characteristics in writing** — zero-allocation hot paths, P99 latency budgets, peak-RAM ceilings, AOT-clean publishing — backed by a contract.
- You want **a named person on the hook** when something breaks in production.

If none of the above applies, the open-source AGPLv3 path is genuinely fine. Use it, file issues, send PRs.

---

## What the open-source license already includes

Just so this is explicit — under AGPLv3, with no commercial engagement, you get the full feature set:

- GGUF / HF safetensors / ONNX (linear + DAG) loaders — Qwen, Llama, Mistral, Mixtral, MoE, BERT-family embeddings, GPT-2.
- `OverfitClient.LoadGguf(path)` turnkey chat facade, `ChatSession`, streaming, sliding-window eviction, tool calling, guaranteed-valid JSON.
- Agentic primitives (ReAct, Critic, CircuitBreaker, SummarizingChatSession).
- Native-AOT-compatible CPU inference, zero per-token allocations on the chat hot path.
- Training: autograd, optimizers, gradient checkpointing, data-parallel trainer, LR schedules.

The commercial license does not gate features. It changes the **legal terms** under which you use those features, and it lets you buy time / support from the author.

---

## Service packages

Three concrete, fixed-scope, fixed-price engagements. Each is a productised offer — not "let's see how it goes" consulting. Each bundles the commercial license for the integrated product.

---

### Package 1 — Private .NET RAG/Agent PoC

> **For .NET teams who need a local copilot / RAG agent on their own documents and infrastructure, without sending data to external APIs.**

**Common triggers:**
- "Our security team rejected the OpenAI integration."
- "Our data is GDPR / HIPAA / SOC2 / NDA-scoped — model has to stay on-prem."
- "We need a copilot/chatbot/triage agent over our knowledge base, ticket system, contracts or manuals."
- "Procurement blocked the SaaS LLM vendor; we need a .NET-native solution."

**Typical scope:**
- Load and serve a local GGUF model (Qwen / Llama / Mistral family — selected with you during kick-off).
- Index your documents into the in-process vector store (text, markdown, PDF — extraction included).
- Wire your business tools as C# delegates with constrained, schema-valid JSON arguments (e.g. customer lookup, ticket creation, calendar, CRM, internal API calls).
- Stand up ASP.NET endpoints: `/chat` (streaming), `/rag/index`, `/rag/query`, `/tools`, `/health`, `/metrics` (Prometheus).
- Deploy to your target environment — Linux container, Windows service, desktop installer, or air-gapped binary.
- Build a small evaluation suite (20–30 representative queries with expected behaviour) so regressions are caught in CI.

**Duration:** 2–3 weeks fixed.

**What you have at the end:**
1. Merged feature branch in **your** repository running on **your** model and **your** documents.
2. Benchmark report on **your** hardware — P50 / P95 / P99 latency per endpoint, tokens/sec, peak RAM, allocations per request.
3. Deployment artefact for your target (Dockerfile + `compose.yaml`, or Windows service installer, or AOT-published single-binary + `appsettings.json`).
4. Written architecture note + 90-minute handover call so your team can extend and maintain it.
5. Commercial license bundled in (covers the closed-source product the PoC was integrated into).

**Pricing model:** Fixed fee per engagement. Quote on request — depends on document corpus size, tool count, and target environment.

---

### Package 2 — Python / ONNX-Sidecar Replacement

> **For teams currently running a Python or external model-server sidecar and wanting to collapse inference into the .NET process.**

**Common triggers:**
- "We maintain two stacks (Python + .NET) for one product."
- "Python container drift breaks our deploys."
- "The network hop to the model server is eating P99 latency."
- "Our infra team wants to retire the Ollama / llama.cpp-server / Triton / vLLM tier."
- "ONNX Runtime works but the P/Invoke overhead is killing us at our throughput."

**Typical scope:**
- Inventory the current model + serving path — which model, which quantisation, how the .NET service calls it, what the SLA is.
- Port to Overfit (GGUF / safetensors / ONNX import — whichever applies).
- Stand up identical functionality as an in-process API in your .NET service.
- Run a side-by-side benchmark — same hardware, same model, same inputs — Python sidecar (or ONNX Runtime, or llama.cpp-server) vs Overfit in-process.
- Remove the sidecar container from your deploy; update CI/CD pipeline; update health checks and monitoring.

**Duration:** 2 weeks fixed.

**What you have at the end:**
1. In-process inference in your .NET service — no more sidecar.
2. **Honest** side-by-side benchmark report — tokens/sec, P50 / P95 / P99, peak RAM, allocations, cold-start time. If the sidecar still wins on a specific metric (e.g. raw decode tok/s on Q4_K_M against hand-tuned AVX-512 llama.cpp), it's stated explicitly with the trade-off named: typically latency / per-token allocation / deploy simplicity wins are Overfit; raw single-stream tok/s wins on quantised models are native AVX-512 sidecars.
3. Deploy-savings report — container count, image size delta, RAM delta, deployment complexity delta, removed services from `compose.yaml` / Helm chart / pipeline.
4. PR removing the sidecar from your infra config + CI/CD + runbooks.
5. Commercial license bundled in.

**Pricing model:** Fixed fee per engagement. Quote on request.

---

### Package 3 — Zero-GC Inference Audit

> **For teams shipping .NET inference (Overfit, ML.NET, ONNX Runtime, custom kernels) where allocations are eating tail latency or AOT publishing is blocked.**

This package does **not** require you to use Overfit. It's a methodology audit applicable to any .NET inference setup. If we discover that switching to Overfit would deliver a 10× win, that's noted in the report; if your current solution is the right one for your constraints, that's noted too.

**Common triggers:**
- "P99 is fine until traffic doubles, then GC kicks in."
- "We can't ship as Native-AOT — too many trim / reflection warnings."
- "Tail latency on the inference path is 10× the median."
- "ML.NET seems to be allocating but we can't see where."
- "BenchmarkDotNet says zero allocs, production says GC every 30 s — we need someone to dig in."

**Typical scope:**
- You provide your hot-path code (or a reproducible repro) and a representative benchmark.
- Profiling on your hardware: BenchmarkDotNet harness + dotMemory / PerfView traces, GC stats per phase.
- Identification of allocation sources, GC-pressure hot-spots, AOT-incompatible patterns (reflection, LINQ in inner loops, hidden boxing, struct copies in `foreach`, etc.).
- Prioritised list — each item with diagnosis (file:line where possible), cost estimate per fix, and projected allocation/latency delta.
- Optional: implementation of the top-3 fixes in your code, re-benchmark, deliver PR with **measured** deltas.

**Duration:** 1–2 weeks fixed (depending on whether the top-3 fixes are included in scope).

**What you have at the end:**
1. Profiling report — BenchmarkDotNet results, dotMemory traces, GC stats from the actual hot path.
2. Prioritised list of allocation / perf / AOT issues with diagnoses and expected fix cost.
3. (Optional, if in scope) PR with the top-3 fixes implemented and benchmarked.
4. AOT-clean publish verification — actual `dotnet publish -r linux-x64 /p:PublishAot=true` run on your code, with any remaining warnings categorised (trim, reflection, etc.).
5. 60-minute walkthrough call.

**Pricing model:** Fixed fee per engagement. Quote on request — depends on whether top-3 fixes are in scope.

---

## Other commercial options

### Standalone commercial license (per-product, perpetual)

For teams who only need the legal clearance — they will integrate Overfit themselves and don't need a service engagement.

- Signed PDF license agreement granting the right to embed Overfit in a named closed-source product, with no AGPLv3 source-disclosure obligation. Covers redistribution to end-users.
- Issued within 5 business days of contract terms agreement.
- One-time fee per product + optional annual maintenance for upgrades. If you stop maintenance you keep the rights you bought; you just don't get newer releases.
- Discounted **Indie / Startup** tier available for companies under €1 M ARR.

### Support retainer (monthly)

For teams running Overfit in production and wanting ongoing access to the author for production support, performance work, and roadmap influence.

| Tier | Response SLA (business hours, CET) | Hours/month included | Roadmap influence |
|------|---|---|---|
| **Standard** | Next business day | 4 h | Bug reports prioritised |
| **Business** | Same business day | 12 h | Priority on requested features |
| **Priority** | Within 4 business hours | 32 h | Joint roadmap planning, named release windows |

"Business hours" = Mon–Fri 09:00–17:00 Europe/Warsaw (CET / CEST). Outside-hours and weekend response is on a best-effort basis, not contractually guaranteed — Overfit is small and direct, not a 24/7 NOC, and this page is honest about that.

**What "hours included" covers:** debugging sessions, code review of your Overfit-using code, performance investigations on your benchmarks, custom kernel work, training-loop tuning, version-upgrade help.

**What it does not cover:** building your end-user product features, generic .NET / ASP.NET consulting unrelated to Overfit, ML strategy / model-selection advice unrelated to runtime fit.

Flat monthly retainer, 3-month minimum. Unused hours do not roll over. Discounted retainer rate during the first 6 months of commercial use.

---

## SLA — what's actually contractual

To be precise about what "SLA" means here, not aspirationally:

- **Response-time SLA** (retainer tiers above) — first human response within the stated window, every business day. Written into the engagement contract.
- **Defect-fix priority** — a defect in shipped Overfit code that you can reproduce will get a named owner (the author) and an acknowledged fix-or-mitigation plan within 2 business days on the Business tier, 1 business day on Priority.
- **Version compatibility** — for the duration of an active retainer, security fixes are backported to the version you're pinned to.
- **Benchmark commitments** — tokens/sec, P99, peak-RAM targets are project-scoped, written into the Package 1 or Package 2 statement of work against your hardware and your model. They are contractual *for that engagement*, not as universal product claims.

What is **not** an SLA:

- 24/7 uptime / pager rotation. If you need that, you need an infra team, not a library vendor.
- Universal tokens/sec or latency targets independent of hardware and model. Those numbers only mean something tied to a specific deployment.

---

## How to buy

1. **Email** `devonbike@gmail.com` with subject `Overfit Commercial Inquiry` and a short description:
   - What you're building (1–2 sentences).
   - Which package you're interested in (Package 1 / 2 / 3, standalone license, or retainer).
   - Approximate target deployment (cloud / on-prem / edge / desktop / air-gapped), model family if known, target latency / RAM if known.
2. **Discovery call** (30–45 minutes, free) — scope check, fit check, honest "Overfit is / isn't right for you" conversation. NDA available on request before the call.
3. **Quote + statement of work** — fixed price, fixed scope (or fixed retainer terms), sent within 3 business days of the call.
4. **Sign + invoice** — once signed, work starts on the agreed date; license documents are issued in parallel with the engagement.

For OSS contributors who want a commercial license on a project they're already shipping under AGPLv3, the process is the same but shorter — usually just the standalone license, no service package needed.

---

## Frequently asked

**Q: Can I evaluate Overfit commercially before buying a license?**
Yes. The OSS AGPLv3 license permits evaluation. The commercial license is needed when you ship to end-users as part of a closed-source product, or run it as part of a closed-source network service.

**Q: We're a startup, can we get a discount?**
Yes — discounted **Indie / Startup** tier on the standalone per-product license for companies under €1 M ARR, and a discounted retainer rate during the first 6 months of commercial use. Service packages (1, 2, 3) are fixed-price; reach out and we'll see what fits.

**Q: Do you offer a perpetual fallback license?**
The standalone commercial license is perpetual on the version you license. Optional annual maintenance gives you new versions; if you stop maintenance, you keep the rights you bought, you just don't get newer releases.

**Q: Can the PoC run fully air-gapped / on-prem / inside our VPN?**
Yes. The whole point of Package 1 is data not leaving your boundary. Deployment to air-gapped or on-prem environments is in scope; the author can work on-site or against a sandbox VPN as the contract specifies.

**Q: Do you sub-contract or white-label?**
No. The author does the work personally. If you need a multi-engineer team, Overfit is the wrong vendor — but the author can recommend partners who already use Overfit.

**Q: What if my deployment doesn't fit any of the three packages?**
Reach out. The three packages cover the most common shapes; bespoke scopes are still possible, just priced and scoped per case rather than from a standard SOW template.

---

**Contact:** `devonbike@gmail.com` — subject line `Overfit Commercial Inquiry`. Expect a reply within one business day.
