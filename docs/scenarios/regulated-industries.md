# Overfit for Regulated Industries

**For teams in healthcare, finance, insurance, and government that cannot send their data to a third-party AI API.**

---

## The Problem

You want to use modern AI — language models, classification, anomaly detection — inside a regulated environment. Patient records. Claims data. Financial transactions. Citizen data. Case files.

And you immediately hit a wall that has nothing to do with model quality:

**The data cannot leave the building.**

- **GDPR / data residency** — personal data cannot be transferred to a processor outside an approved boundary. A call to an externally hosted AI API is a transfer.
- **HIPAA / health data** — protected health information sent to a third party requires a Business Associate Agreement, and most compliance functions will simply decline.
- **Sector rules** — financial regulators, government security classifications, and pharmaceutical IP policy all converge on the same constraint: sensitive inputs do not go to someone else's servers.

So the OpenAI / Anthropic / hosted-API route is closed before the technical evaluation even starts. The only compliant path is **self-hosted inference**.

But self-hosted ML usually means standing up a second technology stack: a Python runtime, a model server, often a GPU cluster — a whole parallel ecosystem to deploy, secure, patch, and audit. For an organisation whose engineering is built on .NET, that is a large and permanent cost.

---

## Why Overfit Fits

Overfit is a deep-learning and inference engine written in **pure C#**. It runs as a library inside your own .NET process, on your own infrastructure.

**No data egress, by construction.** Inference makes zero outbound network calls. There is no Overfit service, no API key, no telemetry phone-home. The model runs in-process; the data never crosses your boundary. This is not a policy you configure and hope holds — it is a property of the architecture. Your compliance review verifies it once, structurally, instead of re-auditing every request.

**No second stack to secure.** No Python runtime, no native binaries, no model-server container. Fewer components means a smaller attack surface and a shorter list of things to patch, scan, and certify. Overfit is one package on the .NET runtime your team already operates — and already knows how to harden.

**Runs on hardware you can actually procure.** CPU inference, no GPU required. In regulated and on-premises environments, GPU procurement and supply are often a multi-month problem of their own. Overfit runs on the ordinary servers and VMs you have already approved.

**Ships as a sealed, single-file artifact.** Native-AOT publishing produces one self-contained executable — straightforward to sign, scan, and move through a change-controlled release process, with no dependency graph resolved at deploy time.

---

## Auditability

Regulated AI is not only about *where* inference runs — it is about being able to explain it afterwards.

- **Deterministic inference.** Greedy decoding is fully deterministic: the same input and the same model weights produce the same output every time. A result can be reproduced for an auditor or a regulator.
- **Versionable models.** Model weights are a plain binary file. Check it into artifact storage, hash it, tag it to a release. "Which model produced this decision" has a concrete, file-level answer.
- **In-process, in-language logging.** Because inference is an ordinary C# method call — not a request to an opaque external service — recording the input, the model version, the output, and a timestamp is ordinary application code. The audit trail lives in your systems, in your format, under your retention policy.

Overfit provides the substrate — deterministic, in-process, versioned. The decision log itself is a few lines of your own code, not an integration with someone else's platform.

---

## What You Can Actually Run

- **Language models** — GPT-2, and the Qwen / Llama / Mistral families — loaded and run entirely in-house, including quantized GGUF weights from Ollama or HuggingFace.
- **In-house fine-tuning** — LoRA fine-tuning adapts a base model to your own corpus (clinical notes, claims language, internal documentation) without that corpus ever reaching an external provider.
- **Classification, regression, anomaly detection** — MLPs, CNNs, LSTMs, and autoencoders for tabular and sequence data, plus ONNX import for models trained elsewhere.

---

## What Overfit Does Not Give You

Honest scope:

- **It is not a hosted service.** You own the deployment, the infrastructure, and the operations. That is the point — but it is real work.
- **It is not GPU-accelerated.** Inference is CPU-only. For small and medium models that is a feature — predictable, inexpensive, no GPU dependency; for very large models it is a ceiling.
- **It is not a compliance certification.** Overfit removes the single largest *technical* blocker — data egress — and shrinks the audit surface. The compliance programme itself is still yours to run.
- **It does not run frontier-scale models.** The sweet spot is small-to-medium models that fit and run well on CPU.

---

## Further Reading

- [Main README](../../README.md) — project overview and benchmarks
- [Finance & low-latency scenario](finance-latency.md) — for latency-critical deployments
- [Edge & IoT scenario](edge-iot.md) — for on-device, disconnected deployment
- [ROADMAP](../../ROADMAP.md) — upcoming features
