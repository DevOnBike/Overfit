# Bielik — an honest assessment (from running it in Overfit)

> Internal notes, not a marketing page. This is an empirical read of
> **Bielik-4.5B-v3.0-Instruct** after loading and running it in Overfit (Q8_0 GGUF, fp16 GGUF, and
> the real bf16 safetensors) across a handful of Polish prompts — RAG, chat, tool calling, JSON,
> and a greedy parity check. **Small sample, one model size.** Not deliberately linked from the
> README/docs index.

## TL;DR

Not "magic" and not "genius" — a **solid, specialised 4.5B model that is clearly better than general
models *at Polish*, while its reasoning and parametric knowledge are typical for its size.** It is
also architecturally **unusual**, in a way that has real consequences for tooling.

Its real edge is **language, not intelligence.**

## Where it is better (with evidence)

- **Native-quality Polish.** Correct grammar, inflection, **diacritics** ("Stolicą", "Kraków",
  "odstąpienie"), idiomatic phrasing. It stays in Polish without being asked. A general Qwen/Llama
  of similar (or larger) size does not write Polish this naturally.
- **Polish domain tasks via RAG.** On a clear question — *"Ile dni ma klient z UE na odstąpienie od
  umowy?"* → *"Klient z UE ma **14 dni** na odstąpienie od umowy, liczone od daty zakupu."* —
  correct, grounded, cited.
- **Agentic mechanics work on its SentencePiece tokenizer.** Constrained tool calling with the exact
  argument schema, and guaranteed JSON, both behaved — so it is usable as a *private Polish agent*,
  not just a chatbot. (A structured refund decision came back as a valid JSON object with a Polish
  `reason`.)

## Where it is weaker / limited (honestly)

- **It hallucinates without grounding.** Plain `/chat`, no RAG, *"czym jest rękojmia?"* →
  *"rękojmia to proces w systemie operacyjnym…"* — confidently wrong on a domain term. A 4.5B has
  little parametric knowledge; **don't trust it without RAG.**
- **Reasoning ceiling like its peers.** The borderline temporal question *"zwrot **po** 10 dniach"*
  flipped to "No" (10 < 14 should be "Yes") — the *same* failure a Qwen-3B shows. So here it is **not
  better**, just typical for ~4–5B.
- **Slower.** ~8 tok/s on Q8_0 (bigger model + heavier quant than a 3B Q4) on CPU.

## Where it is *different* (the most interesting part for loading)

- **Depth-upscaled: 60 layers, narrow (hidden 2048), GQA 16 Q / 2 KV heads.** That is an unusual
  deep-and-narrow shape (SOLAR-style depth upscaling), not a typical "wide" llama — 4.5B params
  packed into 60 layers.
- **Standard SentencePiece (llama) tokenizer, NOT the custom APT4** Polish tokenizer (APT4 is in the
  larger Bielik-PL-11B). So for the 4.5B, the embedded SPM handles Polish well and APT4 is a
  non-issue.
- **Its GGUF is off the beaten path.** The most concrete finding: **HuggingFace transformers' own
  GGUF loader produces garbage on this model** (degenerate repeating tokens, even when fed correct
  input ids) — its depth-upscaled config breaks the dequantiser. **Overfit loads it correctly.**

## Parity result

On the **real bf16 safetensors**, Overfit's `SafetensorsLlamaLoader` (F32) reproduced HuggingFace
transformers **token-for-token, 30/30 greedy tokens** for `"Stolicą Polski jest" → "Warszawa. …"`.
So Overfit's forward pass (RoPE permute, GQA, the 60-layer stack, tensor mapping) is faithful to the
reference — and notably, **Overfit loads Bielik correctly from *both* GGUF and safetensors, where
HF's own GGUF path fails.**

## Verdict

As a **Polish 4.5B for private RAG / agent work**, it is a good choice and a natural hero for the
"Bielik in pure .NET" demo. As a **general reasoning model**, it is average for its weight, needs RAG,
and a larger model will beat it on hard reasoning. Lead demos with clear factual questions, not
borderline temporal ones.

## Caveat & a better way to judge

This is a read from a **small sample on one size (4.5B)**. For firmer conclusions, run Bielik and,
say, Qwen2.5-3B through the *same* set of Polish RAG questions and score accuracy + language quality
side by side.
