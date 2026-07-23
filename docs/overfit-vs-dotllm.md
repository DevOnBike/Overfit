# Overfit vs dotLLM — measured

> **Status:** internal bench note, 2026-07-23. Not linked from the README or `docs/README.md` by design —
> it names a specific competing project and carries caveats that don't belong in launch-facing copy.

Two pure-C# CPU inference engines, measured through **one load driver over one protocol** (`overfit bench`
against each engine's OpenAI-compatible server) so neither reports its own scorecard. After the prompt-cache
work landed on branch `sauron`, Overfit leads on every axis a chat server is judged on — the caveats that
make the numbers trustworthy are stated as plainly as the wins.

## Setup

| | |
|---|---|
| **Model** | Qwen-2.5-3B-Instruct Q4_K_M (`C:\qwen3b\qwen.q4km.gguf`, same file both engines) |
| **Box** | Ryzen 9 9950X3D (Zen 5, 16 physical / 32 logical) |
| **Prompt** | ~25-token chat prompt |
| **Load** | 1 concurrent user, 32 tokens out |
| **Method** | ABAB interleaved, 5 rounds, all servers resident in one wall-clock window, model id verified per port |

## Verdict

Ratios are Overfit over dotLLM in its **default configuration** (prompt cache on); `>1` = Overfit ahead.

| Metric | Overfit | dotLLM (cache on) | Edge |
|---|--:|--:|--:|
| **TTFT** (time to first token) | 0.6 ms | 39 ms | **65×** |
| **ITL** (inter-token latency) | 38.3 ms/tok | 43.0 ms/tok | **1.12×** |
| **E2E** (end to end) | 1190 ms | 1380 ms | **1.16×** |
| **Throughput** | 26.9 tok/s | 23.3 tok/s | **1.15×** |

## Head to head

dotLLM shown in both configurations — its default (prompt cache on) and its true cold prefill
(`--no-prompt-cache`), because comparing our prefill to their cache hit would measure a missing feature
rather than kernel quality.

| Metric | Overfit | dotLLM · cache on | dotLLM · cold prefill | Edge vs default |
|---|--:|--:|--:|--:|
| TTFT | 0.6 ms | 39 ms | 825–992 ms | 65× ahead |
| ITL | 38.3 ms | 43.0 ms | ≈ 38 ms | 1.12× ahead |
| E2E | 1190 ms | 1380 ms | ≈ 2100 ms | 1.16× ahead |
| Throughput | 26.9 tok/s | 23.3 tok/s | ≈ 15 tok/s | 1.15× ahead |

## What changed — the TTFT collapse

Time to first token fell three orders of magnitude across one session, each step a separate mechanism,
each measured before the next was built:

```
session start ──▶ KV prompt cache ──▶ early token emit ──▶ logits cache
    197 ms            74.8 ms              ~40 ms            0.6 ms
```

- **KV prompt cache** reuses the key/value state a previous turn already built instead of re-encoding the
  shared prefix (`CachedLlamaSession.PrefillReusingCache`, `_cacheTokens` indexed by cache position;
  truncation is O(1) bookkeeping).
- **Early token emit** puts each token on the wire *before* the forward pass that prepares the next one —
  one whole weight-pass earlier (the `onSampled` hook in `GenerateNextToken` / `GenerateSpeculative`).
- **Logits cache** keeps the end-of-prompt logits, so a re-sent prompt runs **zero** passes
  (`_promptLogits`, invalidated on eviction / prefix restore).

Each was A/B-isolated in one process via env toggles (`OVERFIT_DISABLE_LOGITS_CACHE` /
`OVERFIT_DISABLE_EARLY_EMIT` / `OVERFIT_DISABLE_SPECULATIVE`). The logits cache measured **1.001×** on
decode — free. Speculation on vs off measured **1.002×** — not a factor either way.

## Who wins where, and why

**Overfit leads:**

- **Prefill kernel — ~4.2×.** Cold prefill of a chat prompt: ~197 ms vs their ~825–992 ms. Each weight row
  is read once from DRAM and amortised across all prompt rows (the batched projection dispatch).
- **Decode — 1.12×.** Both engines are near the memory-bandwidth ceiling here; the edge is real but small.
- **TTFT on a repeated prompt — 65×.** The full cache stack forwards nothing when the prompt matches.
- **Zero-alloc, no native binary.** Single Native-AOT executable, pure managed C#.

**dotLLM's ground:**

- **Prompt caching shipped first.** Its 39 ms default was a cache hit, not a prefill — a feature Overfit
  lacked at the start of this session and now matches.
- **Steadier cold prefill under drift.** Its cold path is slower but its across-round variance was tighter
  than ours on a loaded box.
- **GPU path exists.** dotLLM ships a CUDA backend; Overfit's public identity is CPU-only.

## What makes these numbers trustworthy

- **One box, one model, one prompt shape.** Not a basis for an unqualified "Overfit is faster than dotLLM"
  — always name the model, hardware and prompt shape.
- **The 0.6 ms TTFT is the repeated-prompt shape** (retry, regenerate, load test). Real multi-turn chat
  appends tokens each turn, taking the tail-prefill path: tens of ms, not sub-millisecond. ITL and E2E are
  ordinary decode and hold in every shape.
- **dotLLM's 39 ms is its prompt cache, not its kernel.** Its true cold prefill is 825–992 ms; both are
  shown so the comparison is like-for-like.
- **Every ratio is within-run, interleaved.** Cross-process before/after drifts up to ~30% on this box, so
  all arms were measured in one window with the untouched arms as canaries; ratios are medians.

## Retracted during this session — three measurements that were wrong

Kept here on purpose: the failures are why the surviving numbers are trustworthy.

- ~~"TTFT 6.2× worse than dotLLM."~~ **Measured Ollama**, not Overfit — port 11434 was already taken and the
  load driver silently polled the wrong server. Caught by the model id (`…fp16`) and a 100 ms/token ITL that
  only fp16 weights explain. → guard: verify `/v1/models` against the file you launched, use a free port.
- ~~"Decode 1.78× ahead."~~ dotLLM's arm was measured on **cold mmap rounds**; withdrawn once both arms were
  warmed in one window.
- ~~"Logits cache regressed ITL 37→55 ms."~~ **Machine drift**, not the change — one round spiked to 55 in
  both our arms at once while dotLLM moved too. The in-process ablation had already shown the change was free;
  the cross-process comparison that flagged a "regression" was the mistake.

---

*Full suite 1505 / 0 at time of measurement. Cache stack A/B-isolated per change. Numbers from the 5-round
interleave; ratios are medians, never cross-process.*
