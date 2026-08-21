---
name: xc58-session-stack-contract
description: XC-58 (2026-08-15) — chose contract (a) shared stack + detection throw over (b) stack-per-session; per-stack scratch figures for Qwen-3B/0.5B/Gemma-2 and why "(b) costs memory" is false once right-sized.
metadata:
  type: project
---

`CachedLlamaInferenceEngine` hands ONE `CachedGptStack` to every `CachedLlamaSession` (`:40`, `:132`, `:392`);
each session gets its own `KeyValueCache`, which is what hides it. **Signed contract (a)**: sessions are cheap
views, must not decode concurrently, interleaving on one thread IS supported (each decode step is atomic
w.r.t. the scratch; nothing carries between steps except `_lastFinalHidden`/`_finalHidden`/`_lastLogits`,
which are documented as "last decode through this engine, whoever made it"). Plan:
`docs/specs/xc-58-shared-stack-session-contract-plan.md`.

**Per-`CachedGptStack` scratch, derived from constructors 2026-08-15** (arithmetic, NOT a measured
allocation). Qwen2.5-3B (L36 D2048 H16 kv2 hd128 dFF11008 vocab151936 ctx32768, read from the real GGUF
header): **86.4 MiB** sized to model ctx, **18.9 MiB** sized to ctx 2048; dominant term is
`CachedSingleHeadAttention._scoreScratch = new float[maxSequenceLength]` (`:77`) = L·H·4·ctx = 72 MiB.
KV cache for comparison: 144 MiB @ctx2048 F32. Qwen2.5-0.5B 46.4 MiB (42 of it score scratch — small models
pay proportionally MORE). Gemma-2-2B 15.0 MiB @ctx8192.

**The finding that reverses the naive cost model:** the stack is sized to `config.ContextLength`, never to the
per-session `maxContextLength`. So option (b) + right-sizing is a net **saving** of 67.5 MiB per client on
`overfit serve` (one engine per client); break-even ≈ 4.6 sessions/engine. Never repeat "(b) costs memory"
without the sizing qualifier.

**Other facts established, each by tool not by reading:**
- `CachedSlmSession` (GPT-1/2) already owns its stack per session (`:58-59` → `CachedGpt1ModelAdapter.cs:69`).
  The two engines have opposite contracts and neither says so.
- `CachedLlamaSession.cs:24` says *"Thread-safety: one session per thread."* — a statement in the WRONG
  direction, not silence.
- `QwenInferenceSmokeTests.MultipleSessionsFromSameEngine_Independenet` (`:150`) asserts only "same prompt →
  same token", which cannot distinguish independent from shared. A skipped test whose NAME signs off.
- `find_references(LastHiddenState)` = 2, both tests. `find_references(CreateSession)` = 170; concurrent
  production hosts pool whole clients (`Commands.cs:269-289`) and gate the two shared singletons
  (`OverfitInferenceService.cs:38-40`, `:190`) — so (a) breaks nothing shipped.
- Rejected: a `lock` on the stack. Mutual exclusion IS sufficient for forward-pass correctness, so it is a
  real option — rejected because silent serialisation on a throughput product is a worse failure than a
  throw, and it does not fix the stale-`LastHiddenState` half.

See [[xc49-decode-claim-seam]] / [[xc52-dispatcher-invariant]] for the bounded-wait rule the guard's
two-thread test inherits.
