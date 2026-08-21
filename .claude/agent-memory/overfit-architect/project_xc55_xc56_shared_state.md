---
name: xc55-xc56-shared-state
description: 2026-08-15 XC-56 (process-wide kernel flag) + XC-55 (exposed hidden state) — both signed; rulings, refuted premises and the measurement that closed XC-55
metadata:
  type: project
---

Plan: `docs/specs/xc-55-xc-56-shared-kernel-flag-and-exposed-hidden-plan.md`, SIGNED 2026-08-15.

**XC-56 ruling — serialising the two writers does NOT fix it.** `BatchedQuantProjection.DisableRepackedKernelsForParity`
has a **third**, non-obvious observer: `BatchedQuantProjectionTiledDispatchTests.Dispatch_PrepackedWeight_UsesTiled_EvenWithFlagOff`
is a plain `[Fact]` (no fixture — builds its own Q4_K weight) whose bit-equality assertion reads the flag at
`BatchedQuantProjection.cs:414`. A `[Collection]` over the two `[LongFact]` parity classes leaves it outside.
Decided: `[ThreadStatic]` + a save/restore scope in `TestSupport`. No collection, no runner config.

**The `[ThreadStatic]` admissibility rule, and it generalises:** it is only safe for a field whose correct
default is `default(T)`, because a `[ThreadStatic]` initialiser runs on the FIRST thread only. That is why
only one of `BatchedQuantProjection`'s seven flags can take this fix — `UseTiledPrefillQ4K` (env-derived),
`UseTiledPrefillQ6K` and `UseWeightStationaryQ4K` (`= true`) cannot. Also verified: all four dispatcher reads
happen while building the `TiledContext` passed to `OverfitParallel.For`, never inside a worker body, so the
flag cannot go inert in the parallel region.

**XC-55 ruling — the accessor was never the defect.** Two of the task row's three constraints are false:
`L0_ChatPromptLogits` contains **zero assertions** (so "it still passes" proves nothing), and the 2-token test
asserts only the **argmax**, not the logit value (11.3741 vs an oracle 12.3511, unchecked). The "returns row 0"
candidate is refuted by `BatchedPrefillThreshold = 16` — a 2-token prompt never reaches the batched path.
Cause: `C:\qwen3b\qwen.bin` was re-converted 2026-08-07 after commit `265fd77` added `permute_rope_rows`
(HF rotate-half → adjacent-pair) to `Scripts/convert_llama.py`; the May-2026 constants describe the
pre-permute file. **Position 0 is the identity rotation**, which is why the 1-token oracle still matches and
only position ≥1 diverges — that asymmetry is the signature of a RoPE-convention change.

**Measured 2026-08-15 (this box, Release, `OVERFIT_RUN_LONG=1`), `GgufVsBinaryLayerDivergenceDiagnostics.DoesTheDisagreementDependOnPosition`,
`C:\qwen3b\qwen.gguf` FP16 vs `C:\qwen3b\qwen.bin` FP32** — layer-0 cosine 0.999896 / 0.999631 / 0.999908 at
1/2/3 tokens (no position dependence ⇒ the file on disk has the correct rotary convention); last-layer cosine
0.990525 / **0.947242** / 0.992236. The non-monotonic 2-token dip is unexplained; the committed reading is
that `[BOS, im_start]` is a numerically ill-conditioned probe, which is why the fix moves that assertion to
cosine over the whole vector.

**Two out-of-scope defects found and recommended as new rows** (I filed neither — `docs/TASKS.md` is not mine):
`UseTiledPrefillQ4K` has the same cross-class writer/reader collision (`BatchedQuantProjectionTiledDispatchTests`
vs `RepackedSidecarEngineE2ETests`); and `CachedLlamaInferenceEngine.CreateSession` hands **one shared
`CachedGptStack`** to every session (`CachedLlamaInferenceEngine.cs:392`), so `session.LastHiddenState` returns
whatever any session of that engine decoded last, and two sessions decoding concurrently corrupt each other —
on a `public` API that says nothing about it.

Related: [[xc49-decode-claim-seam]], [[xc52-dispatcher-invariant]].
