# Bug hunt: Sources/Main/LanguageModels/Runtime (decode engine)

- Scope: `Sources/Main/LanguageModels/Runtime` (55 files)
- Reviewed: 2026-08-01/02, UTC timestamp `2026-08-01-2206`
- Commit: `79e9d80`
- Score: **0 points / 0 defects**
- **Ended by the five-minute wall-clock cap, not by finishing the scope.**

## Result

No defect meeting the bar in these instructions (wrong behaviour, silent failure, leak, crash/hang,
false claim, or contract violation) was found in the files actually read. Per the rules of this game,
that is reported honestly rather than padded to reach a score — **but see "What this run does not
mean" below, because the run ended by the clock, not by exhausting the scope.**

## What was reviewed, and how

Breadth first (grep across the whole directory for `catch`, `LastError`/`Failed`/`Warnings` fields,
`norm_topk_prob`/renormalise, `break;`, `TODO`/`FIXME`), then depth on the files that looked highest-risk
by the task's own hints and by size of the most recent commit that touched this directory
(`972ee7d "perf + aot"`, the SIMD/batched-prefill rewrite):

- `KeyValueCache.cs` — full read. Sliding-window `Evict` (memmove shift + `BasePosition` accounting),
  `Advance`/`TruncateTo` bounds, Q8 vs F32 dual-mode read/write surface, `Snapshot`/`RestoreFrom`.
  Bounds checks are consistent (`(uint)x >= (uint)limit` pattern), no leak, no silent truncation.
- `MoeRouter.cs` — `SelectTopK` insertion-sort top-k + both `norm_topk_prob` branches (Mixtral/Qwen
  renormalise vs Qwen1.5-MoE raw-softmax-at-top-k). Tie-breaking verified to actually keep the lower
  index as documented. Numerically stable (global max from the top-1 logit used as the softmax pivot in
  both branches).
- `CachedSingleHeadAttention.cs` — full read. RoPE applied to Q pre-attend and to K before the cache
  write (K stays permanently rotated, matches the documented invariant), GQA K/V-projection-once-per-group
  sharing, Q4_K/Q6_K/Q8_0/F32 per-projection dispatch.
- `CachedMultiHeadAttention.cs` — full read, including the M3 whole-matrix Q4_K attention path
  (`TryDecodeWholeMatrix`/`DecodeHeadWhole`), the GQA head-parallel decomposition, `DecodeBatched` and
  `DecodeBatchedQuant` (whole-Q/whole-KV/whole-O gathering with per-head bias re-application). Bias
  application is consistent between the whole-matrix and per-head paths in both decode and prefill.
- `CachedLlamaSession.cs` (partial, ~450 lines incl. speculative decode, sliding window, prefix reuse) —
  `GenerateSpeculativeCore` (adaptive gate, draft/verify/accept-or-resample/truncate/correction sequence),
  `MakeRoomIfSliding`/`EnableSlidingWindow`, `Prefill`/`PrefillReusingCache`, `StreamGenerateAsync`.
- `SpeculativeSampler.cs`, `PromptLookupDrafter.cs`, `DraftModelSpeculativeDrafter.cs` — full read.
  Rejection-sampling accept/resample, n-gram draft bounds, draft-model KV rollback/resync arithmetic
  (`_draftBase + keep` bounds checked against what was actually fed).
- `MoeFeedForwardBlock.cs` — full read, both single-token `Decode` and batched `DecodeBatched`
  (expert-grouped batched SwiGLU, top-k-slot accumulation order proven to match single-token order).
- `TokenSampler.cs` (~500 lines, most of it) — `Sample`, `ComputeProbabilities`, `SelectTypicalP` /
  `SelectTypicalPFull` (partial-sort-then-fallback-to-full-sort spillover logic), `ApplyRepetitionPenalty`.
- `Q4KGemvKernel.cs` (full, ~1000 lines) — `Gemv`/`GemvParallel`/`GemmTiled`/`GemmTiled512` SIMD kernels
  and their ablation flags; `GemmTiled512`'s odd-column-pair discard logic cross-checked against
  `PairHigh`'s clamp.
- `Q8KvQuant.cs`, `QkNormKernel.cs`, `BatchedAttentionKernel.cs` (incl. the balanced-query-order
  work-stealing bijection), `DecodeProfiler.cs` — full read.
- `SlmSession.cs`, `SlmInferenceEngine.cs`, `CachedSlmSession.cs` — the `Generate`/`GenerateStreaming`
  loops and their stop-token / cache-full termination `break`s (checked for off-by-one in the
  `generated` count reported to the caller).
- `CachedGptStack.cs` (partial) — `PrefillBatchedQuant` vs `PrefillBatchedQuantAllRows` (single-row vs
  per-row final-norm output used by the speculative verifier).
- `BatchedQuantProjection.cs` (partial, ~200 of 733 lines) — axis-selection/tiling heuristics and their
  documented measurements; the already-disclosed non-bit-identity of the repacked kernels (which the
  task said is a known, tested, accepted trade — not re-reported).

## What this run does not mean

**This is a time-capped result, not a scope-complete one.** ~20 of 55 files were read in depth; the
rest were only touched by the opening greps (which found no `catch`, no `LastError`-shaped field, and
no `TODO`/`FIXME` anywhere in the directory — those two specific greps *are* scope-complete and can be
trusted as clean). A zero score here says the reviewer ran out of clock on a directory that is unusually
dense (SIMD kernels, adaptive gating, multi-path dispatch) and does not certify the unread two-thirds.

## Coverage

**Reviewed and found clean** (see file list and specifics above):
`KeyValueCache.cs`, `MoeRouter.cs`, `CachedSingleHeadAttention.cs`, `CachedMultiHeadAttention.cs`,
`SpeculativeSampler.cs`, `PromptLookupDrafter.cs`, `DraftModelSpeculativeDrafter.cs`,
`MoeFeedForwardBlock.cs`, `TokenSampler.cs`, `Q4KGemvKernel.cs`, `Q8KvQuant.cs`, `QkNormKernel.cs`,
`BatchedAttentionKernel.cs`, `DecodeProfiler.cs`, `SlmSession.cs` (Generate loop),
`SlmInferenceEngine.cs` (GenerateStreaming loop), `CachedSlmSession.cs` (Generate loop),
plus the read portions of `CachedLlamaSession.cs`, `CachedGptStack.cs`, `BatchedQuantProjection.cs`.
Directory-wide greps for `catch`/`try` (12 files matched, all doc-comment mentions of "try"/nothing
that actually swallows an exception — the directory contains **zero** `catch` blocks), and for
`TODO`/`FIXME`/`HACK`/`BUG` (zero matches), are exhaustive over all 55 files.

**Not reached** (no meaningful read — start here next time):
`BatchedProjectionKernel.cs`, `CachedAttentionKernel.cs`, `CachedFeedForwardBlock.cs`,
`CachedTransformerBlock.cs`, `Qwen2MoeFeedForwardBlock.cs`, `Q4KWeight.cs`, `Q6KWeight.cs`,
`Q8Weight.cs`, `Q4KDotKernel.cs`, `Q6KDotKernel.cs`, `Q6KGemvKernel.cs`, `Q6KRepack.cs`,
`Q4KRepack.cs`, `StackWeights.cs`, `BlockWeights.cs`, `KvHeadWeights.cs`, `SingleHeadWeights.cs`,
`DecodeWeight.cs`, `SlmRuntimeFactory.cs`, `SlmRuntimeHandle.cs`, `SlmRuntimeMode.cs`,
`KvCacheSnapshot.cs`, `KvCacheDType.cs`, `IKeyValueCacheReader.cs`, `KeyValueCacheReaderExtensions.cs`,
`ISpeculativeDrafter.cs`, `Gpt1SlmModelAdapter.cs`, `CachedGpt1ModelAdapter.cs`,
`CachedLlamaInferenceEngine.cs`, `SingleTokenLayerNormKernel.cs`, `SingleTokenProjectionKernel.cs`,
`FeedForwardActivation.cs`, `PrefillProfiler.cs`, `Q8DotKernel.cs`, `SlmRuntimeFactory.cs`,
the remaining ~700 lines of `CachedLlamaSession.cs`, ~530 lines of `BatchedQuantProjection.cs`, and
`CachedGptStack.cs` beyond the two prefill entry points read.

## What a short score means here

**Ended by time**, not by scope — so this must not be read as "the runtime is clean." It means: in the
roughly three and a half minutes of actual reading available, the highest-risk-looking mechanisms
(KV-cache lifetime/eviction, RoPE rotation placement, MoE gating math, speculative-decode accept/resample
arithmetic, and the newest large SIMD/batched-prefill addition) held up under close reading, and a third
of the directory's files were never opened. The right next step is the "not reached" list above, ideally
with the five-minute cap lifted or split across multiple passes — this directory is dense enough that
5 minutes buys real depth on maybe 8-10 files, not the full 55.
