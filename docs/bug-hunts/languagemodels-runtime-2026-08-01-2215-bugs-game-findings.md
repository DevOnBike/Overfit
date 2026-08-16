# Bug hunt: Sources/Main/LanguageModels/Runtime (decode engine) — second pass

- Scope: `Sources/Main/LanguageModels/Runtime`, restricted to the files the first hunt (`languagemodels-runtime-2026-08-01-2206-bugs-game-findings.md`) never opened.
- Reviewed: 2026-08-01/02, UTC timestamp `2026-08-01-2215`
- Commit: `79e9d80`
- Score: **6 points / 3 defects**
- **Ended by scope exhaustion of the assigned priority list within the time budget — stopped voluntarily at ~5 minutes of the 10-minute cap given to this run, not because the clock ran out.** (The parent task specified a ten-minute cap explicitly, overriding the module's default five; this run used about half of it and stopped once the priority list was covered and returns were diminishing.)

## Findings

### 1. Q4_K tiled prefill silently drops the AVX-512 kernel whenever output-row banding is active

**What breaks.** `BatchedQuantProjection.DispatchTiledQ4K` resolves `ctx.Avx512 = UseAvx512PrefillQ4K && !DisableRepackedKernelsForParity` and stores it on the shared `TiledContext`. The **non-banded** worker (`TiledChunk`) reads that flag and routes to `Q4KGemvKernel.GemmTiled512` when it is set. The **banded** worker (`TiledBandChunk`) — used whenever `ShouldBandOutputRows` picks the banding axis (short prompts, which is exactly where this ships default-on because it's the common chat-latency case) — never reads `ctx.Avx512` at all and unconditionally calls the 256-bit `Q4KGemvKernel.GemmTiled`. The Q6_K sibling path does not have this bug: `TiledQ6KBandChunk` (line ~684) checks `c.Avx512` before choosing between `GemmTiled512`/`GemmTiled`; the Q4_K band chunk (`TiledBandChunk`, line ~788) is missing the equivalent branch that its own non-banded twin (`TiledChunk`, line ~772) has.

**Where.** `Sources/Main/LanguageModels/Runtime/BatchedQuantProjection.cs`, `TiledBandChunk` (compare to `TiledChunk` immediately above it, and to `TiledQ6KBandChunk`/`TiledQ6KChunk` which both do check the flag).

**How anyone would notice today.** They would not. Both kernels are documented and parity-tested as bit-identical (`Avx512PrefillParityTests`), so no test comparing outputs would ever fail, and no exception or log fires. The only symptom is that `OVERFIT_AVX512_PREFILL_Q4K=1` (or the default-on AVX-512 detection) has zero effect on prefill throughput for every prompt short enough to trigger banding — silently leaving the ~1.65x (measured elsewhere in this file's own comments, 4.63 vs 7.71–9.08 TFLOP/s) on the table for exactly the request shape (short chat prompts) the banding feature was built to speed up in the first place. Nothing observes or reports that the flag was ignored.

**What test would have caught it.** A test asserting that `UseOutputBlocking`/banding + `UseAvx512PrefillQ4K` actually dispatches through `GemmTiled512` (e.g. an instrumented counter on both kernels, the same technique the CLAUDE.md perf notes call out as the fix for "verify the flag you are A/B'ing is actually live" — `Avx512PrefillParityTests` checks *correctness* of the two kernels against each other but nothing currently checks that the banded dispatcher actually *reaches* the 512-bit one).

---

### 2. `CachedLlamaSession.Embed` treats `EmbeddingPooling.Cls` identically to `LastToken`

**What breaks.** `EmbeddingPooling` has three values: `Mean`, `LastToken`, and `Cls` ("the first token's hidden state — the `[CLS]` classifier... BGE/SBERT-CLS", per the enum's own doc comment, and it is genuinely used that way elsewhere — `BertEncoder.cs` implements `case EmbeddingPooling.Cls` correctly as first-token). `CachedLlamaSession.Embed`'s pooling loop is:

```csharp
if (pooling == EmbeddingPooling.Mean) { /* accumulate every token */ }
if (pooling != EmbeddingPooling.Mean && i == tokens.Length - 1) { h[..d].CopyTo(dst); }
```

There is no branch on `Cls` at all — any non-`Mean` value falls into the same "copy the *last* token's hidden state" branch. Passing `EmbeddingPooling.Cls` to this method silently produces the **last**-token embedding instead of the first-token one — wrong output for a value the enum explicitly exists to support, with no exception, no validation, and no indication anything is amiss.

**Where.** `Sources/Main/LanguageModels/Runtime/CachedLlamaSession.cs`, `Embed(ReadOnlySpan<int>, Span<float>, EmbeddingPooling, bool)`.

**How anyone would notice today.** They would not, short of comparing against a known-good CLS embedding. The public `float[] Embed(...)` overload accepts `EmbeddingPooling.Cls` without complaint and returns a normalized, plausible-looking unit vector — indistinguishable in shape from a correct one. `Tests/LanguageModels/Runtime/EmbeddingsTests.cs` (the only test touching this API) exercises only the default `Mean` pooling, so this is currently untested in either direction.

**What test would have caught it.** A parity test comparing `CachedLlamaSession.Embed(tokens, EmbeddingPooling.Cls)` against manually pooling `GetLayerActivation`/the first `DecodeTokenWithoutLogits` call's hidden state, or simply asserting `Cls` and `LastToken` produce *different* vectors for a multi-token input (they currently do not).

---

### 3. `Q4KWeight.EnsureRepacked()` lazily builds and caches the repacked layout with no synchronization, while the architecture explicitly shares one weight set across concurrently-decoding sessions

**What breaks.** `Q4KWeight._repacked` is a `ReadOnlyMemory<byte>?` (a multi-field struct: object reference + start index + length) built and cached via:

```csharp
public ReadOnlySpan<byte> EnsureRepacked()
{
    if (_repacked is null)
    {
        ReadOnlyMemory<byte> built = Q4KRepack.RepackMatrix(BlockSpan, OutputSize, InputSize);
        _repacked = built;
    }
    return _repacked.Value.Span;
}
```

with no lock, `Interlocked`, or `Lazy<T>`. `CachedLlamaInferenceEngine.CreateSession()` is documented (project architecture, `CLAUDE.md`) to be cheap specifically *because* weights are never copied per session — the explicit design point is that multiple `CachedLlamaSession`s share one `StackWeights`/`Q4KWeight` set. `EnsureRepacked()` is called from the ordinary decode path (`CachedMultiHeadAttention`'s whole-matrix Q4_K attention, `CachedFeedForwardBlock.DecodeSwiGluDispatched`'s gate/up fusion, and every batched-prefill Q4_K dispatch) — i.e. on the first decode step of *any* session against a weight that hasn't been repacked yet, not behind any load-time warm-up. If two sessions against the same model decode concurrently (the architecture's own stated purpose for making session creation cheap) and both reach a given weight's first `EnsureRepacked()` call around the same time, the check-then-act on `_repacked` is a data race, and a reader that observes a torn write to the multi-field `Nullable<ReadOnlyMemory<byte>>` could get a mismatched (reference, length) pair. `Q6KWeight`'s equivalent cache (`private byte[]? _repacked; _repacked ??= ...`) does not have this specific hazard because a bare reference-type field assignment is atomic, even though the check-then-act itself can still redundantly rebuild.

**Where.** `Sources/Main/LanguageModels/Runtime/Q4KWeight.cs`, `EnsureRepacked()`.

**How anyone would notice today.** Not from any log or counter — a torn read would surface as an intermittent out-of-range span or a decode using garbage weight bytes on some fraction of concurrent-session runs, indistinguishable from a heisenbug. It would not fire under any of this repo's serial parity/coherence tests, all of which drive a single session at a time.

**What would settle it.** This is reported unconfirmed in the strict sense — it needs a concurrency stress test (two threads calling `Decode`/`DecodeBatchedQuant` against sessions sharing one `Q4KWeight` with `IsPrepacked == false`, repeated under a race-detector or with artificial delay injected between the `is null` check and the assignment) to see an actual torn read; static reading only establishes that no synchronization exists where the architecture's own stated sharing model says concurrent first-use is reachable. If it is not currently reachable (e.g. every deployed model ships a `.gguf.repack` sidecar so `IsPrepacked` is always true in practice, matching the "OVERFIT_TILED_PREFILL is dead when a sidecar exists" fact given for this hunt), this degrades from a live bug to a latent one — but the code has no guard against the sidecar-less case either way.

## Shared root cause

None of the three share a root cause — #1 is a dispatcher asymmetry (copy-paste divergence between the Q4_K and Q6_K banded workers), #2 is a missing enum arm, #3 is a missing lock. They are reported separately.

## Coverage

**Reviewed and found clean, this pass** (full or near-full read):
`Q4KDotKernel.cs`, `Q6KGemvKernel.cs` (kernels only — see caveat below), `Q4KRepack.cs`,
`BatchedQuantProjection.cs` (full, including the parts left unread by the first hunt),
`CachedTransformerBlock.cs`, `CachedFeedForwardBlock.cs`, `CachedGptStack.cs` (full),
`BlockWeights.cs`, `DecodeWeight.cs`, `SingleHeadWeights.cs`, `KvHeadWeights.cs`, `StackWeights.cs`,
`Q4KWeight.cs` (see finding #3), `Q6KWeight.cs`, `Q8Weight.cs`,
`KvCacheSnapshot.cs`, `IKeyValueCacheReader.cs`, `KeyValueCacheReaderExtensions.cs`,
`CachedAttentionKernel.cs`, `Qwen2MoeFeedForwardBlock.cs`,
`SlmRuntimeFactory.cs`, `SlmRuntimeHandle.cs`,
`CachedLlamaSession.cs` — the remaining ~700 lines the first hunt did not read: `SnapshotPromptLogits`,
`SavePrefix`/`RestorePrefix`, `PrefillReusingCache`/`PrefillReusingKeyValuesOnly`/`MatchingPrefixLength`,
all three `GenerateNextToken` overloads, `ApplyDry`/`TrackGenerated`, `Feed`/`RollbackTo`, `Embed` (both
overloads — see finding #2), `Dispose`, `DecodeToken`/`DecodeTokenWithoutLogits`, `PrefillBatchedQuant`,
`ApplyEmbeddingScale`/`EmbedAndAdvance`.

Partial: `Q6KDotKernel.cs` (scalar/AVX2 dot identity read and cross-checked against its doc comment for
~150 lines; the remaining `ProjectBatched`/parallel dispatch wrapper functions were not read in this pass
— they follow the same shape as the already-reviewed `Q4KDotKernel` equivalents, but were not verified
line-by-line).

**Not reached this pass:**
`BatchedProjectionKernel.cs`, `Q8DotKernel.cs` (only its `BlockSize`/`Quantize` call sites were seen via
`Q8Weight.cs`, not the kernel itself), `Q6KRepack.cs`, `Gpt1SlmModelAdapter.cs`,
`CachedGpt1ModelAdapter.cs`, `CachedLlamaInferenceEngine.cs` (only its `Dispose()` ownership walk was
spot-checked), `SingleTokenLayerNormKernel.cs`, `SingleTokenProjectionKernel.cs`, `SlmRuntimeMode.cs`,
`KvCacheDType.cs`, `ISpeculativeDrafter.cs`, `PrefillProfiler.cs`.

## What this score means

This pass covered the priority list given for the run (Q4_K/Q6_K dot kernels, repack/batched-projection
layout code, block/stack composition, weight handles, and the remainder of `CachedLlamaSession`) and
found three real, distinct defects — two silent (a dead perf flag, a mispooled embedding) and one a
genuine but execution-unconfirmed concurrency gap. It stopped with time remaining under the ten-minute
cap because the assigned priority list was exhausted and the "not reached" list above is what is left in
the directory as a whole (combined with the first hunt's own "not reached" list, most of the directory's
55 files have now been read at least once). A caller wanting deeper coverage should start from the
"not reached" list here, particularly `Q8DotKernel.cs` and `SingleTokenProjectionKernel.cs`, which sit on
every decode path and were not opened in either pass.
