---
name: xc49-decode-claim-seam
description: XC-49 decode-pool claim-protocol test seam — signed APPROVED 2026-08-14; why a state-writing test seam on OverfitParallel is a hang/crash hazard, and the residual publication-order race found in the fix
metadata:
  type: project
---

**XC-49 signed APPROVED 2026-08-14** (`docs/specs/xc-49-decode-pool-claim-race-test-plan.md`, §A).
Overrode the analyst's recommended seam.

**The general rule this run established: a test seam that WRITES process-global state of a live static pool
is not made safe by "nothing wakes the pool".** The analyst's F3 argued a test seeding `_decodeClaim` /
`_decodeChunkCount` (but never `_decodeGen`) was immune by construction. False — the pool is already awake
because other test classes decode concurrently:

- `Tests/LanguageModels/Runtime/Q4KDotKernelTests.cs:67` `ProjectParallel_IsBitIdenticalToProject` is a plain
  `[Fact]`, `outputSize 96` → real `ForDecode` fan-out. `CachedMultiHeadAttentionTests` has 8 plain `[Fact]`s
  reaching it transitively; likewise `CachedTransformerBlockTests`, `MoeFeedForwardBlockTests`,
  `Qwen2MoeFeedForwardBlockTests`, `CachedFeedForwardBlockBatchedTests`. None fixture-gated.
- **Test classes run in parallel here**: no `xunit.runner.json` in the tree, no `CollectionBehavior` /
  `DisableTestParallelization` in `Tests/`, no parallelism property in `Tests.csproj`; `xunit.v3` 3.2.2
  (`Directory.Packages.props:164`) defaults to parallel collections, one per class. Verified statically, not
  by watching a run.
- Worst outcome is **not** a flaky test: `ForDecode`'s completion wait (`OverfitParallel.cs:665-668`) is a
  **pure spin with no timeout, inside `lock (_decodeGate)`** — clobbering the count or the claim word mid
  dispatch hangs the whole test host. Raising `_decodeChunkCount` above the live dispatch's makes workers
  call `_decodeChunks[i].Body`, a stale **unmanaged function pointer** (`:736`) → process crash.

**Decision: parameterise the protocol on its state** (claim word by `ref long`, chunk count and generation as
parameters) so the test drives a local. Smaller than the type extraction that was rejected, avoids `CS0052`
on the private `PaddedClaim`, and is inert by construction — no doc comment to keep honouring.
`internal … ForTest` accessors that *write* live state were rejected; `StackWeights.ForTest` is not a
precedent for them (it is a pure factory). Its `#pragma OVERFIT001` is the **heap-array-allocation** rule
(`Sources/Analyzers/HeapArrayAllocationAnalyzer.cs:30`) — do not copy the pragma with the shape.

**Residual race found in the just-fixed code, NOT fixed (read-only), routed to its own row.**
`ForDecode` writes `_decodeChunkCount` at `:626` — before the descriptors (`:629-639`) and before the claim
word is republished (`:644`). In that window the count belongs to generation `G+1` while the claim word still
belongs to `G`, so a straggler tagged `G` passes the tag check, sees `oldNext < newCount`, CASes successfully
and executes an unpopulated descriptor while decrementing `G+1`'s completion counter. **The claim word carries
its generation; the exhaustion bound does not.** Reachable because completion counts *chunks executed*, not
*workers that left the drain loop*. Hand-traced only.

Related: `TG-T13` (decode-pool idle-burn test measures the whole process — same "global state, other tests"
shape, on the same pool), `PB-12` (this path's throughput numbers predate the 2026-08-14 fix; any perf claim
about the seam belongs there, not to XC-49).
