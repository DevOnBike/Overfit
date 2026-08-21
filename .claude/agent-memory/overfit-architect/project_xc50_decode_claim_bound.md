---
name: xc50-decode-claim-bound
description: XC-50 (2026-08-14) — decode-pool claim's exhaustion bound packed into the claim word; why the reorder+fence option was rejected, and the correction it forces on the signed XC-49 plan
metadata:
  type: project
---

# XC-50 — the exhaustion bound must ride in the claim word

Plan: `docs/specs/xc-50-decode-claim-bound-generation-tag-plan.md`, written and **SIGNED / STATUS: APPROVED**
by me on 2026-08-14 (no analyst round — I found the defect myself while signing `XC-49`).

**Defect.** `OverfitParallel.ForDecode` publishes `_decodeChunkCount` (`:626`, plain store) before the claim
word carries the new generation (`:644`), so `TryClaimDecodeChunk` checks its tag against one generation and
its bound (`:718`) against another. A straggler's final expected-to-fail claim then succeeds when the next
dispatch is larger. Loud outcome: torn `Body`/`Context`. Quiet outcome: the real chunk never executes and
`ForDecode` returns a partly-written output buffer with no exception.

**Invariant chosen (one sentence).** *Every input to "may this generation take this index" is carried in the
single word the CAS operates on; a claim consults no other mutable state.*

**Decision: pack `(gen tag 32 | count 16 | index 16)` into `_decodeClaim`, delete `_decodeChunkCount`, and
move the claim into its own `internal static` type** so the dispatcher's `private` fields are unreachable from
it (compile-time enforcement, not a review habit). Index must stay in the low bits so `current + 1` remains
the claim; the count bound is established once where `_decodePoolSize` is resolved, never per dispatch.

**Rejected: reorder the count store after the claim publish + make it `Volatile.Write` (2 lines, and it IS
sufficient).** Rejected on **testability**, not elegance: the claim function is unchanged under it, so no test
can distinguish fixed from broken — the property is "these two stores are ordered", which managed code cannot
assert. The ordering argument in this exact method has now been wrong twice (`:278-288`, `:641-643`), both
times reviewed and signed.

**Also rejected (out of scope):** merging `_decodeGen` into the claim word — it rewrites the park protocol
whose coverage gap is open (`TG-T13`; 14.85 idle cores was its last defect).

**Sweep result.** Only two lock-free CAS protocols exist in `Sources/Main` (`Grep Interlocked.CompareExchange`):
this one and `OverfitResourcePool.cs:169` (max-CAS, own word only). The **main** pool is NOT the same shape —
a `SemaphoreSlim` token is consumed before the claim and `_completion` counts every token, so no straggler can
survive into the next dispatch. `_decodeRemaining`'s early reset is safe, but only *as a consequence* of the
claim invariant — never written down before; the plan requires it be stated at the site.

**Correction I owe to a plan I signed.** `XC-49` §A.5 row 4 and §A.6 say "the seam is invariant under any fix
to A3". True for the reorder option, **false for the packed word** (the `chunkCount` parameter disappears).
Also: `XC-49` AC1.1 hand-seeds the word and therefore **would not have caught XC-50** — *a hand-seeded state
cannot catch a publication defect; the test must call the publisher.* Recommended: land XC-50 first and fold
XC-49's seam into its step 1. I did **not** edit the XC-49 file (signed, and not the file I was given).

**Amendment 2026-08-14 (§13) — my `Won't` forbade by wording, not by reasoning, and it had already leaked
into production code** (`DecodeChunkClaim.cs:147-149` cited "the XC-50 plan §9" as the reason a branch is
untested). The rule I should have written the first time:

> **A concurrency test is admissible iff every assertion holds under every legal schedule.** `TG-T12`/`TG-T13`
> are cautionary because they assert an environment-produced quantity (time, CPU), not because they use
> threads. Load then changes only which interleavings are sampled, never the verdict.

Second question every such test must answer *before* it is written: **when the property is violated, is the
result red or hung?** Red → admissible. Hung → it needs a bound, and if the bound would have to live in
production (here: a progress deadline in `ForDecode`'s untimed completion spin, which sits inside
`_decodeGate`, so a hang poisons every later decode), that is its own task.
Probabilistic race tests: a red is a true finding, a green proves nothing — so never coverage, and **never the
predicted victim of a mutation** (an unreadable mutation result is worse than none).

**Two arithmetic corrections against me, 2026-08-14 — both were claims I wrote without evaluating.**
1. I named `0xFFFF_FFFF` / `0x1_0000_0000` as the colliding tag pair. **They do not collide**
   (`(uint)0x1_0000_0000 == 0`). Collisions happen iff generations differ by a **multiple of 2^32** — `G` and
   `G + 2^32`. It propagated from my plan into a brief and into the client conversation before the developer
   caught it. **Evaluate a bit-level claim before writing it down; a modulus is exact where a "days to
   months" estimate is not.** (The wrong pair survived usefully as the *non*-collision assertion.)
2. My §4.1 said an over-wide count fails as "silent truncation". Understated: `((long)(-1) << 16)` is
   `0xFFFF_FFFF_FFFF_0000`, so a **negative** count smears sign bits into the TAG field and the claim is then
   refused by the generation check, not the bound. So: clamp on both sides, and **a test asserting "refused"
   must say WHICH guard refused**, or it passes while pinning nothing.

**Proof shape worth reusing.** Red-first beats a manufactured mutation when the pre-fix code is available:
extract the protocol as-is, drive the exact state, observe `true` where `false` is required. Then the fix gets
a compilable one-expression mutation (publish the new count while leaving the tag) whose victim is named
before the run.
