---
name: xc52-dispatcher-invariant
description: XC-52 (2026-08-14) — decode dispatcher's untested invariant; why ForDecode's completion spin can never carry a throwing deadline (memory safety, not threshold choice), and the capability gate a client-level decode test needs
metadata:
  type: project
---

# XC-52 — the dispatcher's invariant, and the deadline that cannot exist

Plan: `docs/specs/xc-52-decode-dispatcher-invariant-tests-plan.md`, written and **SIGNED / STATUS: APPROVED**
by me on 2026-08-14. Test-only task (plus two `internal` accessors and three comment amendments); I
specified it myself in the `XC-50` plan §14.

## The decision that generalises: an early return from `ForDecode` is memory-unsafe

**Verified by reading all 8 production call sites** (`find_references`, re-run 2026-08-14): every one passes
a **stack local** as `context`, built inside a `fixed` block —
`Q4KGemvKernel.cs:139-148` (`&ctx`), `CachedMultiHeadAttention.cs:253-281`
(`Unsafe.AsPointer(ref context)`, and that struct also holds **managed references**). The pointers inside
those contexts are `fixed` pins that expire with the block.

So the completion spin (`OverfitParallel.cs:725-728`, inside `lock (_decodeGate)`) is not merely a wait — it
is **what enforces the API's contract that `context` stays valid for the call**. Any deadline that *returns*
(throwing, or "fail the pool and return") pops the frame and releases the pins while a worker may still
write through them: a hang becomes silent heap corruption. **The threshold question is never reached.**
Ranked second and still defensible: `Environment.FailFast` (not banned, used nowhere in `Sources/`) — it
never returns, so it cannot corrupt; rejected because the threshold is unjustifiable and killing a
customer's process is a client call. Recorded as a non-blocking question with "no" as the default.

**Why no measurement settles it:** the bounding quantity is not the dispatch's work but the maximum time an
already-claimed worker can be descheduled — CFS throttling per 100 ms period, VM steal, GC pause, a
breakpoint. Unbounded. A threshold surviving those is minutes, and a deadline firing after minutes buys
nothing over `XC-53`'s `--blame-hang`, which costs nothing and touches no hot path.

## The invariant was stated with the wrong extent (my own sentence, corrected)

"`_decodeRemaining == 0` implies no worker is still inside `ExecuteDecodeChunk`" is false in the strict
reading — the decrement is in the `finally`, so a worker is inside the method for the epilogue after it,
touching nothing shared. The testable property is **"when `ForDecode` returns, no worker will read or write
that dispatch's descriptor, `Body`, `Context`, or anything the context points at."**

**Why the client-level assertion is not vacuous:** the test's in-flight counter decrements at the end of the
**body**, strictly before `ExecuteDecodeChunk`'s decrement — so a green is *implied* by the invariant and a
red can only mean a body ran that was never in `_decodeRemaining`'s accounting, i.e. a straggler. That
targeting argument is worth reusing whenever instrumenting a body to test a dispatcher.

## Capability before verdict — a decode test can be green without dispatching

`_decodePoolSize = min(max(1, ProcessorCount − 1), 10)` → **10** here (PC=32, measured), **3** on a 4-vCPU
runner, **1** on a 2-vCPU box — and at 1, or with `OVERFIT_DECODE_POOL=0`, `ForDecode` runs **inline**
(`:625-646`) and every assertion passes without a dispatcher. Fix: `internal static` read-only accessors for
the resolved pool size / enabled flag (`DecodeMaxWorkers` is NOT a substitute — settable public property the
private field does not track) plus a dynamic skip. **`Assert.Skip`/`SkipWhen`/`SkipUnless`/`SkipException`
strings are present in `xunit.v3.assert.dll` 3.2.2** (checked the `#Strings` heap; indicative, not a
signature proof).

## Two facts about this repo's test infrastructure, both checked 2026-08-14

- **`coverlet.runsettings` does NOT exclude `DevOnBike.Overfit.Runtime`** (it excludes Ops, Kernels, Maths,
  Intrinsics, Autograd, Optimizers, Tensors, LanguageModels.Runtime). So `OverfitParallel` and
  `DecodeChunkClaim` run **instrumented** on the Linux CI job — 10-900× on hot loops.
- **Nothing disables xunit parallelism** (no `xunit.runner.json`, no `CollectionBehavior` anywhere in
  `Tests`), so classes run in parallel and neighbours dispatch decode concurrently.

## The ratio rule, added to the admissibility rule

`TG-T12` is cautionary because it asserts a bound **of the same order as the quantity it measures**. A join
backstop three or more orders above a legitimate value cannot be moved by load without the run already being
broken — so it is admissible **as a backstop, reported as one**, never as an assertion. Allowed in a
`[LongFact]` soak (a soak that stops at iteration 40 000 with no name is unusable); refused in the fast
suite, where `XC-53` is the backstop by design.

## Navigator caveat

`find_references(OverfitParallel.ForDecode)` reported the declaration at `OverfitParallel.cs:556`; the file
has it at **`:619`**. The file set was right, the line numbers were stale — confirm every line number with
`Read` before citing it.
