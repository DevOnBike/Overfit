STATUS: APPROVED
Author: overfit-architect (both halves — see §0)
Architecture review: overfit-architect, 2026-08-14 — SIGNED
Date: 2026-08-14
Slug: xc-52-decode-dispatcher-invariant-tests-plan

GATES:
  performance:        NOT_REQUIRED — no performance target, claim or comparison enters this plan, and the
                       decided answer to (c) adds nothing to the hot path. If the developer produces a
                       dispatch DURATION for any reason, it is not a performance claim but it can be read as
                       one: say so in the report and hand the verdict to `overfit-perf-claim-auditor`. This
                       path's published figures are `PB-12` and stay there
  security:           NOT_REQUIRED — no parser, endpoint, gateway, audio/tokenizer decode or externally-fed
                       surface. The `void*` contexts involved originate in first-party kernels and in the
                       new test's own memory
  leak-scan:          NOT_REQUIRED — no config, log, fixture, host name, path or token touched
  AOT:                NOT_REQUIRED — `Tests/AotSmokeTest/Program.cs` read this run: 47 lines, no mention of
                       `OverfitParallel` or the `Runtime` namespace, so `ForDecode` is not reachable. This
                       plan adds no reflection, LINQ, `Activator` or `Expression`, and its only production
                       addition is an `internal` read-only property
  API-compatibility:  NOT_REQUIRED — nothing public is added, removed or changed. `ForDecode` is already
                       `public` and its signature is untouched; the diagnostic accessor is `internal`
  release-readiness:  NOT_REQUIRED — test-only change plus comment amendments. No shipped behaviour, no
                       packaging, no dependency, no version-affecting surface

`verifier`, `reviewer` and `mutation-proof` are **intentionally absent rather than `NOT_REQUIRED`**: a
missing line means "not yet asked", and it is genuinely too early — nothing is implemented. All three apply
once the work lands, and `mutation-proof` is not optional here, because §8's mutation table *is* the
acceptance criterion. The manifest's vocabulary (`Scripts/plan_gate_check.py:31`) has no token for
"required, not yet run", and `INCONCLUSIVE` would falsely imply it ran.

---

# XC-52 — the decode dispatcher's central invariant is a comment; give it a falsifier

Plan file for task `XC-52` (`docs/TASKS.md:183`). Specified by me in
`docs/specs/xc-50-decode-claim-bound-generation-tag-plan.md` §14 and filed verbatim in substance.

## 0. Why the architect wrote both halves, and whether the problem statement is settled

I specified this task, so there is no analysis for an analyst to do that is not already mine. I was asked to
stop if the problem statement turned out to be unsettled. **It is settled in substance and wrong in one
detail**, which §1 corrects rather than sends back: the invariant as written names the wrong extent. That is
a correction to my own sentence, not a missing business requirement, and it does not change what the task is
for. No business rule, acceptance threshold or value judgement is missing, so there is nothing to send to a
client and nothing to stop for.

## 1. The invariant, restated so it can be falsified

**The row says**: *`_decodeRemaining == 0` implies no worker is still inside `ExecuteDecodeChunk`.*

**Finding 1 — that is not the property, and it is not observable.** `ExecuteDecodeChunk`
(`OverfitParallel.cs:751-766`) decrements in its `finally`, so a worker *is* still inside the method for the
few instructions between the decrement and the return — and in that window it touches nothing shared. The
property the design actually rests on, and the only one worth testing, is:

> **When `ForDecode` returns, no worker will subsequently read or write that dispatch's descriptor slot,
> its `Body`, its `Context`, or anything the context points at.**

That is what makes overwriting `_decodeChunks[]` on the next dispatch safe, and it is what makes the API's
contract — `context` is caller-owned and valid for the duration of the call — true. It is observable from a
client: instrument the body's entry and exit.

**Why an assertion on it is not vacuous.** The test's own in-flight counter is decremented at the end of the
**body**, which is strictly before `ExecuteDecodeChunk`'s decrement. So if the invariant holds, "in-flight
== 0 after `ForDecode` returns" is *implied* and can never fire. It can only fire when a body ran that was
never accounted for in `_decodeRemaining` — a straggler executing a descriptor that is not its own. That is
exactly the defect class (`XC-50`'s quiet outcome, and the torn `Body`/`Context` pair recorded at
`OverfitParallel.cs:278-291`), and nothing else.

## 2. Inventory — what exists, and how each line was established this run

| what | evidence |
|---|---|
| `ForDecode` has **8** production call sites | `find_references(OverfitParallel.ForDecode)`, re-run this session: `CachedMultiHeadAttention.cs` ×2, `Q4KDotKernel.cs` ×2, `Q4KGemvKernel.cs`, `Q6KDotKernel.cs`, `Q6KGemvKernel.cs`, `Q8DotKernel.cs`. **The navigator's index is stale on line numbers** — it reports the declaration at `:556` where the file has it at `:619` — so treat its line numbers as hints and its file set as the result |
| Every one of those call sites passes a **stack local** as `context` | Read, not inferred. `Q4KGemvKernel.GemvParallel` builds `var ctx = new GemvContext{…}` inside a `fixed` block and passes `&ctx` (`Q4KGemvKernel.cs:139-148`); `CachedMultiHeadAttention.Decode` builds `var context = new HeadDecodeContext{…}` inside a `fixed` block and passes `Unsafe.AsPointer(ref context)` (`:253-281`). The pointers *inside* those contexts are `fixed` pins that expire with the block, and `HeadDecodeContext` additionally carries managed references reported by that frame. **This is the fact that decides (c)** |
| No test calls `ForDecode` outside `Diagnostics` | `git grep ForDecode -- Tests`: the only call sites are `AttentionQ4KRepackHypothesisTests.cs:145,151`; every other hit is prose |
| Many fast-suite `[Fact]`s reach `ForDecode` **transitively** | `find_callers(ForDecode, depth 4)` — 73 call sites, including plain `[Fact]`s in `CachedMultiHeadAttentionTests`, `CachedTransformerBlockTests`, `Q4KDotKernelTests`, `Q8DotKernelTests`, `Q6KDotKernelTests`, `MoeFeedForwardBlockTests`, `Qwen2MoeFeedForwardBlockTests`, `CachedGptStackTests`. Whether each reaches the *dispatch* or the inline fast path depends on its dimensions |
| Nothing disables xunit parallelism | `Grep` for `CollectionBehavior` / `DisableTestParallelization` / `MaxParallelThreads` across `Tests` (cs, json, csproj): no match; no `xunit.runner.json` exists. xunit.v3 3.2.2 therefore parallelises collections by default — **the new tests will run beside other decode dispatches** |
| The claim protocol *is* covered; the dispatcher is not | `Tests/Core/Runtime/DecodeChunkClaimTests.cs` (7 `[Fact]`s) and `DecodeChunkClaimConcurrencyTests.cs` (1). Both drive `DecodeChunkClaim` over a word the test owns. Neither dispatches |
| `_decodePoolSize` on this box is **10** | `min(max(1, ProcessorCount − 1), 10)` (`ResolveDecodeMaxWorkers`, `:218`, then `ClampDecodePoolSize`), with `ProcessorCount = 32` measured this run. On a 4-vCPU GitHub runner it is **3**; on a 2-vCPU box it is **1**, and `ForDecode` then runs inline (`:642`) |
| The Linux CI job runs this code **instrumented** | `coverlet.runsettings` excludes `Ops`, `Kernels`, `Maths`, `Intrinsics`, `Autograd`, `Optimizers`, `Tensors` and `LanguageModels.Runtime` — **not** `DevOnBike.Overfit.Runtime`, where `OverfitParallel` lives. `CLAUDE.md` puts instrumentation at 10–900× on hot loops |
| CI has no hang backstop | `.github/workflows/ci.yml:52` and `:61` — `dotnet test` with neither `--blame-hang` nor `timeout-minutes`. This is `XC-53` |
| The local mutation harness does have one | `.claude/skills/overfit-mutate/SKILL.md:103-111` passes `--blame-hang --blame-hang-timeout 5m` |
| xunit.v3 3.2.2 can skip dynamically | `Skip`, `SkipWhen`, `SkipUnless`, `DynamicSkipToken` and `SkipException` are all present in the `#Strings` heap of `xunit.v3.assert.dll` 3.2.2. Indicative of the API's presence, **not** proof of a particular signature — the developer confirms at implementation |

## 3. (c) — the decision, and it is not a threshold question

**Decision: `ForDecode`'s completion spin gets NO progress deadline.** The status quo stands; the legibility
of a hang is bought by `XC-53` (a runner flag) and by the tests' own reporting, at zero hot-path cost.

**Finding 2 — the row frames (c) as needing a measured worst-case dispatch duration before a threshold can
be chosen. The threshold question is never reached, because the throwing form is disqualified one step
earlier, on memory safety.**

`ForDecode`'s contract is that `context` and everything it points at stay valid for the duration of the
call, and the completion spin is the *only* thing that enforces it. At all 8 call sites (§2) the context is
a stack local inside a `fixed` block. **Returning early — by throwing or by any other route — pops that
frame and releases those pins while a worker may still be writing through them.** The consequences, in
order of nastiness:

1. The worker writes through `ctx.Output`, a `float*` into an array that is **no longer pinned**. If the GC
   has moved it, that is a write into unrelated heap memory. Silent corruption in the customer's process.
2. The worker re-reads `ctx` itself out of a stack frame the caller has already reused.
3. `HeadDecodeContext` holds **managed references** (`Heads`, `Weights`, `Cache`, `Rope`). Once the frame
   that reported them is gone, nothing keeps them alive on that path.

So a throwing deadline converts a hang into undefined behaviour. **That is strictly worse than the hang**:
the hang is loud, local and diagnosable; heap corruption is none of those. No threshold makes it safe,
which is why no measurement is required to reject it.

### 3.1 The options, ranked, with what each gives up

| | shape | verdict |
|---|---|---|
| **A. No deadline** (decided) | spin until `_decodeRemaining == 0`, as today | **Chosen.** Preserves the memory-safety contract. A liveness defect hangs, inside `_decodeGate`, poisoning every later decode in the process — the honest cost, mitigated by `XC-53` and by §4's reporting |
| **B. Deadline that throws** | after N, throw from the spin | **Rejected on memory safety**, above. Also unimplementable safely in a "fail-safe" variant: marking the pool permanently faulted still returns to the caller, which is the unsafe act |
| **C. Deadline that calls `Environment.FailFast`** | after a generous N, terminate with a dump, without unwinding | **Ranked second and genuinely defensible.** It is the only reaction that does not return, so it cannot corrupt anything, and once the spin fails to terminate the process is already dead (the gate is held forever). Rejected for now because the threshold is unjustifiable (§3.2) and a false positive kills a live production process — a library that can terminate its host is a product decision, not a technical one. `FailFast` is not banned and appears nowhere in `Sources/` today |
| **D. Debug-only assert** | `Debug.Fail` in the spin | **Rejected: inert.** This repository's tests run `-c Release` as a hard rule, so it would never execute where it is needed |
| **E. Deadline behind an internal test hook** | `internal static` opt-in, set by a test | **Rejected.** It is process-global mutable state on a path the parallel fast suite is concurrently using: a neighbour's legitimately slow dispatch trips the hook set by an unrelated test. And whatever the hook does on expiry is B or C, so it inherits their verdicts on top |
| **F. No production change; test-side reporting** | (a)/(b) name the hang before the run dies | **Adopted alongside A** — see §4.3 and §5 |

**What would reopen this**, so the decision is not permanent by default: (i) a stall observed in production
rather than in a mutation run — that would make C's false-positive cost concrete instead of hypothetical and
is the point at which the client should be asked; or (ii) a redesign in which the dispatch owns its context
(a copy, or a pinned pool-owned buffer) rather than borrowing the caller's frame, which removes finding 2's
premise and puts B back on the table. Neither is in this task.

### 3.2 The measurement a threshold would need, named — and why it cannot settle the question

If C is ever chosen, the number required is **not** the dispatch's work. It is the **maximum time a worker
that has already claimed a chunk can be descheduled**, because that is what bounds a legitimate dispatch.
The measurement would have to cover, at minimum:

- the slowest legitimate dispatch, not the median — the tail across model sizes (chunk work from a 2-head
  attention slice to a 12.7 MB FFN row band), worker counts (2, 3, 10, 31) and quantisations;
- a **loaded** box, not an idle one, since the pool is sized to leave exactly one core of headroom
  (`:208-218`) and the dispatcher spins;
- a container under a CFS quota, where throttling is decided per 100 ms period;
- and the instrumented build, since the Linux CI job runs this code under coverlet (§2).

**What would make the number untrustworthy, and does:** none of the above bounds the quantity. VM steal
time, a page fault on a swapped page, a stop-the-world GC pause, a hypervisor migration and a developer's
breakpoint inside a kernel body are all legitimate and all unbounded. A threshold survivable against those
is minutes, and a deadline that fires after minutes buys nothing over `XC-53`'s `--blame-hang`, which costs
nothing and touches no hot path. **A measurement here would be correct and would not decide anything** —
which is the same failure mode as sizing a CPU limit from average cores when CFS throttles on bursts.

I did **not** run this measurement. Under decision A it is not required; under C it is necessary but not
sufficient.

### 3.3 What follows for (a) and (b)

Under A, the honest answer to *"when the property is violated, is the result red or hung?"* is **both,
depending on which half breaks** — and the split must be written into the tests' own doc comments:

| violated property | result |
|---|---|
| a chunk executes twice / an index outside its range / a body paired with a foreign context | **red** (assertions in §4.2) |
| a dispatch returns while one of its bodies is still running | **red**, probabilistically — no false red, detection not guaranteed (§4.4) |
| a chunk is never claimed by anybody (`_decodeRemaining` never reaches 0) | **hung**, inside `_decodeGate`, poisoning the rest of the run. `XC-53` names it in CI; `--blame-hang` names it under the mutation harness today |
| a body dereferences a torn `Context` belonging to another dispatch | **crash or corruption**, not red — see finding 3 |

**Finding 3 — the row calls (a) fast-suite admissible "because it seeds no pool state and its assertions are
schedule-invariant". Both are true and both are incomplete.** Detecting a torn `(Body, Context)` pair
requires *dereferencing* the possibly-foreign pointer, and if the protocol regresses while a neighbouring
test class is dispatching, that pointer belongs to another test's frame. No test-side design removes this:
the fault the defect produces is a wild pointer. It is not a reason to skip the test — a crash with a dump
is a better outcome than silence — but it must be stated, and §4.2 constrains the body to make it as
survivable as it can be.

## 4. (a) — the sequential client-level test

**Fast suite, one `[Fact]`, no `Thread`, no `Task`, no `Sleep`, no timeout, no clock.** It calls `ForDecode`
exactly as production does and seeds no pool state.

### 4.1 Shape

Rounds of back-to-back dispatches on the calling thread, each with **its own body, its own context and its
own work count**. Two structural requirements:

1. **The counts must alternate small→large repeatedly** (e.g. 2, 10, 3, 10, 5, 64, 2, …). `XC-50`'s
   precondition is a dispatch with *more* chunks following one with fewer; a monotone or constant sequence
   never creates it. `chunkCount = Math.Min(_decodePoolSize, totalWork)`, so on a pool of 3 the reachable
   counts are 2 and 3 — enough, and the test must report which it got, because on a pool of 2 the
   precondition is unreachable and the run proves less than it looks.
2. **Every context and every buffer stays alive for the whole test** — one enclosing `fixed`/pinned scope,
   not one per dispatch. A straggler dereferencing a *stale* context then reads memory that is still valid
   and produces a detectable wrong write instead of an access violation. This is a deliberate weakening of
   realism in exchange for a readable failure, and it must be commented as such.

### 4.2 The assertions, and why each holds under every legal schedule

| assertion | why schedule-invariant |
|---|---|
| every index in `[rangeStart, rangeEnd)` of dispatch *d* is written exactly once, by a body of *d* | the dispatch is complete when `ForDecode` returns; chunk ranges partition the range; "exactly once" is a property of the partition and the claim, not of ordering. Counted with `Interlocked.Increment` per index |
| every value written into dispatch *d*'s buffer carries *d*'s stamp | the stamp is a compile-time constant of the body; a mismatch means a body ran against a context that is not its own, which is a protocol violation under any schedule |
| no index outside dispatch *d*'s range is written | same |
| after `ForDecode` returns, *d*'s in-flight body counter is 0 | implied by the invariant (§1); can only fail on an unaccounted body |
| after a dispatch whose body throws, the exception surfaces on the caller **and** the in-flight counter is 0 | `ExecuteDecodeChunk` captures into `Error` and still decrements in its `finally` (`:758-765`), and the spin precedes the rethrow (`:725-733`). The invariant covers the exception path, and nothing tests that today — `For_BodyThrows_…` covers the **main** pool only |

**Bounds-check inside the body, always.** Every body validates the index against its own context's length
*before* writing, and records a violation flag instead of writing when it fails. The failure being tested is
precisely a mismatched `(Body, Context)` pair, and an unchecked write through a mismatched pointer corrupts
the heap instead of reddening a test.

### 4.3 Capability — a green must not be reachable without dispatching

**Finding 4.** On a box where `_decodePoolSize <= 1`, or with `OVERFIT_DECODE_POOL=0`, `ForDecode` runs the
body inline (`:625-646`) and every assertion above passes without a dispatcher ever running. A test that
cannot tell whether it exercised its subject is the two-arms rule's *capability before verdict*.

So: **the test gates on capability and skips dynamically when it is absent** (`Assert.Skip` — present in
xunit.v3 3.2.2, §2; the developer confirms the signature). A skip is visibly not a pass, which is the whole
point; failing instead would be a false red on a legitimate 2-vCPU runner. It also reports the pool size and
the count sequence it actually achieved.

This needs the resolved pool size to be readable. **Add `internal static int DecodePoolSize => _decodePoolSize;`
and `internal static bool DecodePoolEnabled => _decodePool;`** — read-only, no new mutable state, no public
surface, `InternalsVisibleTo` already covers `Tests`. `DecodeMaxWorkers` is *not* a substitute: it is a
settable public property that `_decodePoolSize` does not track.

### 4.4 Falsification power without a clock

To widen the window in which "returns while a chunk is still running" is observable, **make the work uneven
by a deterministic amount** — one chunk of each round does a fixed, large number of arithmetic iterations
while the others do a handful. That changes which interleavings are sampled, never a verdict, so it is
admissible; a `Sleep` or a spin-until-time would not be.

**Budget.** Total under one second uninstrumented, and the developer keeps the per-round work small enough
that the coverlet-instrumented Linux job stays sane (§2): tens of rounds, ranges ≤ 64, and the "slow" chunk
sized in iterations, then measured once and trimmed if the test exceeds the budget.

## 5. (b) — the concurrent-caller soak

`[LongFact]`, in its own class, **labelled a falsification attempt in its name and in its doc comment**.

- N test-owned threads, each looping its own dispatches with its own bodies, contexts and buffers; every
  assertion in §4.2 evaluated per dispatch and per thread. `_decodeGate` serialises the dispatches, so what
  this varies is *which thread* publishes next and how a previous dispatcher's stragglers line up with it.
- **A green proves nothing and may never be cited as coverage; a red is a true finding** (the assertions are
  schedule-invariant, so there are no false positives).
- **It must never be a mutation's predicted victim** — a probabilistic victim makes a mutation result
  unreadable.
- **A join backstop is permitted here and only here**, because a soak that stops at iteration 40 000 with no
  name is unusable. It is a **backstop, not an assertion**: the bound must sit several orders of magnitude
  above a legitimate dispatch, and the failure message must say that it means "did not complete — a liveness
  defect or a frozen box", not "the protocol is violated". The distinction from `TG-T12` is the ratio, and
  it is worth stating as a rule: *`TG-T12` asserts a bound of the same order as the quantity it measures; a
  backstop three or more orders above it cannot be moved by load without the run already being broken.*
- (a) carries **no** such construct: it runs in CI, where `XC-53` is the backstop by design.

## 6. Boundaries and gate answers

| | |
|---|---|
| **Execution path** | **Inference.** The decode dispatch, on the path taken by every quantised GEMV/projection kernel and both `CachedMultiHeadAttention` decode paths |
| **Assembly** | `Tests` for everything new; `Sources/Main` only for two `internal` read-only properties and three comment amendments. No project reference, no dependency |
| **Public surface** | **Unchanged.** `ForDecode`'s signature is untouched; the accessors are `internal` |
| **AOT reach** | Not reachable (gate line). Unchanged |
| **Allocation policy** | Hot path, zero allocations per call — **and this plan must not perturb it**. Nothing is added to `ForDecode`, `ExecuteDecodeChunk`, `TryClaimDecodeChunk` or `DecodeChunkClaim`. The tests allocate freely; they are tests |
| **Ownership / disposal** | No buffer, no `AutogradNode`, no `PooledBuffer<T>`. The test owns its contexts and buffers and keeps them alive for the whole test by design (§4.1) |
| **Threading model** | Unchanged, and must stay unchanged: decode spin pool for decode, `Parallel.For` elsewhere. `docs/measured-baselines.md:375` records **455 µs / 0 B** vs `Parallel.For`'s **2059 µs / 925 KB** on decode, and `:349` the opposite result in `Conv2D`. Cited, not re-verified, and not a target of this plan |
| **Moat side** | Neither. No real-time, GPU or throughput claim is made or implied |
| **Verification oracle** | The dispatcher's own invariant (§1), proven by mutation: §8 names, for each mutation, the predicted victim **and** whether the predicted outcome is red or hung. There is no external reference implementation and none is needed |

**No ADR.** No public API, no assembly placement of a public type, no change to AOT reachability, no on-disk
or on-wire format, no open/commercial boundary, no dependency added to `Main`. The reasoning lives here and,
for (c), at the site (§7).

**Operability.** Library code; nothing runs, nothing is durable, nothing restarts. The nearest operational
question — *what does a user see when this goes wrong* — is answered by §3.3, and the answer for the
liveness half is "the process stops", which is why `XC-53` exists.

## 7. Scope

**Must**
- (a) as specified in §4, including the throwing-body dispatch and the capability gate.
- The two `internal` read-only accessors (§4.3).
- (b) as specified in §5.
- **The invariant comment at `OverfitParallel.cs:655-673` amended to §1's wording and to name the test that
  exercises it.** An invariant recorded as a consequence of another one, with nothing attempting to falsify
  it, is a comment and not a gate — that is what this whole task is about, and leaving the comment saying
  what it says now would reproduce the defect in place.
- **`DecodeChunkClaim.cs:163-169` amended.** It currently says the deadline "is `XC-52`'s decision, not this
  type's". Once (c) is decided, a comment that describes the decision as pending is wrong; it must state
  that there is deliberately no deadline and why (one sentence plus a pointer here), so nobody re-opens it
  by inference.
- **The report records which mutations hung rather than reddened** (§8), because that boundary is a result.

**Should**
- `XC-53` (`--blame-hang --blame-hang-timeout` plus `timeout-minutes`) lands **before** (a) reaches CI. It
  is a runner flag, it is cheap, and it is (a)'s backstop for the one failure mode (a) cannot make red. Not
  a hard dependency: the local mutation harness already passes `--blame-hang`.

**Won't** — each stated as the failure mode it avoids, not as a precedent:
- **No progress deadline in the completion spin** (§3): returning to the caller while a worker may still
  write through the caller's now-unpinned pointers converts a diagnosable hang into silent heap corruption.
- **No test that writes `_decodeGen`, the claim word or `_decodeChunks[]`**: those are process-global and
  the fast suite dispatches decode concurrently from other classes, so a write from a test corrupts a
  neighbour's live dispatch — and the failure lands in the untimed spin, i.e. as a hung run rather than a
  red test.
- **No delay hook added to the dispatch path**: it would put a test-only branch on a per-token hot path, and
  a hook that widens the publication window changes the thing being measured.
- **No assertion on elapsed time, CPU, contention rate or "the retry branch was taken"**: those are
  environment-produced quantities, which is the `TG-T12`/`TG-T13` failure mode — a loaded box moves the
  verdict. §5's join backstop is not an exception to this; it is a backstop and must be reported as one.
- **Nothing touching the park/wake protocol** (`_decodeParkLock`, `Monitor.Wait`/`PulseAll`, the spin
  budget): it is a different mechanism whose last defect cost 14.85 effective cores
  (`OverfitParallel.cs:329-336`), and bundling it makes every mutation result in both areas unreadable. Its
  own row, alongside `TG-T13`.
- **No performance number** (`PB-12`), and no re-measurement of the decode-pool throughput figures.

## 8. How this is proven — mutations, with victims *and* outcomes predicted before the run

`overfit-mutate` protocol; run under `--blame-hang --blame-hang-timeout`, because three of these are
predicted to hang and an unbounded hang costs a machine (during `XC-50` one orphaned host burned 2268
CPU-seconds over 37 minutes).

| # | mutation | predicted victim | predicted outcome |
|---|---|---|---|
| M1 | `chunkEnd = (int)Math.Min((long)chunkStart + perChunk, rangeEnd)` → `… + perChunk - 1 …` in `ForDecode` | (a), exactly-once | **red, deterministically** — pins the coverage half |
| M2 | `Volatile.Write(ref _decodeRemaining.Value, chunkCount)` → `chunkCount - 1` | (a), in-flight-after-return and exactly-once | **red, high probability** — the dispatcher can return with the slow chunk still running. This is the invariant's own mutation |
| M3 | completion spin `!= 0` → `> 1` | (a), same assertions | **red, high probability** |
| M4 | move the `Error?.Throw()` loop above the completion spin | (a), the throwing-body assertions | **red, high probability** — pins that an exception does not shortcut the wait |
| M5 | remove `Interlocked.Decrement` from `ExecuteDecodeChunk`'s `finally` | nobody | **hung** — recorded as the boundary of what any test here can promise, not as a gap |
| M6 | `DecodeChunkClaim.Publish` preserves the incoming index instead of zeroing it | nobody | **hung** (already documented at `DecodeChunkClaim.cs:64-70`) |
| M7 | hoist the `Volatile.Read` out of `TryClaim`'s CAS retry loop | nobody | **hung** (already documented at `DecodeChunkClaim.cs:163-169`) |

**What refutes this test set.** If M1 survives, (a) is not asserting coverage and the task is not finished.
If M2 or M3 survive **repeatedly** (run each at least three times before concluding — a probabilistic victim
demands it), (a)'s falsification power is insufficient and the uneven-work shape in §4.4 must be widened
before this closes; do not lower the claim instead. If M5–M7 redden rather than hang, my model of the
dispatcher is wrong and §3 has to be re-opened.

**(b) is never a predicted victim of any of these** (§5).

## 9. Ordering — highest uncertainty first

1. **The capability probe.** Write (a)'s skeleton with the accessors and the capability gate, and report the
   pool size and the achieved chunk-count sequence on this box and, if possible, from a CI run. If the
   sequence never contains "larger follows smaller", everything below is weaker than it reads and the shape
   must change first.
2. (a): assertions, bodies with their bounds checks, the throwing dispatch.
3. M1 — the deterministic mutation. If it survives, stop and report.
4. M2, M3, M4, three runs each, victims predicted in writing beforehand.
5. M5–M7 under `--blame-hang`, to record the hang boundary. **Run these last and one at a time**; each
   leaves a stuck host that must be confirmed gone before the next.
6. (b), and its own soak run.
7. The two comment amendments (§7 Must).
8. `dotnet build -c Release` on `Main.csproj` and `Tests.csproj` directly — a solution-wide build currently
   fails with `MSB3021`/`MSB3027` because the semantic-navigator host locks its own DLL — then
   `dotnet test -c Release --filter FullyQualifiedName~Runtime`, then the full fast suite, which is the
   second net for a protocol change because so much of it dispatches decode transitively (§2). Whoever runs
   it takes `Global\DevOnBike.Overfit.MachineMeasurement`.

## 10. Risks, and what retires each

| risk | cheapest thing that retires it | when |
|---|---|---|
| (a) never dispatches and is green anyway | the capability gate + the reported count sequence (§4.3) | step 1, and it gates the rest |
| (a) is green because the window is never sampled | M2/M3, three runs each; the uneven-work shape widened if they survive | step 4 |
| A regression crashes the run instead of reddening it (finding 3) | cannot be retired — bounds checks inside the body and lifetime-extended contexts (§4.1-4.2) reduce it; the residual is stated in the tests' doc comments | ongoing |
| A liveness regression hangs CI for six hours | `XC-53` | before (a) reaches CI |
| Someone later reads (b)'s green as coverage | its name, its doc comment, and its exclusion from every mutation prediction (§5) | at implementation |
| (c) is re-opened by inference from a stale comment | the `DecodeChunkClaim.cs:163-169` amendment (§7 Must) | step 7 |
| The fast suite gets slower for everyone | the budget in §4.4, measured once and trimmed | step 2 |

## 11. What remains unverified — read this before treating anything here as met

- **Nothing was built, run, tested or benchmarked this round.** The tree is clean and untouched by me apart
  from this file.
- **The worst-case legitimate dispatch duration was not measured** — deliberately, and §3.2 says why it
  would not settle the decision it is usually asked to settle.
- **The `455 µs / 0 B` figure is cited from `docs/measured-baselines.md:375`, not re-verified**, and
  `OverfitParallel.cs:362-366` already records that this path's throughput numbers predate the 2026-08-14
  claim-protocol change.
- **The "~5 µs/call" dispatch latency in the class doc (`:23-24`) is the MAIN pool's**, on a 32-logical-core
  Ryzen, and is an in-code claim I did not verify. It is used here only for an order-of-magnitude ratio in
  §5, never as a decode-pool figure.
- **`Assert.Skip`'s exact signature was not compiled against** — the strings are in the assembly (§2), which
  is indicative and not proof.
- **Whether each of the transitive `[Fact]` callers actually dispatches** (rather than taking the inline
  path on its dimensions) was **not** checked case by case; the claim made here is only that some do, which
  follows from the pool size being 10 on this box and the ranges in those tests exceeding 1.
- **I did not read `Tests/Core/Runtime/DecodeChunkClaimTests.cs` in full** — only its `[Fact]` names, which
  is what §2's "the claim is covered, the dispatcher is not" rests on.

## Sign-off

**Architecture review: reviewed 2026-08-14 against the code, not against the task row. SIGNED.**
Execution path: **inference**. AOT-reachable: **no**. Allocation policy: **hot path, zero allocations per
call — and unperturbed by this plan**. Assembly: `Tests`, plus two `internal` accessors and three comments
in `Sources/Main`. Public surface: **unchanged**. Ownership: not applicable. Verification oracle: §8, and it
must be run — a surviving M1 is a finding, not a pass, and an M5–M7 that reddens overturns §3.

## BLOCKING QUESTIONS

**None.** (c) is decided on evidence in the code (§3, finding 2), not on a missing threshold, so there is
nothing to send back. One question is *recorded but not blocking*, because its default is the status quo and
neither (a) nor (b) depends on the answer:

> **For the client, if and when a stall is ever seen outside a mutation run:** would you accept the library
> terminating the host process with a dump (`Environment.FailFast`) after a multi-minute decode stall, in
> exchange for a diagnosable failure instead of a silent hang? **Absent an answer I assume no**, and option
> A stands.

---

## Retired figures cited in this plan — added by the main session 2026-08-15

**This plan quotes decode-pool performance figures that `PB-12` retired on 2026-08-14. They are left in
place because a signed plan is a record of what was decided and on what basis, and silently rewriting its
evidence would make the decision unreviewable. Do not carry them forward.**

| cited here | measured 2026-08-14, 24-36 processes, ABAB, canary |
|---|---|
| `455 µs` vs `2059 µs`, "4.5x" | **2.43x** against the capped `Parallel.For` that `OVERFIT_DECODE_POOL=0` actually falls back to, 3.60x against an uncapped one. The published pair was the uncapped comparison, which the product no longer makes on the decode path |
| Qwen3-0.6B `+28%` | **+25.1%** (Q8_0) / **+23.0%** (Q4_K_M) — supported, but no site recorded the quantisation |
| Phi-3.5 `+3%` | **−1.9%**, i.e. neutral to negative; the pool captures only 75% of that model's dispatches |
| Bielik `+11%` | **+3.8%** isolated to its own mechanism; at today's default the same knob is worth +87% because it now also sizes the spin pool |

Two things worth more than the corrections. **`ForDecode` has never had a benchmark class** — all four
figures came from `[ModelFact]` diagnostics that `dotnet test` never runs, single-arm, one process, no
canary. And the client's regression hypothesis was **tested and refuted**: the 2026-08-14 claim/park change
did not cost throughput, and the old figures do not reproduce on the old code either.

Canonical numbers: `docs/measured-baselines.md`. Verdict and method: the `PB-12` row in `docs/TASKS.md`.
