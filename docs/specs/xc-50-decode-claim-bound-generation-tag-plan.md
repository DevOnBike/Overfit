STATUS: REVIEWED  (advanced by the main session 2026-08-14 — see the gate lines below; the
architect deliberately did not advance it, since recording a stage it did not run is the exact
failure the status line exists to prevent, and advancing it is the coordinator's job)
Author: overfit-architect (both halves — see §0 for why there is no analyst round)
Architecture review: overfit-architect, 2026-08-14 — SIGNED
Date: 2026-08-14
Slug: xc-50-decode-claim-bound-generation-tag-plan

GATES:
  performance:        NOT_REQUIRED — no performance target or claim enters this plan. The claim path's
                       operation count is stated structurally in §6 and is **not** a timing claim. The moment
                       anyone asserts this change is neutral or better, that is a performance claim and
                       `overfit-perf-claim-auditor` owns the verdict; the decode-pool figures this file's own
                       comments carry are already filed as `PB-12`
  security:           NOT_REQUIRED — no parser, endpoint, gateway, audio/tokenizer decode or externally-fed
                       surface. The `void* Context` this code dispatches through originates in first-party
                       kernels, not in user input
  leak-scan:          NOT_REQUIRED — no config, log, fixture, host name, path or token touched
  AOT:                NOT_REQUIRED — `Tests/AotSmokeTest/Program.cs` read in full this run (47 lines): it
                       touches `typeof(OverfitClient)`, `SamplingOptions.Greedy` and a `GenerationOptions`
                       constructor and performs no inference, so `OverfitParallel.ForDecode` is not reachable.
                       This plan adds no reflection, LINQ, `Activator` or `Expression`
  API-compatibility:  NOT_REQUIRED — everything added or moved is `private`/`internal`; `OverfitParallel` and
                       `ForDecode` keep their signatures, and no public member of `DevOnBike.Overfit` is
                       added, removed or changed
  verifier:           PASS — `overfit-verifier`, 2026-08-14, second round. Six mutations executed by it,
                       every victim set EXACT against predictions written before each run; the M4 gap it
                       raised in round 1 (a surviving mutation on `Publish`) is closed and now fails on
                       the new assertion, `Expected: 0, Actual: 4`. Suite 2631/0/274
  reviewer:           PASS — `overfit-reviewer`, 2026-08-14. No correctness finding; hand-traced the
                       packed layout at its edges, the self-containment of the claim, the rename-plus-
                       unpack, and independently re-derived the `_decodeRemaining` argument. Two
                       follow-ups, neither a defect: one comment should cite `_decodeGate`'s mutual
                       exclusion as why the premise holds across two `ForDecode` calls, and a CHANGELOG
                       sentence still described the two-field word (fixed by the main session)
  mutation-proof:     PASS — see `verifier`. Round 1 was **FAIL** and is recorded rather than overwritten:
                       one mutation survived, and the suite did not redden but HUNG, stranding an orphan
                       that burned 2268 CPU-seconds and then blocked every build

`release-readiness` has **no line at all**, which is this manifest's way of saying the question was never
asked — a missing line is not `NOT_REQUIRED`. Writing the word "absent" as a verdict is not valid either;
`Scripts/plan_gate_check.py` rejected exactly that on the first attempt at this block.

**`mutation-proof` is not optional here and must not be closed as `NOT_REQUIRED`** — §8's mutations *are* the
acceptance criterion. It is left absent rather than marked because the manifest's vocabulary
(`PASS | FAIL | INCONCLUSIVE | NOT_REQUIRED`, per `Scripts/plan_gate_check.py:32`) has no token for
"required, not yet run", and `INCONCLUSIVE` would falsely imply it ran.

---

# XC-50 — the decode-pool claim's exhaustion bound must be carried by the claim word

Plan file for task `XC-50` (`docs/TASKS.md:181`).

## 0. Why the architect wrote both halves

The defect was found by `overfit-architect` on 2026-08-14 while signing the `XC-49` plan (that plan's §A.3),
hand-traced there, and verified independently by the main session against the working tree. There is no
business question in it and no missing requirement: it is an internal correctness defect in a `private`
synchronisation protocol, with no user-visible option to decide. Dispatching an analyst to re-narrate an
analysis that already exists in a signed plan would add a round trip and no gate. **If a business or scope
question had surfaced while writing this, the correct move would have been to stop and send it back** — none
did.

## 1. The defect, quoted and re-verified this run

`docs/TASKS.md:181`:

> "`_decodeChunkCount` is not generation-tagged while the claim word is … In that window the claim word still
> carries generation `G` while the count already belongs to `G+1` … If the next dispatch has more chunks than
> the last, that comparison wrongly succeeds … Torn descriptor plus early completion — exactly the class the
> CAS was added to remove, in a narrower window."

Re-read from the working tree this run, not from the row. `OverfitParallel.ForDecode`, inside
`lock (_decodeGate)`:

```csharp
var generation = _decodeGen + 1;

_decodeChunkCount = chunkCount;                                   // :626  plain store, NEW count
Volatile.Write(ref _decodeRemaining.Value, chunkCount);           // :627
for (var i = 0; i < chunkCount; i++) { … _decodeChunks[i] … }     // :629-639  NEW descriptors
Volatile.Write(ref _decodeClaim.Value, generation << 32);         // :644  NEW tag, first visible here
Volatile.Write(ref _decodeGen, generation);                       // :645
```

and `OverfitParallel.TryClaimDecodeChunk`:

```csharp
var current = Volatile.Read(ref _decodeClaim.Value);              // :709
if ((uint)(current >> 32) != tag) { index = 0; return false; }    // :711  tag from the WORD
var next = (int)current;                                          // :717  index from the WORD
if (next >= _decodeChunkCount) { index = 0; return false; }       // :718  bound from a SEPARATE FIELD
if (Interlocked.CompareExchange(ref _decodeClaim.Value, current + 1, current) == current) …  // :724
```

**The decision "may this generation take this index" is made from two locations that are published at
different times.** Between `:626` and `:644` the bound belongs to `G+1` and the tag still belongs to `G`.
Every worker's drain loop ends with one `TryClaimDecodeChunk` expected to fail (`DecodeWorkerLoop:824`,
`ForDecode:658`), so a straggler presenting the old tag in that window is the ordinary case, not a contrived
one. With `chunkCount_G = 4` and `chunkCount_{G+1} = 8`: tag matches, `4 >= 8` is false, the CAS succeeds
because the word is unchanged, and the straggler executes `_decodeChunks[4]` — a descriptor either half
rewritten or still holding `G`'s `Body`/`Context`, where `Context` is a `void*` into a stack frame whose
dispatcher has already returned.

**Reachability is ordinary.** `chunkCount = Math.Min(_decodePoolSize, totalWork)` (`:619`) and `totalWork`
differs per dispatch, so "the next dispatch is larger" happens roughly every time attention (few heads) is
followed by an FFN (many outputs) — ~180 dispatches per token.

**Two consequences, and the quiet one is worse.** The loud one is a torn `Body`/`Context` pair — the
`DivideByZeroException` already recorded at `OverfitParallel.cs:278-288`. The quiet one: the straggler
consumes index 4 of `G+1` and decrements `G+1`'s `_decodeRemaining`, so the *real* descriptor for chunk 4 is
never executed by anyone, the dispatcher's spin sees zero, and `ForDecode` returns with a slice of the output
buffer never written — no exception, wrong tokens.

## 2. Inventory — what exists, and how each line was established

| what | evidence |
|---|---|
| Exactly **two** lock-free CAS protocols exist in `Sources/Main` | `Grep` for `Interlocked.CompareExchange` across `Sources/`: `OverfitParallel.cs:724` (this one) and `OverfitResourcePool.cs:169` (a max-CAS over its own word and a local — no second location consulted, not the same shape) |
| `_decodeChunkCount`'s **only** concurrent reader is the claim | `Grep` for `_decodeChunkCount` in `Sources/`: written `:626`, read `:718`, named in the `BOUND:` comment `:655`. `ForDecode`'s error loop `:670` uses the **local** `chunkCount`, not the field |
| The main pool does **not** have this shape | `OverfitParallel.For` (`:534-573`) commits a `SemaphoreSlim` token per chunk *before* a worker claims, and `_completion.Signal()` fires in `ExecuteChunk`'s `finally`. A worker that consumed a token but has not yet incremented `_nextChunk` leaves `_completion` unsignalled, so the dispatcher cannot reach the next dispatch. No straggler can survive into `G+1`. This matches the doc at `:583-588` and was re-derived rather than taken from it |
| The claim protocol is unreachable from tests today | `XC-49` §2, re-confirmed: `TryClaimDecodeChunk`, `_decodeClaim`, `_decodeChunkCount`, `_decodeGen` are all `private static` |
| The 2026-08-14 fix is **committed**, not uncommitted | `git log --oneline -3 -- Sources/Main/Runtime/OverfitParallel.cs` → `9074647`; `git status --porcelain` empty; `git diff HEAD --stat` empty. `XC-49` §10 step 4 assumes the file differs from `HEAD` — that assumption is now stale and a clean-`HEAD` mutation check is available again |
| `ForDecode` has 8 production call sites | `find_references(ForDecode)` per `XC-49` §2 — every quantised GEMV/projection kernel plus both `CachedMultiHeadAttention` decode paths. Not re-run this round |

## 3. The invariant

> **Every input to the decision "may this generation take this index" is carried in the single word the CAS
> operates on. A claim consults no other mutable state.**

This is deliberately stronger than the row's phrasing ("the exhaustion bound must be as generation-tagged as
the claim"), because tagging a second field would leave two things to keep in step; carrying it in the word
leaves one. It is also checkable in one reading: `TryClaimDecodeChunk` must reference no field other than the
claim word.

## 4. Options, and what each gives up

| | **A. Reorder + order the count store** | **B. Carry the bound in the claim word** (decided) | **C. Two-phase drain** |
|---|---|---|---|
| Shape | Move `_decodeChunkCount = chunkCount` to **after** the claim-word publish and make it a `Volatile.Write`, so it is a release store ordered after the tag store | Pack `(generation tag, chunk count, next index)` into `_decodeClaim`'s existing 64-bit word; delete `_decodeChunkCount` | Dispatcher waits for every worker to leave its drain loop before republishing (per-worker epoch) |
| Is it sufficient? | **Yes, but only with the store ordered.** A plain store after a release store may be reordered *before* it — ECMA-335 gives release semantics only against *preceding* accesses, and ARM64's `STLR` likewise does not order a *later* plain store after itself. Two release stores are ordered w.r.t. each other, and the reader's acquire load of the word keeps its count read after it, so `Volatile.Write` placed after `:644` closes it | Yes, and with no ordering obligation at all: a straggler tagged `G` reads a word that is either `G`'s own exhausted word (`next >= count`, both `G`'s) or already `G+1`'s (tag mismatch). No third state is representable | Yes, and far more machinery |
| Cost | Two lines | ~40 lines: pack/unpack, a bound established at pool-size resolution, one field deleted, one comment rewritten | A second spin phase inside `_decodeGate`; rejected |
| **Testability** | **None.** The claim function is unchanged, so no state test can distinguish fixed from broken; the property is "these two stores are ordered", which managed code cannot assert. A future edit that moves the line back or drops the `Volatile` is caught by nobody | A deterministic test **calling the real publish routine** distinguishes them, and the defect has a compilable one-expression mutation (§8) | n/a |
| Residual risk | The correctness argument is a four-step memory-model derivation. **The ordering argument in this exact method has now been wrong twice** — `:278-288` and `:641-643` both asserted a safety property that did not hold, and both were reviewed | The standard release/acquire publication (descriptors before the word, word before `_decodeGen`) remains, and is the pattern this file already gets right | n/a |

**Decision: B.** Not because it is more elegant — it is more code. Because A's correctness is unfalsifiable by
any test this repository can run, on a protocol whose ordering argument has already failed twice, guarding a
failure mode that is silent. A is a legitimate engineering choice for someone who weighs the two lines higher
than the testability; if a reviewer wants to argue for it, the argument above is the honest version of it and
nothing here is hidden.

**What B is not.** It is not a merge of `_decodeGen` into the claim word. That would remove a field and a
store, and it also rewrites the park protocol whose own coverage gap is open (`TG-T13`, and the 14.85-cores
idle-burn incident recorded at `OverfitParallel.cs:300-306`). Out of scope — see §9 Won't.

### 4.1 The shape, to the level a boundary needs and no further

- **Layout: generation tag in the high 32 bits (unchanged), chunk count next, next-index in the low bits.**
  Recommended 32/16/16. Two constraints are load-bearing and the rest is the developer's:
  1. **The index occupies the low bits**, so the claim stays `CompareExchange(ref word, current + 1, current)`
     — the increment must not carry into the count field, which holds while `next < count`.
  2. **The count can never exceed its field, and must never be negative.** `chunkCount = Math.Min(_decodePoolSize, totalWork)`, and
     `_decodePoolSize` is fixed at class init (`ResolveDecodeMaxWorkers`, bounded by
     `Environment.ProcessorCount`), so the bound is established **once, where the pool size is resolved** —
     not checked per dispatch. **Correction, 2026-08-14: "silent truncation" understated it.** A negative
     count does not merely underflow its own field — `((long)(-1) << 16)` is `0xFFFF_FFFF_FFFF_0000`, so the
     sign bits smear into the **tag**, and a claim is then refused by the generation check rather than by the
     bound. Found by `overfit-developer` when a zero-chunk test passed a `chunkCount - 1` mutation for a
     reason it did not intend; re-computed here. Two consequences: the clamp must bound the count on **both**
     sides, and any test that asserts "a claim is refused" must say *which* guard refused it, or it can pass
     while pinning nothing.
- **The claim protocol moves to its own `internal static` type in its own file** (e.g.
  `Sources/Main/Runtime/DecodeChunkClaim.cs`, one top-level type per `OVERFIT-ONETYPE`), exposing a publish
  and a try-claim over `ref long`. **The reason is enforcement, not tidiness**: `_decodeChunkCount` and
  `_decodeGen` are `private` to `OverfitParallel`, so a claim living in another type *cannot* consult them —
  §3's invariant becomes a compile error rather than a review habit. `PaddedClaim` stays `private`; only a
  `ref long` crosses the boundary, so `XC-49` F4's `CS0052` never arises.
  *Escape, stated so this does not become a line-by-line approval:* if the extraction forces something worse
  at implementation time, a `private static` method on `OverfitParallel` taking the same parameters is an
  acceptable downgrade — **the packed word is not negotiable, the file it lives in is.** Record which was
  chosen and why.
- **The loop body moves verbatim** apart from the substitutions the packing requires. A reviewer must be able
  to read the diff as a rename plus the unpack; `overfit-reviewer` should treat any other change inside that
  body as a finding. (Carried over from `XC-49` §A.3, which decided the same discipline.)
- **The generation check stays first**, before the bound check.

## 5. The full sweep — every piece of dispatch state, not just the one reported

The question is not "is `_decodeChunkCount` wrong" but "which state can a *concurrent claimer* read before it
belongs to the generation it is claiming for". Two kinds of reader exist: **claimers** (`TryClaimDecodeChunk`)
and **executors** (`ExecuteDecodeChunk`, reached only through a successful claim).

| state | written | read by | verdict |
|---|---|---|---|
| `_decodeChunkCount` | `:626`, plain store, before the tag | **claimer**, `:718` | **The defect.** Removed by B |
| `_decodeClaim` | `:644`, release store | claimer, `:709`/`:724` | Correct — it is the tag itself |
| `_decodeRemaining` | `:627`, release store, before the tag | executor's `finally` (`:745`), dispatcher's spin (`:665`) | **Safe, but only as a consequence of §3's invariant, and this has never been written down.** Each successful claim decrements exactly once; claims for `G` total exactly `chunkCount_G`; so `_decodeRemaining == 0` implies every `G` execution has already decremented, and no `G` decrement can survive into `G+1`. That argument holds *only* while the claim bound is generation-correct — with today's defect the extra claim is exactly what breaks it. **Requirement: state this at the site**, because the write's position looks wrong and is not |
| `_decodeChunks[i]` (`Start`/`End`/`Body`/`Context`/`Error`) | `:629-639`, plain stores, before the tag | executor, `:736-741` | Safe. Publication is release (the claim word) / acquire (the claimer's `Volatile.Read`), and no execution happens without a claim. This is the ordinary publication pattern and B leaves it intact |
| `_decodeGen` | `:645`, release store, last | worker wake predicate (`:792`, `:811`) | Safe. It is a wake signal only; after B it authorises nothing, which is the point |
| main pool: `_chunkCount`, `_nextChunk`, `_completion`, `_chunks[]` | `:536-550` | `WorkerLoop:873-895` | **Not the same shape** — see §2. No change proposed. Recorded because "fix the class, not the instance" requires the sibling to be checked and it was |

So the class contains exactly one instance, and after B the class is closed by construction: there is no
second mutable location a claim could consult.

## 6. Boundaries and gate answers

| | |
|---|---|
| **Execution path** | **Inference.** The decode pool's per-chunk claim, on the path taken by every quantised GEMV/projection kernel and both `CachedMultiHeadAttention` decode paths |
| **Assembly** | `Sources/Main` only (plus `Tests`). No boundary crossed, no project reference, no dependency added |
| **Public surface** | Unchanged. The new type is `internal`; `OverfitParallel`/`ForDecode` signatures untouched |
| **AOT reach** | Not reachable (see the `GATES:` line — `AotSmokeTest/Program.cs` read in full this run). Unchanged by this plan |
| **Allocation policy** | Hot path, zero allocations per call. Nothing here allocates; no field is added |
| **Ownership / disposal** | Not applicable — no buffer, no `AutogradNode`, no `PooledBuffer<T>`. The `void* Context` remains caller-owned for the duration of the dispatch, which is what `_decodeRemaining` enforces |
| **Threading model** | Unchanged, and **must stay unchanged**: the decode spin pool for decode, `Parallel.For` elsewhere. `docs/measured-baselines.md:375` records the decode figure (**455 µs / 0 B** vs `Parallel.For`'s **2059 µs / 925 KB**) and `:349` the opposite result in `Conv2D`. Neither is a target of this plan and neither was re-verified here; `OverfitParallel.cs:330-337` already records that the decode-pool throughput figures predate the 2026-08-14 change on this exact path (`PB-12`) |
| **Claim-path operation count** (a structural statement, **not** a timing claim) | Before: one `Volatile.Read`, one static field read, one CAS per attempt. After: one `Volatile.Read`, two shift/mask pairs, one CAS. Publish: one fewer store. The claim is already a separate static method, so moving its declaring type adds no call site. Whether the JIT's inlining decision changes is a measurement — `docs/measured-baselines.md:373` records **2.25×** for an extracted method the JIT declines to inline, on a different subject — and **nobody should assert it either way in this plan** |
| **Moat side** | Neither. `OverfitParallel` already ships in the open AGPL surface; no real-time, GPU or throughput claim is made or implied |
| **Verification oracle** | The protocol's own invariant, proven in two directions: the defect demonstrated red on the pre-fix shape, and the fix pinned by a compilable mutation (§8). There is no external reference implementation and none is needed |

**No ADR.** Checked against the list of decisions that need one: no public API (`internal` only), no assembly
placement of a public type, no change to AOT reachability, no on-disk or on-wire format (the packed word is a
`private` in-memory field with no persistence and no cross-process reader), no open/commercial boundary, no
dependency added to `Main`. The reasoning lives here and at the site.

**Operability.** Not applicable — this is library code with no process to operate, nothing durable, nothing to
restart. The nearest operational question, "what does a user see when it goes wrong", is answered in §1 and is
the reason the task exists: today, nothing.

## 7. Interaction with `XC-49` — and a correction to a statement I signed

**`XC-49` §A.5 says "XC-49 proceeds unchanged because the seam is invariant under any fix to it" (row 4), and
§A.6 repeats it. That is now wrong under the decided fix, and it was my sentence.** It holds under option A;
it does not hold under B, for two separate reasons:

1. **The seam's signature changes.** `XC-49`'s decided seam is the claim protocol parameterised on
   `(ref claim word, chunk count, generation)`. Under B the chunk count is *in* the word, so the parameter
   disappears and `XC-49`'s AC1.1/AC1.2 change their seeding (one packed word instead of a word plus a count).
   The **substance** of both ACs and of `XC-49`'s named mutation survives untouched; the seeding lines do not.
2. **`XC-49` AC1.1 would not have caught this defect, and that is a design lesson, not a criticism.** It
   hand-seeds the word to `(G+1) << 32` and asserts a `G`-tagged claim is refused. The XC-50 window is
   precisely the interval *before* that word exists. **A hand-seeded state cannot catch a publication defect;
   the test has to call the publisher.** §8's AC1 does.

**Recommendation to whoever sequences these (a coordination item, not a design question, and it does not block
this plan): land `XC-50` first, and fold `XC-49`'s seam into `XC-50` step 1** — they are the same edit, and
doing it once avoids writing a regression test against a shape that changes the next day. `XC-49` then reduces
to adding its two ACs against the final shape and running its mutation.

**If `XC-49` is landed first anyway**, that is workable but two things must be recorded so nobody reads more
into it than it says: its mutation result covers **the generation-tag defect only**, not this one; and its
plan needs the one-line correction above. **I have not edited `docs/specs/xc-49-decode-pool-claim-race-test-plan.md`** — it
is signed, `APPROVED`, and was not the file I was given. The correction is proposed here and is the main
session's to apply.

## 8. How this is proven — red first, then a mutation

The window is *narrower* than the one the 2026-08-14 fix closed, so a probabilistic stress test is even less
likely to catch it than it was there, and §4 already rejects "the window is small" as an argument in either
direction. Neither the proof nor the refutation is timing-based.

**Step R — demonstrate the defect red, before the fix.** Extract the claim protocol as it stands today
(`(ref long claim, int chunkCount, long generation, out int index)` — this *is* `XC-49`'s seam, unchanged),
then, single-threaded, on a local `long`:

- *Given* `word = (G << 32) | 4` (generation `G`, exhausted at 4 chunks)
- *When* `TryClaim(ref word, chunkCount: 8, generation: G, out var i)` — the state `ForDecode` creates between
  `:626` and `:644`
- *Then* it must return `false`. **It returns `true` with `i == 4` today.**

The developer reports the observed value. This converts a hand-trace into a demonstrated defect and gives the
fix a red-to-green transition, which is stronger than any mutation because the "mutated" code is the real one.

**Post-fix acceptance criteria**, all deterministic, all single-threaded, no `Thread`/`Task`/`Monitor`/`Sleep`
symbol in any of them, no `[LongFact]`, no fixture:

- **AC1 — republishing a later, larger dispatch never revives an exhausted earlier generation.**
  *Given* a local word published for `G` with count 4 and drained to exhaustion (4 claims succeed, the 5th
  returns `false`); *when* the **production publish routine** is called on that same word for `G+1` with count
  8; *then* `TryClaim(…, generation: G, …)` returns `false` and the word is bit-identical to what publish
  wrote. **The state must be reached by calling publish, never by hand-seeding** — that is the whole
  difference between this test and `XC-49` AC1.1 (§7.2).
- **AC2 — exactly `chunkCount` claims per generation, and the bound comes from the word.**
  Publish `G` with count 3, claim four times: `(true,0)`, `(true,1)`, `(true,2)`, `(false, ·)`.
- **AC3 — a claim carrying a superseded tag is refused and consumes nothing** (the word is unchanged, so the
  CAS never fired). This is `XC-49` AC1.1's assertion; whichever task lands first owns it, and it must not be
  written twice.

**Mutations, predicted victim named before the run** (`overfit-mutate` protocol; the tree is clean at `HEAD`
so guard 1's clean-`HEAD` check works normally — see §2):

| # | mutation | predicted victim | why it is the right mutation |
|---|---|---|---|
| M1 | in the publish routine, write the new count while leaving the tag and index in place — one expression, e.g. `word = (word & ~CountMask) \| ((long)count << CountShift)` | **AC1 only** | This *is* XC-50 — it recreates the exact state the defect produced, and it compiles against the fixed shape, which is what makes B provable and A not |
| M2 | publish `chunkCount + 1` | **AC2 only** | proves the bound read by the claim is this dispatch's, not an over-count |
| M3 | revert the claim to a bare `Interlocked.Increment` ignoring the generation | **AC3 only** | `XC-49`'s mutation, carried forward so the earlier fix stays covered |

**What would refute this test set:** if M1 leaves AC1 green — then AC1 is not testing the publication and the
task is not finished; or if M1 turns AC2/AC3 red as well — then the tests are not isolating what they claim
to, and the victim prediction was wrong, which per the skill is itself a finding.

## 9. Scope

**Must**
- The packed claim word (§4.1) and the removal of `_decodeChunkCount`.
- Step R's red observation, reported with the value it returned.
- AC1 and AC2 with mutations M1 and M2 run and reported.
- **The comment at `:641-643` rewritten with the code.** It currently reads *"Publish the claim word first,
  then the generation: … a worker only starts claiming once it has seen the new generation, so it can never
  observe a claim word that still belongs to the previous dispatch."* The second half is false in the
  direction that matters: the hole belongs to a worker still using the **old** generation, which never looks
  at `_decodeGen` again. It must state §3's invariant and why the count travels in the word. **A wrong comment
  on a memory-ordering argument is worse than none**, and this file has now carried two.
- **The `BOUND:` comment at `:655` updated** — it names `_decodeChunkCount`, which this task deletes. A
  `BOUND:` annotation naming a field that no longer exists is the `OVERFIT023` contract rotting in place, the
  same class as `XC-51`.
- **A sentence at `:627`** recording why resetting `_decodeRemaining` before the dispatch is safe (§5), since
  it is the one remaining write whose position looks wrong and is not.

**Should**
- AC3 + M3 carried here **if** `XC-50` lands before `XC-49` (§7), so the earlier fix does not go uncovered in
  the interim.

**Must — added 2026-08-14 by the amendment in §13. Read §13 before implementing these.**
- **Layout edges, tested rather than documented**: `count` of 0, 1, `MaxChunkCount`; a claim at
  `count - 1` and the refusal at `count`; and above all **the top-of-range claim must not carry into the
  count field** — after the last claim of a `MaxChunkCount` dispatch, assert the word's count field is still
  `MaxChunkCount`. That single assertion is what pins layout constraint 1 (§4.1), which is today prose.
  *`count` of 0 and 1 are unreachable from `ForDecode`* (it returns early below `totalWork < 2` and requires
  `_decodePoolSize > 1`); those cases pin the primitive's defined behaviour and must be named so nobody reads
  them as production scenarios.
- **Tag wraparound, asserted as the limitation it is.** The tag is `(uint)generation`, so **two generations
  collide exactly when they differ by a multiple of 2^32** — `G` and `G + 2^32`, e.g. `0xFFFF_FFFF` and
  `0x1_FFFF_FFFF`, both truncating to `0xFFFF_FFFF`. Publish for the later one, claim with the earlier, and
  assert it **is accepted**. The test *documents a collision*, it does not promise safety — name it so
  (`…TagCollides…`), because a test whose green means "the defect boundary is where we said" is misread as a
  guarantee by default. State the boundary as the modulus, which is exact, rather than as the "days to
  months" estimate carried in `DecodeChunkClaim.TryClaim`'s doc comment, which was never measured.
  **Correction, 2026-08-14** — this bullet first named `0xFFFF_FFFF` and `0x1_0000_0000` as the colliding
  pair. **They do not collide**: `(uint)0x1_0000_0000` is `0x0000_0000`. Found by `overfit-developer` while
  implementing it, verified by the coordinator, and re-computed here rather than accepted on report. It had
  already propagated from this file into a brief and into the client conversation, which is the cost of an
  arithmetic claim written without being evaluated. The wrong pair was kept as the **non**-collision
  assertion — adjacent generations spanning `0xFFFF_FFFF → 0x0000_0000` must still be distinguished — so
  both directions are now pinned.
- **`MaxChunkCount`'s clamp pinned without depending on the box.** `_decodePoolSize`'s clamp is inert on any
  real machine (`ResolveDecodeMaxWorkers` already caps at `Environment.ProcessorCount`), so a test must
  exercise the clamping expression or the constant directly. An unreachable clamp nobody drives is the same
  defect class as an untested bound.
- **A multi-threaded exclusivity test over a LOCAL word** — N threads, one `long` the test owns, one
  generation, `count` chunks: exactly `count` successful claims in total, every index in `[0, count)` handed
  out exactly once, none outside. Admissible under §13's rule; constraints and its accepted cost are there.
- **`DecodeChunkClaim.cs:135-156` must be updated with these tests.** That comment currently cites this
  plan's §9 as the reason the lost-CAS retry branch is untested. Amending §9 without amending the comment
  would leave production code citing a rule that no longer exists — the same rot class as `XC-51`.

**Won't**
- **Merging `_decodeGen` into the claim word.** It is the obvious next simplification and it rewrites the
  park protocol, whose coverage gap is open (`TG-T13`) and whose last defect cost 14.85 effective cores on an
  idle box (`OverfitParallel.cs:300-306`). Separate task if anyone wants it.
- **Any change to `OverfitParallel.For`, `WorkerLoop` or the main pool** — checked and not the same shape
  (§5).
- **Re-measuring the decode-pool throughput figures** — `PB-12` owns them.
- ~~**A threaded stress harness.** It cannot prove this and would be a third `TG-T12`/`TG-T13`.~~
  **Amended 2026-08-14 — this wording over-reached and is replaced by §13.** What is still refused: a test
  that *pokes the pool's process-global state* (`_decodeGen`, the claim word, the descriptors) or that adds a
  delay hook to the dispatch path. What is no longer refused: a concurrency test of the extracted protocol
  over a word the test owns (now **Must**, above), and a client-level dispatcher test (a separate task, §14).
- **Widening `PaddedClaim`, `_decodeClaim` or `_decodeGen` to `internal`** — the decided shape removes the
  reason to.

## 10. Ordering — highest uncertainty first

1. **Step R.** Extract the claim as-is, write the failing test, observe red, report the value. *This is also
   `XC-49`'s seam; do it once.* If it comes back **green**, stop and report — the hand-trace would be wrong
   and everything below it is void.
2. Apply the packed word, delete `_decodeChunkCount`, establish the count bound at pool-size resolution.
3. Re-express the test as AC1; add AC2 (and AC3 if `XC-50` leads).
4. Run M1 and M2 (and M3), victims predicted in writing before the run.
5. Rewrite the three comments (§9 Must).
6. `dotnet build -c Release`, then `dotnet test -c Release --filter FullyQualifiedName~OverfitParallel`, then
   the full fast suite — the decode pool is live in every test process and eight plain `[Fact]`s reach
   `ForDecode` transitively (`XC-49` §A.1), so the full suite is the second net for a protocol change.
   Whoever runs it takes `Global\DevOnBike.Overfit.MachineMeasurement`; nothing else may measure concurrently.

## 11. Risks, and what retires each

| risk | cheapest thing that retires it | when |
|---|---|---|
| The hand-trace is wrong and there is no defect | step R — one test, one run, and it is the first thing done | step 1, and it gates everything |
| The fix is wrong in a new way | M1/M2 plus a reviewer diffing the moved loop body as a rename-plus-unpack; the full fast suite as the second net | steps 4 and 6 |
| Silent truncation if `chunkCount` ever exceeds the count field | establish the bound where `_decodePoolSize` is resolved, once, so the state is unrepresentable rather than checked | step 2 |
| The JIT's inlining decision changes and someone claims neutrality | say nothing about timing; `PB-12` owns this path's numbers and `overfit-perf-claim-auditor` owns any verdict | ongoing |
| `XC-49` lands first and its mutation result is read as covering XC-50 | §7 — record the scope of its verdict explicitly | before either lands |
| A future edit reintroduces a second location the claim consults | the type boundary makes it a compile error (§4.1); if the escape is taken, `overfit-reviewer` treats any new field read inside the claim as an automatic finding | ongoing |

## 12. What remains unverified — read this before treating anything here as met

- **Nothing was built, run or benchmarked.** The machine is free, but as architect I had no reason to compile:
  no criterion in this plan is met by a build.
- **The defect is hand-traced, twice by different readers, never executed.** Step R exists to fix that and is
  deliberately the first task.
- **The memory-model claims in §4 option A** (a plain store may move ahead of a release store; two release
  stores are mutually ordered; an acquire load keeps later reads after it) are stated from ECMA-335 §I.12.6
  and the ARMv8 semantics of `STLR`/`LDAR`. I did not find a written statement of the .NET model in this
  repository to cite, and I did not consult external documentation this run. **They are the reason A is
  ranked second, not the reason B is correct** — B's correctness needs no such claim, which is exactly its
  argument.
- **`find_references(ForDecode)` = 8 production call sites is carried from `XC-49` §2 and was not re-run.**
- **The 455 µs / 0 B figure is cited from `docs/measured-baselines.md:375`, not re-verified**, and
  `OverfitParallel.cs:330-337` already records that this path's throughput numbers predate the 2026-08-14
  change.

---

## 13. Amendment, 2026-08-14 — §9's rejection of a "threaded stress harness" over-reached

**What changed and why.** After implementation and a `VERIFIED` verifier run, the client challenged §9's
`Won't` in substance: *if you suspect something, build the falsification yourself; do not ask a reviewer to
be careful.* The challenge lands. §9's bullet was inherited from `XC-49` §5 option A (a harness with a
**delay hook added to the dispatch path** and real background threads racing `ForDecode`) and `XC-49` §A.1
(a test **writing** `_decodeGen`/`_decodeChunkCount`). Both poke process-global state. Neither describes the
two shapes now at issue, and **the extraction this task performed removed the property the rejection rested
on**: `DecodeChunkClaim` (`Sources/Main/Runtime/DecodeChunkClaim.cs:33`) has no field, no thread and no lock,
and operates on a `long` the caller owns — verified by reading it, not taken on report. A rule that forbids by
wording rather than by reasoning is worse than no rule, and this one had already propagated into production
code at `DecodeChunkClaim.cs:147-149`.

### 13.1 The admissibility rule, stated in failure modes rather than precedent

`TG-T12`/`TG-T13` are not cautionary because they are concurrent. They are cautionary because **they assert a
quantity the environment produces** — elapsed time, process CPU — so a loaded box moves the verdict. That is
the property to exclude, not the threads.

> **A concurrency test is admissible here iff every one of its assertions holds under every legal schedule.**
> Load then changes only *which interleavings are sampled*, never the verdict, so it cannot be flaky in the
> `TG-T12` sense. An assertion of the form "within X ms", "the retry branch was taken", or "N% of claims
> contended" is inadmissible — that is a measurement, and measurements go to `PB-12` and
> `overfit-perf-claim-auditor`.

The second question every such test must answer before it is written: **when the property is violated, is the
result red or hung?** The two need different answers and only one of them is a test.

### 13.2 Rulings on the four questions

**Q1 — is a client-level dispatcher test safe against the `TG-T12`/`TG-T13` failure mode?**
Its *verdict* is safe: assertions like "every body sees its own context", "every chunk executes exactly once",
"no dispatch returns while a chunk is running" are schedule-invariant, so there is no false red and no
environment-dependent flip. Assertion failures inside a body are also safe by construction — `ExecuteDecodeChunk`
(`OverfitParallel.cs:734-747`) captures the throw into `_decodeChunks[i].Error`, still decrements in its
`finally`, and `ForDecode` rethrows on the caller — so a violation detected inside a body surfaces as a red
test, not a background-thread crash.

**But it is not hang-proof, and nothing can make it so while the completion spin is untimed and inside the
gate.** The "a chunk is never executed" direction leaves `_decodeRemaining` above zero and parks the caller in
the pure spin at `OverfitParallel.cs:665-668` **while holding `_decodeGate`**, so every later decode in the
process blocks behind it. A deadline on the test's own dispatch thread converts "the run stops" into "one
named test fails, then the run degrades" — worth having, and honestly not a bound. `--blame-hang` is a
backstop, as the coordinator says, not a design.

That is the whole reason this belongs to its own task (§14): **the only real bound is a production change** —
a progress deadline in the completion spin that throws instead of spinning forever — and that is a change to
hot-path failure semantics needing its own reasoning and a measured worst-case legitimate dispatch duration.
It must not be smuggled in under a claim-word fix, and I am not deciding it here.

**Q2 — does varying the chunk count reach the window deterministically?**
No. **Probabilistic, and it must be labelled as a falsification attempt, never as coverage.** Nothing in the
test can force a straggler to be descheduled inside the publication window; it samples interleavings. The
value is asymmetric and that asymmetry is the whole case for it: a red is a true finding (the assertions are
schedule-invariant, so there are no false positives), while a green proves nothing at all. Two consequences
that must be written into the test's own doc comment: it may **never** be cited as evidence that a window is
closed, and it may **never** be the predicted victim of a mutation — a probabilistic victim makes a mutation
result unreadable, which is the one thing `overfit-mutate` cannot tolerate.

It also splits in two, and the split is where most of the value sits:

| shape | verdict |
|---|---|
| **Sequential caller, alternating chunk counts** — one thread dispatching back-to-back `ForDecode` calls with different bodies, contexts and counts. The pool's workers still race each other for chunks, so exclusivity and context-pairing are genuinely exercised; only the second *dispatcher* is absent | **Admissible in the fast suite.** Deterministic in structure, schedule-invariant in assertion, and it reaches `XC-50`'s precondition (a larger dispatch following a smaller one) |
| **Concurrent callers, thousands of iterations** — the soak the coordinator described | **`[LongFact]` only.** Its fast-suite failure mode is a poisoned run (Q1) for a probabilistic gain, and this repository has already paid for a stalled run once: `XC-42`'s orphaned test host held the machine mutex and cost a process-ancestry walk to diagnose |

**Q3 — the park/wake protocol.** **Out of scope here, and it is not `XC-50`'s to fix.** It is a different
mechanism (`_decodeParkLock`, `Monitor.Wait`/`PulseAll`, the spin budget), with its own defect history — the
token leak measured at 14.85 effective cores on an idle box (`OverfitParallel.cs:300-306`) — replaced
wholesale this week on a hand-traced argument. Bundling it makes every mutation result in both areas
unreadable. Its measurement half is already filed (`TG-T13`); its *protocol* half has no row and should get
one. The design question it must answer first, because it decides whether the task is cheap or invasive:
**how is a park/wake observed without measuring the whole process?** Candidates worth costing: pool-local
counters incremented only on park/pulse/wake (never per claim, so off the per-chunk path), and — for
`TG-T13`'s half — thread-scoped CPU time for the `OverfitDecode-*` threads instead of process-wide time,
which is the actual defect in `DecodePoolIdleBurnTests`.

**Q4 — benchmarks.** **`PB-12`, not this plan**, and the verdict on any number produced belongs to
`overfit-perf-claim-auditor`, not to this plan or its author. Three reasons, none of them about effort:
(1) `XC-50` states no performance target, so there is nothing here for a benchmark to settle; (2) the figures
in question — `455 µs / 0 B` vs `2059 µs / 925 KB` (`docs/measured-baselines.md:375`), `+11%`, `+28%`, `+3%` —
are *this path's published numbers*, which is exactly `PB-12`'s subject, and a number produced under `XC-50`
would be read as `XC-50`'s result, conflating a correctness fix with a performance verdict; (3) only one
benchmark process may run at a time (`Global\` mutex, exit code 2), so it is a scheduling decision as well as
a scoping one.
**One requirement to carry into `PB-12` when it runs:** measure the **enclosing operation** — dispatch
overhead and decode tok/s — not a microbenchmark of `Publish`/`TryClaim`. A few nanoseconds of shift-and-mask
inside a microsecond dispatch inside a 40-70 ms token is not a finding, and a microbenchmark that reports one
will be believed. `PB-12` should also carry `A` vs `B` honestly if anyone re-opens §4: neither shape has been
measured.

### 13.3 What this amendment does not change

The deterministic proof of `XC-50` remains `AC1`/`M1` (§8). Nothing added here is a substitute for it, and no
probabilistic test may be cited in its place.

## 14. Proposed follow-on task — the dispatcher's untested invariants

**Not filed by me: I do not write to `docs/TASKS.md`.** Specified here so the main session can file it as
written; the next free id at the time of writing is `XC-52` (`XC-51` is the last row).

> **`XC-52` — the decode dispatcher has no end-to-end test, and its central invariant is a comment.**
> `_decodeRemaining == 0` implies no worker is still inside `ExecuteDecodeChunk`, which is the only reason
> overwriting `_decodeChunks[]` on the next dispatch is safe. `XC-50` documented it as a consequence of the
> claim invariant (`OverfitParallel.cs:636-645`) and nothing exercises it. Every defect on this path this week
> — the torn descriptor, the park-token leak, `XC-50`'s bound-versus-tag window, and two memory-ordering
> comments that were wrong *and signed* — lived in the dispatcher, not in the claim arithmetic, which is now
> the only part with tests.
>
> **Scope**: (a) a sequential client-level test — back-to-back `ForDecode` dispatches with different bodies,
> contexts and chunk counts, asserting every body sees only its own context, every chunk of every dispatch
> executes exactly once, and no dispatch returns while a chunk is still running (fast suite; assertions
> schedule-invariant; seeds no pool state, uses `ForDecode` exactly as production does); (b) the same with
> concurrent callers as a `[LongFact]` soak, labelled a falsification attempt and never a coverage claim;
> (c) **decide** whether the completion spin gets a progress deadline that throws — the only thing that turns
> this whole class from "hung run" into "red test" — which needs a measured worst-case legitimate dispatch
> duration before a threshold can be chosen, and is a change to hot-path failure semantics.
>
> **Explicitly not in it**: the park/wake protocol (Q3 above — its own row, alongside `TG-T13`), and any
> performance number (`PB-12`).
>
> **Architecture note for whoever plans it**: (c) is the irreversible-ish decision in the task and should be
> settled before (a) and (b) are written, because it decides whether a failing dispatcher test is readable.

---

## Sign-off

**Architecture review: reviewed 2026-08-14 against the code, not against the task row. SIGNED.**
**Amended 2026-08-14 (§13) after implementation, in response to a client challenge to §9's `Won't`. The
signature stands and the amendment adds scope: §9's new `Must` block. `STATUS:` was `APPROVED` when this amendment was written,
because neither the developer nor the verifier advanced it (see the report); the main session advanced it to
`REVIEWED` later the same day, so **line 1 is the authority and this sentence is history**. Left rather than
deleted because `overfit-reviewer` found the two contradicting each other, and a plan that quietly rewrites
its own past is worse than one that shows where it moved — the added items are picked up
under the same row, not as a new plan.**
Execution path: **inference**. AOT-reachable: **no**. Allocation policy: **hot path, zero allocations per
call**. Assembly: `Sources/Main`, no boundary crossed. Public surface: **unchanged**. Ownership: not
applicable. Verification oracle: §8, and it must be run — a green M1 is a finding, not a pass.

## BLOCKING QUESTIONS

**None.** There is no client question in this task — no business rule, no acceptance threshold, no value
judgement is missing — and no scope question for an analyst: the row names one defect in one `private`
protocol and §5 establishes that the class contains exactly one instance. The single cross-task item (§7,
`XC-49` sequencing and the correction to a sentence I signed) is a **coordination decision for the main
session**, is stated with a recommendation and a fallback, and does not block implementation of this plan.
