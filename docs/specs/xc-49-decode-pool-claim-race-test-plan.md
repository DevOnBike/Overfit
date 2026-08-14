STATUS: APPROVED
Author: overfit-analyst
Architecture review: overfit-architect, 2026-08-14 — SIGNED with one seam decision that overrides the
                     analyst's recommendation, and one finding on the fix itself (see §A)
Date: 2026-08-14
Slug: xc-49-decode-pool-claim-race-test-plan

GATES:
  performance:        NOT_REQUIRED — no performance claim in this plan; the decode-pool throughput numbers
                       this fix's file also carries are a separate, already-filed task (`PB-12`, owned by
                       `overfit-perf-claim-auditor`) and out of scope here
  security:           NOT_REQUIRED — no parser, endpoint, gateway, audio/tokenizer decode, or externally-fed
                       `unsafe` surface touched by any option under consideration (see §6)
  leak-scan:          NOT_REQUIRED — no config, log, fixture, host name, path or token touched
  AOT:                NOT_REQUIRED — `Sources/Main/Runtime/OverfitParallel.cs` is not reachable from
                       `Tests/AotSmokeTest/Program.cs` (verified: `Grep` for `OverfitParallel`/`Runtime\.` in
                       that file returns no match), and no option under consideration adds a reachable
                       reference
  API-compatibility:  NOT_REQUIRED — every option under consideration is `private`/`internal`; no public
                       member of `DevOnBike.Overfit` is added, removed or changed

`verifier`, `reviewer`, `mutation-proof` and `release-readiness` are **intentionally absent** rather than
`NOT_REQUIRED` — per `overfit-delivery`'s own rule, a missing line means "not yet asked", and it is
genuinely too early to ask: no task under this plan has been implemented. All four apply once a task lands;
`mutation-proof` in particular is not optional here — §9's acceptance criterion **is** the mutation this gate
will check.

---

# XC-49 — a deterministic regression test for the decode-pool claim protocol

Plan file for task `XC-49` (`docs/TASKS.md:180`). Written by `overfit-analyst`, no prior plan for this task.

## 1. What the client asked for, quoted

> "the decode-pool claim protocol has no deterministic regression test, and it must get one."

The `docs/TASKS.md` row, filed by `overfit-reviewer` on 2026-08-14 reviewing the `OverfitParallel` fix:

> "the decode-pool claim race has no deterministic regression test — reverting the fix would probably stay
> green ... `TryClaimDecodeChunk` and every field it touches are `private static`, so nothing can drive the
> claim protocol directly — not even through `InternalsVisibleTo` ... mutating the CAS back to a bare
> increment would most likely leave the suite green ... Do not weaken the assertion to make it pass: the
> failure mode being pinned is a torn descriptor, which is silent by nature."

There is no external client here — the requester is this repository's own review process, and the "user"
whose need matters is whoever next touches this file (`overfit-developer`, or an agent reverting the fix by
mistake) and needs the suite to catch it.

## 2. Inventory — what already exists

| bucket | what | evidence |
|---|---|---|
| **Already exists** | The fix itself, uncommitted in the working tree. Generation+index packed into one 64-bit `_decodeClaim` word, claimed by `TryClaimDecodeChunk` via a single CAS | `git diff -- Sources/Main/Runtime/OverfitParallel.cs`, read in full; `TryClaimDecodeChunk` at `:699-730` |
| **Already exists** | A hand-verified soundness review of the fix (release/acquire ordering traced by hand) | `.claude/agent-memory/overfit-reviewer/project_overfitparallel_decode_pool.md` |
| **Already exists** | Two of the reviewer's three findings on this same diff (stale doc references, "years" overstatement) are **already fixed** in the working tree — confirmed by reading the current file: `:333` now reads "(see `_decodeParkLock`)" with a corrected idle-cost sentence, `ForDecode`'s doc now states plainly "Its claim protocol is NOT the main pool's", and the wraparound comment now reads "days to months" with an explicit correction note | `Sources/Main/Runtime/OverfitParallel.cs:278-288, 583-588, 693-697` |
| **Already exists** | A public, unsafe-function-pointer test call site for `ForDecode` itself (not for the claim protocol) — precedent that the API is already test-reachable at the dispatch level | `Tests/LanguageModels/Diagnostics/AttentionQ4KRepackHypothesisTests.cs:145,151`, confirmed via `find_references(ForDecode)` — 8 production call sites (every quantised GEMV/projection kernel plus both `CachedMultiHeadAttention` decode paths) + this one test file |
| **Already exists** | A precedent for an `internal static … ForTest(...)` test-support member on a hot-path type in exactly this area of the codebase, with its own analyzer-pragma convention | `Sources/Main/LanguageModels/Runtime/StackWeights.cs:88-89`: `#pragma warning disable OVERFIT001 // test-support factory (InternalsVisibleTo), never on a runtime path` `internal static StackWeights ForTest(...)` |
| **Already exists** | The main pool's own claim path (`_nextChunk`, `WorkerLoop`) has full unit coverage, including a torn-chunk check — but it is **structurally different** and does not cover the decode pool at all | `Tests/Core/Runtime/OverfitParallelTests.cs`, in particular `For_NoTornChunks_AllSlicesCovered` (:283-305) — tests `OverfitParallel.For`, not `ForDecode` |
| **Partially exists** | The only corroboration of the decode-pool claim fix is a `[LongFact]` gated on a real model file, which the reviewer already judged non-deterministic | `Tests/LanguageModels/Diagnostics/PrefillCallCountTests.cs` — `[LongFact("7s")]`, requires `C:\qwen3b\...Q4KmGguf`, skipped by default |
| **Does not exist** | Any unit-level seam onto `TryClaimDecodeChunk` or the state it reads (`_decodeClaim`, `_decodeChunkCount`) | confirmed: `TryClaimDecodeChunk`, `_decodeClaim`, `_decodeChunkCount`, `_decodeGen` are all `private static`; `find_references(TryClaimDecodeChunk)` returns **no source symbol** (not reachable, even via `InternalsVisibleTo`) |
| **Does not exist** | Any test in `Tests/Core/Runtime/OverfitParallelTests.cs` that calls `ForDecode` at all | confirmed by reading the file in full |

## 3. Problem / user need / business goal / proposed solution

| | |
|---|---|
| **Problem** | A concurrency defect that produced a `DivideByZeroException` in production and, more dangerously, a *silent* torn-descriptor bug (a later dispatch's completion counter driven down early, returning a partially-written output buffer with no exception) was fixed on 2026-08-14, but nothing in the suite would fail if the fix were reverted. The only corroborating evidence is a `[LongFact]` that happened to over-count calls while the race was firing — a symptom of the race, observed once, on a real model, not a repeatable check of the mechanism |
| **User need** | Whoever next edits `OverfitParallel.cs` — most likely `overfit-developer` acting on a future task, or an agent doing a well-intentioned refactor — needs `dotnet test -c Release` (the fast suite, not a `[LongFact]`) to go red if the CAS-based generation check is weakened or removed |
| **Business goal** | This path runs on **every decode**, on every non-Android platform, by default (`find_references(ForDecode)`: 8 production call sites, every quantised GEMV/projection kernel). A silent torn-descriptor bug here is a wrong or corrupted token returned to a customer with no error — the worst class of defect this product can ship, because nothing signals it happened |
| **Proposed solution** (the client's, one candidate) | "A stress harness driving rapid back-to-back `ForDecode` dispatches with two distinct bodies and contexts, one path delayed" — i.e., reproduce the race with real threads and real timing |

**Success metric.** Not client-supplied (no client persona here); I am setting it from the task's own stated
acceptance test, which is unusually precise for this kind of row: **reverting `TryClaimDecodeChunk`'s CAS to
a bare `Interlocked.Increment` (ignoring `generation`) must turn a specific, named `[Fact]` red, under
`dotnet test -c Release --filter FullyQualifiedName~OverfitParallelTests`, with no `[LongFact]` and no real
model file involved.** This is verifiable by anyone via `overfit-mutate`'s protocol (§9) and is a considerably
sharper bar than "coverage exists."

## 4. Investigation findings

**F1 — the client's own proposed solution (a threaded stress harness) is answerable more cheaply, and I
recommend against it as the primary test.** `TryClaimDecodeChunk` is a pure function of three pieces of
static state (`_decodeClaim.Value`, `_decodeChunkCount`, and the `generation` argument) — it does not need a
second thread to observe the exact defect the fix addresses. The defect was: a claim call carrying a stale
generation succeeding anyway. That is directly reproducible, single-threaded, by constructing the "stale
claim" state by hand and calling the method. A threaded stress harness is a weaker guard for more engineering
cost: any timing-dependent reproduction (`Thread.Sleep`, a delay hook, a `ManualResetEvent` race) is exactly
the flakiness shape this repository already tracks twice over (`TG-T12`, `TG-T13` — both "green in isolation,
red on a loaded box" or vice versa) and the task's own brief says plainly: *"Do not weaken the assertion to
make it pass"* — a probabilistic race test is a standing invitation to do exactly that under CI pressure.

**F2 — a critical, previously-undocumented hazard for ANY seam design here: the decode pool's background
threads are live for the whole test process, always, and must never be woken by a test.**
`OverfitParallel`'s static constructor (`:353-390`) unconditionally spawns `_decodePoolSize` background
threads (`OverfitDecode-{i}`) whenever `_decodePool` is true, which is the **default on every non-Android
platform** (`ResolveDecodePool`, `:328-351`) — including the Windows dev box and Linux CI. These threads run
for the process's entire lifetime, continuously polling `Volatile.Read(ref _decodeGen)` (`DecodeWorkerLoop`,
`:770-829`). **If a test writes to `_decodeGen` directly**, outside the `_decodeGate`-protected `ForDecode`
path, it wakes these real daemon threads immediately, and they will race to call `TryClaimDecodeChunk` and
then `_decodeChunks[index].Body(_decodeChunks[index].Start, ..., _decodeChunks[index].Context)` —
an **unmanaged function-pointer call** (`ExecuteDecodeChunk`, `:732-747`). If the test has not populated
`_decodeChunks[]` with a valid, harmless body for every index the workers might claim, this is calling
through a null or stale unmanaged function pointer, which is undefined behaviour in .NET, not a catchable
exception — the realistic outcome is a crashed test process, not a red test. **This constraint governs every
option in §5**: a safe seam must never let a test write `_decodeGen` except by going through the real,
gated `ForDecode` dispatch. This was not previously written down anywhere I could find (checked
`.claude/agent-memory/overfit-reviewer/project_overfitparallel_decode_pool.md`, the CHANGELOG entry, and the
`docs/TASKS.md` row — none mention it), and it is the single most load-bearing fact for whoever implements
this.

**F3 — F2 has a direct, safe consequence: a pure single-threaded unit test of `TryClaimDecodeChunk` is
possible without ever touching `_decodeGen`, and is therefore immune to the hazard in F2 by construction.**
`DecodeWorkerLoop`'s only wake condition is `_decodeGen` changing (`:792,811`). `_decodeClaim` and
`_decodeChunkCount` are read by `TryClaimDecodeChunk`, which background workers only ever call *after*
already observing a `_decodeGen` change. A test that sets `_decodeClaim`/`_decodeChunkCount` directly and
then calls `TryClaimDecodeChunk` — without touching `_decodeGen` — cannot be raced by the live pool, because
nothing wakes it. This turns the "reproduce a torn-generation claim" problem into ordinary state-based unit
testing: set the claim word to belong to generation `G+1`, then call the method with the stale argument `G`
(exactly what a descheduled straggler would pass), and assert it returns `false` and does not advance the
claim word — which is precisely the invariant the fix establishes and the bare-increment version violates.

**F4 — widening the raw fields (`_decodeClaim`, `_decodeChunkCount`) to `internal` has one concrete
compile-time cost the request's phrasing doesn't surface: it cannot be done without also widening the
private nested `PaddedClaim` struct.** `_decodeClaim`'s declared type is `PaddedClaim`
(`Sources/Main/Runtime/OverfitParallel.cs:955-959`), a `private` nested struct. C# raises `CS0052`
("inconsistent accessibility") if a field's accessibility exceeds its type's — so `internal static PaddedClaim
_decodeClaim` requires `PaddedClaim` to become `internal` too, which then also makes its layout
(`[StructLayout(LayoutKind.Explicit, Size = 128)]`, the cache-line padding) part of the internal-visible
surface for no reason connected to what the test actually needs (a `long`, not the struct). A small
accessor-method seam (§5, option D) avoids this entirely.

**F5 — the API-compatibility gate genuinely does not apply here, checked rather than assumed.** All three
options in §5 either add `internal static` members or change no signature at all (option A adds nothing;
options B–D add or widen `internal` members only). `XC-34`'s comparator (`Tests/TestSupport/Assemblies/`)
verdicts only on public surface (`AC-TYPE-REMOVED` etc.), and `internal` is not part of it — confirmed by
reading `XC-34`'s own row, which frames every finding in terms of `DevOnBike.Overfit`'s **public** API.

## 5. The seam — options compared, not silently picked

| | **A. Threaded stress harness** (client's proposed solution) | **B. Widen fields to `internal`** | **C. Extract claim protocol into its own type** | **D. Thin `internal` test-only accessors** (recommended) |
|---|---|---|---|---|
| What it is | Real background threads, one delayed via a hook, racing `ForDecode` dispatches | `_decodeClaim`, `_decodeChunkCount`, `TryClaimDecodeChunk` → `internal` | New type (e.g. a `DecodeClaimProtocol` struct) owns the word + CAS; `OverfitParallel` holds an instance | 2–3 tiny `internal static` wrapper methods (`...ForTest` suffix), production fields/types stay `private` |
| Determinism | Timing-dependent unless a delay hook is added (adds a hot-path branch — see below) | Deterministic (pure state test, per F3) | Deterministic (pure state test, per F3) | Deterministic (pure state test, per F3) |
| Hazard from F2 | **Real** — must coordinate around live background threads touching real `_decodeGen`; getting this wrong crashes the process, not just the test | None, if the test follows F3's rule (state-only, never touches `_decodeGen`) | Same as B | Same as B |
| Hot-path cost | A delay hook (even a null-checked `Action?`) adds a branch to the busiest call in the file — called per chunk claim, ~180 dispatches/token | None — `internal` vs `private` is a visibility-only change, zero codegen difference | None on the happy path; the extraction itself is the cost (see below) | None — new methods are never called from production code |
| Compile-time cost | Needs a public/internal hook field added to production code | `PaddedClaim` must also become `internal` (F4) — widens more than the test needs | Touches `ForDecode`, `DecodeWorkerLoop`, `ExecuteDecodeChunk` — every reader of this recently-fixed, correctness-critical file | Additive only; nothing existing is touched |
| Blast radius | New: `Sources/Main` (hook field/branch) | `Sources/Main` (2 field decls + 1 struct decl + 1 method signature) | `Sources/Main` (restructure of ~150 lines around the claim state) | `Sources/Main` (2–3 new short methods, nothing existing edited) |
| Review risk | Highest — a synchronization primitive added to review a synchronization fix, on code fixed *yesterday* | Low | **Highest of the four** — re-touches just-fixed, hand-verified-sound concurrency code for a testability goal, not a correctness one | Lowest — no existing line changes |
| Precedent in this codebase | None found | None found for this exact shape | None found | `StackWeights.ForTest` (`internal static … ForTest(...)`, `#pragma OVERFIT001` convention) — F0 above |
| Reusable beyond this task | Possibly, as a general end-to-end race harness | No | Yes, if a second claim-protocol consumer appears — none does today | No, and it should not try to be |

**Recommendation: D**, for the reasons above, with the F2 constraint written into the seam's own doc comment
so a future edit does not "helpfully" make the accessor take a `_decodeGen` value too. **This is a technical
recommendation, not a decision** — it touches `Sources/Main`, so per `docs/specs/README.md` it needs the
architect's signature before `overfit-developer` may act on it, and options B/C remain open for the architect
to override if a broader testability need is anticipated elsewhere.

Sketch of the two new members (illustrative, not authoritative — the developer may reshape signatures, and
**a `Volatile.Read` inside the accessor is required for the same reason it's required in the production
method: the word may be written by a real dispatch immediately before or after the accessor call under
concurrent test execution**):

```csharp
// Test-only. NEVER writes _decodeGen — doing so wakes the live decode-pool background threads that run for
// the whole process lifetime (see XC-49 plan §4 F2) and can crash the test process via a null/stale unmanaged
// function-pointer call. Restrict this seam to the claim word and the chunk count.
#pragma warning disable OVERFIT001 // test-support accessor (InternalsVisibleTo), never on a runtime path
internal static void SeedDecodeClaimForTest(long generation, int chunkCount)
{
    _decodeChunkCount = chunkCount;
    Volatile.Write(ref _decodeClaim.Value, generation << 32);
}

internal static long ReadDecodeClaimForTest() => Volatile.Read(ref _decodeClaim.Value);

internal static bool TryClaimDecodeChunkForTest(long generation, out int index) =>
    TryClaimDecodeChunk(generation, out index);
#pragma warning restore OVERFIT001
```

## 6. Gate answers

| gate | answer |
|---|---|
| **Execution path** | Inference — this is the decode hot path's own dispatch primitive. The new test-only members are never called from production code (naming convention `...ForTest`, matching `StackWeights.ForTest`); they add no code to the execution path itself |
| **Verification oracle** | The mutation itself: reverting `TryClaimDecodeChunk`'s body to a bare `Interlocked.Increment` (ignoring `generation`) must turn the new `[Fact]` red. This is a correctness-of-a-concurrency-primitive oracle, not a numerical-parity one — see §9 for the exact `overfit-mutate` harness |
| **AOT reach** | Not reachable from `Tests/AotSmokeTest/Program.cs` today (verified, §2) and none of the options add a reference from it. No AOT gate implication |
| **Allocation policy** | The new test is not a hot-path call; `TryClaimDecodeChunk` itself is unchanged (already 0-alloc, verified by inspection — it's a `while(true)` CAS loop with no heap access). Test-only accessor methods are likewise 0-alloc but this is incidental, not a stated requirement, since they never run on the measured path |
| **Moat side** | Neither — this is internal test infrastructure for an already-open-AGPL-surface primitive (`OverfitParallel` ships in `DevOnBike.Overfit`). No real-time/GPU/performance claim is made or implied by this plan |

## 7. What is not settled fact

| type | item |
|---|---|
| **Fact** | The fix (generation-packed CAS) is present and correct by hand-traced release/acquire reasoning — `.claude/agent-memory/overfit-reviewer/project_overfitparallel_decode_pool.md`, not re-verified independently in this round (out of scope for a testability plan; the fix's correctness is not in question, only its coverage) |
| **Fact** | `TryClaimDecodeChunk`, `_decodeClaim`, `_decodeChunkCount`, `_decodeGen` are all `private static`, unreachable via `InternalsVisibleTo` today — verified via `find_references` returning "no source symbol" |
| **Fact** | The decode pool's background threads are live for the whole test process by default on non-Android — verified by reading `ResolveDecodePool`/the static constructor (§4 F2) |
| **Assumption** | Default xUnit test collection parallelism (classes may run in parallel with each other; methods within one class run sequentially) means no *other* test class calls `ForDecode` concurrently with the new test under the fast suite (`AttentionQ4KRepackHypothesisTests`, the only other caller, is `[LongFact]`-gated and skipped by default). If this assumption is wrong, the new test should reset `_decodeChunkCount`/`_decodeClaim` to a safe state (chunk count 0) in a `finally` block regardless — cheap insurance, and I recommend doing it whether or not the assumption holds |
| **Assumption** | The new test belongs in `Tests/Core/Runtime/OverfitParallelTests.cs`, beside the existing main-pool tests, rather than a new file — matches this repository's "one file per subject, domain first" convention and the file's existing scope (`OverfitParallel` correctness) |
| **Decision** | Option D (thin internal accessors) is the recommended seam; not yet a Decision until the architect signs it (§5) |
| **Constraint** | The test must never write `_decodeGen` directly (F2) — this is not a style preference, it is a process-crash hazard |
| **Risk** | If option D's accessors are ever called from production code by mistake, they bypass `_decodeGate`'s lock entirely and would corrupt live decode state. Mitigation: the `...ForTest` naming convention (precedented), and the fact that nothing in `Sources/Main` has a reason to call a raw claim-seed method — `overfit-reviewer` should flag any production call site to these methods as an automatic finding |
| **Risk** | A future edit adds a `_decodeGen`-writing convenience to the same test-only accessor group "for completeness," reintroducing F2's hazard silently. Mitigation: the doc comment on the seam states the constraint explicitly (§5's code sketch) |
| **Open question** | Whether option D's accessors should live under `#if DEBUG`-style test-only compilation guards in addition to `internal` — not evaluated here; precedent (`StackWeights.ForTest`) does not use one, so I assume the same is acceptable unless the architect says otherwise |

## 8. Scope

**Must**
- A deterministic, non-`[LongFact]` test that fails when `TryClaimDecodeChunk`'s generation check is removed (the mutation named in §9), and passes against the current fix.
- The minimal testability seam that makes that test possible (§5, option D recommended, subject to architect sign-off).
- The seam's own doc comment states the F2 constraint (never write `_decodeGen` from a test) so it cannot be silently violated later.

**Should**
- A second assertion state, distinct from "claim succeeds": exhaustion within one generation (`next >= _decodeChunkCount` → `false`, index unclaimed) — this is a genuine branch of `TryClaimDecodeChunk` untested today and cheap to add alongside the stale-generation case.

**Could**
- A supplementary, real-thread stress test (the client's original proposal, §4 F1) as an end-to-end belt-and-braces check, `[LongFact]`-gated so it never contributes flakiness to the fast suite. Not required because the deterministic unit test already pins the exact mechanism; only worth doing if the team wants confidence beyond the CAS logic itself (e.g., that `_decodeGate`'s locking still composes correctly with the fix under real contention) — that is a different, broader property than what XC-49 asks for.
- Extending coverage to the exhaustion-then-superseded interleaving (claim exhausts generation G, generation G+1 begins, a call still tagged G arrives) — covered implicitly by the stale-generation test if index 0 of G+1 differs from where G left off, but not asserted as a distinct case.

**Won't (this time)**
- Any change to the park-protocol fix (`_decodeParkLock`/`Monitor`) — that is a *different* defect (Defect B in the reviewer's own split) with its own open coverage gap already filed separately (`TG-T13`, blocked on scoping the measurement to the pool's own threads rather than the whole process).
- Re-measuring the decode-pool throughput numbers this file's comments cite — already filed as `PB-12`, owned by `overfit-perf-claim-auditor`, explicitly out of scope for a testability plan.
- Option B or C from §5, unless the architect prefers them — not built by default because option D is cheaper and lower-risk for the same guarantee.
- Widening `PaddedClaim` or any other production type's accessibility "while we're in there" — F4 names the exact cost; not part of this task.

## 9. Acceptance criteria (Given/When/Then) and the mutation that proves them

**Story 1.** As `overfit-developer` (or any future editor of `OverfitParallel.cs`), I want a fast, deterministic
test that fails if the decode-pool claim protocol stops checking its generation, so that reverting the fix
by accident is caught by `dotnet test -c Release` rather than by a customer seeing a torn or corrupted token.

- **AC1.1 — a claim tagged with a superseded generation is refused and consumes nothing.**
  - *Given* the claim word holds generation `G+1` with next-index `0` (simulating that a later real dispatch
    has already begun) and chunk count `3`,
  - *When* `TryClaimDecodeChunkForTest(G, out index)` is called (the stale generation, as a descheduled
    straggler would present),
  - *Then* it returns `false`, `index` is `0` (out-param default), and `ReadDecodeClaimForTest()` still reads
    exactly `(G+1) << 32` — the CAS never fired, so no index of the new generation was silently burned.
  - **Oracle**: this is the property; there is no external reference implementation to compare against —
    the mutation in §9.1 is what proves the assertion is load-bearing rather than vacuous.

- **AC1.2 — a claim tagged with the current generation succeeds and advances by exactly one.**
  - *Given* the claim word holds generation `G` with next-index `0`, chunk count `3`,
  - *When* `TryClaimDecodeChunkForTest(G, out index)` is called three times in a row,
  - *Then* it returns `(true, 0)`, `(true, 1)`, `(true, 2)`, then `(false, 0)` on a fourth call — chunk
    exhaustion within the same generation, the `Should`-tier branch from §8.

**Mutation (`overfit-mutate` protocol, to be run once the seam and test exist):**

| field | value |
|---|---|
| Target | `Sources/Main/Runtime/OverfitParallel.cs` |
| Anchor | the body of `TryClaimDecodeChunk` between `var tag = (uint)generation;` and the closing brace (exact text taken from the file at implementation time — must match once) |
| Mutated | a bare `Interlocked.Increment`-based claim that ignores `generation` entirely (the pre-fix shape, reconstructable from the `git diff` already read in this plan, §2) |
| Expected victim | `OverfitParallel_TryClaimDecodeChunk_RefusesStaleGeneration` (AC1.1's test — exact name is the developer's to pick, but it must be the one predicted **before** the run, per the skill) |
| Filter | `FullyQualifiedName~OverfitParallelTests` |

**What would refute this test's value**: if it goes red for a reason other than the generation check (e.g., a
typo in the seam unrelated to the CAS), or if AC1.2's happy-path test also fails under the same mutation
(expected — a bare increment still claims 0,1,2 correctly for the *matching* generation; only AC1.1
distinguishes the fix from the bug, which is why AC1.1, not AC1.2, is the named victim).

## 10. Ordering

1. Architect reviews §5 and signs an option (highest uncertainty first — this is the one open design
   question in an otherwise fully-specified task).
2. Add the seam (§5D or the architect's alternative), with the F2 constraint in its doc comment.
3. Add AC1.1 and AC1.2 to `Tests/Core/Runtime/OverfitParallelTests.cs`.
4. Run the mutation in §9 against the **implemented** code (not against this plan's illustrative sketch) —
   guard 1 of `overfit-mutate` needs a byte-snapshot substitute rather than a clean-`HEAD` check, since the
   target file already differs from `HEAD` by design (the uncommitted fix this test protects).
5. `dotnet build -c Release` + `dotnet test -c Release --filter FullyQualifiedName~OverfitParallelTests`
   green, mutation red-then-restored, report.

No vertical slicing beyond this — the whole task is one small, single-file addition with one clear
correctness question (does the test fail when it should), not a feature with a "smallest working version."

## 11. Value against cost

| | |
|---|---|
| **Value** | Not independently measurable (no throughput/latency claim) — the value is risk reduction on a path that runs on every decode by default, previously provable only by a `[LongFact]` that needed a multi-GB model file and happened to catch the bug by a side effect, not by design. `value: not stated` in the client's own terms because there is no external client; I judge it high given the failure mode (silent data corruption in a customer-facing token stream) but that is a judgement, not a measurement |
| **Structural cost** | Checked: touches one production file (`OverfitParallel.cs`, 2–3 new lines' worth of additive `internal` methods), one existing test file (2 new `[Fact]`s), no new dependency, no new fixture, no `[LongFact]`, no new oracle-building cost (the mutation *is* the oracle). `find_references(ForDecode)` = 8 production call sites confirms the path's importance but the plan does not touch any of them |
| **Uncertain** | Whether the architect prefers option B or C over D for reasons beyond this plan's scope (e.g., a second future consumer of the claim protocol that would make extraction pay for itself) |
| **Recommendation** | Do now. Cheapest of the four options, closes a named, reviewer-flagged gap on a default-on hot path, and the acceptance criterion is unusually crisp (a specific mutation must fail a specific test) — there is little room for this to be "worth doing" and still not get built to spec |

## 12. Traceability

| goal | user need | task | acceptance criterion | verified by |
|---|---|---|---|---|
| Catch a silent torn-descriptor regression before it ships | `overfit-developer`/future editor needs a fast, reliable signal | Add seam (§5) + AC1.1/AC1.2 (§9) | AC1.1, AC1.2 | `dotnet test -c Release --filter FullyQualifiedName~OverfitParallelTests` + the named mutation going red |
| Do not add flakiness while closing the gap | this repository already carries `TG-T12`/`TG-T13` from timing-dependent tests | Prefer the deterministic state-machine test (F1/F3) over a threaded stress harness | AC1.1 has no `Thread.Sleep`, no timing dependency | inspection of the test's own body — no `Thread`/`Task`/`Monitor` symbol in it |
| Do not introduce a new hazard while adding the seam | F2's crash risk is real and would be worse than the gap it closes | Seam never writes `_decodeGen` (§5's code sketch, doc comment) | seam's own doc comment states the constraint | code review (`overfit-reviewer`) checking for any `_decodeGen` write outside `ForDecode` |

---

## BLOCKING QUESTIONS

No questions for a client — this task has none; the requester is this repository's own review process, and
the "user" is the next editor of this file. Both questions below are technical and go to whoever signs this
plan (`overfit-architect`) or picks it up (`overfit-developer`).

**For the architect**

1. **Option B/C/D (§5) — which seam, or is D acceptable as scoped?** I recommend D on cost and risk grounds
   (F4, review-risk row); B is viable but wider than needed and hits `CS0052` on `PaddedClaim` (F4); C is a
   bigger, riskier re-touch of code fixed and hand-verified only yesterday, for a testability goal rather
   than a correctness one.
   *If unanswered I will assume:* option D, as sketched in §5, is what `overfit-developer` implements.
2. **Should AC1.2 (exhaustion within a generation) be required in this pass, or is AC1.1 alone sufficient to
   close `XC-49`'s stated gap?** The task's own row is specific to the stale-generation defect; AC1.2 tests a
   different, currently-untested branch of the same method that I found while reading it, not something the
   row asked for.
   *If unanswered I will assume:* both ship together, since the marginal cost of AC1.2 is one more `[Fact]`
   in the same file against state already being seeded for AC1.1.

---

## SUGGESTED IMPROVEMENTS TO MY ROLE

One item, from this run specifically. The `overfit-delivery` skill's `GATES:` manifest format is documented
in two places (`.claude/skills/overfit-delivery/SKILL.md`) but **no plan file in `docs/specs/` currently uses
it** — I checked all eleven existing plans (`Grep` for `STATUS:|GATES:` across `docs/specs/`) and found
`STATUS:` lines in six, but a populated `GATES:` block in none. `XC-30`'s row in `docs/TASKS.md` says the
manifest was added 2026-08-13, so either no plan has passed through a gate since, or the convention has not
yet been exercised end to end. Worth a self-check the next time `overfit-release-readiness` or
`Scripts/plan_gate_check.py` runs against a real plan, since this file is the first test of the format and I
had no existing example to match against — I built the partial-manifest convention (only list what is
determinable now; omit what has not been asked) from the skill's own prose rather than from precedent.

---

# §A. Architecture review — `overfit-architect`, 2026-08-14

Reviewed against the code, not against the plan's description of it. Nothing was built, run or benchmarked:
a `[LongFact]` release gate holds the box and `Directory.Build.targets` refuses builds with
`error OVERFITMEASURING`. Everything below is established by reading source and by the semantic navigator;
§A.7 lists what that leaves unverified.

## A.1 Review verdict — numbered findings

**A1 — F3 is wrong, and it is the finding that decides the seam. Option D is not "deterministic (pure state
test)"; it writes process-global state that other tests in the same run are actively using, and its worst
outcome is a hung or crashed test host rather than a red test.**

F3 argues immunity from F2 because nothing wakes the pool if the test never writes `_decodeGen`. That
argument covers only *"the test wakes a sleeping pool"*. It does not cover *"the pool is already running,
because another test class is decoding right now"* — which is the case in the fast suite:

| evidence | established by |
|---|---|
| `Q4KDotKernelTests.ProjectParallel_IsBitIdenticalToProject` is a plain `[Fact]` with `outputSize = 96`, and `Q4KDotKernel.ProjectParallel` dispatches through `ForDecode` | `Tests/LanguageModels/Runtime/Q4KDotKernelTests.cs:67-94` read; `Sources/Main/LanguageModels/Runtime/Q4KDotKernel.cs:523` from `find_references(OverfitParallel.ForDecode)` |
| `CachedMultiHeadAttentionTests` has **8 plain `[Fact]`s**, all reaching `ForDecode` transitively via `CachedMultiHeadAttention.Decode` | `find_callers(Q4KGemvKernel.GemvParallel)`; `CachedMultiHeadAttention.cs:281`. Same for `CachedTransformerBlockTests` (7), `CachedFeedForwardBlockBatchedTests`, `MoeFeedForwardBlockTests`, `Qwen2MoeFeedForwardBlockTests` — none fixture-gated, none `[LongFact]` |
| test classes run **in parallel**: no `xunit.runner.json` anywhere in the tree, no `CollectionBehavior`/`DisableTestParallelization` attribute in `Tests/`, no parallel-related property in `Tests/Tests.csproj`; `xunit.v3` 3.2.2 defaults to parallel collections, one collection per class | `Glob`, `Grep`, `Directory.Packages.props:164` |

So the analyst's §7 "Assumption" (*"no other test class calls `ForDecode` concurrently"*) is **false**, and the
`finally`-reset offered as cheap insurance does not address it — a reset after the fact cannot undo an
interleaving. Three concrete outcomes, all from a test that never touches `_decodeGen`:

1. **Hang.** Test sets `_decodeChunkCount = 3` while a live dispatch is running with `chunkCount = 10`
   (`_decodePoolSize` is 10 on this box per `TG-T13`). Its workers and dispatcher then refuse every index
   `>= 3`, `_decodeRemaining` never reaches zero, and the dispatcher sits in the **pure spin with no timeout**
   at `OverfitParallel.cs:665-668` — *inside* `lock (_decodeGate)`, so every later decode blocks behind it.
   The run does not fail; it stops.
2. **Hang or partial output.** Test writes `_decodeClaim.Value` mid-dispatch: a colliding tag re-issues
   indices already executed (double execution, `_decodeRemaining` driven negative → same spin), a
   non-colliding tag refuses everything (same spin).
3. **F2's undefined behaviour, without ever writing `_decodeGen`.** Test sets `_decodeChunkCount` *above* the
   live dispatch's count; live workers claim indices the dispatch never populated and
   `ExecuteDecodeChunk` (`:736`) calls `_decodeChunks[index].Body` — a stale unmanaged function pointer with a
   foreign `Context`. Process crash, not a catchable exception.

The plan is right that F2 is a process-crash hazard and right that it was undocumented. It is wrong that
option D avoids it. Option D **narrows** it — and adds a hang mode that F2 did not have.

**A2 — Decision on the seam (question 1): none of A–D as written. The seam is the claim protocol
parameterised on its state, so the test drives a word it owns.** Detail in §A.3.

**A3 — a residual race in the fix itself: the claim word carries its generation, the exhaustion bound does
not.** Hand-traced, not executed. `ForDecode` publishes `_decodeChunkCount` at `:626` — a plain write, before
the descriptor loop `:629-639` and before the claim word is republished at `:644`. Between those points
`_decodeChunkCount` belongs to generation `G+1` while `_decodeClaim` still belongs to `G`, and
`TryClaimDecodeChunk`'s two guards are checked against different generations:

| step | state |
|---|---|
| dispatch `G` completes (`chunkCount 4`), `_decodeRemaining == 0`, dispatcher releases `_decodeGate` | `_decodeClaim = (G<<32)|4`, `_decodeChunkCount = 4` |
| a background worker still draining `G` is descheduled inside `TryClaimDecodeChunk(G, …)` after `var next = (int)current;` (`:717`) | it holds `current = (G<<32)|4` |
| dispatch `G+1` (`chunkCount 8`) takes the gate, writes `_decodeChunkCount = 8` (`:626`) and `_decodeRemaining = 8` (`:627`), starts writing descriptors | `_decodeClaim` **still** `(G<<32)|4` |
| the straggler resumes: tag `G` matches, `4 >= 8` is false, CAS `(G<<32)|4 → (G<<32)|5` succeeds | it executes `_decodeChunks[4]` — stale or half-written — and decrements `G+1`'s `_decodeRemaining` |

That is the same pairing-and-early-completion defect the fix was written to remove, surviving in a narrower
window. It is reachable because completion is counted in *chunks executed*, not in *workers that have left
the drain loop*: nothing stops a straggler re-entering the claim after `_decodeRemaining` hits zero.
`chunkCount = Math.Min(_decodePoolSize, totalWork)` (`:619`) varies dispatch to dispatch, so "the next
dispatch is larger" is ordinary, not exotic.

**This is out of scope for XC-49 and must not be fixed under it** (§A.6). It is a defect for its own row, and
it is *evidence for* the seam in A2 rather than against it: whatever closes it still has to answer "may this
generation take this index", from those same three inputs.

**A4 — the §5 sketch's `#pragma warning disable OVERFIT001` is unnecessary.** `OVERFIT001` is the
heap-array-allocation rule (`Sources/Analyzers/HeapArrayAllocationAnalyzer.cs:30`); `StackWeights.ForTest`
needs it because that factory does `new BlockWeights[layerCount]`. The claim seam allocates nothing.
Copying a precedent's pragma along with its shape leaves a suppression that says something untrue about the
code under it.

**A5 — the plan's own open question, answered: no `#if DEBUG`.** The suite runs `-c Release`
(`CLAUDE.md`, and `Tests/README.md`'s discipline), so a DEBUG-only seam is absent from the build under test.

**A6 — I agree with F1 and it is the answer to "would this become a third `TG-T12`/`TG-T13`".** With the A2
seam, no: the test reads and writes only locals, so there is no shared quantity for a loaded box to move —
determinism is structural rather than tuned. With option D it would be a third, and worse than either
existing one: `TG-T12` and `TG-T13` *fail* under load, option D can *hang*.

**A7 — the AOT verdict is right; its evidence needed upgrading.** A grep of `AotSmokeTest/Program.cs` for
`OverfitParallel` establishes only the absence of a *direct* reference, and AOT reachability is transitive.
Read in full (47 lines): the smoketest touches `typeof(OverfitClient)`, `SamplingOptions.Greedy` and a
`GenerationOptions` constructor, and performs no inference. `ForDecode` is not reachable, and nothing in
this plan adds reflection, LINQ, `Activator` or `Expression` in any case.

## A.2 System context and boundaries

| | |
|---|---|
| **Execution path** | **Inference.** `TryClaimDecodeChunk` is the decode pool's per-chunk claim, called from `ForDecode`'s caller-participation drain (`:658`) and `DecodeWorkerLoop`'s drain (`:824`) — the only two call sites in the solution (`Grep` over `Sources/`, and `find_references` returns no symbol because it is `private`). Reached from 8 production call sites via `ForDecode` (`find_references`) |
| **Assembly** | `Sources/Main` only, plus `Tests`. No boundary crossed, no new project reference, no new dependency |
| **Public surface** | Unchanged. The seam is `internal`, visible to `Tests` and `Benchmarks` through the existing `InternalsVisibleTo` |
| **AOT reach** | Not reachable (A7). Unchanged by this plan |
| **Allocation policy** | Hot path. The seam adds no allocation and no atomic operation; the production call sequence (`Volatile.Read`, `Interlocked.CompareExchange`) is unchanged |
| **Ownership / disposal** | Not applicable — no buffer, no `AutogradNode`, no `PooledBuffer<T>` |
| **Moat side** | Neither. `OverfitParallel` already ships in the AGPL surface; no real-time or throughput claim is made |

## A.3 The seam — decided

**The claim protocol becomes a pure function of the state it decides on, and production passes its own state
in.** Inputs: the claim word (by `ref`), the chunk count, the generation being claimed for. Output: the
verdict and the index. Nothing in it reads a static field.

That is the whole boundary. The exact shape is the developer's — a forwarding `private` method keeping both
call sites untouched, or changing the two call sites directly; either satisfies it.

**Why this and not D.** A seam that cannot reach production state cannot corrupt production state. Option D's
`SeedDecodeClaimForTest` is a loaded gun pointed at a live pool (A1); this one is inert by construction, so
the F2 constraint stops being a doc comment somebody has to keep honouring and becomes a property of the
signature. It is also strictly *smaller* than the option C the analyst rejected: no new type, no restructure
of `ForDecode`/`DecodeWorkerLoop`/`ExecuteDecodeChunk`, no accessibility change to `PaddedClaim` (F4's
`CS0052` never arises — the seam takes a `long`, not the padded struct).

**What it costs, stated plainly.** It edits a method fixed and hand-verified 24 hours ago, on a path with 8
production call sites. That cost is paid down by three constraints the developer carries:

1. **The loop body moves verbatim.** The only permitted differences are the two identifier substitutions
   (the claim-word field → the `ref` parameter, the count field → the parameter). A reviewer must be able to
   diff it as a rename, and `overfit-reviewer` should treat any other change inside that body as a finding.
2. **The claim word must be passed by `ref`, never by value or `in`.** By value compiles and silently
   CASes a copy.
3. **The generation check stays first** (`:711` before `:718`). It is what makes a per-call `chunkCount`
   parameter safe: a count can only change under `_decodeGate`, which implies a new generation, which the tag
   check rejects — *except* inside the publication window A3 names, which this plan does not touch and does
   not make worse.

**No `[MethodImpl(AggressiveInlining)]`, and no perf argument for or against the shape.** The forwarder is
loop-free and the callee is the same loop that exists today; whether that costs a call frame is a
measurement, and this path's throughput numbers are *already* flagged as un-remeasured since the 2026-08-14
change — `PB-12` (`docs/TASKS.md:342`) owns that, citing `OverfitParallel.cs:330-337`. Do not fold a
measurement into this task, and do not defend the refactor on performance grounds; the moment anyone asserts
it is neutral or better, that is a claim and `overfit-perf-claim-auditor` owns the verdict.

The test then drives a `long` it declares itself. AC1.1 and AC1.2 as written in §9 survive unchanged in
substance — seed a local instead of a static — and so does the named mutation and its predicted victim: a
bare increment that ignores `generation` makes AC1.1 return `true` and advance the local word, failing both
of its assertions, while AC1.2 still passes. No `finally` reset is needed, because there is nothing to reset.

## A.4 Quality requirements as parameters

| requirement | parameter | how measured | against what |
|---|---|---|---|
| Deterministic | zero reads or writes of `OverfitParallel` static state from the test; zero `Thread`/`Task`/`Monitor`/`Sleep` symbols in its body | inspection of the test body, and `Grep` for those symbols in the new test | `TG-T12`/`TG-T13`, both of which measure or mutate process-wide state and both of which fail only under load |
| Fast-suite | no `[LongFact]`, no model fixture, no environment variable | the test runs under `dotnet test -c Release --filter FullyQualifiedName~OverfitParallelTests` on a fixture-less Linux CI box | `Tests/README.md` discipline; CI has no fixtures |
| Load-bearing | the named mutation turns AC1.1 red, and **only** AC1.1 among the two | `overfit-mutate`, victim predicted before the run (§9) | a green mutation is a finding, not a pass |
| Costs the hot path nothing measurable | no allocation, no added atomic, no added field read on the claim path | inspection; any timing claim is deferred to `PB-12` | the decode pool's own 455 µs / 0 B vs `Parallel.For` 2059 µs / 925 KB, cited at `OverfitParallel.cs:763-765` — **not re-verified here and not a target of this plan** |

## A.5 Risks and order

| risk | cheapest thing that retires it | where |
|---|---|---|
| The refactor changes behaviour while claiming to be a rename | reviewer diffs the moved loop body as a two-identifier rename; the full suite is the second net | after step 2 |
| The mutation does not fail (fix uncovered after all) | run it — it is the acceptance criterion, not a follow-up | step 4, and it gates the task |
| A future edit adds a state-writing accessor beside the pure one, reopening A1 | none needed structurally; a doc line on the seam saying *why* it takes parameters is enough. `overfit-reviewer` treats any new `internal` member that writes `_decodeClaim`/`_decodeChunkCount`/`_decodeGen` as an automatic finding | ongoing |
| A3's residual race is real and someone conflates it with this task | file it as its own row; XC-49 proceeds unchanged because the seam is invariant under any fix to it | before implementation starts |

Order is the analyst's §10, with step 1 now closed by this section.

## A.6 Won't — added by the architect

- **Do not fix A3 under XC-49.** It is a correctness change to a synchronisation protocol and needs its own
  task, its own reasoning about publication order, and its own test. Bundling it would make the mutation
  result unreadable, because the test and the thing it tests would move in the same commit.
- **Do not widen `PaddedClaim`, `_decodeClaim`, `_decodeChunkCount` or `_decodeGen`** to `internal` — the
  decided seam removes the reason to.
- **Do not add the `#if DEBUG` guard** (A5) or the `OVERFIT001` pragma (A4).

## A.7 What remains unverified — read this before treating any criterion as met

Nothing in this plan has been compiled or run, by the analyst or by me. Specifically:

- **The mutation in §9 has never been executed.** It is a specified criterion, not a met one. Until
  `overfit-developer` runs it and reports the victim, the claim "the fix is covered" is unsupported.
- **The parallel-execution finding (A1) is established from static configuration**, not from observing a
  run: absence of `xunit.runner.json`, of `CollectionBehavior`, and of any parallelism property in
  `Tests.csproj`, plus xUnit v3's documented default. I did not watch two classes overlap.
- **A3 is hand-traced.** I could not build a reproduction, and a race that needs a straggler inside a
  ~microsecond publication window may be rare in practice. Rare is not absent, and the failure is silent.
- **The 455 µs / 0 B and the `+28%` / `+3%` decode-pool figures are cited, not re-verified**, and `PB-12`
  already records that they predate the change on this exact path.

## A.8 Gates

The analyst's `GATES:` manifest is correct as written and I have not edited it. Re-checked, with two notes:

- **AOT — `NOT_REQUIRED` stands, on stronger evidence** (A7 above, whole file read rather than grepped).
- **performance — `NOT_REQUIRED` stands, with a trigger**: no performance claim is made or permitted here;
  if the implementation or its review asserts the refactor is neutral or faster, that assertion is a claim
  and the gate fires. The standing obligation to re-measure this path is `PB-12`, not this task.
- **security / leak-scan / API-compatibility — `NOT_REQUIRED` stands.** The change is `internal`, inside
  `Sources/Main`, touches no parser, endpoint, config or public member.
- `verifier`, `reviewer`, `mutation-proof` are deliberately absent and **will** fire — `mutation-proof` is
  this task's acceptance criterion, not an extra. `Scripts/plan_gate_check.py` only demands the manifest from
  `IMPLEMENTED` onward, so the omission is correct at `APPROVED` and must be filled by the gates themselves.

Expected diff: `Sources/Main/Runtime/OverfitParallel.cs` and `Tests/Core/Runtime/OverfitParallelTests.cs`.
Two files, one assembly, no public surface.

## A.9 Answers to the analyst's blocking questions

1. **Seam** — neither B, C nor D. The state-parameterised pure claim helper of §A.3. D is rejected on the
   evidence in A1, not on preference.
2. **AC1.2** — ship both. Once the helper is pure it costs one `[Fact]` over a local `long`, and it is what
   distinguishes *"refused because superseded"* from *"refused because exhausted"*; without it AC1.1 alone
   cannot show the refusal came from the generation check.

## A.10 ADR

None. No public API, no assembly placement, no AOT reachability change, no persisted format, no dependency,
no open/commercial boundary — an `internal` test seam inside one file is not an ADR-class decision. The
reasoning that must survive lives in `OverfitParallel.cs`'s own comments, which is where this file already
keeps it.

## A.11 Sign-off

**Architecture review: signed 2026-08-14 by `overfit-architect`.** Execution path: **inference**.
AOT-reachable: **no**. Allocation policy: **hot path** (no allocation added; no timing claim permitted).
Implementation may start on §A.3 plus §9's acceptance criteria, subject to §A.6.
