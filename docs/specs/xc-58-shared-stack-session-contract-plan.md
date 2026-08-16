STATUS: APPROVED
Author: overfit-architect (no analyst round was run — see §0)
Architecture review: overfit-architect, 2026-08-15 — SIGNED
Date: 2026-08-15
Slug: xc-58-shared-stack-session-contract-plan

GATES:
  performance:        NOT_REQUIRED — no performance target, claim or comparison enters this plan. The guard
                       is one uncontended `Interlocked` operation per stack entry; the smallest published
                       decode interval on this path is 37.2 ms/token (`CachedLlamaSession.cs:434`), so the
                       addition is ~1e-6 of the budget and is not a claim anybody needs to measure. If the
                       developer produces any before/after timing, it is a performance claim and the verdict
                       belongs to `overfit-perf-claim-auditor`, not to this plan
  security:           NOT_REQUIRED — no parser, endpoint, gateway, tokenizer/audio decode or externally-fed
                       surface. The change adds a flag and a throw on an in-process API
  leak-scan:          NOT_REQUIRED — no config, log, fixture, host name, path or token touched
  AOT:                NOT_REQUIRED — the addition is `System.Threading.Interlocked` plus a throw: no
                       reflection, LINQ, `Activator`, `Expression`, `Array.Copy` or raw `ArrayPool`. Reach
                       is therefore irrelevant to the verdict. For the record, `Tests/AotSmokeTest/Program.cs`
                       touches only `typeof(OverfitClient)`, `SamplingOptions.Greedy` and `GenerationOptions`
                       — it never calls `CreateSession`, so the Llama decode path is not rooted through it
  API-compatibility:  NOT_REQUIRED for the *surface* — nothing public is added, removed or changed; the
                       guard is `internal`, the exception type is the existing public `OverfitRuntimeException`.
                       **But this is a behaviour change on a previously-"working" path** and it must appear in
                       `CHANGELOG.md` under `[10.1.0]`; see §7. `Scripts/api_compat_check.py` will not see it,
                       which is exactly why the CHANGELOG line is mandatory rather than optional
  release-readiness:  NOT_REQUIRED — no packaging, dependency or version-affecting surface. The ship verdict
                       for 10.1.0 is in §8 and is **ship**

`verifier`, `reviewer` and `mutation-proof` are intentionally absent rather than `NOT_REQUIRED`: a missing
line means "not yet asked", and nothing is implemented. All three apply once the work lands, and
`mutation-proof` is not optional — §6's mutation table is the acceptance criterion.

---

# XC-58 — one `CachedGptStack` serves every session of a `CachedLlamaInferenceEngine`: decide the contract

Plan file for task `XC-58` (`docs/TASKS.md:186`).

## 0. What this document is, and what it is not

There was no analyst round. The task was handed to me directly with the defect already verified by the main
session, and the question asked was a **contract** question, which is mine. I have not invented a business
requirement to fill the gap: the two places where the answer is a product decision rather than a technical
one are marked as such in §9 and one of them is called out in the report.

**The defect is real and I re-verified it independently.** `CachedLlamaInferenceEngine` declares
`private readonly CachedGptStack _stack;`, builds it once in the private constructor
(`_stack = new CachedGptStack(config.NLayers, config.DModel, …)`), and passes that same instance to every
`CachedLlamaSession` it constructs in `CreateSession` — `CachedLlamaInferenceEngine.cs:40`, `:132`, `:392`.
Each session does get its own `KeyValueCache` (`KeyValueCache.Create(…)` inside `CreateSession`), which is
what makes the sharing invisible.

## 1. Review verdict — numbered findings

**F1. The defect is worse-documented than the task states: the existing doc points the reader at the wrong
contract.** `CachedLlamaSession`'s summary says, verbatim, *"Thread-safety: one session per thread."*
(`CachedLlamaSession.cs:24`). That sentence tells a reader that N sessions on N threads is the supported
shape — which is precisely the shape that corrupts. It is not silence; it is a statement in the wrong
direction, and it must be replaced, not merely supplemented.

**F2. The two engines in this repository have opposite contracts, and neither says so.** The GPT-1/2 engine
gives every session its own stack: `CachedSlmSession(GPT1Model model) : this(new CachedGpt1ModelAdapter(model))`
(`CachedSlmSession.cs:58-59`) and `CachedGpt1ModelAdapter` builds `new CachedGptStack(…)` per instance
(`CachedGpt1ModelAdapter.cs:69`). So `CachedSlmSession` is already option (b) and `CachedLlamaSession` is
option (a). `CachedLlamaSession`'s own doc block is headed *"Differences from `CachedSlmSession` (GPT-1/2)"*
and lists four differences; this one is not among them. Whichever contract is chosen, that list is wrong
today.

**F3. There is a test named for the property that does not test it.**
`QwenInferenceSmokeTests.MultipleSessionsFromSameEngine_Independenet` (`:150`, `[LongFact("7s")]`) creates two
sessions from one engine, gives them **the same prompt**, and asserts the two greedy tokens are equal. That
assertion holds identically whether the sessions are independent or share every buffer — it cannot
distinguish the two. A skipped test whose name signs off on independence is worse than no test, because it
will be cited. This must be fixed in the same change (§5, T4).

**F4. The first consequence is contained; I confirmed the bound rather than inheriting it.**
`find_references` on `CachedLlamaSession.LastHiddenState` returns exactly two call sites, both tests
(`MergeDivergenceTests.cs:50`, `QwenLayer0CompareTests.cs:88`). No production caller. The same staleness
applies to the engine-level interpretability API (`EnableActivationCapture` / `GetLayerActivation` /
`LogitLens` all delegate to `_stack`), which is honest by placement — it is on the engine — but under two
sessions it silently answers about whichever session decoded last. The doc must say so.

**F5. The second consequence is not bounded that way, and the shipped server already worked around it by
hand.** `OverfitInferenceService` carries `private readonly SemaphoreSlim _ttsGate = new(1, 1);` and
`_embedGate = new(1, 1)` with the comment *"A SentenceEmbedder has one scratch arena; the TTS engine is
single-instance. Serialize each."* (`:38-40`), and `SynthesizeAsync` awaits that gate before
`SpeechExchange.Handle` (`:190`). Without it, two concurrent `POST /v1/audio/speech` requests would each run
`using var session = _llm.Engine.CreateSession();` (`OrpheusVoiceEngine.cs:135`) on **one** engine and decode
concurrently through one stack. The mitigation exists, it is correct, and the reason recorded beside it is
not the real one. That is a contract nobody wrote down being rediscovered empirically — which is the
argument for writing it down now.

**F6. The stack is not only shared, it is sized to the wrong context — and this reverses the naive cost
model for option (b).** `CachedLlamaInferenceEngine` passes `config.ContextLength` as the stack's
`maxSequenceLength` (`:138`), never the per-session `maxContextLength` that `CreateSession` accepts. The
dominant per-stack buffer is `CachedSingleHeadAttention._scoreScratch = new float[maxSequenceLength]`
(`:77`), allocated per head per layer. Numbers in §4. Consequence: `overfit serve` runs **one engine per
client** (`Commands.cs:264` — *"One client per session — each owns its KV cache (extra RAM); the model
weights are shared via mmap … N sessions ≈ 1× weights + N× KV"*), so it already pays the full stack N times,
and that comment omits the term. This is a separate task, not this one (§9, Q3).

**F7. Nothing shipped decodes two sessions of one engine concurrently.** Checked, not assumed:
`find_references` on `CreateSession` returns 170 sites, of which the non-test production callers are
`OverfitClient.LoadGguf`/`LoadPretrained`, `OverfitClient.EmbedSession` (`:343`), `QwenChatModel`,
`HuggingFaceChatModel`, `OrpheusVoiceEngine.GenerateAudioCodes` (`:135`) and `Demo/AgentDemo`. The two
that put a second session on a live engine — `EmbedSession` and Orpheus — are used sequentially within a
call, and the only concurrent host (the ASP.NET server) pools whole clients (`Commands.cs:269-289`) and gates
the two shared singletons. This is the evidence behind the ship verdict in §8.

**F8. What I could not check.** I did not build or run anything — a second agent (`XC-54`) is using the
machine, and every figure below is arithmetic over constructors I read, not a measured allocation. §4 says
how to measure it. Separately, `Tests/TestResults/` contains two hang dumps from today
(`testhost_27128_20260815T110107_hangdump.dmp`, `testhost_16748_20260815T112956_hangdump.dmp`). I did not
open them and they are almost certainly the other agent's run; I mention them only because a hang dump on
this branch has a known shape (`XC-49`/`XC-52`) and somebody should look.

## 2. The decision

**Contract (a): a session is a cheap view over the engine's stack. Sessions of one engine must not decode
concurrently. Interleaving them on one thread is supported; overlapping them in time is not.**

Reasons, in order of weight:

1. **It is what the code already promises everywhere else.** The one concurrent host in the tree serialises
   its shared engines by hand (F5). (a) writes down the contract that shipped code has already been built to.
2. **(b) is not a documentation change, it is a behaviour and memory change**, and it wants its own measured
   pass — including the right-sizing in F6, without which (b) is strictly worse for RAM at one session per
   engine, and with which it is strictly better (§4). Bundling an unmeasured memory change into a release
   being cut is the exact shape this repository has been burned by.
3. **Silent corruption is the failure to remove.** (a) plus a detector converts "wrong logits, no signal"
   into a named exception at the moment of misuse. That is the whole value of this task; the doc alone is
   not worth shipping.

**The precise contract, in the words that go into the XML doc:**

- Sessions created from one `CachedLlamaInferenceEngine` share the engine's transformer scratch. **At most
  one session of a given engine may be inside a decode, prefill or projection at any instant.**
- Sequential and cooperatively-interleaved use is supported: a decode step is atomic with respect to the
  shared scratch, and no shared state carries between steps except the members named below. Two sessions
  taking turns on one thread produce exactly the results they would produce alone.
- `CachedLlamaSession.LastHiddenState`, `CachedGptStack.GetLastFinalHidden`, `GetLastLogits`,
  `CachedLlamaInferenceEngine.GetLayerActivation` and `LogitLens` reflect **the most recent decode through
  this engine, by whichever session made it**. On an engine with more than one session they do not answer a
  question about "this session".
- To decode concurrently, create one engine per concurrent stream (what `overfit serve` does), or serialise
  (what `OverfitInferenceService` does).

**Rejected: serialise instead of throwing (a `lock` on the stack).** It is cheap and it would make the
current API honest — mutual exclusion is *sufficient* for correctness of the forward pass, since each decode
step is atomic with respect to the scratch. It is rejected because for a product whose identity is CPU
throughput, silently serialising two streams the caller believes are parallel is a worse failure than an
exception: the customer sees 1× throughput on an idle machine and nothing tells them why. It also does not
fix the `LastHiddenState` half. Recorded here so it is not re-proposed as a discovery.

**Rejected for now: (b), a stack per session.** Deferred with a number, not a shrug — see §4. It is the
contract a reader assumes and it is probably where this ends up, but it must arrive with F6's right-sizing
and a measured peak-RAM pass.

## 3. Boundaries and responsibilities

| question | answer |
|---|---|
| Execution path | **Inference.** `InferenceEngine`-family, caller-owned buffers, zero allocations per decode. No `ComputationGraph`, no `AutogradNode`, no tape. |
| Assembly | `Sources/Main` only. No new assembly, no new project reference, no new dependency. |
| Public surface | **Nothing new.** The guard is `internal` (tests reach it through the existing `InternalsVisibleTo`). The thrown type is the existing public `OverfitRuntimeException`. |
| Ownership / disposal | No new buffer, so no ownership tag to assign. The guard is a single `int` field on `CachedGptStack`, whose lifetime is the engine's. |
| Allocation policy | **Hot path — zero allocations per call.** The guard must not allocate: `try` / `finally` around an `Interlocked` flag, no closure, no `IDisposable` scope object that boxes. `PrefillAllocationTests` is the existing pin and must stay green. |
| AOT reach | Irrelevant to the verdict (§ GATES). `Interlocked` + `throw` is AOT-safe unconditionally. |
| Threading model | Not changed. No new `Parallel.For`, no `OverfitParallelFor`, no worker. The guard is held across the parallel projection kernels the stack already dispatches — that is one logical call, not re-entry. |
| Moat side | Open (AGPL). Correctness and a documented contract; nothing near the commercial line. |
| Source of truth for state | Unchanged. There is no durable state and no restart behaviour: this is a library type. |

## 4. Quality requirement: what (b) would cost, measured against the constructors

**Derived, not measured** — arithmetic over `CachedGptStack.cs:88-123`, `CachedTransformerBlock.cs:92-122`,
`CachedMultiHeadAttention.cs:103-131`, `CachedSingleHeadAttention.cs:73-87`, `CachedFeedForwardBlock.cs:56-69`,
with `Q8DotKernel.BlockSize = 32`, `Q4KDotKernel.SuperBlockElements = 256`, `Q4KDotKernel.GroupSize = 16`.
Model shapes read out of the real GGUF headers on this box on 2026-08-15 (`C:\qwen3b\`, `C:\gemma\`), not
recalled: Qwen2.5-3B-Instruct is `L=36, D=2048, H=16, KV=2, headDim=128, dFF=11008, vocab=151936,
ctx=32768`.

Per `CachedGptStack` instance, Qwen2.5-3B-Instruct:

| sized to | total | of which `_scoreScratch` | `_headOutputs` | FFN | logits + activations |
|---|---|---|---|---|---|
| model context (32768) — **what is built today** | **86.4 MiB** | 72.0 MiB | 4.5 MiB | 3.9 MiB | 0.9 MiB |
| session context (2048) — what F6's right-sizing would build | **18.9 MiB** | 4.5 MiB | 4.5 MiB | 3.9 MiB | 0.9 MiB |

For scale, the same session's `KeyValueCache` is **144.0 MiB** at ctx 2048 (F32) / 36.0 MiB (Q8), and
2304 MiB at ctx 32768. Gemma-2-2B: 15.0 MiB per stack at its 8192 context. Qwen2.5-0.5B: 46.4 MiB, of which
42.0 MiB is score scratch — the small model pays *proportionally more*, which matters for the low-end-hardware
identity.

**Read the table before concluding that (b) costs memory.** As the code stands, (b) adds 86.4 MiB per extra
session on a 3B model — +60% on top of a 144 MiB KV cache, and the reason to defer it. With F6's right-sizing
it costs 18.9 MiB per session and **removes** 86.4 MiB from the engine, so for the shipped `overfit serve`
shape (one session per engine) it is a net saving of 67.5 MiB per client. Break-even is around 4.6 sessions
per engine. "(b) costs memory" is true only for the version nobody would write.

**How to measure it** (for whoever takes the (b) pass, and it must be measured before that pass is designed):
`GC.GetTotalMemory(forceFullCollection: true)` either side of `new CachedGptStack(…)` with the real config,
or `GC.GetAllocatedBytesForCurrentThread()` around it, on the dev box, Release, no coverage
(`--collect` instrumentation makes this codebase 10x-900x slower and is meaningless here). Peak, not steady
state — this is a load-path allocation and peak is what decides whether a model fits.

## 5. The work

Ordered. T1 is the whole point; T2-T4 make it provable and honest.

**T1 — the guard.** A non-reentrant mutual-exclusion flag on `CachedGptStack`, entered on every method that
**writes** shared scratch, released in a `finally`. On a failed entry, throw `OverfitRuntimeException` whose
message names the cause and the two remedies (one engine per concurrent stream, or serialise).

Entry points that must be guarded — this list is the contract, and a missed one is the likely real defect:
`Decode`, `DecodeWithoutLogits`, `PrefillBatched`, `PrefillBatchedQuant`, `PrefillBatchedQuantAllRows`,
`ProjectLogits`, `ProjectLogitsFrom`, `ProjectLogitsBatched`, `LogitLensFromHidden`.

Readers stay **unguarded**: `LastFinalHidden`, `GetLastFinalHidden`, `GetLastLogits`, `GetLayerActivation`,
`ActivationCaptureEnabled`, `EnableActivationCapture`. Guarding a reader would turn the documented staleness
of §2 into a throw, which is a different and larger behaviour change.

**Two hazards the developer must design around, both already in the file:**

- **Nesting.** `Decode` calls `DecodeWithoutLogits` and then `ProjectLogits` (`CachedGptStack.cs:188`, `:190`);
  `ProjectLogits` calls `ProjectLogitsFrom` (`:506`); `LogitLensFromHidden` calls `ProjectLogitsFrom` (`:663`).
  A naive flag would self-trip on every single decode. Resolve it **structurally** — guarded facade, private
  unguarded core — not with a reentrancy counter: a counter that tracks depth without tracking the owner
  admits a second session while the first is inside, which is the bug the guard exists to catch. The shape
  is the developer's; the property is not.
- **Release on the exception path.** The flag must be cleared in a `finally`. Without it, one shape
  exception mid-decode leaves the engine permanently refusing every call — a live-lock dressed as a contract
  violation.

**T2 — the guard's own tests** (`Tests/LanguageModels/Runtime/Blocks/CachedGptStackTests.cs` already builds
synthetic stacks with `dModel: 2, layerCount: 1` and plain `[Fact]`, so these run in CI with no model
fixture — verified this run).

- `T2a` — one test **per guarded entry point** in the T1 list: with the guard held, that entry throws
  `OverfitRuntimeException`; after release, it succeeds. Same-thread, deterministic, no sleeps, no threads.
  This requires the guard to be reachable from the test — `internal`, which `InternalsVisibleTo` already
  covers.
- `T2b` — atomicity: two threads, a `Barrier`, each attempting entry; exactly one succeeds. **Every wait in
  this test carries a timeout and asserts the return value** (`ManualResetEventSlim.Wait(TimeSpan)` /
  `Barrier.SignalAndWait(TimeSpan)`). This is not stylistic: the fast suite runs classes in parallel and this
  repository's habit of unbounded completion waits has already produced hangs rather than red tests
  (`XC-49`, `XC-52`). A hang is a worse outcome than the defect.

**T3 — the documentation.** Replace `CachedLlamaSession.cs:24`'s *"Thread-safety: one session per thread."*
with the contract of §2, and state it on `CachedLlamaInferenceEngine.CreateSession` too — that is the method
whose name implies independence, and the doc a caller actually reads. Add the shared-stack difference to
`CachedLlamaSession`'s *"Differences from `CachedSlmSession`"* list (F2). Add the staleness note to
`LastHiddenState` and to the three interpretability members on the engine (F4).

**T4 — fix the false witness.** `QwenInferenceSmokeTests.MultipleSessionsFromSameEngine_Independenet` must
either be renamed to what it tests or, better, be made to test what it claims under contract (a): two
sessions, **different** prompts, decoded in an interleaved sequence on one thread, each session's token
sequence identical to the same session run alone. That is true under (a), false if the KV caches were shared,
and it is the assertion the name has been signing for.

**Not in scope, deliberately:** any change to how many stacks exist, any change to scratch sizing, any change
to `LastHiddenState`'s value. Those are (b) and F6.

## 6. Invariant, falsifier, and the mutations

**Invariant.** No two sessions of one engine are inside `CachedGptStack`'s writing entry points at the same
instant; if they try, the second one throws instead of proceeding.

**What would refute it.** A decode returning without a throw while another thread is inside the same stack.

| # | mutation | must turn red | deterministic? |
|---|---|---|---|
| M1 | guard entry always succeeds (delete the throw) | every T2a case | yes |
| M2 | never release (drop the `finally`) | T2a's "succeeds after release" half | yes |
| M3 | guard `Decode` but not one of the other eight entry points | the T2a case for that entry point | yes — this is why T2a is per-entry-point rather than one test |
| M4 | replace the atomic exchange with a plain read-then-write | T2b | **no — probabilistic** |

**M4 is stated as untestable-in-the-strong-sense on purpose.** T2b will usually catch a non-atomic flag and
may not on a given run; there is no deterministic test for a memory-ordering property here, and this
repository has twice signed an ordering argument that was wrong. What the plan can require is that the entry
be a single `Interlocked.CompareExchange`/`Exchange` and that the code say why. What it must not do is record
"tested" for a property that is reasoned.

**Verification oracle for the change as a whole:** the existing decode parity and coherence suites must be
unchanged — the guard alters no arithmetic, so any movement in `Q4KMDecodeParityTests`, `Q8DecodeParityTests`
or `BatchedPrefillParityTests` means the guard is not where it was meant to be. Plus `PrefillAllocationTests`
green (zero-alloc pin).

## 7. CHANGELOG — mandatory, not optional

Under `[10.1.0]`, in a behaviour-change (not API-break) entry, worded so a reader can tell whether it affects
them:

> Sessions created from one `CachedLlamaInferenceEngine` share the engine's transformer scratch and must not
> decode concurrently. Doing so previously corrupted both sessions' forward passes silently; it now throws
> `OverfitRuntimeException`. Sequential and interleaved use is unaffected. For concurrent streams, create one
> engine per stream.

`Scripts/api_compat_check.py` cannot see this — the surface is unchanged — which is precisely why the line is
required. A caller whose code "worked" now gets an exception; that is the intended outcome and it must be
findable.

## 8. Ship verdict for 10.1.0 — **ship, with (a)**

**Yes, 10.1.0 can ship with (a). I checked what it would break rather than assuming.** F7: no shipped path
puts two sessions of one engine into a concurrent decode. `overfit serve` pools whole clients, one engine
each; the two shared singletons (TTS, embedder) are already gated by `SemaphoreSlim(1,1)`; `EmbedSession` and
Orpheus use their second session sequentially within one call.

**Who could break.** A customer who built exactly the shape `CreateSession`'s name invites — several sessions
from one engine, decoded from a thread pool. Today that customer gets silently wrong output. After this
change they get an exception naming the cause, which is strictly better information, and their fix is one
line (one engine per stream). There is no scenario in which the throw takes away working behaviour: the
behaviour it interrupts was already producing corrupted logits.

**What would change the verdict:** evidence of a shipped or customer path that decodes two sessions of one
engine concurrently *and is currently believed to work*. I found none.

## 9. Open questions

**For the client / product owner**

- **Q1.** Is a thrown `OverfitRuntimeException` on concurrent use the behaviour you want, versus silently
  serialising? I recommend the throw (§2) and the plan is written for it. If you prefer serialising, say so
  before T1 — it changes the implementation, not just the message, and it is hard to withdraw once shipped.
  *If unanswered, I proceed with the throw.*
- **Q2.** Should contract (a) be treated as the permanent answer, or as an interim one with (b) scheduled?
  My reading is interim: (b) is what a reader assumes, and §4 shows it is affordable once the scratch is
  right-sized. This changes nothing in T1-T4 either way. *If unanswered, I record (a) as the shipped contract
  with (b) open.*

**For whoever owns the backlog**

- **Q3.** F6 (stack scratch sized to the model's max context, not the session's — 86.4 MiB against 18.9 MiB
  on Qwen-3B, paid once per engine and therefore once per `overfit serve` client) is a separate defect with a
  measured size. It should get its own task row rather than riding along here. I have not filed it — filing
  is not mine.

## 10. Definition of Ready

Problem understood; boundaries and responsibilities assigned (§3); execution path stated (inference); no
durable state to own; the one quality requirement that mattered is quantified against the constructors and
its measurement method named (§4); the risk that the guard self-trips or dead-locks the engine is named with
its resolution (§5); the irreversible part (a throw on a public path) is small, deliberate, and recorded in
the CHANGELOG (§7); nothing blocking is open. **Ready.**

**Architecture review:** reviewed on 2026-08-15 against the source. Execution path: **inference**.
AOT-reachable: **not through the smoketest, and irrelevant — the addition is AOT-safe regardless**.
Allocation policy: **hot path, zero allocations per call**.
