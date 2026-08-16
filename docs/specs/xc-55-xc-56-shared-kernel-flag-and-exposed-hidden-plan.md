STATUS: APPROVED
Author: overfit-architect
Architecture review: overfit-architect, 2026-08-15 — SIGNED
Date: 2026-08-15
Slug: xc-55-xc-56-shared-kernel-flag-and-exposed-hidden-plan

GATES:
  performance:        NOT_REQUIRED — no performance target, claim or comparison enters this plan. Part A adds
                       `[ThreadStatic]` to one `internal static bool` read ~5x per layer per PREFILL (never on
                       the per-token decode path — §A.4). A TLS read against a Q4_K GEMM is not a claim worth
                       measuring; if the developer produces a prefill timing for any reason it is not a
                       performance claim, but say so in the report and hand the verdict to
                       `overfit-perf-claim-auditor` rather than judging it
  security:           NOT_REQUIRED — no parser, endpoint, gateway or externally-fed surface. Part B reads a
                       model fixture the tests already load
  leak-scan:          NOT_REQUIRED — no config, log, host name or token touched. Part B records a fixture
                       path (`C:\qwen3b\qwen.bin`) that four test files already name in source
  AOT:                NOT_REQUIRED — `[ThreadStatic]` is AOT-clean and adds no reflection, LINQ, `Activator`
                       or `Expression`. `Tests/AotSmokeTest/Program.cs` does not reach
                       `BatchedQuantProjection`; NOT re-read this run, so treat that as carried from
                       `xc-52`'s reading of the same 47-line file rather than as freshly verified
  API-compatibility:  NOT_REQUIRED — nothing public is added, removed or changed. `DisableRepackedKernelsForParity`
                       is `internal`; `CachedLlamaSession.LastHiddenState` keeps its signature AND its meaning
  release-readiness:  NOT_REQUIRED — one attribute on an internal test hook, one test-support type, and test
                       changes. No shipped behaviour, no packaging, no dependency

`verifier`, `reviewer` and `mutation-proof` are **absent rather than `NOT_REQUIRED`**: a missing line means
"not yet asked". All three apply once the work lands, and `mutation-proof` is not optional — §A.6 and §B.6
*are* the acceptance criteria.

---

# XC-55 and XC-56 — a process-wide kernel flag, and a hidden state that was never the defect

Plan file for `XC-56` (`docs/TASKS.md:186`) and `XC-55` (`docs/TASKS.md:187`).

**One file for two tasks because they share nothing but a session.** They are independent: different
subsystems, different fixes, different acceptance criteria. Do them in either order. What they do share is a
lesson this repository keeps re-learning, recorded once in §C: *a per-instance accessor over process-wide
mutable state reads as a defect in whatever happens to be looking at it*.

---

## Part A — `XC-56`: the shared kernel flag

### A.1 Review findings on the task row

**Finding A1 — the row's diagnosis is correct and I add nothing to it.** Re-verified by reading:
`BatchedQuantProjection.DisableRepackedKernelsForParity` (`Sources/Main/LanguageModels/Runtime/BatchedQuantProjection.cs:56`)
is a plain `internal static bool`, and both scopes restore a constant —
`PrefixKvCacheParityTests.cs:37` and `BatchedPrefillParityTests.cs:358` are each
`public void Dispose() => BatchedQuantProjection.DisableRepackedKernelsForParity = false;`. The row's two
defects are two defects and are fixed separately below.

**Finding A2 — serialising the two parity classes does NOT fix the reported failure, and this is the ruling
the task turns on.** `find_references` on the field returns five production reads. One of them,
`BatchedQuantProjection.cs:414`

```csharp
var tiled = (w.IsPrepacked || UseTiledPrefillQ4K) && !DisableRepackedKernelsForParity
```

is on the path taken by `BatchedQuantProjectionTiledDispatchTests.Dispatch_PrepackedWeight_UsesTiled_EvenWithFlagOff`
(`Tests/LanguageModels/Runtime/BatchedQuantProjectionTiledDispatchTests.cs:22`, calling `Dispatch` at `:61`).
That test is a plain **`[Fact]`** whose only gate is `Avx2.IsSupported && Fma.IsSupported` — it builds its own
512x64 Q4_K weight, needs no fixture, and therefore **runs in the fast suite, in its own collection, in
parallel with everything**. Its assertion is bit-equality against `Q4KGemvKernel.GemmTiled` (`:76-81`), so a
concurrent `true` on the flag takes it off the tiled path and reddens it — which is exactly the
`6.63813305` vs `6.63813257` reduction-order signature the row recorded, and exactly why it passes alone.

**A collection containing the two parity classes leaves this third class outside it.** So the client's
"run them sequentially" is *necessary-if-you-choose-it* and not *sufficient*, and the mechanism has to be
chosen on that basis rather than on the two writers alone.

**Finding A3 — the collision is a `[LongFact]`-gate phenomenon, not a plain-`dotnet test` one.** Both writers
are `[LongFact]` (`PrefixKvCacheParityTests.cs:43`; `BatchedPrefillParityTests` is five `[LongFact]` plus one
tokenizer-only `[Fact]` that never touches the flag). So they execute only under `OVERFIT_RUN_LONG=1`, and
that is the run in which the fast-suite reader above shares the process with them. Any reproduction or
regression check must set that variable; a green `dotnet test -c Release` proves nothing about this defect.

**Finding A4 — the fast suite's ~22 s is not at stake either way, and no option in this plan touches it.**
Recorded so the "measure the cost of disabling parallelism" branch can be closed without measuring: none of
the options below disables parallelism.

**Finding A5 — a sibling flag has the same defect, is out of scope, and needs its own row.**
`UseTiledPrefillQ4K` (`:31`) is written by the fast-suite `[Fact]` above (`:57`, sets it `false`) and by
`RepackedSidecarEngineE2ETests` (`Tests/LanguageModels/Loading/RepackedSidecarEngineE2ETests.cs:39`, a
`[ModelFact]` that sets it `true` and whose no-sidecar arm *needs* it true to reach the path it asserts
bit-equality on). Different classes, so they run in parallel. **This one cannot take Part A's fix** — see
A4's constraint in §A.4 — so do not widen this task to cover it. Recommend a new `XC-` row.

### A.2 System context

| | |
|---|---|
| Execution path | **inference**, prefill only — `DisableRepackedKernelsForParity` is read by `DispatchQ6K` (`:376`), `DispatchQ4K` (`:414`), `DispatchTiledQ4K` (`:524`), `DispatchTiledQ6K` (`:618`) and `CachedMultiHeadAttention.DecodeBatchedQuant` (`:680-681`). Not on the per-token decode path |
| Assembly | one attribute in `Sources/Main`; everything else in `Tests` |
| Public surface | unchanged — the field is `internal`, visible to `Tests` through `InternalsVisibleTo` |
| AOT reach | not reachable from `Tests/AotSmokeTest`; `[ThreadStatic]` would be AOT-clean regardless |
| Allocation policy | hot path (prefill), zero allocations per call — unchanged; a TLS read allocates nothing |
| Ownership | not applicable, no buffer |
| Moat side | neither; internal test hook |

### A.3 The options, and what each gives up

| option | fixes defect 1 (restore) | fixes defect 2 (sharing) | cost |
|---|---|---|---|
| **A. save/restore only** | yes | **no** — a concurrent reader still sees the wrong kernel | none |
| **B. + `[Collection]` over the two writers** | yes | **no** (Finding A2) | serialises two `[LongFact]` classes; buys nothing |
| **C. + `[Collection]` over writers *and every reader*** | yes | yes, **while the membership list is right** | membership is "any test whose result depends on kernel layout" — unenforceable, silently violated by the next test added |
| **D. `[ThreadStatic]` on the flag** | (needs A too) | **yes, structurally** | production semantics change; a silent-inert failure mode, named in §A.4 |
| **E. thread the layout choice through the call** | n/a | yes, and cleanly | adds a parameter to `Dispatch` and to three block-level batched entries across six production call sites, for a test hook |

**Decision: A + D. No collection, no runner config, no change to suite parallelism.**

D is chosen over C because C's correctness is a list somebody has to keep, and this repository already
carries two open rows (`TG-T12`, `TG-T13`) of "an assertion that holds only under some schedules". D is
chosen over E on proportionality: E is the design I would pick if this were a product knob, and it is not —
it is a test hook whose only writers are two test classes. **Revisit E** if a third flag needs per-caller
scoping, or the first time an engine is driven from a thread other than the one holding a scope.

### A.4 Design

**A1. `[ThreadStatic]` on `DisableRepackedKernelsForParity` (`BatchedQuantProjection.cs:56`).**

Two facts make this admissible, and both were checked rather than assumed:

- **Every read is on the calling thread, before any fan-out.** `:414` and `:376` are plain reads in the
  dispatcher; `:524` and `:618` are evaluated while *constructing* `TiledContext` / `TiledQ6KContext`, which
  is then passed by pointer to `OverfitParallel.For` (`:529`, `:539`, `:626`, `:636`). No worker body reads
  the field. So `[ThreadStatic]` cannot make the flag inert inside the parallel region — which is the
  failure this repository has already paid for once with `OVERFIT_TILED_PREFILL` (a dead flag ran both A/B
  arms identically).
- **The field's correct default is `default(bool)`.** A `[ThreadStatic]` field initialiser runs on the first
  thread only. `DisableRepackedKernelsForParity` has no initialiser, so nothing is lost. `UseTiledPrefillQ4K`
  (`:31`, `= Q4KGemvKernel.TiledPrefillEnabled`), `UseTiledPrefillQ6K` (`:39`, `= true`) and
  `UseWeightStationaryQ4K` (`:27`, `= true`) all do have one, and would silently lose it on every other
  thread. **This is why exactly one field changes**, and the doc comment must say so — otherwise the next
  reader "tidies" the attribute onto its neighbours.

**Named failure mode, to be written into the field's doc comment rather than assumed away:** the flag must be
set on the thread that drives the prefill. A test that opens the scope and then runs the engine from another
thread — an `async` continuation, `Task.Run`, `GenerateStreamAsync`'s `await Task.Yield()` — silently gets the
repacked kernel and the parity assertion then compares two different kernels. Verified today: neither
`PrefixKvCacheParityTests.cs` nor `BatchedPrefillParityTests.cs` contains `async` or `await`, so it cannot
happen now.

**A2. `Tests/TestSupport/NonRepackedKernelScope.cs`** — one `readonly struct`, the *only* writer of the flag
in the test tree:

- constructor stores the previous value in a field and sets `true`;
- `Dispose()` restores the stored value.

Delete both private duplicates (`PrefixKvCacheParityTests.cs:33-38`, `BatchedPrefillParityTests.cs:354-359`)
and the factory at `BatchedPrefillParityTests.cs:352`. The comment at `PrefixKvCacheParityTests.cs:27-31`
asked for exactly this once a third case appeared; carry its substance into the new type's doc, including the
`*.gguf.repack` sidecar sentence, which is the part that explains why the hook exists at all.

**A3. Nothing else.** No `xunit.runner.json`, no `[Collection]`, no `CollectionDefinition` — the test tree
has none today (`git grep` returns zero `[Collection(` and zero `CollectionDefinition`), and introducing the
mechanism here would advertise a guarantee that is not the operative one.

### A.5 Quality requirements

| requirement | parameter | how measured | baseline |
|---|---|---|---|
| the fix does not slow prefill | no per-token cost at all; one TLS read replaces one static read, ~5x per layer per prefill | not measured, and deliberately: the read sits beside a Q4_K GEMM measured at 0.61-1.78 TFLOP/s (`BatchedQuantProjection.cs:372-374`) | none needed; a claim either way would need `Sources/Benchmark` and there is no claim |
| the fast suite stays fast | unchanged — nothing is serialised | `dotnet test -c Release` wall time before/after | ~22 s, from the task assignment; not independently re-measured this run |

### A.6 Acceptance criteria and the mutation each pins

**The lead's criterion cannot be met by any option in §A.3, and the plan says so rather than pretending.**
"A mutation restoring `false` must redden these tests under a parallel run" is unreachable, because at scope
entry the previous value *is* `false` in every current run — save/restore and restore-`false` are
observationally identical unless a scope is nested or a second writer exists. So the two defects get two
separate pins, each with a test that can actually see it. Both are fast, model-free and deterministic.

| # | mutation | must redden | if it stays green |
|---|---|---|---|
| M1 | `NonRepackedKernelScope.Dispose()` → `= false` | **new** `NonRepackedKernelScope_Nested_RestoresOuterValue`: open outer, open inner, dispose inner, assert flag is still `true`; dispose outer, assert `false` | the scope test is vacuous — defect 1 is unpinned |
| M2 | remove `[ThreadStatic]` from the field | **new** `NonRepackedKernelScope_IsNotVisibleFromAnotherThread`: hold the scope, start a thread, read the flag there, assert `false` | the isolation is unpinned and the fix is a coincidence |
| M3 | none — regression check | with `OVERFIT_RUN_LONG=1`, `PrefixKvCacheParityTests` + `BatchedPrefillParityTests` + `BatchedQuantProjectionTiledDispatchTests` in **one** run, **three times**, all green | one green trio is also what a benign schedule looks like |
| M4 | none — negative control for M3 | the same trio on a build with `[ThreadStatic]` reverted must be red **at least once** in the same session | M3 proves nothing; the collision was never reproduced on this box |
| M5 | none — anti-inert control | `BatchedPrefill_MatchesSingleToken_OnRealQwen` passes (`maxDiff < 1e-2`, `BatchedPrefillParityTests.cs:176`) | the flag has gone inert: without it the same comparison measures ~0.44 (`:184`), so a pass is the evidence that the hook is still live |

M4 is the one to run first. It costs one revert-build and one trio run, and without it M3 is unfalsifiable.

---

## Part B — `XC-55`: the exposed hidden state

### B.1 Ruling, first, because it changes what the task is

**The accessor is not the defect. The oracle constants describe a different model file, and the mechanism has
a name.** Evidence in §B.2-B.4. The row's instruction — do not close this by re-deriving the oracle — has been
honoured: re-recording is now the *right* fix, but only because the question "which tensor is returned" has
been answered first, and the answer has to be written down beside the new numbers or the next reader is back
where this started.

### B.2 Finding B1 — the row's "wrong row" candidate is refuted for this test's path

`L0_TwoToken_HiddenStateVsPython` calls `session.Reset([151643, 151644])`. `Reset(ReadOnlySpan<int>)` is
`Reset()` + `Prefill` (`CachedLlamaSession.cs:279-283`), and `Prefill`'s batched branch requires
`promptTokens.Length >= BatchedPrefillThreshold` (`:224`) where `BatchedPrefillThreshold` is **16** (`:1129`).
Two tokens take the single-token loop (`:241-259`): token 0 through `DecodeTokenWithoutLogits`, token 1 —
the last — through `DecodeToken`.

Both end in `CachedGptStack.DecodeWithoutLogits`, whose last three lines are the whole answer
(`CachedGptStack.cs:242-246`):

```csharp
// Save hidden state BEFORE final norm.
// LastFinalHidden matches Python: x before rms_norm(x, fg2, eps).
new ReadOnlySpan<float>(current, 0, DModel).CopyTo(_lastFinalHidden);

ApplyFinalNorm(current, weights, _finalHidden);
```

`_lastFinalHidden` and `_finalHidden` come from the **same** `current`, and `ProjectLogits` (`:504-508`)
projects `_finalHidden`. So on this path the exposed tensor *is* the tensor the logits were computed from,
for the *last* token, and there is no row index to get wrong. Row indexing exists only in
`PrefillBatchedQuant` (`:403`, `rows - 1`) and `PrefillBatched` (`:335`, `rows - 1`), neither of which this
test reaches — and both of which already take the last row.

### B.3 Finding B2 — two of the row's three constraints do not hold

**(a) "The logits are right" is weaker than stated.** The test asserts `Assert.Equal(198, top1.i)` — the
**argmax**, nothing more (`QwenLayer0CompareTests.cs:114-115`). The top-1 *value* reads `11.3741` against the
docstring's `12.3511`, ~8% apart, and **nothing asserts it**. An argmax survives a genuinely different hidden
state whenever the model is confident; it is not evidence that the vector matches.

**(b) "`L0_ChatPromptLogits` still passes, so the forward has not drifted since May" is vacuous.** That test
(`QwenLayer0CompareTests.cs:124-153`) contains **no `Assert` at all** — it writes ten logits to the output
helper and ends. It passes on any forward pass whatsoever, including one that returns zeros.

The one surviving constraint is the useful one: `L0_LogitsAfterReset` **does** assert
(`Assert.Equal(33975, top1.i)` and `|v - 15.5608| < 0.1`, `:53-55`) and passes. That is a real one-token
oracle match — at **position 0**.

### B.4 Finding B3 — the mechanism, and the measurement that supports it

`C:\qwen3b\qwen.bin` has mtime **2026-08-07 14:29**; the oracle constants were recorded 2026-05-17. Commit
`265fd77` (2026-08-07) added `permute_rope_rows` to `Scripts/convert_llama.py` — the HF `rotate-half` →
adjacent-pair permute of the Q and K weights *and biases*. Its docstring states the property that decides
this task:

> "Getting this wrong is close to invisible: **position 0 is the identity rotation**, so the model still
> answers short prompts, and the damage grows with context. Measured 2026-08-07 on Qwen2.5-3B, the .bin
> written WITHOUT this permute and the GGUF read with the correct convention agreed to cosine 0.999896 on a
> one-token prompt and drifted to 0.9789 at three tokens."

`Scripts/forward_multitoken.py:18-33` rotates **adjacent pairs** while reading the `.bin`, so the oracle
assumes the permuted layout. Constants recorded in May against a pre-permute file therefore match at
position 0 and diverge from position 1 — which is precisely the observed pattern: `L0_LogitsAfterReset`
(1 token, position 0) green, `L0_TwoToken` (position 1) red.

**Measured this run**, `GgufVsBinaryLayerDivergenceDiagnostics.DoesTheDisagreementDependOnPosition`,
`OVERFIT_RUN_LONG=1`, Release, this dev box, 2026-08-15, `C:\qwen3b\qwen.gguf` (FP16) against
`C:\qwen3b\qwen.bin` (FP32):

| tokens | cosine layer 0 | cosine last layer |
|---|---|---|
| 1 | 0.999896 | 0.990525 |
| 2 | **0.999631** | **0.947242** |
| 3 | 0.999908 | 0.992236 |

Layer 0 agrees at every length, so **the file on disk carries the correct rotary convention** — the .bin was
re-converted after the permute landed, and the May constants describe the file it replaced.

**The residual I cannot explain and am not going to explain away.** Last-layer cosine dips to 0.947 at
exactly two tokens, against 0.990 and 0.992 either side. It is non-monotonic, so it is not depth-accumulated
precision, and the two arms differ in weight precision (FP16 GGUF vs FP32 bin) so it is not a defect claim
either. The reading I will commit to is narrower: **`[BOS, im_start]` is a numerically ill-conditioned
probe** — two special tokens, out of distribution — and its final hidden state moves under perturbations that
leave one- and three-token prompts alone. That has a direct consequence for the fix, in §B.5.

### B.5 Finding B4 — a second shared-state defect, same family as Part A, out of scope

`CachedLlamaInferenceEngine.CreateSession` passes the engine's **single** `_stack` to every session it
creates (`CachedLlamaInferenceEngine.cs:40`, `:132`, `:392`). `_lastFinalHidden`, `_finalHidden`,
`_currentHidden`/`_nextHidden` and `_layerActivations` all live there. So `session.LastHiddenState` returns
the last hidden state decoded by **any** session of that engine — and two sessions of one engine decoding
concurrently corrupt each other's forward pass, on a `public` API with nothing saying so.

Not the cause of this failure (the test has one session), and **not in scope**: it is a public-API contract
question, not a test fix. Recommend a new `XC-` row. It does constrain the spike below — a self-consistency
check is only valid on a single-session engine.

Also worth one line to whoever next reads that file: `CachedGptStack.LastFinalHidden` (`:595`) returns
`_lastFinalHidden` (**pre**-norm) while `CachedGptStack.GetLastFinalHidden(Span<float>)` (`:597-598`) copies
`_finalHidden` (**post**-norm). Two members one prefix apart return different tensors, and `GetLastFinalHidden`
is `public` on a `public class`, so a rename is an API break and an ADR. **Fix the doc comment, not the name.**

### B.6 Design

**B1 — the spike, and it comes first because it is the only oracle-free discriminator.**
Add to `QwenLayer0CompareTests` (or run as a throwaway first, then keep it — see below) a **self-consistency**
assertion that needs no Python:

```
engine.LogitLens(session.LastHiddenState, lens);   // public: CachedLlamaInferenceEngine.cs:209
// lens must reproduce session.LastLogits
```

`LogitLens` takes a pre-final-norm hidden and applies the same `ApplyFinalNorm` + `ProjectLogitsFrom` the real
decode uses (`CachedGptStack.cs:644-650`, documented `:638-640`). One session per engine (§B.5).

- **Fails** → the accessor exposes a different or overwritten tensor; §B.6-B2 is wrong and the fix moves into
  `CachedLlamaSession`/`CachedGptStack`. Stop and re-plan.
- **Passes** → the accessor is exonerated and the oracle is the stale side. Proceed.

**Keep it**, at both prompt lengths — 2 tokens (single-token loop) and ≥16 (batched, `PrefillBatchedQuant`'s
`rows - 1`). It is the assertion that pins "the accessor returns the tensor the logits came from" *for all
time and without an oracle*, which is what nobody had. It is nearly tautological against the code as read
today, and that is the point: it is a change detector for exactly the edit that would break it.

**B2 — re-record the oracle, with its provenance.** Run `Scripts/forward_multitoken.py --bin C:\qwen3b\qwen.bin`
and replace the constants in `L0_TwoToken_HiddenStateVsPython`'s docstring and body. Beside them record, in
the docstring: the date, the file's mtime/size (`2026-08-07 14:29`, 13 593 731 124 bytes), and one sentence
saying the previous constants predate the `265fd77` RoPE row permute. Numbers without that line will be
re-litigated the next time the fixture is re-converted, which is the failure this task *is*.

**B3 — assert the logit value, not only the argmax.** The `11.3741` vs `12.3511` gap has never been checked
by anything. Add the assertion with the re-recorded value and a tolerance chosen the way
`L0_LogitsAfterReset` chose its `0.1`.

**B4 — change the hidden-state comparison from absolute to cosine, and say why.** §B.4's 0.947 says this
prompt's final hidden moves under FP16-vs-FP32 weight precision; an absolute `maxDiff < 0.05` over four
components will therefore break again on any precision-touching change, and the next reader will have no way
to tell that from a real defect. Use cosine over the **whole** `DModel` vector with a bound stated in the
docstring, and keep a loose absolute check on the four components as a sanity floor. This is the same lesson
`BatchedPrefillParityTests.cs:194-200` already records for argmax on an out-of-distribution prompt — link it
rather than restating it.

**What this plan does NOT authorise:** re-converting or otherwise touching `C:\qwen3b\qwen.bin`. It is
multi-gigabyte, outside the repo, and nothing here needs it changed.

### B.7 Acceptance criteria and the mutation each pins

| # | mutation | must redden | if it stays green |
|---|---|---|---|
| M6 | `CachedLlamaSession.LastHiddenState` → `_stack.GetLastFinalHidden`-equivalent, i.e. return the **post**-norm `_finalHidden` | B1's self-consistency assertion (the lens double-norms) | B1 is vacuous; it is pinning nothing about which tensor is exposed |
| M7 | in `CachedGptStack.PrefillBatchedQuant` (`:403`), change `rows - 1` to `0` | B1's **batched-length** assertion | the batched row index — the row's own named candidate — is still unpinned, and only the 2-token case was ever covered |
| M8 | perturb the re-recorded `logit[198]` constant by 0.5 | B3 | the value assertion's tolerance is wider than the thing it is asserting |
| M9 | none — regression | `L0_TwoToken_HiddenStateVsPython` and `L0_LogitsAfterReset` both green under `OVERFIT_RUN_LONG=1` | — |

**M7 is the one that matters most**, because it is the hypothesis this task was opened on. If B1 at a batched
length does not redden under it, the plan has replaced one unpinned claim with another.

---

## C. Risks, and what retires each

| risk | retired by | order |
|---|---|---|
| A2's ruling is wrong and some *other* fast test also depends on kernel layout | already bounded: the flag is read only for Q4_K/Q6_K weights (`:376`, `:414`); the two other fast tests reaching `Dispatch` — `CachedFeedForwardBlockBatchedTests`, `Qwen2MoeFeedForwardBlockTests` — build **F32** weights, which take `kind == 3` (`:360-363`) and never read the flag | done, by reading |
| `[ThreadStatic]` goes inert in the parallel region | done: all four dispatcher reads happen while building the context passed to `OverfitParallel.For`; M5 is the standing control | done, by reading |
| the collision does not reproduce on this box, so M3 is unfalsifiable | **M4, the negative control — run it first** | first |
| B1 is tautological and pins nothing | M6 and M7 | with B1 |
| the re-recorded oracle breaks again on the next fixture change | B2's provenance line and B4's cosine bound; this is mitigated, not eliminated | — |

## D. Decisions, and why there is no ADR

Nothing here is hard to reverse. `[ThreadStatic]` on an `internal` field is one attribute; the TestSupport
scope is test-only; `LastHiddenState` keeps its signature and its meaning. **No ADR.** The two genuinely
ADR-shaped questions this work surfaced — the engine-wide shared `CachedGptStack` behind per-session
accessors (§B.5), and whether `GetLastFinalHidden` should be renamed — are both out of scope and both need
their own rows before anyone designs them.

## E. Operability

Not applicable: library and test change, nothing runs.

## Sign-off

**Architecture review: reviewed 2026-08-15 against the code, and against one measurement taken this run
(§B.4). SIGNED.**
Execution path: **inference** (Part A prefill; Part B diagnostics over prefill). AOT-reachable: **no**.
Allocation policy: **hot path — unperturbed**. Assembly: one attribute in `Sources/Main`, everything else in
`Tests`. Public surface: **unchanged**. Ownership: not applicable. Verification oracles: §A.6 and §B.7, and
they must be run — **a surviving M1, M2, M6 or M7 is a finding, not a pass**, and M4 comes before M3.

**What I did not check, stated so it is not read as clean:** I did not execute the failing trio myself, so
Part A's collision is carried from the task row's measurement rather than reproduced here — which is exactly
why M4 exists. I did not re-read `Tests/AotSmokeTest/Program.cs` this run. I did not run
`Scripts/forward_multitoken.py`; §B.4's conclusion rests on the layer-0 table above plus the converter's own
docstring, not on new oracle numbers.

## BLOCKING QUESTIONS

**None.** Both parts are decided on evidence in the code and, for §B.4, on a measurement taken this run. Two
questions are **recorded but not blocking**, because their default is the status quo and no part of this plan
depends on the answer:

1. *For the client* — the two shared-state defects deliberately left out (§A.5's `UseTiledPrefillQ4K`,
   §B.5's engine-wide `CachedGptStack`): file them as rows now, or leave them until one of them costs a red
   test? **Absent an answer I assume "file them now, fix them later"** and have changed neither.
2. *For the analyst* — `L0_ChatPromptLogits` asserts nothing (§B.3). Is a test that only prints in scope for
   this task? **Absent an answer I assume no** and have left it alone; it is a separate decision about
   whether it should assert or be deleted.
