# XC-26 — the 30-second blocking rent in `CompleteChat`

Plan file for task `XC-26` (`docs/TASKS.md:154`). **Written by `overfit-architect` on 2026-08-12.**

**No analyst round preceded this file, and the analyst sections are therefore absent rather than empty.**
The `docs/TASKS.md` row is the brief and is quoted below where it is load-bearing. Problem, users, success
metric, value-against-cost and acceptance criteria are analyst-owned and I have not written them; where the
decision needed one, it is recorded as an assumption or as a blocking question, never as a requirement I
invented.

---

## 1. What I verified, with the decisive lines

Everything below was read, not inferred. Symbol questions were resolved with `mcp__overfit-navigator__*`,
text questions with `Grep`.

| claim | evidence |
|---|---|
| the rent blocks for up to 30 s | `Sources/Server.AspNet/Services/OverfitInferenceService.cs:26` `RentTimeout = TimeSpan.FromSeconds(30)`, passed at `:92` `_pool.TryRent(RentTimeout, cancellationToken, out lease)` |
| the wait is a thread block | `Sources/Main/Serving/OverfitResourcePool.cs:123` `if (!_slots.Wait(timeout, cancellationToken))` — `SemaphoreSlim.Wait`, not `WaitAsync` |
| the token is honoured | same line; a cancelled wait throws and is **not** counted a rejection (no `Interlocked.Increment(ref _totalRejected)` on that path) |
| timeout ⇒ rejection, not exception | `OverfitResourcePool.cs:125-127` increments `_totalRejected`, sets `lease = default`, returns `false`; caller maps it to 503 at `OverfitInferenceService.cs:94` |
| production callers of `TryRent` | `find_references` → **one**: `OverfitInferenceService.CompleteChat` (`:92`). The other 13 hits are 1 `<see cref>` and 12 test sites in `Tests/Serving/OverfitResourcePoolTests.cs` |
| `CompleteChat` is public interface API | `Sources/Server.AspNet/Services/IOpenAiInferenceService.cs:45`, `void`. `find_implementations` → 2: `OverfitInferenceService`, and `FakeInferenceService` in `Tests/Server/OverfitAspNetServerIntegrationTests.cs:269`. `find_references` → 1 call site, `Sources/Server.AspNet/Endpoints/ChatEndpoints.cs:38` |
| default pool size is **1** | `Sources/Cli/Commands.cs:256` `int sessions = 1`, `:300` `new OverfitResourcePool<OverfitClient>(clients)` |
| the request thread is held for the whole generation anyway | `Sources/Server.AspNet/AspNetResponseSink.cs:28-30`: *"the request thread that entered the endpoint is held for the whole generation. That is the server's design"*; enabled by `EndpointHelpers.EnableSynchronousIO` at `ChatEndpoints.cs:37` |
| `Sources/Main` has essentially no asynchronous surface | one `async` member — `CachedLlamaSession.StreamGenerateAsync` (`:913`, `async IAsyncEnumerable<int>`) — plus `OverfitClient.SendAsync` (`:299`), which is `Task.Run` over the synchronous `Send`. No `ValueTask` anywhere in `Sources/Main` |
| `Sources/Main` already carries **nine** written "synchronous on purpose" constraints | `Mp3Reader.cs:89`, `VoiceProfileStore.cs:25`, `WavAudioSink.cs:92`, `ScalerParams.cs:38`, `OverfitClient.cs:209`, `BertConfigReader.cs:20`, `LlamaConfigReader.cs:33`, `QLoRAFineTuner.cs:67-76`, `CharacterTokenizer.cs:157` |
| OVERFIT040's gate is name+signature sibling resolution on the called type | `Sources/Analyzers/SynchronousIslandAnalyzer.cs:168-187` `HasAsyncSibling` → `called.ContainingType.GetMembers(called.Name + "Async")` |
| OVERFIT040 is `suggestion` **globally** | `.editorconfig:572-573`, `[*.cs]`; the note at `:567-571` records 12 remaining sites and *"it is NOT a sweepable rule"* |
| `Serving/` is AOT-compiled in the shipped binary | `Sources/Cli/Cli.csproj` `<PublishAot>true</PublishAot>`, and `Cli` references `Server.AspNet` + `Main`. It is **not** reached by `Tests/AotSmokeTest/Program.cs`, which touches only `OverfitClient`, `SamplingOptions` and `GenerationOptions` |

Two things I checked because they would have changed the answer, and which turned out **not** to be defects:

- **Return ordering is correct.** `Return` does `_available.Add(item)` *before* `_slots.Release()`
  (`OverfitResourcePool.cs:157-159`), so a released waiter always finds an item; the discarded result of
  `_available.TryTake` at `:133` is safe.
- **Shutdown does not race disposal.** `Commands.Serve` disposes the pool in a `finally` *after*
  `OverfitAspNetServer.Serve` returns (`Sources/Cli/Commands.cs:383-388`), i.e. after Kestrel has drained.
  A parked waiter therefore sees `RequestAborted` and the `OperationCanceledException` catch at
  `OverfitInferenceService.cs:98-102`, not an `ObjectDisposedException` from `_slots.Dispose()`.

---

## 2. Review findings

**F1 — the benefit of an async rent stops at the gate, and the site next door already says so.**
`AspNetResponseSink.cs:28-30` states that the request thread is held for the whole generation by design.
`TryRentAsync` would therefore change *when* the thread is taken, not *whether*: from
`queue + generation` to `generation`. It cannot improve chat throughput, which is bounded by `pool.Size`
either way. Anyone presenting this as a throughput win is wrong.

**F2 — what it would actually buy is isolation, and nobody has measured that it is needed.**
At saturation the threads parked in `TryRent` are `concurrent_requests − pool.Size`, and with the default
`sessions = 1` that is *every* concurrent request but one. Those threads do nothing, and thread-pool
injection past `ProcessorCount` is slow, so a chat burst can delay unrelated endpoints — `/metrics`,
`/v1/models`, `/health`. The server already exports `dotnet_threadpool_queue_length` and calls it
*"an early thread-starvation signal"* (`Sources/Server.AspNet/Endpoints/MetricsEndpoints.cs:101-103`), so
the harm is one this codebase already believes matters. **It has never been observed here.** That makes it
a hypothesis, not a design input.

**F3 — the quantity that would settle XC-26 is already measured and then thrown away.**
`OverfitResourcePool.Metrics` computes `meanWaitMs` (`OverfitResourcePool.cs:77-79`), but `PoolStatus`
(`Sources/Server.AspNet/Services/PoolStatus.cs:12`) carries only `Size, Active, Available, RejectedTotal,
PeakActive`, and `MetricsEndpoints.cs:61-65` exports exactly those five. **The mean queue wait never leaves
the library.**

**F4 — and that mean is biased low in the exact direction that hides this problem.**
`_waitTicksTotal` is only added to on the *success* path (`OverfitResourcePool.cs:130`, after the early
`return false` at `:127`). A rent that waits the full 30 s and is rejected contributes **zero** to the mean
wait; so does a cancelled one, which throws. The metric therefore excludes every one of the longest waits,
and dilutes what remains with uncontended near-zero rents. A reader would see a healthy mean wait on a
server that is timing out.

**F5 — the analyzer's blind spot is structural and will recur.**
`HasAsyncSibling` (`SynchronousIslandAnalyzer.cs:168`) can only see a blocking API that the BCL has already
paired with an `…Async`. **Every first-party blocking primitive this repository writes will be invisible to
OVERFIT040 forever**, and its callers with it. `TryRent` is the first instance, not a special case.

**F6 — part of artefact (b) is already written; what is missing is the cost and the decision.**
`IOpenAiInferenceService.cs:21-27` already explains why two of three methods are awaitable and one is not.
What no site states is *what the synchronous one costs* (30 s, default pool of 1, one thread per queued
request) and that it was **decided** rather than defaulted. `OverfitResourcePool.cs:105-109` currently
defers the choice to XC-26 — that sentence goes stale the moment XC-26 closes.

**F7 — allocation and AOT do not decide this, and should not be used to.**
A `TryRentAsync` would allocate only when it actually suspends: `SemaphoreSlim.WaitAsync` completes
synchronously when a count is available, which is the common case, and a `ValueTask<…>` returns without
allocating there. When it *does* allocate, we were about to wait seconds. Rent is once per request, not per
token, so `Sources/Main/README.md`'s per-call zero-allocation contract is not engaged. Async state machines
are Native-AOT-clean and touch none of the six banned symbols. **Rejecting (a) on allocation or AOT grounds
would be a wrong reason for a defensible answer.**

---

## 3. Decision

> **Option (b): accept the block, write the constraint at the sites, and do not add `TryRentAsync` now.**

The reason is not that (a) is unsound — per **F7** it is feasible, allocation-acceptable and AOT-safe. The
reason is the trade:

**What (a) costs, permanently.** Two public surfaces change and neither can be withdrawn.
1. `DevOnBike.Overfit` (the shipped library) gains a task-returning rental. `out lease` cannot cross an
   `await`, so the shape must change — `ValueTask<Lease?>` or a `TryRentResult` struct. This would be the
   library's first asynchronous *primitive*; today it has one streaming iterator and one `Task.Run` wrapper,
   and **nine** sites that say in writing "synchronous on purpose".
2. `IOpenAiInferenceService.CompleteChat` must become `Task CompleteChatAsync` — a public interface in
   `Server.AspNet`, 2 implementations, 1 call site.

**What (a) buys, measured: nothing yet.** Per **F1** it cannot improve throughput, and per **F2** the
isolation benefit is a hypothesis about a burst nobody has run against this server. This repository does not
accept an unmeasured performance or capacity argument, and I am not going to be the exception to that on the
strength of reasoning that sounds right.

**(a) is deferred, not refused.** The terms on which it should be reconsidered are in §6, and the spike that
would settle it is cheap. If the spike shows real starvation, (a) becomes the right change and gets an ADR
(public API in the shipped package is on the ADR list).

**No ADR is written for this decision.** Choosing *not* to add public API is reversible; choosing to add it
is not. An ADR becomes required if and only if (a) is later taken.

---

## 4. The artefact — what gets written, and where

Three sites. The wording below is a proposal, not a dictation; the developer may reword as long as each
site states the **cost** and that it is a **decision**.

**4.1 `Sources/Main/Serving/OverfitResourcePool.cs`, the existing OVERFIT040 block at `:98-116`.**
Lines `105-109` currently say the choice "is filed as XC-26 … which is where the choice belongs". Replace
that deferral with the decision. Keep everything else, including the paragraph at `:111-116` explaining why
this diagnostic is the only automated signal for the shape — that paragraph is still true and is now the
justification for **F5**'s follow-up. Proposed replacement for the deferral:

```text
// WHY IT IS NOT FIXED HERE — decided under XC-26, 2026-08-12, not deferred. The fix would be a
// `TryRentAsync` on this type, and it would change what resource is held during the wait (a
// continuation instead of a thread), NOT the wait itself. It cannot make the server faster: chat
// concurrency is bounded by `Size` either way, and the request thread is held for the whole
// generation regardless — see `Server.AspNet/AspNetResponseSink.cs`, which records that constraint.
// What it would buy is isolating unrelated endpoints from a chat burst, and that has never been
// observed on this server. Against that: `out lease` cannot cross an `await`, so the shape would
// change, and this is public API of the shipped `DevOnBike.Overfit` package — permanent, and the
// library's first asynchronous primitive.
```

**4.2 `Sources/Server.AspNet/Services/OverfitInferenceService.cs`, above `CompleteChat` (`:87`).**
This is the site the analyzer cannot reach, so a comment is the *only* record that will ever exist here.

```text
/// <para><b>This rent blocks the calling thread for up to 30 seconds, and that is a decision (XC-26,
/// 2026-08-12), not an oversight.</b> `OverfitResourcePool.TryRent` has no asynchronous form, so unlike
/// the two gates below there is nothing to await. With the CLI's default `--sessions 1` every concurrent
/// request but one queues here, each holding a thread that is doing nothing; on timeout the request is
/// shed with 503 and counted in `overfit_pool_rejected_total`.</para>
///
/// <para><b>Why it is not awaited:</b> making it awaitable buys back only the QUEUE. Once through the
/// gate the thread is held for the whole generation anyway — `AspNetResponseSink` writes each token from
/// inside the model's synchronous decode callback — so this cannot raise chat throughput, which is
/// bounded by the pool size. The cost of the change is a permanent asynchronous rental on the shipped
/// library's public API. Reopen it if `dotnet_threadpool_queue_length` shows real starvation.</para>
```

**4.3 `Sources/Server.AspNet/Services/IOpenAiInferenceService.cs:21-27`** — the existing paragraph explains
the split correctly but not its price. One added sentence naming the 30 s and pointing at `CompleteChat`.

**Sweep after editing.** Grep each whole file for `XC-26` and for the deferral wording before finishing —
an amendment that leaves the old "the choice belongs elsewhere" sentence in a different paragraph is worse
than no amendment, because the file then states both.

---

## 5. Boundaries and standing rules for this change

| | |
|---|---|
| **Execution path** | **inference**, request-scoped. Not the per-token hot path: rent happens once per request |
| **Assemblies touched** | `Sources/Main` (comment only), `Sources/Server.AspNet` (comments only). No new dependency, no dependency direction change |
| **Public API surface** | **unchanged** — that is the point of choosing (b) |
| **AOT reachability** | unchanged. `Serving/` is AOT-compiled via `Cli.csproj` (`PublishAot=true`) but is not reached by `Tests/AotSmokeTest`; comments alter neither |
| **Allocation policy** | unchanged (no code change) |
| **Ownership** | unchanged. `Lease` is a caller-disposed `readonly struct`; the pooled item's lifetime is the pool's when `ownsItems` is true |
| **Threading model** | unchanged: one thread per in-flight completion, concurrency bounded by `pool.Size` |
| **Moat side** | open. The OpenAI-compatible server is public surface. **Nothing here is to be described as "real-time"** and none of it references the gateway |
| **Verification oracle** | comment-only, so the oracle is the existing suite staying green plus a whole-solution build with **zero new diagnostics**. There is no behaviour to test, and no test should be added pretending otherwise |

---

## 6. Quality requirements, as parameters — and the spike that would reopen (a)

Each is measurable; none is asserted.

| parameter | how it is measured | status |
|---|---|---|
| threads parked in `TryRent` under a chat burst | `process_num_threads` and `dotnet_threadpool_queue_length` from `/metrics`, sampled during a burst of N concurrent chat requests at `--sessions 1` | **never measured** |
| latency of an unrelated endpoint during that burst | wall-clock of `GET /v1/models` and `GET /metrics` while the burst runs | **never measured** |
| observed queue wait | `PoolMetrics.MeanQueueWaitMs` — **computed but not exported** (F3), and biased low (F4) | not observable today |
| 503 shed rate | `overfit_pool_rejected_total`, already exported (`MetricsEndpoints.cs:65`) | observable |

**SPIKE-1 (prerequisite, ~1 task): export the queue wait, and fix its bias.** Add the wait to `PoolStatus`
and to `/metrics`; count timed-out and cancelled waits into the wait total, or export them separately so the
longest waits stop being invisible. Small, contained, and it is the instrument every other question here
needs. **It should be filed as its own task and is not part of XC-26's closure.**

**SPIKE-2 (settles (a)): one burst, two readings.** With SPIKE-1 landed, run N concurrent chat requests
against `--sessions 1` and read `dotnet_threadpool_queue_length` plus the latency of `GET /v1/models` during
the burst. **Reopen (a) if and only if unrelated endpoints are measurably delayed.** Note the standing rule
that applies to whoever runs it: this is a load measurement on a live server, so both arms must be shown
*capable* of a verdict before either is read.

**Any claim of a speedup, a latency improvement or a throughput ratio arising from either spike is a
performance claim and the verdict belongs to `overfit-perf-claim-auditor`, not to this plan.** Nothing in
§3 rests on one: the decision rests on an unmeasured *benefit* and a permanent *cost*, which is a scope
argument, not a performance argument.

---

## 7. Recommendation — make the analyzer see this class of cost (F5)

This is, in my judgement, worth more than either branch of XC-26, and it should be **filed as its own task,
not folded into this one.**

**Shape:** an attribute — e.g. `[OverfitBlocking("…")]` in `DevOnBike.Overfit.Diagnostics`, alongside the
existing `OverfitHotPathAttribute` (`Sources/Main/Diagnostics/OverfitHotPathAttribute.cs:25`) — which
`SynchronousIslandAnalyzer.HasAsyncSibling` treats as equivalent to "has an async sibling". Marking
`TryRent` would then make the rule fire on `CompleteChat`, and on every future first-party blocking
primitive's callers. The precedent for an attribute in `Main` driving analyzer behaviour is already
established: `OverfitPerfAnalysis.cs:90-93` matches `OverfitHotPathAttribute` by name + namespace chain.

**Two honest weaknesses, which is why this is a recommendation and not a decision:**

1. **The signal would land in a channel nobody reads.** OVERFIT040 is `suggestion` globally
   (`.editorconfig:572-573`) and **a build does not print suggestions** — the `.editorconfig` note at
   `:559-565` records exactly that trap. A new report on `CompleteChat` would be invisible unless it gets
   its own diagnostic id with its own severity, which is a larger decision than it first looks.
2. **The attribute is public API in `Sources/Main`** and would need an ADR, plus the standing analyzer
   contract: an entry in `AnalyzerReleases.Unshipped.md`, a severity in `.editorconfig`, and a test — a
   rule with no test has only been shown not to fire on clean code.

---

## 8. Operability (the server, unchanged by this plan)

- **How an operator notices:** `overfit_pool_rejected_total` rising is the shed signal;
  `overfit_pool_available_sessions` at 0 is saturation. Neither shows the *wait*, which is F3.
- **How an operator fixes it without editing state:** `overfit serve … --sessions N` raises pool size;
  the cost is N× KV cache (weights are shared via mmap — `Sources/Cli/Commands.cs:275-276`).
- **On restart:** the pool is rebuilt from freshly loaded clients; there is no durable state and nothing to
  recover. In-flight rents are drained by Kestrel before the pool is disposed (verified, §1).
- **When a dependency is unreachable:** not applicable — the pool has no external dependency.
- **The alert that cannot fire when the thing is dead:** `overfit_pool_rejected_total` disappears with the
  process, so "the server has stopped shedding" and "the server has stopped" read identically. Any alert on
  it needs `absent()`, the same lesson the guard already paid for.

---

## 9. Files this plan authorises a developer to change

Comments only, three files:

- `Sources/Main/Serving/OverfitResourcePool.cs` — amend the OVERFIT040 block at `:98-116`
- `Sources/Server.AspNet/Services/OverfitInferenceService.cs` — add the paragraph above `CompleteChat`
- `Sources/Server.AspNet/Services/IOpenAiInferenceService.cs` — one sentence at `:21-27`

**Not authorised by this plan:** any signature change, any new public member, any pragma (there is nothing
to suppress — the rule never fires at `CompleteChat`), and any edit to `docs/TASKS.md`, whose status column
is the lead's.

---

## BLOCKING QUESTIONS

Nothing here blocks the (b) artefact — §4 can be written today. These block the **spikes** and the
reconsideration of (a).

**For the client**

1. **Is the server expected to serve concurrent users at all, or is it a single-user local runtime?** The
   CLI defaults to `--sessions 1`. If concurrency past one user is not a supported scenario, SPIKE-2 should
   not be run and (a) is closed permanently rather than deferred.
   *If unanswered I will assume:* small-team concurrent use is in scope (the `--sessions` flag, the 503
   shed path and the Docker image all imply it), so (a) stays deferred rather than closed.
2. **Is 30 s the intended client-visible ceiling for "server busy"?** It is a hard-coded constant
   (`OverfitInferenceService.cs:26`), not configurable. A shorter timeout would reduce the held-thread cost
   without any API change at all — but the right value is a product question about what a client should
   wait for, not a technical one, and I will not pick it.
   *If unanswered I will assume:* 30 s stands and is documented as a decision, not tuned.

**For the analyst / task owner**

3. **Is SPIKE-1 (export the queue wait, fix its low bias — F3/F4) in scope for XC-26, or a separate task?**
   I have written it as separate, because XC-26's brief is a comment-or-API decision and SPIKE-1 changes
   observable output. Say if you want it folded in.
   *If unanswered I will assume:* separate task.
4. **Should the analyzer generalisation (§7) be filed?** It is the only one of these that prevents the
   class of defect recurring, and it is the one thing here I would spend budget on first.
   *If unanswered I will assume:* filed as a task, unassigned.

---

## Sign-off

**Architecture review: SIGNED, 2026-08-12 — option (b).**

- **Execution path:** inference, request-scoped (not the per-token hot path).
- **AOT-reachable:** `Serving/` is compiled into the AOT-published `overfit` binary; not reached by
  `Tests/AotSmokeTest`. Unchanged by this plan.
- **Allocation policy:** unchanged; no code change.
- **Beyond the standing rules in `CLAUDE.md`:** only the sweep obligation in §4 (grep the whole file for the
  superseded deferral wording) and the §9 authorisation boundary.

Option (a) was rejected **for now** because its benefit is unmeasured (F1, F2) while its cost — a permanent
asynchronous rental on the shipped library's public API plus a public interface change in `Server.AspNet` —
is not. It is not rejected on allocation or Native-AOT grounds (F7); those would be wrong reasons. §6 names
the measurement that would reopen it.
