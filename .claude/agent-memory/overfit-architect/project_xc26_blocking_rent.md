---
name: xc26-blocking-rent
description: XC-26 SIGNED 2026-08-12 — chose (b) write the constraint, rejected TryRentAsync; plus the OVERFIT040 structural blind spot and the pool's low-biased wait metric.
metadata:
  type: project
---

`OverfitInferenceService.CompleteChat` blocks up to 30 s in `OverfitResourcePool.TryRent`
(`SemaphoreSlim.Wait`). Signed option **(b)** — write the constraint at the site — in
`docs/specs/xc-26-blocking-rent-plan.md`. `TryRentAsync` deferred, **not** refused.

**Why (a) was rejected, and the reason people get wrong:** an async rent cannot improve throughput —
`AspNetResponseSink.cs:28-30` records that the request thread is held for the *whole generation* by design,
so awaiting the rent buys back only the QUEUE. The cost is two permanent public surfaces (`out lease` cannot
cross an `await`, so `DevOnBike.Overfit` gains its first async primitive; `IOpenAiInferenceService.CompleteChat`
must become task-returning). **Do NOT reject it on allocation or AOT grounds** — `ValueTask` + `WaitAsync`'s
synchronous fast path make allocation a non-issue, rent is once per request not per token, and async state
machines are AOT-clean. Those would be wrong reasons for a defensible answer.

**Durable facts found on the way:**
- `Sources/Main` has essentially no async surface: one `async IAsyncEnumerable` (`CachedLlamaSession.StreamGenerateAsync`)
  + `OverfitClient.SendAsync` (`Task.Run` wrapper), and **nine** written "synchronous on purpose" constraints.
- **OVERFIT040 structural blind spot:** `SynchronousIslandAnalyzer.HasAsyncSibling` resolves `{Name}Async` on
  the *called type*, so **every first-party blocking primitive is invisible forever**, callers included.
  Proposed fix = an `[OverfitBlocking]` attribute treated as an async-sibling equivalent (precedent:
  `OverfitHotPathAttribute` matched by name in `OverfitPerfAnalysis.cs:90`). Weakness: OVERFIT040 is
  `suggestion` globally and **builds do not print suggestions**, so it needs its own id/severity to be seen.
- `PoolMetrics.MeanQueueWaitMs` is computed in `Main` and **dropped at the `PoolStatus` boundary** — never
  exported to `/metrics`. Worse, `_waitTicksTotal` is only added on the SUCCESS path, so timed-out and
  cancelled waits contribute zero: the mean systematically excludes the longest waits.
- CLI default is `--sessions 1`, so at any concurrency > 1 every request but one queues.

Related: [[amendment-sweep]] (the plan requires grepping the whole file for the superseded deferral wording).
