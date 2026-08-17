---
name: decode-pool-wake-rendezvous
description: How to assert a parked decode worker was woken without asserting a race outcome, and the ordering mutation that stays GREEN anyway
metadata:
  type: reference
---

**To make a concurrency assertion schedule-invariant, make the desired outcome NECESSARY FOR TERMINATION
rather than likely.** "A pool worker executed a chunk" is a race outcome — `ForDecode` makes the caller
participate (`OverfitParallel.cs:764`), so a legal schedule has the caller draining everything. Fix: every
chunk body blocks until a body has entered on a thread named `OverfitDecode-*`. The caller then blocks in
its first chunk, cannot claim another, and >=1 chunk stays claimable by a worker and nobody else.
Implemented in `Tests/Core/Runtime/DecodePoolWakeTests.cs`.

- **Bodies must set an abort flag and RETURN on deadline, never throw or spin on.** With N chunks and a
  throwing body the caller pays N deadlines; with abort-and-return it pays exactly one and the dispatch
  returns normally, so the verdict is an xUnit assertion instead of a hang.
- **Managed `Thread.ThreadState` works and is not Windows-only** — a parked decode worker reads
  `Background, WaitSleepJoin`, a spinner reads `Background`. Get the `Thread` objects by capturing
  `Thread.CurrentThread` inside a barrier-held `ForDecode` body (10 of 10 workers in 2 rounds). This is the
  cross-platform alternative to the `ProcessThread` route in
  [[decode-pool-thread-cpu-attribution]] — verified on Windows only.
- Park poll after the identification dispatches: **22-27 ms** to all 10 parked. Healthy wake: microseconds.

**Measured 2026-08-16, mutations on `OverfitParallel.cs`:**

| mutation | result |
|---|---|
| `Monitor.PulseAll(_decodeParkLock)` removed | new test RED in 30 s; **rest of the suite 0 of 2659 failed** |
| `Volatile.Write(ref _decodeGen, …)` moved AFTER the pulse (classic lost wakeup) | **GREEN** — 9 of 10 chunks still on distinct workers |

**The GREEN is a property of the mutation, not of the test.** `PulseAll` fires inside the lock, so a woken
waiter cannot return from `Wait` until the dispatcher releases it — and the very next instruction writes the
generation. The window is a few instructions wide, so it is a race that essentially never fires rather than
a permanent failure, and no schedule-invariant assertion can catch it. **The publication order is held by a
comment (`OverfitParallel.cs:734`) and by nothing executable.**
