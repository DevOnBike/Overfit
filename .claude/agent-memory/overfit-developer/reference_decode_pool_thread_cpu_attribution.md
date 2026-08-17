---
name: decode-pool-thread-cpu-attribution
description: How to attribute per-thread CPU to the OverfitDecode-* pool from a test, and why scoping does NOT remove the starved-spinner false-green bias
metadata:
  type: reference
---

Per-thread CPU attribution for the decode pool IS available on Windows, and the route is
identification-by-execution, not a thread-name lookup from outside.

- `ProcessThread` carries no name, but `ProcessThread.TotalProcessorTime` and `.ThreadState` work for the
  current process and every pool tid resolves (measured: 0 of 10 missing from `Process.Threads`).
- Get the ids by dispatching `OverfitParallel.ForDecode` from the test and reading `GetCurrentThreadId()`
  (kernel32 P/Invoke) plus `Thread.CurrentThread.Name` **inside the body**. The managed name is the string
  the pool assigned (`OverfitParallel.cs:460`), so it is exact and not an OS thread name.
- **The calling thread executes a chunk too** (observed as `.NET TP Worker`), so filter it out.
- A bounded barrier in the body — hold every chunk until all have arrived — is what forces the dispatch
  across distinct threads; without it one worker greedily drains the lot. All 10 identified in 1-3 rounds.
- `Process.Refresh()` is required between samples or every sample repeats the first.

**Scoping does NOT remove the false-GREEN bias from external load** — this was believed and is wrong. A
starved spinner accumulates less CPU per wall-second whatever the counter's scope. Measured with the park
mutation (`OverfitParallel.cs:868` spin budget → `int.MaxValue`): **9.97 cores idle box, 4.35 cores under
96 external spinners on 32 logical** — 2.3x suppression. What IS load-invariant is
`ProcessThread.ThreadState`: **100.0% `Wait` healthy, 0.0% under the mutation, 0.0% under the mutation
with the box oversubscribed 3x**. Assert on the parked fraction if you need immunity to box load.

Same-window divergence of the two instruments, with 4 burner threads inside the process: pool **0.01**
cores against process **4.01**. That is the number that shows scoping bought something.

See [[flaky-xgboost-alloc-test]] for the other environment-produced-quantity traps in this suite.
