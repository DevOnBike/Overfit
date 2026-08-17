---
name: measurement-mutex-orphan
description: Benchmark exit 2 / "did not return valid JSON" both trace to a live orphan holding Global\DevOnBike.Overfit.MachineMeasurement; AbandonedMutexException does NOT cover it.
metadata:
  type: reference
---

`Global\DevOnBike.Overfit.MachineMeasurement` is held by exactly two things, verified 2026-08-13 by
grep of the whole tree: `Sources/Benchmark/Program.cs:114` and `Tests/MeasurementExclusion.cs:65`,
byte-identical names. A benchmark that cannot take it exits **2** (`BusyExitCode`, `Program.cs:117`),
which is deliberately distinct from a benchmark failure.

**The nuance that turns a five-step diagnosis into a one-step one.** `Program.cs:49-52` catches
`AbandonedMutexException` and treats it as acquired — but *abandoned* means the owning **thread exited
without releasing**. A still-alive orphan process is **not** abandoned: `WaitOne(TimeSpan.Zero)` simply
returns `false` and there is no exception at all. So **exit 2 with no exception means "something alive is
holding it", never "something died holding it"** — go looking for a live process, not for a stale lock.

**XC-42, 2026-08-13 (reported to me by the team lead, not observed by me).** A stalled test run's
`testhost` was killed, but **`DevOnBike.Overfit.Tests.exe` survived as an orphan** — xUnit v3 makes the
test project an executable, so it is a *grandchild* the kill does not reach — and it kept the mutex.
Every later run then refused to start with `Test process did not return valid JSON (non-object)`, a
message that points at the discoverer and not at a held lock, which is what made it expensive. After
killing the orphan, discovery returned 2892 tests in 16.0 s.

**Related, and it is why this keeps happening: `PB-7` (OPEN, `docs/TASKS.md:329`) — `dotnet build` does
NOT join the machine-exclusion scheme.** Only the test harness and the benchmark hold the mutex;
`Directory.Build.props` has no `OVERFIT_MEASUREMENT_OWNER` guard. Consequence I caused on 2026-08-13: I
checked for a running benchmark, saw a live `testhost`, judged a single-project build harmless, and built
— which rewrote assemblies the test host had loaded and became the prime suspect for its stall. **A live
`testhost`/`vstest` means do not BUILD, not merely do not measure**; the failure lands on whichever side
you are not watching. Two design traps recorded on `PB-7` for whoever implements it: `dotnet test` builds
first, so a hold-the-lock design deadlocks the very commands it protects; and a per-project probe fires 26
times.

See also [[navigator-index-stale-after-move]] for the other reason a build here exits non-zero for
reasons that are not the code.
