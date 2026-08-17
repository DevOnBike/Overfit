---
name: traps-this-box
description: Measurement traps confirmed by my own runs on this box, with the numbers that confirmed them.
metadata:
  type: reference
---

# Traps confirmed on this box (Ryzen 9 9950X3D, 32 logical), 2026-08-14

- **The decode-pool liveness probe is free and decisive.** `OverfitParallel.CountDispatches` +
  `DispatchCount` are incremented ONLY inside `OverfitParallel.For` (`OverfitParallel.cs:554`), never in the
  spin-pool branch. So a decode window reading 0 dispatches/token proves the pool is live, and reading N/token
  proves it is not — settling trap #3 ("the lever was not live") in one line instead of a disassembly.
- **Run 1 of a timed set is systematically the slowest** even after a 16-token warm-up: Qwen3 Q8_0
  63.93 / 69.38 / 70.75 within one process. Take best-of-N, never mean-of-N, and never a single run.
- **Per-arm spread within one process is 1.8-14.8%** across my 24 processes; the cross-process paired ratio
  is far tighter (Qwen ±2.4 points, Phi ±2.0, Bielik ±1.6). **Pair the arms and compare ratios**, do not
  compare arm means across the whole session.
- **The documented ±3-4% cross-process floor held.** `docs/performance-discipline.md:152` — my canary
  (single-thread FP chain, same process, before and after the timed block) drifted −3.1% to +4.4% and did not
  correlate with the arm.
- **`Stopwatch.StartNew()` is 40 B** and lands inside an allocation window if you put it there — my dispatch
  harness reported "40 B" for a zero-allocation path until I traced it to my own scaffolding. Trap #4 in
  miniature.
- **The benchmark mutex is `Global\DevOnBike.Overfit.MachineMeasurement`.** Any harness I write takes it and
  exits 2 if held, exactly as `Sources/Benchmark/Program.cs:133` does.
- **A scratch project under `D:\Overfit\.claude\` must carry an empty `Directory.Build.props`** or it inherits
  the repo-root in-repo analyzers, and OVERFIT008 makes raw `Parallel.For` a build error — i.e. the arm under
  measurement fails to compile.
