---
name: reference-measurement-guard-mechanics
description: Two measured mechanics of the machine-exclusion guard — BenchmarkDotNet's generated build DOES compile Main and needs OVERFIT_MEASUREMENT_OWNER, and `-list classes` fires the xunit v3 framework ctor in 225 ms without running tests.
metadata:
  type: reference
---

Measured 2026-08-14 while building the `PB-7` build guard and the `XC-42` marker file.

**BenchmarkDotNet's generated per-job build reaches `Main.csproj`, and inherits the parent's
environment.** A build guard in `Directory.Build.targets` that refuses while
`Global\DevOnBike.Overfit.MachineMeasurement` is held therefore kills every benchmark: mutating out the
`Environment.SetEnvironmentVariable("OVERFIT_MEASUREMENT_OWNER", pid)` line in
`Sources/Benchmark/Program.cs` produced `// Build Error: ... OVERFITMEASURING` and
**`executed benchmarks: 0`** — while `dotnet run` still exited **0**. A script keying on the exit code
cannot tell a benchmark run that measured nothing from one that worked.

**`DevOnBike.Overfit.Tests.exe -list classes` constructs the xunit v3 test framework — so
`MeasurementExclusion` runs and can refuse — without executing a single test.** Measured 225 ms with the
mutex held externally, exit code 2, full message on stderr. That is the safe way to exercise the refusal
path from inside the suite: if the guard ever stops refusing, the child prints class names instead of
running the suite recursively.

`BeforeTargets="CoreCompile"` still fires when the project is up to date, so the guard works on an
incremental build too — verified as its own arm.

Related: [[reference-measurement-mutex-orphan]], [[reference-test-output-and-anchors]].
