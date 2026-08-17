---
name: flaky-xgboost-alloc-test
description: Zero-allocation tests flake under the loaded full suite — two named instances beyond CLAUDE.md's flaky pair (XgboostParityTests.Prediction_IsZeroAllocation_AfterWarmup, AveragePoolAndDagTests.ResNetBlock_DAG_InferenceAllocatesZeroBytes).
metadata:
  type: reference
---

`DevOnBike.Overfit.Tests.Trees.XgboostParityTests.Prediction_IsZeroAllocation_AfterWarmup` failed once and
only once in six consecutive `dotnet test -c Release` runs on 2026-08-13 — on the **first** run after a full
solution rebuild. Five later runs on the identical tree were green (2610/0/273), including one with the
run's new test filtered out.

The published flaky pair is `PromptCacheReuseTests` and `RealEstateFullCycleTests`; this is a third, and
"one red is dismissible once you can name it" does not cover a name that is not on the list.

**Why cross-test interference is NOT the mechanism**: the assertion reads
`GC.GetAllocatedBytesForCurrentThread()` (`Tests/Trees/XgboostParityTests.cs:146,155`), which is
thread-isolated, so allocations on other xunit collections' threads cannot pollute it. The warm-up calls
`PredictBatch` / `PredictBatchParallel` / `Predict` once each before the counter is read, so a cold-JIT
first run is the shape that fits.

**Not captured, and worth capturing next time**: the byte count in the `AssertAllocation` message. The run
filter kept only the failing test's name, so how far over the floor it was is unknown — a small overshoot
and a large one point at different causes.

**A second, unrelated zero-alloc test with the same shape, 2026-08-15**:
`DevOnBike.Overfit.Tests.Integrations.Onnx.AveragePoolAndDagTests.ResNetBlock_DAG_InferenceAllocatesZeroBytes`
(`Tests/Integrations/Onnx/AveragePoolAndDagTests.cs:255`) went red in **1 of 4** full-suite runs inside a
`XC-58` mutation matrix — in an arm whose mutation was in `CachedGptStack`, which the ONNX DAG path does not
touch. It passed **5 of 5** when run alone with `--filter` on the restored tree. That isolation result is
weak evidence about behaviour under a loaded parallel suite, which is the condition it failed in.

**Practical rule**: when a mutation arm reddens a zero-allocation test in a subsystem the mutation cannot
reach, treat it as suite noise and say so — but name it, because the arm's victim set is evidence and an
unexplained extra victim reads as a real coupling.

Related: [[test-output-and-anchors]].
