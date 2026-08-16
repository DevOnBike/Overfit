# Bug hunt — `Sources/Main/Diagnostics`

**Scope:** `Sources/Main/Diagnostics` (7 files: `ValueStopwatch.cs`, `OverfitTelemetry.cs`, `Throughput.cs`,
`OverfitHotPathAttribute.cs`, `Contracts/{DotNetMemorySnapshot,EvolutionGenerationMetrics,MapElitesIterationMetrics}.cs`),
plus caller-following for `ValueStopwatch`, `Throughput`, `OverfitTelemetry` as instructed.

**Timestamp (UTC):** 2026-08-02 19:42
**Commit:** `996a161` (branch `gimli`)
**Score:** 3 defects confirmed = **6 points**
**Ended by:** scope — the directory is small (7 files) and was read in full; caller-following covered every
non-test call site of `ValueStopwatch`, `Throughput` and `OverfitTelemetry` found by grep. Stopped well
inside the 10-minute cap with the scope exhausted of the highest-yield leads, rather than running the clock
out.
**README:** `Sources/Main/Diagnostics/README.md` exists and was read first. It documents the
`Stopwatch.StartNew` ban (matches `BannedSymbols.txt`) and the `for`/`foreach`-vs-declared-type measurement
(matches `Sources/Main/README.md` and `Sources/Analyzers`) — no drift found there.

---

## Findings

### 1. `[OverfitHotPath]`'s own doc comment overstates which rules it escalates

**What breaks:** `OverfitHotPathAttribute.cs`'s XML doc says: *"the in-repo Overfit performance analyzers
(`OVERFIT001`–`OVERFIT014`) escalate from their per-directory configured severity to a hard build error"*
inside a marked member. That is false for two rules inside the claimed range: `OVERFIT008`
(`RawParallelForAnalyzer`) and `OVERFIT012` (`FinalizerAnalyzer`) never call
`OverfitPerfAnalysis.Report`/`HotPathRule` — their `SupportedDiagnostics` never even lists `OVERFIT900`. All
other rules in 001–014 do route through the shared `Report()` helper and correctly escalate.

**Where:** `Sources/Main/Diagnostics/OverfitHotPathAttribute.cs` (doc comment) vs.
`Sources/Analyzers/RawParallelForAnalyzer.cs`, `Sources/Analyzers/FinalizerAnalyzer.cs`,
`Sources/Analyzers/OverfitPerfAnalysis.cs` (`Report`/`IsInHotPath`).

**How anyone would notice today:** they would not. A raw `Parallel.For` or a finalizer introduced inside a
member marked `[OverfitHotPath]` builds clean at whatever severity the containing directory's
`.editorconfig` happens to have for `OVERFIT008`/`OVERFIT012` (often just a warning, easy to miss in build
output), not the `OVERFIT900` hard error the attribute promises. This is not hypothetical: the attribute is
live on real decode/inference code — `Sources/Main/LanguageModels/Runtime/CachedSlmSession.cs`,
`Sources/Main/Inference/SequentialInferenceBackend.cs`, `Sources/Main/Inference/InferenceEngine.cs` all carry
it — so the gap sits exactly on the zero-allocation decode path the attribute exists to protect.

**What test would have caught it:** a compile-time analyzer test that asserts, for every `DiagnosticId` in
the claimed `OVERFIT001`–`OVERFIT014` range, that its `SupportedDiagnostics` includes `OverfitPerfAnalysis.HotPathRule`
(or equivalently that its analyzer calls `Report`) — i.e. a reflective sweep instead of hand-picked
per-analyzer tests, so a future rule added to the range without wiring it up fails the build.

---

### 2. `TensorStorage` telemetry conflates ArrayPool-backed "pooled" storage with GC-array "unpooled" storage

**What breaks:** `OverfitTelemetry.RecordTensorStorageCreated(length, elementSizeBytes, bool borrowed)` only
distinguishes two buckets — `borrowed:true` → `TensorStorageBorrowedCreated`, `borrowed:false` →
`TensorStoragePooledCreated`. But `TensorStorage<T>` actually has *three* lifetime strategies: pooled
(`PooledBuffer<T>`, rented from `ArrayPool`), unpooled (`new T[length]`, GC-managed, used deliberately for
long-lived weights to avoid pool bucket retention — see the `Unpooled()` doc comment right above it), and
borrowed (native arena). Both the pooled ctor (line 45) and the private `Unpooled` ctor (line 61) call
`RecordTensorStorageCreated(..., borrowed: false)`, so **GC-array weight storage is counted under the
"pooled" metric**, and its disposal likewise lands in `TensorStoragePooledDisposed` (via
`RecordTensorStorageDisposed(_isBorrowedMemory)`, which for unpooled storage is `false`). There is no counter
that isolates true ArrayPool pressure from GC-array allocation.

**Where:** `Sources/Main/Diagnostics/OverfitTelemetry.cs` `RecordTensorStorageCreated`/`RecordTensorStorageDisposed`;
call sites `Sources/Main/Tensors/Core/TensorStorage.cs` lines 45, 61, 100, 147.

**How anyone would notice today:** they would not — this is exactly the "matched on the wrong field" shape
the prompt warns about. `overfit.tensor_storage.pooled.created`/`.pooled.disposed` look like they answer "is
the ArrayPool being churned/leaked?" (the README right above `Unpooled()` explicitly frames the pool-vs-GC
distinction as mattering for retention behaviour), but for any model that loads weights via `Unpooled` (the
documented, intended path for long-lived weights) the counter is polluted with one-time large allocations
that have nothing to do with pool health. A dashboard or regression test built on this counter would report
"pool churn" that is actually just model load, or miss real pool churn hidden under the same bucket.

**What test would have caught it:** a unit test creating one `TensorStorage.Unpooled(n)` and one
`new TensorStorage(n)` (pooled) instance, disposing both, then asserting via a `MeterListener` that
`tensor_storage.pooled.created` increased by exactly 1 (only the pooled one) — it would fail today because it
increases by 2.

---

### 3. Eleven of the ~30 `OverfitTelemetry` instruments are declared, documented, and never fed

**What breaks:** `KernelDurationMs`, `ModuleDurationMs`, `GraphBackwardDurationMs`, `ModuleAllocatedBytes`,
`GraphAllocatedBytes`, `AllocationBytes`, `KernelCount`, `ModuleCount`, `GraphCount`, `TapeOpCount`, and
`NativeMemoryBytes` are all created via `Meter.CreateHistogram`/`CreateCounter`/`CreateUpDownCounter` with
real descriptions ("Execution time of low-level kernels.", "Backward pass duration.", "Native / unmanaged
memory **tracked** by Overfit diagnostics.", etc.), but grep across `Sources/Main` finds **zero** call sites
that ever invoke `.Record(...)` / `.Add(...)` on any of them. Only `GraphRecordTotalCount`, the
`TensorStorage*` counters, and the `Evolution*`/`MapElites*` metrics are actually fed (via
`ComputationGraph.cs`, `TensorStorage.cs`, `MapElites.cs`, `EvolutionRunner.cs`).

**Where:** `Sources/Main/Diagnostics/OverfitTelemetry.cs` lines 29–87 (comment above line 89 even calls them
"Existing runtime metrics", suggesting a prior instrumentation pass that was removed or never finished
wiring these up).

**How anyone would notice today:** they would not. These instruments are exported under the
`DevOnBike.Overfit` meter and would show up in any Prometheus/OTel dashboard subscribed to the meter as a
flat zero (histograms) or absent series (counters) forever — indistinguishable from "kernel duration is
truly negligible" or "no native memory is ever tracked." This is precisely the failure mode the task
description warns about: a confident-looking metric name and description with nothing behind it, exactly
the kind of thing that gets quoted for a year once it lands in a README or a Grafana panel.

**What test would have caught it:** a `MeterListener`-based regression test enumerating every instrument on
`OverfitTelemetry.Meter` and asserting each one records at least once over a representative smoke run
(kernel op, module forward, graph backward, an allocation) — `Tests/Diagnostics/DiagnosticsRegressionTests.cs`
already exists and exercises some of the meter's tracing machinery, so this would be a natural extension
rather than new infrastructure.

---

## Shared root cause

Findings 2 and 3 are both instances of the same underlying pattern: **`OverfitTelemetry`'s public counters
are trusted at face value by anyone who reads their names/descriptions, but nothing in this directory (or
its tests) verifies that the description matches what actually increments the counter.** Finding 2 is a
wrong wiring (mislabeled bucket); finding 3 is missing wiring (no bucket touched at all). A single
`MeterListener`-driven "every declared instrument gets exercised and lands in the bucket its name implies"
test would have caught both.

Finding 1 is unrelated (analyzer wiring, not telemetry), but is the same *shape* of defect: a doc comment
asserting a guarantee ("escalates to a hard error") that the code silently does not provide for two of the
named rules.

---

## Coverage

**Reviewed and found clean:**
- `ValueStopwatch.cs` — `GetElapsedTime()` throws (loud, not silent) on a default/uninitialized instance;
  the tick conversion (`TimeSpan.TicksPerSecond / Stopwatch.Frequency`) matches the BCL's own
  `Stopwatch.GetElapsedTime` formula and is correct on both high- and low-frequency timers. All call sites
  found (`MapElites.cs`, `EvolutionRunner.cs`, `ChatSession.cs`, `CircuitBreaker.cs`,
  `OfflineTrainingJob.cs`, `DataPipeline.cs`, `OrpheusVoiceEngine.cs`, `OverfitSkillRunner.cs`,
  `MnistTrainingTests.cs`) pair `StartNew()`/`GetElapsedTime()` correctly with no reuse-after-read misuse.
- `Throughput.cs` — division-by-zero guarded in `RatePerSecond`/`FractionOfCeiling`; `MatmulFlops` multiplies
  as `long` throughout, no overflow for realistic shapes; STREAM/FLOP counting conventions match their doc.
  (Note: it currently has zero callers in `Sources/Main` or `Tests` — only `Sources/Benchmark` uses it — so
  its correctness has not been exercised by anything except by inspection here.)
- `OverfitTelemetry.cs` — `Enabled`/`TraceEnabled` gating is checked consistently before every record call;
  `Counter<T>`/`Histogram<T>`/`UpDownCounter<T>.Add`/`.Record` are documented thread-safe by the BCL, so no
  cross-thread correctness issue found beyond the two findings above; `OpCodeNames` cache construction is
  correct and genuinely zero-allocation on the hot path as its comment claims.
- `Contracts/EvolutionGenerationMetrics.cs`, `Contracts/MapElitesIterationMetrics.cs` — plain data carriers,
  no derived arithmetic to get wrong; the `InvalidFitnessCount`/NaN-vs-zero doc claim on
  `MapElitesIterationMetrics` was traced into `GridEliteArchive.Insert` and holds (NaN/±∞ fitness is
  rejected via `float.IsFinite` before being stored, with an explicit comment on why — matches the doc
  exactly).
- `Contracts/DotNetMemorySnapshot.cs` — the fields it does use (`TotalAllocatedBytes`, `LiveManagedBytes`,
  `WorkingSetBytes`, `PrivateMemoryBytes`, GC counts) are read directly from `GC`/`Process` with no derived
  arithmetic. Minor, not scored: the `Now` field (a `ValueStopwatch`) is assigned at construction but never
  read anywhere in the one caller found (`Tests/Data/Mnist/MnistTrainingTests.cs`) — dead, but not wrong.
- `OverfitHotPathAttribute.cs` — the attribute itself is inert (no runtime behaviour, as documented); the
  `AttributeUsage` targets are consistent with how it's applied at the three real call sites found.

**Not reached:** nothing of substance — the directory is fully read, and caller-following covered every
non-test reference to `ValueStopwatch`, `Throughput`, and `OverfitTelemetry.*` found by grep across
`Sources/Main`. Not independently re-verified: `Sources/Benchmark`'s use of `Throughput` (out of scope —
benchmarks aren't reviewed here), and the full `Sources/Analyzers` test suite for the OVERFIT001–014 range
(finding 1 was confirmed by reading the analyzer source directly, not by running its tests, per the
no-build/no-test constraint).

## What a short score means here

This ended by scope, not by the clock — the directory is small enough that "found 3, stopped" reflects the
directory being in reasonably good shape rather than an unfinished search. Two of the three findings
(2 and 3) are squarely in the "broken measurement apparatus" category the task called out as highest-value;
finding 1 is a genuine contract violation on a live hot-path attribute. All three would need real fixes
before their surrounding claims (README lines, XML docs, dashboard-worthy metric descriptions) can be
trusted at face value.
