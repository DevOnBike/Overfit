# Bug hunt: Sources/Main/Evolutionary

- **Scope:** `Sources/Main/Evolutionary` (Strategies, Selection, Crossover, Mutation, Fitness,
  Evaluators, Storage, Runtime, Adapters, `SeededXorShiftRandom`, all `Abstractions`) — all 29
  `.cs` files in the directory were read.
- **Timestamp (UTC):** 2026-08-01-2226
- **Commit:** `79e9d80` (branch `gimli`)
- **Score:** 4 defects confirmed = **8 points**
- **Ended by:** scope. All files in the directory were read before the ten-minute cap (stopped
  at ~4 minutes elapsed); the low score is evidence the module is in good shape, not that time
  ran out.

---

## Findings, ranked by damage

### 1. `GenerationalGeneticAlgorithm.Tell()` has no guard against being called without `Ask()`/`Initialize()` — silently runs on uninitialized pooled memory

**What breaks:** `Tell(ReadOnlySpan<float> fitness)` never checks `_initialized`, and there is no
`_hasPendingPopulation`-style flag at all. If a caller drives the `IEvolutionAlgorithm` interface
directly (legal — nothing in the interface enforces Ask-before-Tell) and calls `Tell()` before
`Initialize()`/`Ask()`, the method proceeds straight into `RankPopulation()` → `StoreBestGenome()`
→ `RebuildPopulation()`, reading `_workspace.Population`. That buffer is a `PooledBuffer<float>`
constructed with `clearMemory: false` (`EvolutionWorkspace` ctor, `GenerationalGeneticAlgorithm`
ctor line `new EvolutionWorkspace(populationSize, parameterCount, clearMemory: false)`), i.e. it
contains whatever a previous tenant of the ArrayPool rental left behind — not zeros, arbitrary
leftover floats. `StoreBestGenome()` will happily copy that garbage into `_bestParameters`, and
`GetBestParameters()` returns it as if it were a real trained result. No exception, no signal.

**Where:** `Sources/Main/Evolutionary/Strategies/GenerationalGeneticAlgorithm.cs`, `Tell()`
(around line 223); compare with `Ask()` a few lines above, which does call
`ThrowIfNotInitialized()` and has an XML doc block explicitly titled "Guards the silent-failure
mode" for exactly this scenario — but the guard was only wired onto `Ask()`.

**Why this is a real gap, not a hypothetical:** the two sibling strategies in the same directory
both close this exact hole on `Tell()`: `OpenAiEsStrategy.Tell()` throws
`OverfitRuntimeException("Tell() was called without a matching Ask().")` when `!_hasPendingPopulation`,
and `SeparableCmaEsStrategy.Tell()` has the identical check. `GenerationalGeneticAlgorithm` is the
odd one out among the three `IEvolutionAlgorithm` implementations in this directory.

**How anyone would notice today:** they would not. The run does not crash; it produces a
population update and a "best genome" derived from uninitialized memory, and `GetBestParameters()`
returns it with no distinguishing signal from a legitimate result.

**What test would catch it:** a unit test that constructs a fresh `GenerationalGeneticAlgorithm`,
calls `Tell(someFitness)` without ever calling `Initialize()` or `Ask()`, and asserts it throws
`OverfitRuntimeException` — mirroring the existing (implicit, by inspection) contract that
`OpenAiEsStrategy` and `SeparableCmaEsStrategy` already enforce and could plausibly already have
equivalent tests for.

---

### 2. `GridEliteArchive.TryGetCellIndex` silently maps a NaN descriptor component into cell 0 instead of rejecting it

**What breaks:** the bounds check `if (value < min || value > max) { cellIndex = -1; return false; }`
is false for both branches when `value` is `NaN` (all comparisons against NaN are false), so a NaN
descriptor component is **not** rejected as out-of-bounds. Execution falls through to
`normalized = (value - min) * _descriptorInvRange[d]` (NaN), then
`bin = value == max ? ... : Math.Clamp((int)(normalized * _binsPerDimension[d]), 0, bins - 1)`.
`value == max` is false for NaN, so `(int)(NaN * bins)` is taken — converting a NaN `float` to
`int` in .NET yields `int.MinValue` on the SSE2 conversion path used by the JIT — and
`Math.Clamp(int.MinValue, 0, bins - 1)` clamps it to `0`. The candidate is silently filed into bin
0 (or cell 0 along that axis) as if its behaviour had actually been measured there, competing for
that cell's slot against real occupants and contributing its (possibly perfectly valid) fitness to
`QdScore`.

**Where:** `Sources/Main/Evolutionary/Storage/GridEliteArchive.cs`, `TryGetCellIndex()` (lines
464–501), reached from `Insert()` (line 401 onward).

**Why this matters given the code's own stated philosophy:** `Insert()` explicitly guards the
*fitness* argument for exactly this failure mode — `EliteInsertStatus.InvalidFitness` exists
specifically, per its own doc comment on `EliteInsertStatus`, "so callers can detect and fix
their fitness function instead of silently dropping samples." The same evaluator contract
(`IBehaviorDescriptorEvaluator<TContext>.Evaluate(..., Span<float> descriptor)`) can just as
easily emit a NaN descriptor component (e.g. a divide-by-zero in a custom behaviour metric), and
for that input path there is no equivalent signal — it is accepted as a fully valid measurement
into a real cell, which is a worse outcome than "silently dropping," because it corrupts the
archive's contents rather than declining to touch it.

**How anyone would notice today:** they would not. `Insert` returns `InsertedNewCell` or
`ReplacedExistingCell` just as it would for a legitimate candidate; `QdScore` and `Coverage`
both look normal.

**What test would catch it:** call `Insert(parameters, fitness, descriptor)` with one descriptor
component set to `float.NaN` and assert the returned status is `OutOfBounds` (or a new
`InvalidDescriptor` status) rather than `InsertedNewCell`.

---

### 3. `CenteredRankFitnessShaper.Shape()` throws when reused with a smaller population than a previous call, despite being documented for exactly that reuse pattern

**What breaks:** the class's own remarks say the ranking buffer "grows monotonically with the
largest population seen so far, so steady-state calls perform zero managed allocations" — i.e.
the type is designed to be shared across calls with varying population counts. But `Shape()` only
grows `_ranking` when `_ranking.Length < count`; when a later call has a **smaller** `count` than
a previous one, the full (larger) `_ranking` array is passed to
`PartialSort.SortIndices(ranking, rawFitness, ascending: true)` without slicing it down to
`count`. `PartialSort.SortIndices` throws `ArgumentException("indices and values must have the
same length.")` whenever `values.Length != indices.Length` — which is exactly this case, since
`rawFitness.Length == count < ranking.Length`.

**Where:** `Sources/Main/Evolutionary/Fitness/CenteredRankFitnessShaper.cs`, `Shape()` (lines
32–74); the mismatched call is at the `PartialSort.SortIndices(ranking, rawFitness, ascending: true)`
line, and the throwing check is `Sources/Main/Maths/PartialSort.cs`, `SortIndices()` line 118.

**How anyone would notice today:** immediately, via a crash — but only the first time the shaper
is reused across a shrinking population size, e.g. one `CenteredRankFitnessShaper` instance handed
to two `OpenAiEsStrategy`/`GenerationalGeneticAlgorithm` instances with different population sizes,
or reused for a second, smaller run. Within a single strategy instance, `PopulationSize` is fixed
and this path is never hit — so it is dormant unless a caller follows the reuse pattern the class's
own doc comment invites.

**What test would catch it:** call `Shape()` on a single `CenteredRankFitnessShaper` instance
first with a length-N fitness span, then with a length-M < N span, and assert it does not throw.

---

### 4. `IEvolutionCheckpoint`'s XML doc says checkpoints do not capture RNG state — every implementation in the directory captures it and guarantees bit-identical resume

**What breaks:** the interface doc says: *"They do NOT capture the internal state of the
`System.Random` instance — after `Load` the RNG restarts from a fresh seed... a resumed run is
statistically equivalent to continuing the original, but not bit-identical."* This is false for
all three concrete implementations reviewed:

- `OpenAiEsStrategy.Save`/`Load` (schema v3) explicitly writes/reads `_rngState`.
- `SeparableCmaEsStrategy.Save`/`Load` explicitly writes/reads `_rngState`.
- `GenerationalGeneticAlgorithm.Save`/`Load` (schema v2) calls `_rng.SaveState(writer)` /
  `_rng.LoadState(reader)`, and its own inline comment says: *"Schema v2 added rngState so a
  resumed run is bit-identical, not merely statistically equivalent."*

All three go out of their way to persist and restore the full generator state precisely so a
resumed run **is** bit-identical — directly contradicting the interface's own doc, which is the
first place a new implementer or caller would look to learn the contract.

**Where:** `Sources/Main/Evolutionary/Abstractions/IEvolutionCheckpoint.cs` (remarks block, lines
19–27) versus `OpenAiEsStrategy.cs` (Save/Load, `_rngState`), `SeparableCmaEsStrategy.cs`
(Save/Load, `_rngState`), `GenerationalGeneticAlgorithm.cs` (Save/Load, `_rng.SaveState`/`LoadState`).

**How anyone would notice today:** they would not notice a behavioural bug — the actual behaviour
is *better* than documented. The damage is the reverse: a caller who trusts the interface doc
might avoid depending on bit-exact resume (e.g. for a reproducibility-sensitive experiment or a
regression test that diffs a resumed run against an uninterrupted one), or a future implementer of
a fourth `IEvolutionAlgorithm` might reasonably choose not to persist RNG state at all, reading the
interface as license not to, and quietly regress reproducibility for that implementation.

**What test would catch it:** none directly — this is a doc/code mismatch, not a runtime
assertion. Fixing the interface doc to match the implementations (or vice versa) is the action;
a documentation-consistency check isn't something a unit test enforces here.

---

## Shared root cause

Findings 1 and 3 are both instances of the same underlying pattern: **a documented invariant
("initialize before use", "grows monotonically for safe reuse") that is enforced or upheld in
some implementations of a shared abstraction but not all of them.** `OpenAiEsStrategy` and
`SeparableCmaEsStrategy` both guard Tell-without-Ask; `GenerationalGeneticAlgorithm` does not.
`CenteredRankFitnessShaper`'s buffer-growth strategy handles growing but not shrinking reuse.
Worth treating as one review pass ("does every implementation of `IEvolutionAlgorithm` actually
enforce the Ask/Tell/Initialize ordering it documents?") rather than as two unrelated bugs, if
that's useful when scheduling the fix.

Findings 2 and (indirectly) the `EliteInsertStatus.InvalidFitness` design in finding 2's writeup
share a cause too: input validation was added carefully for `fitness` but the equally
evaluator-controlled `descriptor` argument was not given the same treatment on the same code path.

---

## Coverage

**Reviewed and found clean** (no defect found; read in full):

- `Strategies/OpenAiEsStrategy.cs` — Ask/Tell RNG consumption, noise-offset round-trip between
  Ask and Tell (the perturbation-index correctness the task flagged), Adam/SGD update, checkpoint
  schema v2/v3 handling, `ThrowIfNotInitialized` guard on `Ask`, Tell-without-Ask guard.
- `Strategies/SeparableCmaEsStrategy.cs` — full generation update, RNG, checkpoint, both guards.
- `Storage/PrecomputedNoiseTable.cs` — parallel Box-Muller fill determinism (each partition seeded
  deterministically from `(masterSeed, from)`, independent of thread-scheduling order — the
  "parallel path breaks reproducibility" hypothesis the task suggested checking does **not** hold
  here), `GetSlice`/`SampleOffset` bounds checks.
- `Evaluators/ParallelPopulationEvaluator.cs` — per-genome fitness writes are positionally
  independent (`fitPtr[i] = ...`), so parallel dispatch order does not affect the result; no shared
  mutable state crossing iterations except the caller-owned `ThreadLocal<TContext>` context, which
  is documented as caller-managed.
- `Runtime/EvolutionRunner.cs`, `Runtime/MapElites.cs` — Ask/Tell/Reset/Save/Load orchestration,
  cross-checks between `_bestEliteCellIndex` and archive occupancy on Load.
- `SeededXorShiftRandom.cs` — Lemire bounded-generator correctness, zero-state fixed point handled.
- `Storage/EvolutionWorkspace.cs`, `Storage/GridEliteArchive.cs` (Insert/Save/Load/Clear paths other
  than the NaN-descriptor gap above), `Storage/EliteInsertStatus.cs`.
- `Adapters/NeuralNetworkParameterAdapter.cs`, `Crossover/SbxCrossoverOperator.cs`,
  `Mutation/GaussianMutationOperator.cs`, `Selection/TournamentSelectionOperator.cs`,
  `Selection/UniformEliteParentSelector.cs`.
- All `Abstractions/*.cs` interfaces except the doc issue noted in finding 4.

**Not reached:** none — all 29 files in the directory were read.

---

## What the score means

The hunt ended by scope, not by the clock (stopped at roughly 4 of the allotted 10 minutes). Four
confirmed defects (8 points) against a 21-point target is a real result for this module: the
directory is unusually disciplined about determinism (three independent xorshift32 + Box-Muller
implementations, all correctly seeded and checkpointed; a noise table whose parallel fill is
provably order-independent) and about NaN handling on the *fitness* side. The gaps that exist are
narrow and specific — one missing guard, one missing input-validation branch, one buffer-reuse
edge case, and one stale doc comment — not systemic. This is not a padded list to reach a number;
it is what a full read of the module turned up.
