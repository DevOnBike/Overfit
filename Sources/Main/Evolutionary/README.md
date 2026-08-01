# `Evolutionary` — gradient-free optimisation

Population-based search for problems where there is no gradient: a simulator in the loop, a discrete
structure, a black-box objective. `IEvolutionAlgorithm` is the seam and `EvolutionRunner` drives it.

| Directory | Contents |
|---|---|
| `Strategies` | `OpenAiEsStrategy`, `SeparableCmaEsStrategy`, `GenerationalGeneticAlgorithm`. |
| `Selection`, `Crossover`, `Mutation`, `Fitness` | The operators, each behind an interface. |
| `Evaluators` | `ParallelPopulationEvaluator` — the expensive part, parallel by default. |
| `Storage` | `GridEliteArchive` (MAP-Elites cells), `PrecomputedNoiseTable`, `EvolutionWorkspace`. |
| `Runtime` | `EvolutionRunner`, `MapElites`. |
| `Adapters` | `NeuralNetworkParameterAdapter` — flattens a model's parameters into a search vector. |

## Two implementation choices worth knowing

**`PrecomputedNoiseTable`.** OpenAI-ES perturbs parameters with Gaussian noise and needs the *same*
noise reproduced when the update is applied. A shared table indexed by seed means a perturbation is a
32-bit integer rather than a full parameter-sized vector — the difference between a workable and an
unworkable memory profile at model scale.

**MAP-Elites keeps a grid, not a best.** `IBehaviorDescriptorEvaluator` places each candidate in a cell
by *behaviour*, and each cell keeps its own elite. The output is a map of qualitatively different
solutions instead of one winner, which is what you want when the objective does not capture everything
you care about.

`SeededXorShiftRandom` is the deterministic generator: a search that cannot be replayed cannot be
debugged.
