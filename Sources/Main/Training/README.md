# `Training` — the training loop and its parallel form

`TrainingEngine` is the single-process loop: forward, loss, backward, optimiser step, with
`ITrainingLoss` / `ITrainingOptimizer` / `ITrainingBackend` as the seams and `Delegate*` adapters for
the one-off cases. `LearningRateSchedule` covers cosine decay and warmup.

`DataParallelTrainer` is the throughput lever: `DataParallelReplica`s hold their own graph and
parameters, gradients are averaged, and `DataParallelLearningRate` adjusts for the effective batch
size. It is public API, not an internal experiment.

## The two levers are different levers

- **Memory** is bought with gradient checkpointing (`CheckpointSegment` in `../Autograd`): **24×** peak
  RAM reduction on a 12-layer GPT-1, paid for with a recomputed forward pass.
- **Throughput** is bought with data parallelism here.

They compose, and neither substitutes for the other. Reaching for data parallelism to fix an
out-of-memory failure adds replicas, each with its own copy of the thing that did not fit.

## Path discipline

Training allocates and records a tape; that is correct and expected. What must not happen is the
reverse — an inference call routed through this path — see `../Inference`. `Array.Copy` is banned
across the assembly; use `Span<T>.CopyTo`.
