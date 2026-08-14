# `Runtime` — process-level helpers

Small utilities about the process rather than the maths: parallelism, GC behaviour, disposal, and
environment knobs.

| Type | Purpose |
|---|---|
| `OverfitParallel` | The parallel-for used on the decode path. |
| `GcLatencyScope` / `GcHandleScope` | Scoped GC latency mode and pinned handles. |
| `CompositeDisposable` | One `using` for several owned things. |
| `OverfitEnvironment` | Environment-variable knobs, read in one place so they are greppable. |
| `PooledBuffer<T>` | Pooled scratch as a `using` scope. Lives in `Tensors/`, not here. A plain `struct` on purpose, so it can be a class field or captured. |

## `OverfitParallel` versus `Parallel.For` — measured, and it goes both ways

On the decode path `OverfitParallel.ForDecode` runs **2.3-2.7× faster** than `Parallel.For` and allocates
**0 B** against its hundreds of KB, because it reuses a spin pool instead of dispatching per call.
**Which `Parallel.For` you compare against is most of that ratio**: 2.43× against an arm capped at
`DecodeMaxWorkers` — which is what `OVERFIT_DECODE_POOL=0` actually falls back to — and 3.60× against an
uncapped one. The **4.5×** stated here until 2026-08-14 quoted the uncapped pair and is retired. The
numbers, the box and the model this does **not** hold for (Phi-3.5 is neutral to negative) live in
[`docs/measured-baselines.md`](../../../docs/measured-baselines.md), re-audited 2026-08-14 under `PB-12`.

That does **not** generalise. Migrating Conv2D onto it measured **+13% wall-clock** on MNIST and was
reverted; Conv2D stays on `Parallel.For`. The decode pool assumes dedicated cores and is sensitive to
background load, which is exactly what a training step does not have.

The rule: `OverfitParallel` for decode, `Parallel.For` elsewhere, and a measurement before moving
anything across that line.

## `ValueStopwatch` lives in `../Diagnostics`

`Stopwatch.StartNew` is banned in this assembly — it allocates. Use `ValueStopwatch.StartNew()` and
`GetElapsedTime()`. The ban is surgical: the static `Stopwatch.GetTimestamp` and `Stopwatch.Frequency`
remain allowed.
