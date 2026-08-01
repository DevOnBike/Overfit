# `Runtime` — process-level helpers

Small utilities about the process rather than the maths: parallelism, GC behaviour, disposal, and
environment knobs.

| Type | Purpose |
|---|---|
| `OverfitParallel` | The parallel-for used on the decode path. |
| `GcLatencyScope` / `GcHandleScope` | Scoped GC latency mode and pinned handles. |
| `CompositeDisposable` | One `using` for several owned things. |
| `OverfitEnvironment` | Environment-variable knobs, read in one place so they are greppable. |
| `PooledArray` | `ref struct` wrapping `ArrayPool` rent/return as a `using`. |

## `OverfitParallel` versus `Parallel.For` — measured, and it goes both ways

On a fair sustained benchmark of the decode path, `OverfitParallel.ForDecode` ran **455 µs and 0 B**
against `Parallel.For` at **2059 µs and 925 KB** — 4.5× faster and allocation-free, because it reuses
a spin pool instead of dispatching per call.

That does **not** generalise. Migrating Conv2D onto it measured **+13% wall-clock** on MNIST and was
reverted; Conv2D stays on `Parallel.For`. The decode pool assumes dedicated cores and is sensitive to
background load, which is exactly what a training step does not have.

The rule: `OverfitParallel` for decode, `Parallel.For` elsewhere, and a measurement before moving
anything across that line.

## `ValueStopwatch` lives in `../Diagnostics`

`Stopwatch.StartNew` is banned in this assembly — it allocates. Use `ValueStopwatch.StartNew()` and
`GetElapsedTime()`. The ban is surgical: the static `Stopwatch.GetTimestamp` and `Stopwatch.Frequency`
remain allowed.
