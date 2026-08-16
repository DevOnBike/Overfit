# `Diagnostics` — measuring the engine without perturbing it

Timing and telemetry primitives that are safe to leave in a hot path.

| Type | Purpose |
|---|---|
| `ValueStopwatch` | Allocation-free elapsed time. `StartNew()` → `GetElapsedTime()`. |
| `Throughput` | Tokens or items per second, computed without allocating. |
| `OverfitTelemetry` | The counters the engine publishes. |
| `OverfitHotPathAttribute` | Marks a method the analyzer holds to hot-path rules (`OVERFIT900`). |

## `Stopwatch.StartNew` is banned here

`Stopwatch` is a class, so `StartNew()` allocates — in a loop that measures per-token decode, the
measurement changes what it measures. `ValueStopwatch` is a struct over the same timestamps.

The ban is deliberately surgical: `Stopwatch.StartNew` and the constructor are banned, while the static
`Stopwatch.GetTimestamp()` and `Stopwatch.Frequency` remain allowed, because those are exactly what an
allocation-free timer is built from.

## `[OverfitHotPath]`

Marking a method with it opts into the strictest analyzer tier: no allocation, no LINQ, no interface
dispatch where a concrete type would do. It is a claim about the method, so apply it where the claim is
true and enforce it, rather than sprinkling it as documentation.

Related measurement, since it contradicts a plausible intuition: over an array, `for` versus `foreach`
is **not** a lever in .NET 10 — about 2 ns, and the direction reverses with size. The lever is the
**declared type**: iterating an interface costs 2.4× (foreach) to 4.6× (indexing) plus 32 B for the
enumerator. Do not "tidy" a `T[]` field into `IReadOnlyList<T>` on a hot path.
