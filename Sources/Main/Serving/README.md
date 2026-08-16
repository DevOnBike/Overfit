# `Serving` — sharing an engine between concurrent requests

`OverfitResourcePool` hands out engine instances to concurrent callers and takes them back;
`PoolMetrics`, `ServingLoadReport` and `ServingRequestSample` are what it reports about itself.

## Why a pool rather than one engine

An inference engine holds per-session mutable state — KV cache, position counter, scratch buffers — so
it is not reentrant. Two requests through one instance interleave their state and produce two wrong
answers. Copying the *weights* per request is unaffordable; copying the *session* is cheap, since the
runtime keeps weights as spans into shared storage rather than duplicating them (see
`../LanguageModels/Runtime`). The pool is the boundary that makes that distinction operational.

## Metrics exist because saturation looks like slowness

Without pool metrics, a queue that is full presents as latency and is indistinguishable from a model
that got slower. `PoolMetrics` separates wait time from work time, which is the difference between
"add replicas" and "make the model faster" — opposite fixes.

The ASP.NET host in `../../Server.AspNet` exposes these; `Sources/Main/Anomalies` is the subsystem that
would then decide whether the numbers are anomalous, on a different deployment.
