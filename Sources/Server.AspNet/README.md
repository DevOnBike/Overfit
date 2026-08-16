# `Server.AspNet` — the HTTP host

An ASP.NET Core host over the engine: an OpenAI-shaped API, metrics, and the background services that
run continuously. This is where the library becomes a process.

| Directory | Contents |
|---|---|
| `Endpoints` | `ChatEndpoints`, `EmbeddingsEndpoints`, `ModelsEndpoints`, `SpeechEndpoints`, `MetricsEndpoints`, `DocsEndpoints`. |
| `Services` | Hosted services and DI registration — inference, metrics, the anomaly guard. |

`OverfitOpenAiApi` is the compatibility surface, so existing OpenAI clients can point at this host
unchanged. `ServerMetrics` and `LatencyHistogram` publish Prometheus-shaped metrics; the exposition is
hand-rolled because the OpenTelemetry exporter drags reflection, which this project keeps out of the
AOT-published paths.

## The anomaly guard runs here

`AnomalyGuardService` is the loop that makes `Sources/Main/Anomalies` a running thing: read a window,
evaluate, report, wait, repeat. Three properties of it are deliberate and each was a defect first:

- **A failed cycle is logged and skipped, never fatal.** A monitoring guard that dies because
  Prometheus was briefly unreachable has replaced the problem it was bought to detect with its own, and
  takes the host down with it.
- **It reports when it cannot see.** A metric no pod exports produces no findings, which is
  indistinguishable from health. Blind metrics are logged by name, and every cycle emits a summary line
  including the quiet ones — otherwise an absence of incidents is ambiguous between "nothing happened",
  "the guard was blind" and "the guard was not running".
- **It passes a durable store.** Without one, every open incident reopens after a restart and the
  operator is paged again for problems they were already told about. This was missing while the library
  had supported it all along.

`AnomalyGuardRegistration` keeps the one `IConfiguration.Bind` call isolated in a single documented
place, because `Bind` is IL2026/IL3050 and would otherwise contaminate the AOT-safe path; the CLI
route parses JSON through a source-generated context instead.
