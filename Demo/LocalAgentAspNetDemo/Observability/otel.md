# Observability

## What ships today

Metrics are instrumented with the built-in [`System.Diagnostics.Metrics`](https://learn.microsoft.com/dotnet/core/diagnostics/metrics) (`Meter`) API in `MetricsCollector`, and exported to a Prometheus scrape endpoint at **`GET /metrics`** by [OpenTelemetry](https://learn.microsoft.com/aspnet/core/log-mon/metrics/metrics) (`AddOpenTelemetry().WithMetrics(... .AddPrometheusExporter())` + `app.MapPrometheusScrapingEndpoint()` in `Program.cs`). This is the idiomatic ASP.NET Core path — no hand-rolled exposition text.

The instruments are fed from the same `GenerationStats` the engine reports:

| Instrument (Meter) | Prometheus name | Type | Meaning |
|---|---|---|---|
| `overfit.build.info{model,fingerprint,mmap}` | `overfit_build_info` | gauge | Static info about the loaded model (value always 1) |
| `overfit.model.load` | `overfit_model_load_seconds` | gauge | Time to load the model at startup |
| `overfit.requests{endpoint}` | `overfit_requests_total` | counter | Requests per endpoint (`chat`, `rag`, `agent`, `chat_json`) |
| `overfit.generations` | `overfit_generations_total` | counter | Total model generations |
| `overfit.prompt.tokens` | `overfit_prompt_tokens_total` | counter | Prompt tokens processed |
| `overfit.generated.tokens` | `overfit_generated_tokens_total` | counter | Tokens generated |
| `overfit.allocated` | `overfit_allocated_bytes_total` | counter | Bytes allocated during generation — Overfit's headline is ≈ 0 B/token |
| `overfit.decode.rate{endpoint}` | `overfit_decode_rate_per_second` | histogram | Decode throughput per generation (distribution, not just last value) |
| `overfit.tool.calls{tool}` | `overfit_tool_calls_total` | counter | Tool calls dispatched, per tool name |
| `overfit.rag.search` | `overfit_rag_search_seconds` | histogram | In-process retrieval (embed + cosine scan) latency, with latency-shaped buckets |

The Prometheus exporter handles exposition (`_total` suffixes, `_bucket`/`_sum`/`_count` for histograms, `# UNIT`/`# HELP`/`# TYPE` lines, label escaping) per the OpenMetrics spec.

`compose.yaml` runs a Prometheus container that scrapes this endpoint (`Observability/prometheus.yml`). Open `http://localhost:9090` and graph e.g. `overfit_allocated_bytes_total / overfit_generated_tokens_total` to watch the per-token allocation stay near zero, or `histogram_quantile(0.95, rate(overfit_rag_search_seconds_bucket[5m]))` for retrieval-latency p95.

## Going further (optional)

Because the instrumentation already uses the standard `Meter` API, adding more exporters is just configuration — no code change to `MetricsCollector`:

- **OTLP / distributed tracing**: add `OpenTelemetry.Exporter.OpenTelemetryProtocol` (+ `OpenTelemetry.Instrumentation.AspNetCore` for request traces) and chain `.AddOtlpExporter()` onto the OpenTelemetry builder.
- **Grafana / dashboards**: point Grafana at the Prometheus container, or scrape `/metrics` directly.
