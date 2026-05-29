# Observability

## What ships today

`GET /metrics` exposes [Prometheus text exposition format](https://prometheus.io/docs/instrumentation/exposition_formats/) — scrapeable with no exporter dependency. The metrics are recorded by `MetricsCollector` from the same `GenerationStats` the engine reports:

| Metric | Type | Meaning |
|---|---|---|
| `overfit_build_info{model,fingerprint,mmap}` | gauge | Static info about the loaded model (value always 1) |
| `overfit_model_load_seconds` | gauge | Time to load the model at startup |
| `overfit_requests_total{endpoint}` | counter | Requests per endpoint (`chat`, `rag`, `agent`, `chat_json`) |
| `overfit_generations_total` | counter | Total model generations |
| `overfit_prompt_tokens_total` | counter | Prompt tokens processed |
| `overfit_generated_tokens_total` | counter | Tokens generated |
| `overfit_allocated_bytes_total` | counter | Bytes allocated during generation — Overfit's headline is ≈ 0 B/token |
| `overfit_decode_tokens_per_second` | gauge | Decode throughput of the most recent generation |
| `overfit_tool_calls_total{tool}` | counter | Tool calls dispatched, per tool name |
| `overfit_rag_searches_total` | counter | RAG retrieval searches |
| `overfit_rag_search_seconds_total` | counter | Cumulative in-process retrieval (embed + cosine scan) time |

`compose.yaml` runs a Prometheus container that scrapes this endpoint (`Observability/prometheus.yml`). Open `http://localhost:9090` and graph e.g. `overfit_allocated_bytes_total / overfit_generated_tokens_total` to watch the per-token allocation stay near zero.

## Adding OpenTelemetry (optional)

The metrics above are deliberately hand-rolled so the demo has **no extra package dependencies** and stays AOT-clean. If you want OTLP export / distributed tracing in your own deployment, the idiomatic .NET path is:

1. Add `OpenTelemetry.Extensions.Hosting` + `OpenTelemetry.Exporter.OpenTelemetryProtocol` (and `OpenTelemetry.Instrumentation.AspNetCore` for request traces).
2. Replace `MetricsCollector`'s fields with a `System.Diagnostics.Metrics.Meter` and `Counter<long>` / `Histogram<double>` instruments.
3. Register `builder.Services.AddOpenTelemetry().WithMetrics(m => m.AddMeter("Overfit.LocalAgent").AddOtlpExporter())`.

That swap is intentionally left out of the starter to keep the dependency surface minimal — wire it in when your observability stack calls for it.
