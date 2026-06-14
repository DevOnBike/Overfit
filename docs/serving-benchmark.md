# Serving benchmark (`overfit bench`)

A concurrent load test for Overfit's OpenAI-compatible HTTP server — or any other
OpenAI-compatible streaming endpoint — that reports the metrics production serving is actually
judged on, and folds them into a single **holistic score**.

Most "is it fast?" claims about an inference engine are single-user, single-request numbers. What a
team deploying a server actually cares about is how it holds up when **many users hit it at once**:
how long until the first token, how steady the token stream is, how much total throughput it sustains,
and how many requests fail. `overfit bench` measures exactly that, the same way the GPU-serving world
measures its leaderboards — so a pure-.NET CPU server can be put on the same axes and compared
apples-to-apples.

## Run it

Start a server, then point the benchmark at it:

```bash
# terminal 1 — the server under test
overfit serve qwen2.5-3b --port 11434

# terminal 2 — drive it with 16 concurrent users, 128 requests
overfit bench --url http://127.0.0.1:11434/v1 --model qwen2.5-3b --users 16 --requests 128
```

Common options:

| Option | Default | Meaning |
|---|---|---|
| `--url` | `http://127.0.0.1:11434/v1` | Base URL (the part before `/chat/completions`). |
| `--model` | `local` | Sent as the request's `model` field. |
| `--users` / `-u` | `8` | Concurrent virtual users. |
| `--requests` / `-n` | `64` | Total requests in the measured window (≥ `--users`). |
| `--max-tokens` | `128` | Caps decode length so inter-token latency is measured over a real stream. |
| `--prompt` | (a short fixed prompt) | The user prompt every request sends. |
| `--warmup` | `8` | Warm-up requests sent before the measured window (not scored). |
| `--cost-units` | `1` | Cost in the score denominator (e.g. CPU cores or server count). |

It works against **any** OpenAI-compatible streaming endpoint, so you can run the same command against
a different provider for a side-by-side comparison (mind each provider's terms and rate limits).

## What it measures

| Metric | Meaning | Goal |
|---|---|---|
| **TTFT** | Time to first token: request submission → first streamed chunk. Dominated by prompt prefill + queue depth. | lower |
| **ITL** | Inter-token latency: mean gap between streamed tokens during decode. Reflects per-step overhead (dispatch, memory bandwidth, kernel time). | lower |
| **E2E** | End-to-end latency: submission → last token. | lower |
| **Throughput** | Aggregate streamed tokens/second across all concurrent users over the measured window. | higher |
| **Goodput** | `throughput × (1 − error_rate)` — throughput that actually reached a client. | higher |
| **Error rate** | Fraction of requests that failed (HTTP error, timeout, transport drop). | lower |

All latencies are reported at **p50 / p95 / p99**. Warm-up requests are excluded so the scored window
reflects steady state, not cold start.

## The score

```text
score = (goodput × concurrency) / (TTFT_p95_s × ITL_p95_s × cost_units)
```

Higher is better. It is deliberately the same *shape* as a GPU inference leaderboard score: it rewards
high throughput, high concurrency and low tail latency **simultaneously**, and divides out the serving
cost — so adding capacity without a proportional latency win does **not** raise the score. Efficiency,
not brute force, is what moves the number.

The scoring math (`ServingLoadReport`, `ServingRequestSample`) lives in `Sources/Main/Serving/` and is
unit-tested (`Tests/Serving/ServingLoadReportTests.cs`); the load driver is `Sources/Cli/ServingBenchmark.cs`.

## Why this exists

Overfit's core thesis is that the per-token **dispatch tax** — the cost of getting from "decode step N"
to "decode step N+1" — dominates LLM serving latency, and that a no-Python, zero-allocation, in-process
.NET engine removes that tax structurally rather than papering over it. This benchmark is how that claim
gets *measured* instead of asserted: same metrics, same score shape as the GPU world, on CPU.

## Honest caveats

- The client adds a small amount of its own overhead (HTTP, SSE parse) that is included in E2E/ITL;
  keep the load driver on a separate machine, or at least a separate process, from the server for the
  cleanest numbers.
- "Tokens" here are counted as **streamed content chunks**, which tracks tokens closely but is not a
  tokenizer-exact count.
- A score is only comparable across runs on the **same hardware** with the **same prompt and
  `max_tokens`** — it is a relative dial for your own optimization loop and for like-for-like
  comparisons, not an absolute cross-hardware constant.
