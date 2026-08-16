# PromQL for the anomaly lab

Queries to paste into <http://127.0.0.1:9090/graph> (run `monitoring\forward.cmd` first). Grouped by which
decision each one serves, because that is the distinction the whole
[blueprint](../docs/aiops/aiops-cluster-anomaly-guard.md) turns on — a statistic answers *"is this difference
real?"*, and that is the wrong question when the service is already on fire.

Every query below was run against the live lab before being written down.

---

## Is anything actually being scraped

Start here. A dashboard full of flat lines looks identical whether the signal is genuinely flat or the
target was never found.

```promql
up{namespace="overfit"}
```

Three series at `1`. If a pod is missing, <http://127.0.0.1:9090/targets> says why.

---

## Peer group — what `PeerGroupOutlierDetector` consumes

The detector compares members of one group **at the same instant**, so it needs no history and works from
the first minute after install. These are its literal inputs.

**One series per replica** — the shape the detector takes:

```promql
process_resident_memory_bytes{namespace="overfit"}
```

**The spread, which is what the detector judges.** On identical idle replicas this measured **0.06 %** — the
detector's noise floor on real data rather than on synthetic noise:

```promql
(max(process_resident_memory_bytes{namespace="overfit"})
 - min(process_resident_memory_bytes{namespace="overfit"}))
 / avg(process_resident_memory_bytes{namespace="overfit"}) * 100
```

**Load-independent signals** — comparable across peers with no application instrumentation at all, straight
from kube-state-metrics and cAdvisor. These are the ones `PeerSignalKind.LoadIndependent` covers:

```promql
kube_pod_container_status_restarts_total{namespace="overfit"}
kube_pod_status_ready{namespace="overfit", condition="true"}
container_oom_events_total{namespace="overfit"}
```

---

## Load-sensitive signals — only valid once normalised

Raw CPU or memory per pod is **not** comparable when traffic is uneven: a replica serving three times the
requests legitimately uses three times the CPU, and comparing the raw numbers reports correct behaviour as a
fault. `PeerSignalKind.LoadSensitive` therefore requires a per-pod work metric, and reports
`InsufficientData` rather than guessing when one is absent.

**The work metric:**

```promql
rate(overfit_chat_requests_total{namespace="overfit"}[5m])
```

**Cost per unit of work** — the normalised comparison, and the same number the FinOps framing uses
("this version made every request 12 % more expensive"):

```promql
rate(process_cpu_seconds_total{namespace="overfit"}[5m])
  / rate(overfit_chat_requests_total{namespace="overfit"}[5m])
```

**Memory per request:**

```promql
process_resident_memory_bytes{namespace="overfit"}
  / rate(overfit_chat_requests_total{namespace="overfit"}[5m])
```

> Both are `NaN` while the request rate is zero, and that is correct — no work means no unit cost. It is
> also why the load generator exists.

**Do not read unit cost as load-invariant — it is not.** Measured here with the load generator at three
skews (`OVERFIT_LAB_SKEW=2`, `4`, `8`), ~600 requests per run:

| Achieved skew | requests/s | raw CPU | **CPU per request** |
|--:|--:|--:|--:|
| 2.3x | 87.9 % | 57.5 % | **26.2 %** |
| 4.4x | 161.6 % | 116.8 % | **44.3 %** |
| 8.4x | 223.7 % | 185.1 % | **51.6 %** |

The residue **grows with the imbalance and never vanishes**, and in every run the busiest replica showed the
**lowest** cost per request — fixed per-process overhead (background threads, GC, idle polling) amortises
over more requests. Run these two side by side after a skewed load and the effect is visible immediately:

```promql
rate(process_cpu_seconds_total{namespace="overfit"}[5m])
rate(process_cpu_seconds_total{namespace="overfit"}[5m]) / rate(overfit_chat_requests_total{namespace="overfit"}[5m])
```

Consequence: when traffic is materially uneven, base the verdict on a **load-independent** signal and keep
unit cost as supporting evidence. See §13.5 amendment 9 in the blueprint.

---

## Topology without the API server

The claim in §11 of the blueprint that lets the MVP drop the Kubernetes list/watch layer entirely: the
Pod → ReplicaSet → Deployment chain and the rollout timestamp are already in Prometheus.

```promql
kube_pod_owner{namespace="overfit", owner_kind="ReplicaSet"}
kube_replicaset_owner{namespace="overfit"}
kube_deployment_created{namespace="overfit"}
```

**Old versus new ReplicaSet** — the canary split, before any rollout tooling is involved:

```promql
sum by (owner_name) (kube_pod_owner{namespace="overfit", owner_kind="ReplicaSet"})
```

---

## Throughput and model-rollout A/B

The metric set §11.1 settles on: hard physics plus a deterministic quality signal, and deliberately no
LLM judge.

```promql
rate(overfit_generated_tokens_total{namespace="overfit"}[5m])
rate(overfit_prompt_tokens_total{namespace="overfit"}[5m])
histogram_quantile(0.95, sum by (le) (rate(overfit_chat_ttft_seconds_bucket{namespace="overfit"}[5m])))
histogram_quantile(0.95, sum by (le) (rate(overfit_chat_response_time_seconds_bucket{namespace="overfit"}[5m])))
```

**Per replica**, which is what an A/B actually needs:

```promql
histogram_quantile(0.95, sum by (le, pod) (rate(overfit_chat_ttft_seconds_bucket{namespace="overfit"}[5m])))
```

---

## Hard rules — the circuit breaker, not the statistic

§3d keeps these on absolute thresholds on purpose. A test answers *"is this difference real?"*, which is
the wrong question when the service is already failing; these fire immediately and never consult a p-value.

```promql
sum(overfit_pool_rejected_total{namespace="overfit"})
overfit_pool_available_sessions{namespace="overfit"} == 0
increase(kube_pod_container_status_restarts_total{namespace="overfit"}[10m]) > 0
```

---

## Trend — what `TrendDetector` is for

Slow drift that never crosses a static threshold until it is late. The Prometheus-native version of the
same question, useful as the sanity check on our own detector:

```promql
deriv(process_resident_memory_bytes{namespace="overfit"}[30m])
predict_linear(process_resident_memory_bytes{namespace="overfit"}[30m], 4 * 3600)
```

> `predict_linear` is also the honest competitor. It is one recording rule, it ships with Prometheus, and it
> catches the headline memory-leak demo on its own — which is precisely why the blueprint's promise is
> **scope and noise reduction**, not "we detect it earlier". Our contribution is that the finding arrives
> attributed to one replica, correlated with a rollout, and grouped with its siblings into one incident.

---

## Reading the endpoint directly

Sometimes the question is "what does the server actually export", not "what does Prometheus think".

```powershell
kubectl port-forward -n overfit svc/overfit-server 8080:8080
curl http://127.0.0.1:8080/metrics
```

That goes through the Service, so it lands on an arbitrary replica. For one specific pod:

```powershell
kubectl port-forward -n overfit pod/<pod-name> 8081:8080
```

The output is **sorted by metric name** (`MetricsEndpoints.Render`, pinned by
`Metrics_AreOrderedByName_WithEachFamilyIntact`) — which exists so that diffing two replicas' endpoints
against each other produces a diff of *values*, not of ordering.
