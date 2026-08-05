# Kubernetes lab

Local cluster setup for developing the anomaly detectors in `Sources/Main/Statistics/` against real metrics
instead of synthetic arrays. Development infrastructure for the candidate direction in
[`docs/aiops/aiops-cluster-anomaly-guard.md`](../docs/aiops/aiops-cluster-anomaly-guard.md) — not a product component,
not referenced from the README.

## Order

```powershell
cd monitoring
.\install.cmd          # Prometheus + Grafana + kube-state-metrics + node-exporter, namespace `monitoring`
.\forward.cmd          # Grafana :3000 (admin / overfit), Prometheus :9090

cd ..\overfit
.\build.cmd            # only if `overfit:latest` is missing — Native-AOT build, several minutes
.\deploy.cmd           # three replicas of the Overfit server + a ServiceMonitor, namespace `overfit`
```

Tear down in reverse: `overfit\undeploy.cmd`, then `monitoring\uninstall.cmd`.

Everything lives in its own namespace and its own Helm release. Whatever else the cluster is running is not
touched — worth stating, because the cluster this was built against already had RabbitMQ, a Kafka operator
and other workloads on it.

## Why three replicas of our own server

It is the only workload here that emits **Overfit's own** metrics, and three identical pods serving one
model is exactly the input `PeerGroupOutlierDetector` takes: same image, same model, same traffic. Any
spread between them is either genuine skew or the detector's own noise floor.

That number is now measured rather than assumed. Three idle replicas of Qwen-0.5B Q4_K:

```
overfit-server-...-bxczj    1060.5 MB
overfit-server-...-zzkqz    1061.1 MB
overfit-server-...-vgsrl    1061.1 MB
```

**A 0.6 MB spread across 1060 MB — 0.06%.** Useful in both directions: it says the peer detector has an
enormous margin before a real fault becomes ambiguous, and it says that a threshold tuned on synthetic data
with realistic-looking noise would have been tuned for a world far messier than this one.

## What the server exports

Confirmed flowing into Prometheus, one series per pod:

```
overfit_chat_requests_total          overfit_pool_active_sessions
overfit_embedding_requests_total     overfit_pool_available_sessions
overfit_generated_tokens_total       overfit_pool_peak_active_sessions
overfit_prompt_tokens_total          overfit_pool_rejected_total
overfit_speech_requests_total        overfit_pool_size
```

Plus the standard per-pod cAdvisor and process metrics, and the kube-state-metrics topology
(`kube_pod_owner`, `kube_replicaset_owner`) that §11 of the blueprint depends on.

## What this lab cannot tell you

- **One node.** Noisy neighbours across nodes and scheduling pressure cannot be reproduced.
- **No traffic.** Idle replicas produce flat metrics; the RED signals (rate, errors, duration) stay at zero
  until something drives the server.
- **No faults.** Metrics are real, incidents are not. Validating a detector needs an injected fault or —
  per the blueprint's M0 — a recorded Prometheus snapshot from a cluster that had a real one. This is where
  the code is developed, not where it is proven.
