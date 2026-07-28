# Overfit AIOps development lab

A local Prometheus + Grafana + kube-state-metrics install, so the detectors in
`Sources/Main/Statistics/` can be developed against real cluster metrics instead of synthetic arrays.

> **Not a product component and not linked from the README.** This is development infrastructure for the
> candidate direction described in [`docs/aiops-cluster-anomaly-guard.md`](../../docs/aiops-cluster-anomaly-guard.md).

## Use

```powershell
.\install.cmd      # Helm install into the `monitoring` namespace (first run pulls images)
.\status.cmd       # pods, PVCs, and what is not healthy
.\forward.cmd      # Grafana on :3000 (admin / overfit), Prometheus on :9090
.\uninstall.cmd    # remove everything the install created, and nothing else
```

Everything lands in its own namespace and its own Helm release. If the cluster is already running other
workloads, none of them are touched — worth stating because the cluster this was first built against had a
RabbitMQ cluster and a Kafka operator on it.

## Why these components

| Component | Why it is here |
|---|---|
| **kube-state-metrics** | The one that is not optional. §11 of the blueprint drops the Kubernetes list/watch layer from the MVP because the topology is already in Prometheus: `kube_pod_owner`, `kube_replicaset_owner` and `kube_deployment_created` give the Pod → ReplicaSet → Deployment chain and the rollout timestamp in a single PromQL query, with no API-server access at all. |
| **node-exporter** + cAdvisor | Per-pod CPU, memory, throttling and restarts — the load-independent signals `PeerSignalKind` can compare without any application instrumentation. |
| **Prometheus** | The metric source, and the shape the ingestion has to accept: histograms, not raw per-request values (blueprint §3f). |
| **Grafana** | Reading the data while developing. The product exports `overfit_anomaly_score` and expects Grafana to stay the UI rather than shipping its own. |
| **Alertmanager** | The blueprint's output path is an Alertmanager webhook, so the lab should contain the thing the product talks to. |

## Choices worth knowing before changing them

**Retention is 15 days.** That covers peer-outlier and canary (which need no history at all) and the trend
detector (hours to days). It does **not** cover the seasonal baseline, which wants 2–4 weeks — raise
retention *and* disk together when that detector is being worked on.

**Scrape interval stays at 15 s.** Lowering it would fill canary windows faster and would be cheating: the
detectors have to work at the resolution customers actually run, and a lab that is quietly finer than
production hides sample-count problems until the pilot.

**Control-plane scrape targets are disabled** (etcd, scheduler, controller-manager, kube-proxy). On Docker
Desktop they are not reachable, and leaving them enabled produces permanently-red targets that teach you to
ignore the targets page.

**The chart's default alert rules are off.** They are written for production clusters and fire constantly on
a single-node laptop. This lab's job is to produce clean metrics for our own detectors, not to be a working
alerting install.

## What to look at first

Once `forward.cmd` is running, these are the queries the design actually depends on:

```promql
# Topology without the API server — the §11 claim, verified
kube_pod_owner{owner_kind="ReplicaSet"}

# A peer group: every pod of one workload, same metric, same instant.
# This is exactly the input PeerGroupOutlierDetector takes.
sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="default"}[5m]))

# Old versus new ReplicaSet — the canary comparison, before any rollout tooling exists
kube_replicaset_owner{owner_kind="Deployment"}
```

A StatefulSet with several replicas is the most useful subject available: identical pods, identical
workload, so any spread between them is either real skew or the detector's own noise floor. That is the
cheapest possible check on whether the peer detector's thresholds are sane before it ever sees a fault.

## Limits of this lab

- **One node.** Anything about noisy neighbours across nodes, or scheduling pressure, cannot be reproduced.
- **No traffic generator.** Idle pods produce flat metrics; the RED signals (`request rate`, `error rate`,
  `duration`) will not exist until something is instrumented and driven.
- **Docker Desktop's `hostpath` storage** is not a realistic disk. Do not measure I/O behaviour here.
- **Metrics are real but the faults are not.** Validating a detector needs either an injected fault or, per
  the blueprint's M0, a recorded Prometheus snapshot from a cluster that had a real incident. This lab is
  where the code is developed, not where it is proven.
