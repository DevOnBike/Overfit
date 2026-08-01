# `k8s/lab` — the measurement cluster

Manifests for the environment the anomaly guard is measured in. Three workloads in namespace `lab`:

| Manifest | What it deploys |
|---|---|
| `workload.yaml` | 12 replicas of `Demo/LabWorkload`, plus its Service and ServiceMonitor. |
| `load-driver.yaml` | `Demo/LabLoadDriver` as a single pod. |
| `anomaly-guard.yaml` | The guard itself as a pod, configured from a ConfigMap, state on an `emptyDir`. |
| `guard.Dockerfile` | Runtime-only image for the guard, built from a host publish. |

## Choices that are not arbitrary

**512 Mi memory limit, no CPU limit.** At 256 Mi a 5 MB/s leak becomes an OOM kill in about 40 seconds,
which tests the OOM path rather than the trend detector. A CPU limit would put every pod into CFS
throttling under the driver's load and make the throttle signal a property of the manifest.

**Twelve replicas.** Peer statistics need a group; four replicas cannot support a statement about
between-pod spread, and the previous lab was capped at four by a 1.07 GB model per pod.

**The guard runs as a pod, not from a workstation.** It is the deployment shape a customer gets, and it
is the only configuration in which a guard restart — and therefore incident-state durability — can be
tested at all.

**Pods carry `role: member`.** Peer cohorts are *declared* through a pod label the customer names in one
config line, not discovered. Discovery by ReplicaSet was tried and reverted: a canary is its own
ReplicaSet, so partitioning that way leaves it alone and below the minimum peer count — the guard goes
blind on precisely the pod that is different.

## Discipline while a run is in progress

A measurement run in this namespace takes hours. **Do not redeploy, scale or restart anything in `lab`
while a count is running** — the run's own denominator is the number of cycles the guard completed, and
a restart silently resets it. Grafana is not started during a measurement either; Prometheus is
required (the guard reads it), Grafana is a discretionary consumer of the same instrument.

Build and monitoring stacks live in `../overfit` and `../monitoring`.
