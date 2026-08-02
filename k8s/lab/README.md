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

**A warning from `kubectl` is a failed action, not noise.** `apply` and `patch` accept a field they do not
recognise, print a warning, and exit 0. On 2026-08-02 the day-one-events harness patched this namespace's
HPA with `spec.targetCPUUtilizationPercentage` — the `autoscaling/v1` name on an `autoscaling/v2` object.
kubectl warned `unknown field`, reported `patched (no change)`, and returned success; the phase then spent
38 minutes observing a cluster that never scaled, and would have reported a clean HPA scale-down that never
happened. Read every warning, fix it, and **read the state back and assert it** before anything downstream
measures. A phase that cannot confirm its own premise must abort with a message rather than produce a
number — the same rule as verifying that an A/B flag is actually live.

Build and monitoring stacks live in `../overfit` and `../monitoring`.
