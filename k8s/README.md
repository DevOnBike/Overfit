# The lab, case by case

What is in this directory, and the exact command for each scenario the anomaly guard is exercised
against. Every manifest carries its own reasoning in its header — this file is the index, not a
duplicate of it.

**The thing to know first: most faults are not YAML.** The lab workload injects them at runtime over
HTTP, and that is deliberate. Injecting a fault by editing a manifest costs a pod restart, and a restart
is itself a fault — cold working set, a bumped restart counter, no history inside the evaluation window —
so the guard's reaction could never be attributed to the fault you meant to inject. `Demo/LabWorkload/FaultState.cs`
holds the state; `Demo/LabWorkload/Program.cs` maps the endpoints.

## Bringing the lab up

| step | command |
|---|---|
| monitoring stack (Prometheus, Alertmanager, kube-state-metrics, node-exporter) | `k8s\monitoring\install.cmd` |
| the workload — 12 replicas | `kubectl apply -f k8s/lab/workload.yaml` |
| the load driver (runs **inside** the cluster) | `kubectl apply -f k8s/lab/load-driver.yaml` |
| the guard | `kubectl apply -f k8s/lab/anomaly-guard.yaml` |
| port-forwards for the tests | `k8s\monitoring\forward.cmd` (Prometheus on 9090, 9098, 9099) |

The workload image is built from `Demo/LabWorkload` with the publish step first — the build context is
that directory, **not** the repository root. `k8s/lab/workload.yaml`'s header says why.

`k8s\overfit\deploy.cmd` and `forward-replicas.cmd` bring up the *inference server* lab instead, which is
a separate and older subject; `k8s/lab/workload.yaml`'s header records the three measured reasons it was
replaced.

## Fault cases — HTTP, against one pod

`kubectl -n lab port-forward pod/<name> 8080:8080`, then POST. `GET /fault` describes the current state
of that pod, and `POST /fault/clear` returns it to healthy.

| case | command | what it exercises |
|---|---|---|
| latency | `POST /fault/latency?ms=200&jitter=50` | the three latency percentile channels; the classic "slow but correct" pod |
| stall | `POST /fault/stall?probability=0.05&seconds=30` | **requests that never complete.** Note the guard cannot currently see this: latency percentiles only count *finished* requests, so a hung request never enters the histogram |
| errors | `POST /fault/errors?rate=0.1` | `ErrorRate` |
| memory leak | `POST /fault/leak?bytesPerSecond=1048576` | `MemoryWorkingSetBytes` and the trend family; the slow-burn case a single window cannot catch |
| cpu burn | `POST /fault/cpu?msPerRequest=50` | `CpuUsageRatio`, and the work-adjustment question — cost per request rises while request rate does not |
| OOM | `POST /fault/oom` | `OomEventsRate` and `ContainerRestarts`. Allocates **native** memory with every page touched — measured 2026-08-08: a real `OOMKilled`, exit 137, in under 5 s. Managed allocation cannot do this: the container-aware GC throws at 75% of the limit before the kernel acts (`AN-D6`). **Measured: the peer family is structurally blind to a single OOMKill** — see D2 in `docs/aiops/aiops-backlog.md` |
| crash | `POST /fault/crash` | `ContainerRestarts`, restart-loop detection |
| clear | `POST /fault/clear` | back to healthy |

## Cluster-shape cases

| case | how | notes |
|---|---|---|
| healthy baseline | the lab as brought up above | 12 replicas, no faults. This is what every floor was calibrated against |
| CPU throttling | `kubectl apply -f k8s/overfit/fault-cpu-throttle.yaml` | A degraded replica with a hard CPU limit, in the `overfit` namespace. **It is the only pod in the cluster that carries a CPU limit**, which is why `CpuThrottleRatio` is otherwise permanently unbound: `container_cpu_cfs_throttled_periods_total` only exists on containers that have one |
| replica churn | `k8s\lab\scale-experiment.cmd` | 12 -> 15 -> 12. Measured 2026-08-07: **0 incidents in 11 static cycles, 5 in the 7 cycles spanning the change** |
| HPA | not built | A5's other half. Manual scaling is indistinguishable from HPA *to the guard*, which is why the scale experiment substitutes for it — but it does not cover HPA's own metric traffic or its scale-down stabilisation window |
| StatefulSet | not built | A5's undone half. Members with their own volumes are not interchangeable, so peer comparison is structurally questionable there in a way scaling does not test |

## Guard configuration

`k8s/anomaly-guard/guard.lab.json` and `guard.lab-workload.json` are the guard's metric bindings and
floors. `k8s/lab/anomaly-guard.yaml` carries the deployed ConfigMap; the two must agree, and
`Tests/bin/lab-guard.json` is a local copy used by the replay diagnostics.

**No floor in these files came from literature.** Each was measured on this population — see
`docs/aiops/aiops-backlog.md` for the worked negative, where the literature's 25% sustained-CPU threshold
never fired because the real peak was 19.8%.

## What the monitoring stack is, and why it is not in this directory

Helm installs it from `k8s/monitoring/values.yaml`, so the Deployments and StatefulSets under the
`monitoring` namespace are generated rather than authored. Nothing is missing; there is nothing to
capture. `status.cmd` reports what is up, `uninstall.cmd` removes it.

## A note for whoever measures with this lab

The box that runs the cluster is usually the same box running the tests. Two effects, both measured:

- A full `[LongFact]` suite run peaked at 21.7 GB and **evicted Prometheus**, which broke a 24-hour
  measurement's cycle continuity. `Scripts/longfact_gate.py` refuses to start while the marker file
  `Tests/bin/fp-run-clean-start.txt` is fresh — but a bare `dotnet test` bypasses that guard.
- `node_exporter` runs inside the docker-desktop VM and cannot see host processes at all. To ask whether
  the host starved the lab, use PSI — `container_pressure_cpu_waiting_seconds_total` — which records the
  effect regardless of where the cause lives.
