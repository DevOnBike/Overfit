---
name: lab-prometheus-measurement-environment
description: How to get a defensible window out of the lab Prometheus - port-forward ports, the stale Tests/bin/lab-guard.json trap, and the fault-indicator screen that says whether a window is clean
metadata:
  type: reference
---

Checked 2026-08-10 while running AN-D1 spike 2.

**Reaching it.** `kubectl port-forward -n monitoring svc/overfit-lab-prometheus <local>:9090`.
`k8s\monitoring\forward.cmd` forwards 9090, 9098 and 9099; different diagnostics default to different ports
(`LabFloorCalibrationDiagnostics` uses 9098, `AnomalyGuardReplayDiagnostics` 9090). `LabFact` probes all
three, so any one being up makes the test run.

**`.claude/lab.py` does not exist** despite `CLAUDE.md` describing it as the home of `kubectl`/`prom`/
`guard_pod`/`inject`/`apply_and_read_back`. Checked by listing `.claude/` on 2026-08-10.

**`Tests/bin/lab-guard.json` goes stale silently and nothing checks it.** The copy on this box was from
2026-08-01: no `workload`, none of the five custom channels, and different thresholds
(`CpuUsageRatio.minGap` 0.000159 vs the deployed 0.00041). Every lab diagnostic falls back to that path.
Re-pull before any measurement:

```
kubectl -n lab get configmap anomaly-guard-config -o jsonpath="{.data.guard\.json}"
```

**Screening a window for injected faults.** `guard.lab-workload.json`'s own `_comment` blocks record the
healthy and faulted values, so a window can be screened without guessing:

| indicator | healthy | faulted |
|---|---|---|
| `rate(container_pressure_cpu_waiting_seconds_total[2m])` | max 0.00003 | 0.48682 (starvation) |
| `rate(labapp_lock_contentions_total[2m])` | peak 0.0952/s | 11.6/s |
| `rate(labapp_exceptions_total[2m])` | exactly 0 | 40/s |
| `labapp_active_requests` | 0, ten-minute maxima 0/1/2 | 6+ (hung requests) |
| `increase(kube_pod_container_status_restarts_total[Nm])` | 0 | a restart inside the window |

Those four runtime channels only exist from 2026-08-09, and the CPU limit (1000m) that PSI was calibrated
against arrived the same day — so a window older than that is a **different workload configuration**, not
just older data, and the healthy levels above do not describe it.

**Retention is not what the tsdb metrics suggest.** `prometheus_tsdb_lowest_timestamp_seconds` read
2026-07-28 while `count(up)` had a six-and-a-half-hour hole (2026-08-09 23:43Z to 2026-08-10 06:13Z) from a
host outage. Always plot coverage as contiguous runs before choosing a range, and group by ReplicaSet
generation — the lab rolled seven generations in one day on 2026-08-09, and the longest single-generation
stretch is usually a few hours, not a day.

Related: [[gap-change-floor-unit]].
