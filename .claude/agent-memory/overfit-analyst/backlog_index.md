---
name: backlog-index
description: Condensed index of *-backlog.md files — item, status, whether it carries a diagnosis
metadata:
  type: project
---

## docs/aiops/aiops-backlog.md (compiled 2026-08-05, anomaly guard only)
Scored A (verification, no code) / B (client docs) / C (tuning, config only) / D (detection gaps, real +
measured) / E (deferred by decision). Does NOT mention telemetry-rendering format, Meter, or OTLP at all —
this backlog is entirely about detection quality (false-positive rate, peer/trend/rules families), not about
how `/metrics` is exposed.

Key rows with a diagnosis already attached (re-examine the reason before re-doing the work):
- A2 DONE 2026-08-05: `absent()` needed for the "guard has stopped" alert — `time()-timestamp>900` alone goes
  `inactive` when the pod (and its series) is gone. Shipped `k8s/lab/guard-alerts.yaml`.
- D1 DIAGNOSED 2026-08-06: peer family reports a fixed pod property as a recurring anomaly — missing
  mechanism (per-pod learned offset), not a wrong threshold; raising the floor would blind a real leak.
- D2/D3: peer structurally blind to a single OOMKill / to a cluster-wide simultaneous CPU rise — needs a
  mechanism, not a threshold.
- E1 multi-scope slices 3-5 deferred until a client says the fleet is unmanageable (cost is small, risk of a
  shared id counter / shared state file is high).

Only one other `*-backlog.md` found in this repo as of 2026-08-06 (`docs/aiops/aiops-backlog.md`); no
sibling backlogs elsewhere yet.
