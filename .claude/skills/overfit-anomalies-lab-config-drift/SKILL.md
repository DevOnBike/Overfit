---
name: overfit-anomalies-lab-config-drift
description: Compare what the repository says about the anomaly-guard lab against what the cluster is actually running — ConfigMap bindings, custom channels, container resources, applied-but-uncommitted fixes — and report the divergence in both directions. Use before trusting any claim of the form "the guard watches X" or "the limit is Y", before a measurement whose premise is a config value, and after any change applied with kubectl. Read-only; it never applies anything.
---

# What the repo says versus what the cluster runs

**Both directions are defects and only one of them is obvious.** A change committed and not applied is a fix
nobody is protected by. A change applied and not committed is a fix that disappears the next time somebody
runs `kubectl apply` — and it disappears silently, because the manifest looks authoritative.

Both happened on 2026-08-10, and neither was found by a check. They were found because somebody thought to
look, which is not a method:

- the lab workload's CPU limit was corrected to **1 core in the cluster** and left at **`200m` in
  `k8s/lab/workload.yaml`** — re-applying would have restored a quota the CPU-starvation fault cannot reach,
  turning an experiment into a differently-named one (`XC-14`). *Closed the same day: the manifest now carries
  `cpu: "1"` with the mechanism written next to it. Cited here as the shape, not as an open divergence;*
- `ScrapeCoverage` was implemented, mutation-proven and **absent from the live ConfigMap**, so "built" and
  "protecting this cluster" were two different claims that read the same in a status report (`AN-D9`);
- and a `TASKS.md` row said a fix was "not yet applied" hours after it had been applied and verified.

## Procedure

Import the helpers rather than re-deriving them:

```python
import sys
sys.path.insert(0, r"D:\Overfit\Scripts")
from lab import NAMESPACE, kubectl_out, workload_pods
```

Compare, and print both sides of every difference:

1. **Guard configuration.** `kubectl get configmap -n lab anomaly-guard-config -o jsonpath={.data.guard\.json}`
   against `k8s/anomaly-guard/guard.lab-workload.json`, section by section — `metrics`, `customMetrics`,
   `thresholds`. Compare the **parsed structures**, never the raw text or a diff: the ConfigMap is JSON
   embedded in YAML, and an edit that looks right in a diff can still produce a document the guard reads
   differently.
2. **The custom-channel set specifically.** Which names exist live, which exist in the repo. This is where
   "implemented but not deployed" hides.
3. **Container resources.** CPU and memory limits and requests on every workload pod, against the manifest.
   A limit is a premise of several measurements — CFS throttling exists only on a container carrying a
   quota — so a drift here silently changes what an experiment means.
4. **Deployment shape**: replica count, image, `automountServiceAccountToken`, `securityContext`.

## Rules

- **Read-only.** Never `apply`, never `patch`, never restart. If a divergence should be resolved, say which
  direction and let a human choose — the cluster may be right and the repo wrong, which was the case for the
  CPU limit.
- **Assert every fetch is non-empty before building on it.** `kubectl` reports a malformed query on stderr
  and returns empty stdout with a zero exit code in some shapes, and an empty result is indistinguishable
  from "no divergence" downstream.
- **Do not report "in sync" for anything you did not compare.** Name what you checked and what you skipped.
  A partial comparison reported as a clean bill is worse than no comparison, because it stops the next person
  looking.
- **Check `docs/TASKS.md` claims about deployment state while you are there.** Rows saying "applied" or "not
  applied" are the third artefact in this comparison and they rot fastest.
