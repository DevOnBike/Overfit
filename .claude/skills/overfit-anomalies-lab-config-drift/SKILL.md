---
name: overfit-anomalies-lab-config-drift
description: Compare what the repository says about the anomaly-guard lab against what the cluster is actually running — ConfigMap bindings, custom channels, container resources, applied-but-uncommitted fixes — and report the divergence in both directions. Use before trusting any claim of the form "the guard watches X" or "the limit is Y", before a measurement whose premise is a config value, and after any change applied with kubectl. Read-only; it never applies anything.
model: sonnet
color: blue
---

# What the repo says versus what the cluster runs

Read the live configuration out of the cluster, read the manifests out of the repository, and report every
place they disagree — in both directions. Read-only throughout.

## Why both directions are defects, and only one of them is obvious

A change committed and not applied is a fix nobody is protected by. A change applied and not committed is a
fix that **disappears the next time somebody runs `kubectl apply`** — silently, because the manifest looks
authoritative.

| Problem | Symptom | Consequence |
|---|---|---|
| Committed, not applied | Code exists, tests pass, channel absent from the live ConfigMap | "Built" and "protecting this cluster" read the same in a status report |
| Applied, not committed | Cluster behaves correctly | The next apply silently restores the old value |
| Resource drift | Manifest and pod disagree on a limit | A limit is the premise of a measurement; the experiment quietly becomes a different one |
| Stale deployment claim in docs | A row says "not yet applied" hours after it was | Somebody re-does the work, or skips it |

All four happened on 2026-08-10, and none was found by a check. They were found because somebody thought to
look, which is not a method:

- the lab workload's CPU limit was corrected to **1 core in the cluster** and left at **`200m` in
  `k8s/lab/workload.yaml`** — re-applying would have restored a quota the CPU-starvation fault cannot reach,
  turning an experiment into a differently-named one (`XC-14`). *Closed the same day: the manifest now carries
  `cpu: "1"` with the mechanism written next to it. Cited here as the shape, not as an open divergence;*
- `ScrapeCoverage` was implemented, mutation-proven and **absent from the live ConfigMap** (`AN-D9`);
- a `TASKS.md` row said a fix was "not yet applied" hours after it had been applied and verified.

## When to Use

- Before writing any claim of the form "the guard watches X" or "the limit is Y"
- Before a measurement whose premise is a config value — a quota, a cadence, a binding
- After **any** change applied with `kubectl`, including one you applied yourself
- Before a report that says a channel is deployed
- When a fault does not produce the effect its size predicts

## When Not to Use

- To fix the divergence. This skill never applies, patches or restarts — it reports and names the direction
- To verify a detector fires (use `overfit-anomalies-lab-two-arms`)
- To calibrate anything from lab data (use `overfit-anomalies-lab-window`)
- On a cluster you have not confirmed is the lab. Verify the context before the first query

## Inputs

| Input | Required | Description |
|---|---|---|
| Live ConfigMap | Yes | `anomaly-guard-config` in namespace `lab`, key `guard.json` |
| Repo config | Yes | `k8s/anomaly-guard/guard.lab-workload.json` |
| Manifests | Yes | `k8s/lab/workload.yaml`, `k8s/lab/anomaly-guard.yaml` |
| Live pods | Yes | Workload and guard pods, for resources and deployment shape |
| `docs/TASKS.md` | Yes | The third artefact — rows claiming applied / not applied |

## Workflow

### Step 1: Import the helpers rather than re-deriving them

```python
import sys
sys.path.insert(0, r"D:\Overfit\Scripts")
from lab import NAMESPACE, kubectl, kubectl_out, guard_pod, workload_pods
```

Re-pasting these is how a defect propagates: the broken unpacking `_, target, _ = kubectl(...)` — which
returns two values — was pasted three times in one evening and failed three times.

### Step 2: Guard configuration, compared as parsed structures

`kubectl get configmap -n lab anomaly-guard-config -o jsonpath={.data.guard\.json}` against
`k8s/anomaly-guard/guard.lab-workload.json`, section by section — `metrics`, `customMetrics`, `thresholds`.

**Compare the parsed structures, never the raw text or a diff.** The ConfigMap is JSON embedded in YAML, and
an edit that looks right in a diff can still produce a document the guard reads differently.

### Step 3: The custom-channel set specifically

Which names exist live, which exist in the repo. This is where "implemented but not deployed" hides, and it
is the one comparison a structural diff of the whole file will bury.

### Step 4: Container resources

CPU and memory limits and requests on every workload pod, against the manifest. A limit is a premise of
several measurements — **CFS throttling exists only on a container carrying a quota** — so a drift here
silently changes what an experiment means.

### Step 5: Deployment shape

Replica count, image, `automountServiceAccountToken`, `securityContext`.

### Step 6: The documentation claims

Check `docs/TASKS.md` rows saying "applied" or "not applied" while you are there. They are the third artefact
in this comparison and they rot fastest.

### Step 7: Report the direction, not the fix

Say which side should move and let a human choose. The cluster may be right and the repo wrong — which is
exactly what happened with the CPU limit.

## Validation

- [ ] Every fetch was asserted non-empty **before** anything was built on it
- [ ] Configuration was compared as parsed structures, not as text
- [ ] The custom-channel set was compared by name, not only as part of the whole document
- [ ] Resources were read from the live pods, not from the Deployment spec alone
- [ ] `docs/TASKS.md` deployment claims were checked against what was found
- [ ] The report names what was **not** compared
- [ ] Nothing was applied, patched or restarted

## Common Pitfalls

| Pitfall | Solution |
|---|---|
| Empty `kubectl` result read as "no divergence" | It reports a malformed query on stderr and returns empty stdout with exit 0 in some shapes. Assert non-empty first |
| Reading the terminating guard pod | Right after a rollout the old pod is still listed and its log is empty. `lab.guard_pod()` returns the live one |
| `kubectl apply` reported success for a field it dropped | An `unknown field` warning with exit 0 is a **failed** action. Read the state back and assert — `lab.apply_and_read_back()` exists for this |
| "In sync" reported for something not compared | Name what you checked and what you skipped. A partial comparison sold as a clean bill stops the next person looking |
| Diffing the ConfigMap as text | JSON embedded in YAML. Parse both sides |
| Fixing the divergence while you are there | Read-only. The direction is a human decision, and it went the unobvious way at least once |
