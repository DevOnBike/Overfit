# AIOps — the cluster anomaly guard

Everything about the guard that watches a Kubernetes namespace and reports what changed. Moved here from
`docs/` on 2026-08-05, when eleven of its forty-four documents turned out to be about this one subsystem.

**Where to start depends on what you are doing.**

## Deciding whether to run it

- [`aiops-client-readiness.md`](aiops-client-readiness.md) — what it sees, what it cannot, and what it
  costs. The honest version, including the blind spots.
- [`aiops-client-flow.md`](aiops-client-flow.md) — the engagement stage by stage.
- [`aiops-day-one-events.md`](aiops-day-one-events.md) — what a cluster actually does in the first hours,
  measured. **The first day after deployment has a structurally raised false-positive rate**, because the
  seasonal baseline needs a previous day and a fresh guard has none.

## Building on it

- [`aiops-detection-pipeline.md`](aiops-detection-pipeline.md) — the pipeline as built, not as planned.
  The long one, and the one to read before changing a detector.
- [`aiops-cluster-anomaly-guard.md`](aiops-cluster-anomaly-guard.md) — product and technical blueprint.
- [`aiops-operator-feedback.md`](aiops-operator-feedback.md) — acknowledgement, suppression, and the
  `--real` label that constrains future floor proposals.
- [`aiops-multi-scope-design.md`](aiops-multi-scope-design.md) — several namespaces in one process.
  **Deferred by decision**: one instance per namespace costs 130 MB and no code, gives failure isolation at
  a process boundary and allows per-namespace RBAC. Build this when a client says the fleet is
  unmanageable, not before.
- [`aiops-canary-blueprint.md`](aiops-canary-blueprint.md) — automated canary analysis, the adjacent
  product.

## Working out what to do next

- [`aiops-backlog.md`](aiops-backlog.md) — every open task with ROI, difficulty and risk, and a column for
  whether anybody is paying for it **today**. That column exists because the largest single waste on this
  subsystem was chasing the channel that dominates the *historical* statistics and produces no incidents
  now.
- [`aiops-repair-plan.md`](aiops-repair-plan.md) — the six defects found by reading rather than by running,
  all fixed.
- [`gp-anomaly-baseline.md`](gp-anomaly-baseline.md) — a Gaussian-process baseline sketch. Proposed, not
  implemented.

## Deliberately left in `docs/`

- [`../silence-review.md`](../silence-review.md) — a review method for any code whose product is the
  absence of an alarm. Born here, but it applies to anything that can fail by staying quiet, and it has its
  own agent.
- [`../autoresearch-program.md`](../autoresearch-program.md) — automated search instead of hand-tuning.
  The worked example is this subsystem's synthetic generator, but `CLAUDE.md` cites it as the general
  method, and the rule it teaches ("what value" → search, "what mechanism" → no search will find it) is not
  about anomalies.
