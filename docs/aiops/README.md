# AIOps — the cluster anomaly guard

Everything about the guard that watches a Kubernetes namespace and reports what changed. Moved here from
`docs/` on 2026-08-05, when eleven of its forty-four documents turned out to be about this one subsystem.

**This is a document map, not a summary — read the file it points at before acting on a claim from it.**
Written after a session that read the wrong document twice and spent 200 lines against a mechanism the code
does not implement, because it was not obvious which file was authoritative. If a question has one right
answer, this page says which file holds it.

## Start here — four questions people actually arrive with

| Question | Go to |
|---|---|
| What is this and should we run it? | [`aiops-business-case.md`](aiops-business-case.md) |
| How does it work? | [`aiops-architecture.md`](aiops-architecture.md) |
| What does it actually watch, and in what unit? | [`aiops-metrics-catalogue.md`](aiops-metrics-catalogue.md) |
| Which failures will this catch? | [`aiops-coverage-map.md`](aiops-coverage-map.md) |
| How do I add a metric? | [`aiops-adding-a-metric.md`](aiops-adding-a-metric.md) |
| How is an `AN-*` / `RS-*` / `PS-*` task actually run? | [`aiops-task-protocol.md`](aiops-task-protocol.md) — the nine steps and the incident behind each; `CLAUDE.md` links here rather than repeating it |
| What is broken right now? | [`docs/TASKS.md`](../TASKS.md) — **not a document in this directory.** `aiops-backlog.md` below is history and reasoning, not current status |

## The four documents written 2026-08-09 and how they relate

- **[`aiops-architecture.md`](aiops-architecture.md)** (by `overfit-architect`) — pieces, ownership, failure
  modes, the two execution paths (`AnomalyGuard.RunCycle` vs. `LiveMonitoringPipeline`) that never meet at
  runtime. **Authoritative over `aiops-detection-pipeline.md` wherever the two disagree** — it was checked
  against `Sources/Anomalies/**` on 2026-08-09 and says so explicitly, with a named point of disagreement
  (incident matching, below).
- **[`aiops-business-case.md`](aiops-business-case.md)** (by `overfit-analyst`, this file's author) — should
  you run it, what it costs, what day one gets you. Buyer-facing; every number names the document it came
  from rather than restating a measurement.
- **[`docs/security/aiops-anomaly-guard-trust.md`](../security/aiops-anomaly-guard-trust.md)** (by
  `overfit-ciso`) — **lives outside this directory**, in `docs/security/`, not `docs/aiops/`. A security
  reviewer will not think to look here for it; point them at the path above directly.
  **Placement note found in the file itself**: it was written to be copied into
  `docs/aiops/aiops-architecture-security.md` (a file `aiops-architecture.md` already cites as a sibling),
  but that copy has not happened — `docs/aiops/aiops-architecture-security.md` does not exist as of
  2026-08-09. Until it does, the security document's only location is under `docs/security/`.
- **[`aiops-detection-pipeline.md`](aiops-detection-pipeline.md)** — pre-existing, not new tonight, but the
  one most affected by the three above landing. See its own entry below.

## Deciding whether to run it

- [`aiops-business-case.md`](aiops-business-case.md) — for an engineering manager or platform lead: the
  problem it solves, what day one/hours/two days each unlock, measured cost per instance, fleet arithmetic.
  Buyer-facing summary; defers to the three documents below for the numbers it quotes.
- [`aiops-client-readiness.md`](aiops-client-readiness.md) — the detailed capability version of the same
  question: what it sees, what it cannot, what it needs (scrape interval, kube-state-metrics), and why the
  shipped advice is "shadow mode first, do not arm on day one." This is where a number in the business case
  ultimately comes from.
- [`aiops-client-flow.md`](aiops-client-flow.md) — the engagement stage by stage: what a client hands over,
  what happens each stage, what is finished versus still in flight (arming still waits on a full-day
  false-positive measurement, as of this document's last edit — check its own status line before quoting
  that as current).
- [`aiops-day-one-events.md`](aiops-day-one-events.md) — measured, not designed: what a rollout, a manual
  scale, and HPA scaling actually did to the guard on the lab cluster in one experiment. Evidence for the
  "day one is noisier" claim used above, not a design document.
- [`docs/security/aiops-anomaly-guard-trust.md`](../security/aiops-anomaly-guard-trust.md) — for a security
  reviewer or the engineer answering their questionnaire: what leaves the cluster (nothing but Prometheus
  reads), what a compromise of the pod yields, permissions, data at rest.

## Building on it / changing a detector

- [`aiops-architecture.md`](aiops-architecture.md) — read this first when the question is "what are the
  pieces and where does this change land." System context, assembly boundaries, the four detection families
  each with what they answer and what they cannot see, the two execution paths that never meet.
- [`aiops-metrics-catalogue.md`](aiops-metrics-catalogue.md) — every channel the guard supports today: source
  series, `MetricSourceKind`, the PromQL it actually becomes, unit, stack-neutral vs. .NET-specific, which
  detector families can act on it, its calibrated floor with the conditions it was measured under, and its
  known failure mode. Read this before binding a new deployment or trusting a number a channel reports.
- [`aiops-coverage-map.md`](aiops-coverage-map.md) — the same information reorganised **per failure mode**
  instead of per channel: OOM kill, memory leak, CPU throttling vs. starvation, hung requests, lock
  contention, scrape saturation, a bad rollout, and so on, each marked detected (live-verified vs.
  fixture-proven only), structurally undetectable, or uncovered today. Read this before answering "which
  failures will this catch."
- [`aiops-detection-pipeline.md`](aiops-detection-pipeline.md) — 1151 lines, the deepest thing here: how a
  metric becomes an incident, and the measurement behind every calibrated threshold. Read this before
  changing a threshold or a detector's arithmetic. **Known stale on one point, corrected by the file above**:
  it describes incident matching as "same primary subject + subject overlap," treating overlap as a gate;
  the code (`Sources/Anomalies/Incidents/IncidentTracker.cs:24-38`) removed that veto, and matching is now
  primary-subject alone, with overlap only ranking candidates (`AN-D10` in `docs/TASKS.md`). Otherwise
  treat it as authoritative for the "why this number" question — that is not `aiops-architecture.md`'s job.
- [`aiops-adding-a-metric.md`](aiops-adding-a-metric.md) — the runbook, for an operator or a developer acting
  on one, when a new metric must be watched. Step 0 is which vehicle (`CustomMetricBinding` vs. a new
  `MetricIndex` member); Step 4 is the table operators keep asking for — does this need training, per
  mechanism, with the answer "no" for four of five and the two-calendar-day floor for seasonal history.
- [`aiops-operator-feedback.md`](aiops-operator-feedback.md) — design note, not shipped: what an
  acknowledge/suppress button should mean, given three candidate semantics that are three different products.
  Written before the code; check `docs/TASKS.md` for whether a choice has since shipped.
- [`aiops-multi-scope-design.md`](aiops-multi-scope-design.md) — design for several namespaces in one process.
  **Deferred by decision**, stated in its own text: one instance per namespace costs 130 MiB and no code today;
  build this when a client says the fleet is unmanageable, not before. Nothing in it is measured — the cost
  arithmetic is arithmetic, and it says so.
- [`aiops-repair-plan.md`](aiops-repair-plan.md) — six defects found by reading the code on 2026-08-01, not by
  running it, each with damage, fix, and the test that would have caught it. Historical record of a completed
  pass — check `docs/TASKS.md` for whether each is still open.
- [`gp-anomaly-baseline.md`](gp-anomaly-baseline.md) — sketch for a Gaussian-process baseline to A/B against
  the learned/GPT detector. Proposed, not implemented, not scheduled.

## Working out what is open right now

- [`docs/TASKS.md`](../TASKS.md) — **the only place status lives.** `aiops-backlog.md` below reads like a
  status list and is not one.
- [`aiops-backlog.md`](aiops-backlog.md) — domain prose and measured history: what was tried, what it cost,
  what a fix changed. **Commentary, not status** — its rows and `docs/TASKS.md`'s rows diverged on 2026-08-08
  when the same item was edited in both, which is the reason this distinction exists at all. If a row here
  disagrees with `docs/TASKS.md`, `docs/TASKS.md` wins. Read this file for *why* something was done or
  deferred; read `docs/TASKS.md` for whether it still is.

## Strategy notes for unshipped candidate products — do not quote to a client

- [`aiops-cluster-anomaly-guard.md`](aiops-cluster-anomaly-guard.md) — internal product/technical blueprint
  for "Watch," dated 2026-07-24, revision 1. Says of itself: "not linked from the README... it describes a
  candidate second product, not a feature of the inference engine." Carries per-cluster/site-licence pricing
  (§23) and market positioning. Its implementation-map pointer (`aiops-detection-pipeline.md`) is accurate;
  its product framing is not launch copy.
- [`aiops-canary-blueprint.md`](aiops-canary-blueprint.md) — internal design/strategy note for automated
  canary analysis ("Compare"), dated 2026-07-23 rev 2. Says of itself: "not linked from the README — it names
  a competing system (Kayenta) and carries market/strategy judgements that don't belong in launch-facing
  copy." Superseded as a standalone product by the file above, but **remains authoritative for its own §3
  (statistics) and §4 (decision-logic holes)**, which the shipped peer-group outlier detector reuses.

Both carry pricing, competitor names and go-to-market judgement — the same category of content the Redaction
Gateway's own docs are kept out of `README.md`/`ROADMAP.md` for. Treat them the same way: internal only,
never linked from public-facing docs, never quoted verbatim to a prospect.

## Deliberately left in `docs/`

- [`../silence-review.md`](../silence-review.md) — a review method for any code whose product is the absence
  of an alarm. Born here, but it applies to anything that can fail by staying quiet, and it has its own agent.
- [`../autoresearch-program.md`](../autoresearch-program.md) — automated search instead of hand-tuning. The
  worked example is this subsystem's synthetic generator, but `CLAUDE.md` cites it as the general method, and
  the rule it teaches ("what value" → search, "what mechanism" → no search will find it) is not about
  anomalies.

---

**Files I could not find a live purpose for beyond "historical record":** `aiops-repair-plan.md` and
`aiops-operator-feedback.md` are both pre-code design/finding notes with no visible status marker of their
own — each depends on `docs/TASKS.md` to say whether it is still open, and neither points there itself.
Not a defect in either file, but a gap in the set: nothing *inside* `docs/aiops/` tells a reader that status
has moved to `docs/TASKS.md` except this map.

**Genuine near-duplication, not just overlap:** `aiops-client-flow.md` and `aiops-client-readiness.md` split
cleanly on their own terms ("how a week goes" vs. "what it can see") and both say so explicitly in their own
opening lines — that is cross-referencing, not duplication. The one pair that *is* real duplication of
content rather than of topic: `aiops-cluster-anomaly-guard.md` §13.5 (peer-group outlier preconditions) and
`aiops-canary-blueprint.md` §3 (the same statistics, since the guard's peer detector reuses the canary
engine's design unchanged) — both documents say this about each other, so it is acknowledged rather than
accidental, but a reader fixing the statistics needs to know both files describe the same mechanism.
