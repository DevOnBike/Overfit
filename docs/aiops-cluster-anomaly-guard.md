# Overfit AIOps — Cluster Anomaly Guard (product & technical blueprint)

> **Status:** internal product/technical blueprint, revision 1 (2026-07-24). Not linked from the README or
> `docs/README.md` by design — it describes a candidate second product, not a feature of the inference engine.
>
> **Companion document:** [`aiops-canary-blueprint.md`](aiops-canary-blueprint.md) covers the statistical
> engine in depth (Mann-Whitney, effect size, decision failure modes, ingestion cost). What that document calls
> the *Canary Sniper* is **Compare** here — one mode of a larger product, scheduled after the core detection is
> proven.

**Working name:** Overfit AIOps — Cluster Anomaly Guard
**Direction:** a local, Prometheus-native anomaly-detection engine for Kubernetes
**Deployment:** one .NET 10 / Native AOT process running inside the cluster
**Governing rule:** explainable algorithms decide; a local LLM may only summarise the evidence they gathered

---

## 0. What this revision changed

The original draft is intact in substance. Nine amendments were folded into the sections they belong to,
rather than appended, because a correction filed away from the claim it corrects does not get read:

| # | Section | Change |
|---|---|---|
| 1 | §11, §18 | **Kubernetes list/watch removed from MVP** — topology comes from kube-state-metrics labels |
| 2 | §19, §26 | **False-positive measurement moved from day 71 to day 10** via an offline replay harness |
| 3 | §3 | **The headline promise changed** from "earlier than static alerts" to scope + noise reduction |
| 4 | §13.5 | **Peer-group outlier gets a hard precondition** — a per-pod work metric, or it is disabled |
| 5 | §13.2 | **Zero-inflated metrics excluded from median/MAD** — wrong tool class, not an edge case |
| 6 | §13.3 | **Baseline poisoning gets a mechanism** — freeze updates on a series with an open incident |
| 7 | §8, §17, §22 | **Three missing pieces added** — feedback/suppression loop, the engine's own SLO and kill switch, per-namespace profiles |
| 8 | §23 | **Per-cluster pricing is wrong for the best segment** — site licence becomes the MSP default |
| **9** | **§13.5** | **Amendment 4 corrected by measurement (2026-07-28): normalising by work is necessary and NOT sufficient** — residual spread 26 / 44 / 52 % at skew 2.3 / 4.4 / 8.4x, ranking inverted in every run |

---

## 1. Strategic decision

Worth building — but not as a full observability platform, and not as a general "AI SRE" that stores all
telemetry, diagnoses every incident and changes the cluster on its own.

The right shape is a light component installed next to an existing Prometheus. It analyses a bounded set of
cluster metrics, builds adaptive baselines, compares like-for-like Kubernetes objects, detects trend and
operating-point changes, correlates anomalies with rollouts and cluster events, groups many signals into one
incident, and hands the result to Grafana, Alertmanager, webhooks and ticketing. It sends no metrics outside
the organisation and requires no Elastic, OpenSearch, Python or cloud AIOps service.

Product line:

| Mode | What it does | When |
|---|---|---|
| **Watch** | continuous cluster anomaly detection | first |
| **Explain** | incident narrative, timeline, evidence | with Watch |
| **Compare** | baseline versus canary during a rollout | after Watch is proven |
| **Verify** | did the rollback or fix actually restore health | after Compare |

Watch ships first. Compare and Verify only after the core detection has demonstrated value on real data.

> **Honest note on focus.** This document answers "both Overfit and AIOps" without pricing that answer.
> Overfit is a CPU inference engine at release readiness; Cluster Anomaly Guard is a Kubernetes observability
> product with a discovery-heavy go-to-market into platform teams, MSPs, finance and the public sector. The
> shared surface is the language, the comparator (§13.10) and the single-AOT-binary story — roughly 5% of the
> code and 0% of the buyer. The 90-day plan in §26 is a full-time plan. This is the argument for running §19's
> M0 *before* anything else: two weeks of interviews plus an offline replay costs little and does not require
> putting the inference engine down.

---

## 2. Goal

Local early detection of problems that have not yet crossed a static threshold; that affect a single pod, node
or new ReplicaSet; that are a slow trend; that deviate from the object's own history; that are buried across
several independent alerts; that appeared after a deployment, configuration or infrastructure change — and
that must be analysed entirely on-premise or air-gapped.

The product strengthens the existing stack rather than replacing it: Prometheus stays the store, Grafana stays
the UI, Alertmanager stays the router, and Overfit supplies adaptive signals, incidents and justifications.

---

## 3. The promise

> **Detect unusual Kubernetes behaviour with far less noise and a precise blast radius, without sending
> metrics outside the cluster and without installing another heavy observability platform.**

**Amendment — do not lead with "earlier than static alerts".** The flagship demo (a memory leak confined to
new pods, §24) is caught by a one-line recording rule that appears in every Prometheus alerting guide:

```promql
predict_linear(container_memory_working_set_bytes[1h], 4*3600) > container_spec_memory_limit_bytes
```

A competent SRE will raise exactly this in the first meeting, and "we detect it earlier" is a contestable
claim to stake the product on. The defensible differentiator is not the detection — it is that the finding
arrives **scoped to the new ReplicaSet, correlated with the 14:03 rollout, and grouped with the throttling and
p95 signals into a single incident with a timeline**. A threshold rule produces a signal; this produces a
scope and a probable cause.

Earlier detection remains a real secondary benefit (§4.3 trends), but noise reduction and scope are the claims
that survive scrutiny.

Technical positioning: *Prometheus-native anomaly detection for Kubernetes — self-hosted, explainable, one
.NET binary.*
Sales positioning: *Keep Prometheus and Grafana. Add a local engine that understands what is unusual for your
cluster.*

---

## 4. Problems solved

**4.1 Static thresholds have no context.** "Memory above 2 GB" cannot say whether 2 GB is unusual *for this
workload*, whether the growth tracks traffic, whether it is one pod, whether only the new version behaves this
way, or whether memory is climbing steadily well below the limit. Hard thresholds remain necessary as SLO
safety rails — they must not be the only detection method.

**4.2 Alert fatigue.** One fault fires throttling, latency, restart, HPA, queue and timeout alerts. These
belong in one incident with one timeline.

**4.3 Slow leaks and trend changes.** Steady growth in memory, latency, queue lag or unit cost should be
caught before the limit, with an estimated time to critical.

**4.4 A single broken pod or node.** If eleven replicas behave alike and one deviates, that is detectable with
no history at all — which is what makes it valuable during cold start (§14).

**4.5 No deployment context.** Better than "CPU anomalous": *14:03 v2.17 deployed → 14:05 new pods took
traffic → 14:07 throttling only in the new ReplicaSet → 14:09 p95 up 34% → old ReplicaSet normal.*

**4.6 Missing data read as health.** Healthy, Anomalous, Warming Up, Insufficient Data and Error are distinct.
Absence of samples must never resolve to healthy.

**4.7 Privacy and air-gap.** Many organisations cannot ship metrics to SaaS, install cloud agents, or run
Elastic solely for anomaly detection.

---

## 5. The niche

**Self-hosted, Prometheus-native anomaly detection for Kubernetes, delivered as a small auditable component
running entirely inside the cluster.**

It combines Kubernetes topology awareness, Prometheus as the existing data source, no new telemetry store,
self-hosting and air-gap, explainable algorithms, a small deployment footprint, Native AOT, no Python,
ready-made signal packs, and a later path into canary deployments.

**Entry wedge.** The customer already runs Kubernetes, Prometheus, Grafana and Alertmanager, does not want to
change the stack, needs adaptive alerts, outlier pods, trends and rollout correlation — and the metrics cannot
leave the organisation.

Datadog, Grafana Cloud, Elastic and OpenSearch all confirm the demand exists. This product is not meant to be
broader than them. It is meant to be lighter, local, and trivial to add to an existing Prometheus.

---

## 6. Competitive landscape (verified 2026-07-24)

| Product | What it does | Its strength | Room for Overfit |
|---|---|---|---|
| Datadog Watchdog | automatic baselines, anomaly alerts, service impact, partial RCA | whole-platform context | no SaaS, no egress, no migration |
| Grafana Cloud ML / AI | forecasting, anomaly detection, peer outliers, Sift | Grafana + many sources | fully local, no Grafana Cloud |
| Elastic ML | anomaly-detection jobs for hosts and pods | mature analytics stack | no Elastic Stack requirement |
| OpenSearch AD | near-real-time detection via Random Cut Forest | self-hosting, streaming | no copying metrics into OpenSearch |
| Robusta / HolmesGPT | alert grouping, enrichment, AI investigations | fast post-alert triage | deterministic detection *before* the alert |
| Prometheus / Alertmanager | rules, thresholds, routing | the industry standard | adaptive baselines and correlation without replacing it |
| Argo Rollouts | progressive delivery, AnalysisTemplate | rollout orchestration | statistical provider / webhook, post-MVP |
| Flagger | canary analysis with threshold ranges | ready-made canary automation | distribution comparison, effect size, Inconclusive |

**The competitor the table omits: a few recording rules and an afternoon.** Prometheus ships
`predict_linear`, `holt_winters`, `quantile_over_time` and `stddev_over_time`. Before a line of code exists,
there has to be an answer to "why not just write rules", and it has to be stronger than "because ours is ML".
The answer is §3: scope, correlation and grouping — none of which a recording rule produces.

**Where not to compete:** whole-telemetry storage, log management, tracing, hundreds of integrations, a query
language, rich dashboards, full incident management, universal automatic RCA.

**Where it can win:** small install, fast time-to-value, no data migration, no egress, no new TSDB, ready-made
Kubernetes profiles, explainable detectors, auditability, OEM for MSPs.

---

## 7. Target customers

**Ideal profile.** Production Kubernetes; Prometheus or a compatible API; several to several dozen clusters;
many hand-written alerts; no mature anomaly detection; needs self-hosting or air-gap; keeps Grafana; an
overloaded SRE/DevOps team; willing to pilot on one cluster.

| Segment | Why it fits | Caution |
|---|---|---|
| **MSPs and software houses** | one product across many client clusters, measurable time saved, OEM licence, fast access to technical buyers | best first segment — but see the pricing correction in §23 |
| Finance and payments | high cost of outage, privacy, own clusters, paid pilots plausible | long security review |
| Industry, energy, telecom | on-premise and edge, limited connectivity, repeatable installs | procurement cycles |
| Healthcare, public sector | egress forbidden, air-gap, formal requirements | larger but slower contracts |
| .NET product companies | natural credibility for Overfit, reachable through the .NET community | smaller clusters, smaller budgets |

| Persona | Problem | Value |
|---|---|---|
| Head of Platform | many clusters, inconsistent alerting | policy packs, less manual configuration |
| SRE Lead | alert fatigue, slow triage | earlier detection, grouping |
| CTO at an MSP | rising cost of client operations | OEM, automation |
| Security / Compliance | egress forbidden | locality, audit trail |
| DevOps Engineer | hundreds of PromQL rules | ready-made signals and baselines |
| FinOps Lead | cost regressions | unit-cost module, post-MVP |

---

## 8. What the product does

**Prometheus.** Range and instant queries; TLS and token auth; retry, backoff, timeout; empty results, NaN and
late samples; counters to rates; resampling; query-cost limits; a short cache; self-cost metrics; a `doctor`
mode for availability and cardinality.

**Kubernetes.** Mapping across cluster, namespace, Deployment, ReplicaSet, StatefulSet, DaemonSet, Job,
CronJob, Pod, Container, Node and HPA — owner references, old versus new ReplicaSet, image changes, OOMKilled,
eviction, failed scheduling, readiness, replica-count changes, rollouts and pod-to-node relations. **See §11
for how much of this the MVP gets from metric labels rather than the API server.**

**Detection.** Hard safety rules; rolling robust baseline; seasonal baseline; peer-group outlier; trend; change
point; minimum samples; minimum duration; cooldown; warming-up; per-signal configuration.

**Grouping.** Several signals on one workload; pod versus whole deployment; a node's problem and its pods;
rollout correlation; deduplication; incident lifecycle; probable contributing changes — without claiming
confirmed root cause.

**Output.** Prometheus metrics; Alertmanager; webhook; JSON API; `status` and `explain` CLI; sample Grafana
dashboards; a justification attached to every detection.

**Amendment — the feedback loop is missing, and it is the second most valuable feature after grouping.** There
is no way for a user to say "this was expected". Without it, alert fatigue returns wearing a new costume —
which is the very problem being sold against. The minimum viable form:

- a **suppression rule** keyed by the deduplication key (workload + signal + detector), with a TTL and a
  required reason string;
- **"expected" feedback** that widens the expected range for that series rather than silencing it outright, so
  a genuinely larger deviation still fires;
- suppressions are **configuration, not hidden state** — they live in the ConfigMap and are visible in
  `explain`, because an invisible mute is how monitoring quietly stops working.

---

## 9. What the product must not do

**In the MVP:** its own TSDB; dashboards instead of Grafana; logs and traces; a per-node agent; analysing
every metric and every label; a query language; patching Kubernetes; automatic rollback; automatic HPA
changes; automatic restarts; full RCA; an LLM as an alarm source; Python; GPU; Elastic or OpenSearch as a
dependency; a zero-allocation claim over Prometheus JSON; a Healthy state when data is missing; a decision on
insufficient samples.

**After the MVP:** auto-remediation only as a separate audited layer; every cluster modification behind manual
approval; the LLM never influences severity; an anomaly score is not a statement of business impact.

---

## 10. Signal packs

| Pack | Signals |
|---|---|
| **Node** | CPU utilisation and saturation, memory pressure, available memory, filesystem usage and trend, inode usage, disk latency, network errors, TCP retransmits, readiness, evictions |
| **Pod / container** | CPU, throttling ratio, working set, RSS, memory slope, restarts, OOMKilled, readiness, network I/O, filesystem growth, terminated containers |
| **Workload** | ready vs desired replicas, unavailable replicas, pending pods, failed scheduling, CrashLoopBackOff, job failures, HPA oscillation, rollout duration, differences between ReplicaSets |
| **Application (RED)** | request rate, error rate, duration, p50/p95/p99, timeout rate, rejection rate, throughput per replica |
| **Dependency** | database latency, connection pool, queue depth, consumer lag, cache hit ratio, external HTTP errors, retry rate |
| **Control plane** *(post-MVP)* | API server latency, etcd latency, leader changes, scheduler errors, controller workqueue, admission webhook latency |

The MVP ships **10–15 signals across at most two packs**, not the full matrix. The list above is the roadmap
of what the packs eventually contain, not the first release.

---

## 11. Architecture

```text
Prometheus HTTP API
        │
        ▼
Query engine ── retry / backoff / cache / cost budget
        │
        ▼
Topology resolver ── kube-state-metrics labels (MVP) │ list/watch (post-MVP)
        │
        ▼
Time-series normalizer ── grid, gaps, NaN, counter resets, rates, units
        │
        ▼
Detector pipeline ── hard rules → robust baseline → peer outlier → trend/change point
        │
        ▼
Incident grouper ── dedup, scope, severity, lifecycle, rollout correlation
        │
        ▼
Output ── Prometheus metrics │ Alertmanager │ webhook │ JSON │ CLI │ (optional local LLM summary)
```

**Amendment — no Kubernetes list/watch in the MVP.** The original scope included a topology cache built on
list/watch with owner references, cache rebuild and watch reconnect. That is a substantial subsystem, it
carries its own failure class, and **it is largely redundant at this stage: the topology is already in
Prometheus.**

`kube_pod_owner`, `kube_replicaset_owner` and `kube_deployment_created` from kube-state-metrics give the
Pod → ReplicaSet → Deployment chain and the ReplicaSet creation time in a single PromQL query, and cAdvisor
already carries `namespace` / `pod` / `container` labels on every series. That covers rollout correlation and
scope — the two things the demo depends on — with no connection to the API server at all.

What this removes from the MVP: RBAC across ten resource kinds, watch reconnect, cache rebuild, cache-coherence
bugs, and the API-server permission conversation in every security review. What it costs: the details labels do
not carry — Helm release annotations, image digests, eviction reasons. Those justify list/watch **after M5**,
when the value is already proven.

**State.** Ring buffers, medians, MAD, hourly baselines, detector state, active incidents, topology cache and
deduplication keys, persisted as a versioned snapshot on a persistent volume. No SQLite in the MVP.

**Kubernetes events are context only.** `resourceVersion`, ReplicaSet creation time, pod-template changes,
image digest, owner references, status and annotations are all stronger evidence than the event stream.

---

## 12. Incident model

**Statuses:** Healthy · Anomalous · Warming Up · Insufficient Data · Error.

**Severity** derives from anomaly score, duration, number of affected objects, SLO proximity, direction of
change, traffic impact, hard symptoms, and correlation with a new version.

**Contents:** id, status, severity, cluster, namespace, workload, affected pods, signal, detector, observed
value, expected range, score, start and end, sample count, evidence, related changes, limitations, data status.

```json
{
  "severity": "high",
  "status": "anomalous",
  "scope": { "cluster": "production-eu", "namespace": "checkout", "workload": "checkout-api" },
  "signal": "container_memory_working_set_bytes",
  "detector": "trend-and-peer",
  "actual": 3180000000,
  "expectedRange": { "lower": 1100000000, "upper": 1900000000 },
  "confidence": 0.97,
  "dataStatus": "sufficient",
  "firstObserved": "2026-07-24T08:43:00Z",
  "evidence": [
    "Memory is 67% above the expected range for this workload at this hour.",
    "The increase began 4 minutes after deployment v2.17.0.",
    "Only pods from the new ReplicaSet are affected; the old ReplicaSet is within range."
  ],
  "limitations": ["Seasonal baseline is still warming up (6 days of history)."]
}
```

Exported alongside it, so the existing Grafana and Alertmanager remain the interface:

```text
overfit_anomaly_score{namespace="checkout",workload="checkout-api",signal="memory",detector="trend"} 0.97
```

---

## 13. Algorithms

**13.1 Hard rules** — effective from minute one: all replicas unavailable, sustained 5xx, OOMKilled, node
NotReady, restart storms, disk full, no canary samples, Prometheus unreachable, latency past a hard SLO.

**13.2 Median and MAD** — the base robust baseline for memory, CPU, latency, traffic and peer comparison.

> **Amendment — MAD = 0 is not an edge case, it is the wrong tool class.** Error rate, restart count,
> OOMKilled and failed-scheduling counts are *zero-inflated*: the median is 0, the MAD is 0, and no fallback
> repairs that, because the problem is the choice of statistic rather than a degenerate input. These metrics
> need **count-based tests (Poisson / binomial)** on the event count in a window, not a robust scale estimate.
> **Zero-inflated signals must not be routed through the median/MAD detector at all** — they get hard rules
> (§13.1) plus a count test, and the router enforces that by signal type.

**13.3 Rolling baseline** — short, medium and long windows.

> **Amendment — baseline poisoning needs a mechanism, not a warning.** "Adaptation must be slow so the
> detector does not learn the outage" names a real failure without fixing it: a two-hour incident silently
> becomes the new normal. The mechanism: **freeze baseline updates for any series with an open incident**, and
> resume only after the incident closes; on close, discard the incident window from the history rather than
> folding it in. Slowness alone cannot solve this — it only makes the poisoning take longer.

**13.4 Seasonal-naive** — compare against the same hour of day and day of week; expected value from the
median, tolerance from the MAD of residuals. A far better first step than Prophet, LSTM or heavy forecasting.

**13.5 Peer-group outlier** — pods of a Deployment, pods of a StatefulSet, nodes of a pool, and old versus new
ReplicaSet. MVP: at least three series, median/MAD, duration, quorum.

> **Amendment — a hard precondition, not a nice-to-have.** "Eleven pods alike, one different" assumes even
> load balancing. With sticky sessions, uneven sharding, a hot tenant or keep-alive skew, one pod legitimately
> does three times the work, and the detector fires on **correct** behaviour. This is the traffic-mix hole of
> the canary blueprint (§4.3 there) reappearing on a different axis.
>
> **Peer detection requires a per-pod unit-of-work metric** — memory per request, CPU per request — and must
> be **disabled, reporting `Insufficient Data`, when one is unavailable.** Raw per-pod resource comparison is
> offered only for signals where uneven load cannot explain the deviation (restart count, OOMKilled, readiness).

> **★ Correction (2026-07-28), measured in the lab: normalising by work is necessary and NOT sufficient.**
> The amendment above quietly assumed that dividing by a work metric restores comparability. It does not.
> Three identical replicas in `k8s/`, one deliberately given a multiple of the others' traffic, ~600 requests
> per run, three skews:
>
> | Achieved skew | requests/s spread | raw CPU spread | **CPU per request** |
> |--:|--:|--:|--:|
> | 2.3× | 87.9 % | 57.5 % | **26.2 %** |
> | 4.4× | 161.6 % | 116.8 % | **44.3 %** |
> | 8.4× | 223.7 % | 185.1 % | **51.6 %** |
>
> Division removes most of the apparent difference and leaves a residue that **grows with the imbalance and
> never vanishes** — and in every run the busiest replica came out with the **lowest** cost per request. The
> per-replica numbers show why:
>
> ```
> skew 2.3   7.27  7.26  |  5.52
> skew 4.4   7.57  6.82  |  4.74
> skew 8.4   7.06  6.62  |  4.01
>            └ lightly loaded ┘  └ busy ┘
> ```
>
> The lightly-loaded replicas hold ~6.6–7.6 CPU-seconds per request in every run; the busy one falls
> monotonically as its share rises. That is fixed per-process overhead (background threads, GC, idle polling)
> amortised over more requests — a replica is cheaper per request precisely because it is busier.
>
> **The shape was predicted before it was measured.** An affine cost model, `cost ≈ fixed + marginal × work`,
> puts the residue proportional to `(1 − 1/skew)`, which saturates. Normalised to the largest skew the model
> predicts 0.571 / 0.857 / 1.0; measurement gave 0.508 / 0.859 / 1.0 — within 0.2 % at the middle point,
> ~11 % off at the smallest. Consistent with the model, and not a fit of it: a single global
> `(fixed, marginal)` pair does not reproduce all three runs, so the mechanism is right and the magnitude is
> not yet characterised.
>
> **Consequence for the MVP.** A peer comparison on unit cost under uneven traffic will still flag a
> difference — it will point at the *wrong* member, which is worse than flagging nothing. Three things follow:
>
> 1. **The verdict should rest on a load-independent signal** (restarts, OOMKilled, readiness, throttling)
>    whenever traffic is materially uneven. Unit cost stays as supporting evidence, not as the decision.
> 2. **The honest model is affine, not proportional** — `cost ≈ fixed + marginal × work`. Fitting the two
>    coefficients across the peer group and comparing the *marginal* term is what "cost per request" was
>    supposed to mean; plain division conflates it with the fixed term. That is a real piece of work, not a
>    parameter change.
> 3. **Traffic-mix parity is not only a canary problem.** §4.3 raised it for the canary axis; it is at least
>    as sharp here, and it cannot be waved away by normalising.
>
> *Caveat kept deliberately: one run per skew, 180 s each, two-minute rate windows, on a single-node lab
> where all three replicas share a host. The direction, the monotonic growth and the inverted ranking are
> solid across three points; the absolute residue is not, and no threshold should be derived from these
> numbers without repeating them on a multi-node cluster.*

> **Status: implemented.** `PeerGroupOutlierDetector` in `Sources/Main/Statistics/`, on the §13.10 comparator,
> 14 tests. Leave-one-out: each member's window against the pooled windows of the others, through
> `ITwoSampleComparer`. `PeerSignalKind` makes the load precondition structural — a load-sensitive signal
> submitted without `Work` returns `InsufficientData` instead of guessing. Scratch is pooled; the only
> allocation is the caller's findings buffer.
>
> **A masking bound found while building it, worth stating because it is not obvious.** In a leave-one-out
> design the effect available to a deviating member is capped by the fraction of clean siblings — roughly
> **(n − k) / (n − 1)** for *k* deviants among *n* peers. So a **one-sided detector fails silently exactly
> where it matters most**: with eight of ten pods regressed, each one's effect falls to ~0.11, under any usable
> threshold, and the group reads as **Healthy** — the §4.1 "silently approves" failure, reappearing inside a
> detector rather than a decision rule.
>
> The fix is to test **both directions**. In that same scenario the two untouched pods stand out sharply
> *below* their peers, so the group stays visible; and when members deviate on both sides at once, the group
> has no single norm and the verdict is `Inconclusive` rather than a confident list. What remains genuinely
> undecidable is **attribution** — a relative method has no external reference, so "eight regressed" and "two
> are idle" produce identical evidence. The finding therefore states the direction and stops; resolving it
> needs the workload's own history. **This is the concrete reason peer detection and the trend/baseline
> detector are complementary rather than alternatives**, and why the MVP ships both.

Post-MVP: DBSCAN over the shape of time windows.

**13.6 EWMA** — smoothing and fast adaptation. Never the sole source of a critical alarm.

**13.7 CUSUM** — small, persistent shifts in latency, CPU per request, error rate, cache miss, queue lag.

**13.8 Page-Hinkley** — a light online change-point detector. The MVP ships **Page-Hinkley or CUSUM, not both
as public options**.

**13.9 Robust slope / Theil–Sen** — memory leaks and trends. Output carries growth rate, period, estimated
time to limit, confidence and scope.

> **Status: implemented.** `TrendDetector` in `Sources/Main/Statistics/`, 16 tests. Three rank-based pieces,
> deliberately mirroring the peer detector's gate rather than inventing a second vocabulary: **Theil-Sen**
> (magnitude — the median of every pairwise slope, so ~29% of the window can be garbage before it breaks),
> **Mann-Kendall** (significance — the same concordant-minus-discordant machinery as Mann-Whitney, applied
> against time), **Kendall's tau** (effect size — the direct analogue of Cliff's delta). Pooled scratch, zero
> GC allocation.
>
> **Autocorrelation correction is not optional, and this is the finding worth carrying forward.** Mann-Kendall
> assumes independent observations; consecutive scrapes of memory or latency are nearly identical. Feeding a
> correlated series to a test that assumes independence **manufactures significance** — a plain random walk,
> which by construction has no trend at all, produces a p-value that looks overwhelming. An uncorrected trend
> detector on real Kubernetes data therefore fires constantly, delivering exactly the alert fatigue §4.2 sells
> against. The implementation estimates lag-1 autocorrelation on the detrended residuals and inflates the
> variance by (1+ρ)/(1−ρ) — the AR(1) effective-sample-size factor — reporting ρ on the result so a reader can
> see how much was discounted. A test pins this: a random walk with ρ > 0.5 must come back `Healthy`.
>
> **Two thresholds, because either alone misleads.** Tau measures how *consistent* the movement is — a series
> creeping up 1% but never once dipping scores near 1.0 and is not worth waking anyone — so a second gate
> requires the fitted change to reach a fraction of the series' own median across the window. Expressed
> relatively, one number serves bytes, seconds and counts alike.
>
> **Reported honestly:** time-to-limit is projected from the *end* of the window (the observed period has
> already consumed part of the headroom), is null when the series moves away from the limit, and is null beyond
> a decade rather than printing a number nobody can act on. A series sitting at zero — error counts, restarts —
> has no usable relative scale, so the size gate is skipped and the reason says so, instead of dividing by
> something arbitrarily small.

**13.10 Mann-Whitney U** — for baseline versus canary. Requires minimum samples, effect size, traffic-mix
validation, `Inconclusive`, multiple-comparison correction and a single-look verdict.

> **Amendment — this is already built, and it shortens the MVP rather than lengthening it.**
> `Sources/Main/Statistics/` ships `MannWhitneyU` (mid-ranked ties, tie-corrected variance, continuity
> correction, four ingestion shapes including O(K) over histogram buckets, zero allocation) behind
> `ITwoSampleComparer`, with `TwoSampleComparison.IsRegression(maxPValue, minEffectSize, minimumSamplesPerArm)`
> encoding the significance + effect + sample-floor gate directly. 36 tests, benchmarked.
>
> More importantly, **§13.5 peer-group outlier is the same operation**: one population (this pod) against
> another (its siblings). Median/MAD is a point-estimate detector — "is the current value an outlier"; the
> comparator is a distribution test — "is this pod's window different from its siblings' window". The latter is
> more robust, is already written and tested, and using it for peer detection removes a second independent set
> of thresholds to tune for the same question. **Peer-group outlier should ship on the existing comparator**,
> which makes Compare (§20) a thin addition rather than a new subsystem.

**13.11 Cliff's delta** — a p-value is not enough; a regression counts when the difference is both credible
and past a business effect threshold.

**13.12 Bonferroni and Benjamini-Hochberg** — Bonferroni in Compare's MVP; BH and an aggregate score later.

**13.13 Random Cut Forest** *(post-MVP)* — only once the simple detectors have demonstrated limitations on
real data and a reduction in false positives can be shown.

**13.14 LLM for Explain only** — it never changes status or severity and never triggers an action; it receives
structured evidence and returns prose.

---

## 14. Cold start

| History | Available mechanisms |
|---|---|
| 0–24 h | hard rules, peer outliers, fast trend, rolling baseline |
| 1–7 days | local daily baseline |
| 2–4 weeks | daily and weekly seasonality |
| > 4 weeks | stable seasonality, capacity trends |

Do not promise full adaptivity from minute one. The ordering above is deliberately **monotone in data
requirement**, so every stage demonstrates value on the day it is installed.

---

## 15. Cardinality

Metric and label allowlists; a hard series cap; per-namespace and per-workload limits; detector TTL; top-N
workloads; normalised route labels; no user IDs, request IDs or raw URLs; a report of rejected series; a
dry-run `doctor`; bounded concurrency; a time and data budget per cycle.

Route-label stratification is exactly what causes high-cardinality blow-ups, so strata are **hard-capped**:
the top *N* routes by traffic share, everything else folded into `Other` — and `Other` is itself watched for
composition shift, because a catch-all whose mix changes re-creates the bias stratification was meant to
remove. See the canary blueprint §4.3 for the full argument.

**Overfit must not overload Prometheus.**

---

## 16. External libraries

The MVP can be built with **no runtime dependencies**. .NET 10 supplies HTTP, JSON, TLS, collections, channels,
memory, cryptography, Native AOT and SIMD.

Implemented in-house: median, quantiles, MAD, EWMA, CUSUM, Page-Hinkley, robust slope, Mann-Whitney U (done),
Cliff's delta (done), Bonferroni, Benjamini-Hochberg. A large maths library is not needed.

**Kubernetes API:** direct REST, service-account token, CA, minimal JSON models. The official .NET client only
after verifying Native AOT, trimming, reflection, size and watch reconnect — and under §11 the MVP does not
need it at all.

**Configuration:** the runtime reads JSON; Helm generates the ConfigMap; **no YAML parser in the application**.
This is a deliberate AOT decision — YAML parsers are reflection-heavy and are a common source of trim warnings.

**Library acceptance criteria:** pure-managed, AOT-clean, trimmable, actively maintained, small surface,
compatible licence, benchmarked, and a clear reduction in risk.

---

## 17. Security

**RBAC:** `get`/`list`/`watch` only, on namespaces, nodes, pods, events, deployments, replicasets,
statefulsets, daemonsets, jobs, cronjobs and HPAs. No create, update, patch, delete, exec, logs or secrets.
Under §11 the MVP needs even less than this.

**Prometheus:** TLS, token or mTLS, secret masking, and `insecure skip verify` never a default.

**Distribution:** signed images, SBOM, provenance, CVE scanning, an air-gap bundle, LTS, a documented upgrade
path.

**Amendment — the engine needs its own SLO and a kill switch.** Selling a monitoring component into an
air-gapped bank invites one question immediately: *what happens if your thing takes down our Prometheus?*
§15's query budgets are the mechanism; what is missing is the commitment and the escape hatch:

- a **published resource envelope** — CPU, RSS and queries-per-minute ceilings the process will not exceed,
  enforced in code and exported as metrics, not stated in a datasheet;
- **self-throttling** — when Prometheus query latency rises past a threshold, the engine widens its own
  interval rather than retrying harder;
- **a kill switch** — one ConfigMap flag (and one CLI verb) that stops all querying and detection while
  leaving the process up, so an operator can neutralise it during an incident without an uninstall;
- **a clean uninstall** — removing the Helm release leaves no CRDs, webhooks or admission hooks behind.

---

## 18. MVP

**Goal:** prove that the product surfaces a real problem with materially less noise and a precise scope,
compared with threshold rules.

| # | Item | Note |
|---|---|---|
| 1 | Prometheus HTTP API client | retry, backoff, budgets |
| 2 | Topology from kube-state-metrics labels | **replaces the list/watch cache — §11** |
| 3 | 10–15 signals | two packs, not five |
| 4 | Hard rules | §13.1 |
| 5 | Rolling median / MAD | zero-inflated signals excluded — §13.2 |
| 6 | Peer outlier | **done** — on the existing comparator, two-sided, work-metric precondition enforced (§13.5) |
| 7 | One change-point / trend detector | **done** — Theil-Sen + Mann-Kendall, autocorrelation-corrected (§13.9) |
| 8 | Deployment / ReplicaSet correlation | from labels |
| 9 | Workload / node grouping | one incident, one dedup key |
| 10 | Data statuses | **done** — `DetectionStatus`, shared by every detector (§12) |
| 11 | `/metrics` endpoint | `overfit_anomaly_score` |
| 12 | Alertmanager webhook | |
| 13 | CLI `doctor`, `status`, `explain` | |
| 14 | State snapshot | versioned, on a PV |
| 15 | Suppression / feedback loop | §8 amendment |
| 16 | Offline replay harness | **moved forward from M2 — §19** |
| 17 | Helm chart | |
| 18 | Grafana dashboard | one, not a suite |

**Deliberately out of the MVP:** Argo, rollback, remediation, Random Cut Forest, multivariate detection, LLM
summary, a UI, multi-cluster, FinOps, forecasting, logs, traces, an own TSDB, and Kubernetes list/watch.

---

## 19. Milestones

**M0 — Validation.** 10–15 interviews; 3–5 historical incidents; Prometheus data; a description of the
Kubernetes events; measured detection and triage time; two pilot partners.

> **Amendment — the offline replay harness belongs here, not in M2.** The single riskiest assumption in this
> product is the false-positive rate, and the original plan measured it at day 71, after everything was built.
> That is backwards.
>
> Take a two-week Prometheus snapshot from a pilot (`promtool tsdb dump`, or plain range queries), run
> candidate detectors over it **offline** — no Kubernetes, no Helm, no deployment — and count how many alerts
> each would have fired, against the incidents the customer already knows about. That is roughly a week of
> work and it either validates or kills the product in week 2 instead of week 13. It is the same discipline
> this repository applies to performance: measure before you build, not after.

**Exit criteria:** three replayable incidents, two pilots, a confirmed cost of the problem, and a
false-positive count from real data that is not disqualifying.

**M1 — Ingestion.** Range queries, retry, auth, resampling, rates, missing data, query budgets, query packs,
`doctor`. *Exit:* runs against kube-prometheus-stack, no query storm, replay works, CPU/RAM measured.

**M2 — Detectors.** Rolling MAD, peer comparison, EWMA, Page-Hinkley or CUSUM, robust slope, statuses,
warming-up. *Exit:* synthetic tests, replay, expected ranges, false positives under control.

**M3 — Kubernetes awareness.** Label-derived topology, old versus new ReplicaSet, image changes, timeline,
OOM, eviction, scheduling. *Exit:* correct scope, correct affected replicas. *(list/watch deferred past M5.)*

**M4 — Incidents and output.** Dedup, grouping, severity, lifecycle, exporter, Alertmanager, webhook, Explain,
JSON, suppression. *Exit:* several signals form one incident with a stable dedup key.

**M5 — Private beta.** Helm, dashboards, docs, runbook, signed images, profiles, upgrade. *Exit:* 2–3 clusters,
a real incident, measured noise or triage reduction, no cardinality problem.

**M6 — Commercial 1.0.** Commercial licence, air-gap bundle, SBOM, support, LTS, compatibility matrix,
backup/restore, security documentation.

---

## 20. After the MVP

**Compare** — baseline versus canary: CPU per request, latency, an error circuit breaker, Mann-Whitney, effect
size, minimum samples, `Inconclusive`, single-look verdict, traffic-mix validation. The full treatment lives in
[`aiops-canary-blueprint.md`](aiops-canary-blueprint.md); with §13.10's comparator and §13.5's peer detection
already shipped, this is largely an integration.

**Argo Rollouts** — Overfit as a provider or webhook; Argo still owns the traffic.

**Verify** — did latency, errors and resource use return to baseline after a rollback. No improvement means
the rollback did not remove the problem — a loop almost nobody closes.

**FinOps regression** — CPU-seconds and RAM-seconds per request, cost per transaction, per namespace, per
version.

**Multi-cluster** — a central console over anomaly events without shipping raw metrics; tenant isolation; an
MSP dashboard.

**LLM summary** — local model, structured evidence only, cited signals, no influence on the decision.

**Auto-remediation** — ticket, runbook and suggested command first, behind manual approval. Automatic patching
only as a separate module.

---

## 21. Quality testing

**Data:** synthetic scenarios, real incidents, normal windows, traffic growth without a fault, maintenance
windows, deployment regressions, shared-infrastructure effects, missing data, and Overfit restarts.

**Scenarios:** single-pod memory leak; all pods growing with traffic; throttling in a new ReplicaSet; node
pressure; nightly batch; weekend drop; spike; Prometheus timeout; HPA oscillation; queue lag; OOM; scheduling
failure; canary with no traffic.

**Metrics:** true incidents; false positives per cluster per day; detection delay; alert-grouping ratio; time
to explanation; share of `Insufficient Data`; query cost; CPU; RAM; rejected series; user-confirmed incidents.

**Hypotheses** (fewer than one useless incident per day; earlier trend detection; 50% fewer notifications
through grouping; never Healthy without data) are **hypotheses**. Do not publish any of them before the pilots
produce numbers.

---

## 22. Profiles

| Profile | Confidence | Samples | Effect threshold | Use |
|---|---|---|---|---|
| **Strict** | high | many | small | production, high tolerance for `Inconclusive` |
| **Balanced** | default | default | default | the default |
| **Fast Feedback** | lower | fewer | large | staging, internal systems |

Profiles set effect threshold, confidence, minimum samples, duration and cooldown. **No marketing "sigma
multiplier"** — a σ knob promises a normal-distribution trade-off the data cannot honour.

**Amendment — profiles must be assignable per namespace.** A single global profile is wrong for a real
cluster: production wants Strict and staging wants Fast Feedback, in the same cluster, on the same install.
Resolution order: signal override → namespace → cluster default.

---

## 23. Monetisation

| Tier | Contents |
|---|---|
| **Community** | AGPL-3.0, basic Watch, Prometheus, three detectors, Helm, community support |
| **Commercial** | commercial licence, signed images, SBOM, air-gap bundle, support, enterprise signal packs, certified compatibility matrix |
| **Enterprise** | multi-cluster, central console, SSO/RBAC, audit, custom packs, OEM/MSP, LTS, SLA, private fixes, onboarding |

AGPL plus a commercial licence needs a lawyer before any of this is published.

**Metering.** Per cluster/year, per node tier, site licence, or MSP/OEM. **Never** per metric, per series, per
GB or per anomaly — per-series pricing would charge the customer for the cardinality this product tells them
to control, and it is internally incoherent with §15.

| Offer | Hypothesis |
|---|---:|
| Single-cluster pilot | 5–15 k EUR |
| Commercial Starter | 3–8 k EUR / cluster / year |
| Enterprise | 20–80 k EUR / organisation / year |
| MSP / OEM | 25–150 k EUR / year |
| Audit / onboarding | 5–20 k EUR |

> **Amendment — per-cluster is the wrong default meter for the best segment.** §7 names MSPs and software
> houses as the strongest first segment; at 3–8 k per cluster, an MSP with forty client clusters faces
> 120–320 k EUR for an anomaly detector sitting next to a free Prometheus. They will write recording rules
> instead. The pricing model punishes exactly the buyer the strategy depends on.
>
> **Make a site / organisation licence the default for MSPs and software houses from day one**, with
> per-cluster reserved for single-cluster enterprises. The MSP/OEM line partially acknowledges this already —
> it should be the primary path for that segment, not the exception.

**30-day pilot:** install, 10–15 signals, baselines, peer detection, Alertmanager, two historical incidents,
a false-positive count, an alert cleanup, and an ROI report. First revenue most likely comes from pilots and
onboarding, not licences.

---

## 24. Finding customers

**Discovery list of 30:** DevOps Lead, SRE Lead, Head of Platform, CTO of a software house, MSP owner,
on-prem administrator, FinOps, Security Architect.

Questions: how do you detect problems that never cross a threshold; how many alerts do you ignore; how long
does establishing scope take; what did you catch too late; do you use cloud observability; may metrics leave
the organisation; do you keep historical incidents; would you pilot alongside Prometheus; what result counts
as success; who approves the purchase; what pricing model is acceptable; do you require air-gap.

**The best single question:** *show me the last incident your current alerts did not catch early enough.*

**Channels:** LinkedIn, CNCF Slack, the Prometheus and Argo communities, DevOpsDays, KCD, Kubernetes/DevOps/.NET
meetups, MSPs, Rancher/OpenShift/AKS partners, direct outreach, a GitHub demo, a technical blog.

**Materials:** a two-minute demo, an architecture diagram, a sample incident report, a compatibility matrix, a
security overview, an air-gap guide, a competitive comparison, an ROI calculator, an alert-noise case study, a
resource-footprint sheet, a Helm quickstart, and a transparent limitations page.

**Sales demo.** A v2 rollout leaks memory only in the new pods. The threshold has not fired yet. Overfit
detects the trend, **scopes it to the new ReplicaSet**, groups memory, throttling and latency, and publishes
one incident with a timeline — while the old ReplicaSet stays visibly within range. Per §3, the punchline is
the scope and the single incident, not the timing.

---

## 25. Continue / stop criteria

**Continue if:** two teams want a pilot; they share incident data; the product finds a real problem; noise
drops measurably; air-gap matters to them; Prometheus-native is an argument they make themselves; someone
accepts a paid pilot; an MSP sees the OEM angle.

**Narrow or stop if:** everyone expects a full logs/metrics/traces platform; nobody wants a local component;
anomaly detection has no budget line; customers are content with Datadog/Grafana; configuration effort exceeds
the value; false positives stay high after tuning; there is no data to validate against; air-gap turns out to
be marginal; nobody pays for a pilot.

---

## 26. Revised 90-day plan

| Days | Work |
|---|---|
| **1–15** | interviews; obtain Prometheus snapshots; define 15 signals; two pilots; incident format |
| **16–25** | **offline replay harness + first false-positive count on real data** — the go/no-go gate |
| **26–45** | ingestion, resampling, MAD, peer outlier on the existing comparator, change point |
| **46–60** | label-derived topology, ReplicaSet timeline, scope, grouping, suppression |
| **61–75** | metrics endpoint, Alertmanager, CLI, dashboard, snapshot, Helm |
| **76–90** | pilot, tuning, false positives, detection delay, ROI, beta decision |

The change from the original plan is days 16–25: the riskiest assumption is tested on real data before the
detectors are productionised, and the Kubernetes watch layer that consumed days 36–55 is gone (§11).

---

## 27. Recommendation

Build **Overfit AIOps — a local, explainable Kubernetes anomaly-detection engine for existing Prometheus
installations.** It does not replace Prometheus or Grafana, stores no telemetry, detects deviation from
history, outlier pods and nodes, and slow trends, understands Kubernetes topology, correlates anomalies with
rollouts, groups alerts, runs without egress, ships as one small .NET binary, and later extends into canary
comparison and rollback verification.

**The single goal that matters:** on real data, surface a valuable incident with a precise scope and
materially less noise than the existing alerts — and explain why it was raised.

**Do M0 first.** Two weeks, two pilots, one offline replay. Everything after that is contingent on what those
numbers say.

---

## 28. Sources

- Datadog Watchdog Alerts — <https://docs.datadoghq.com/watchdog/alerts/>
- Datadog Watchdog — <https://docs.datadoghq.com/watchdog/>
- Grafana Machine Learning — <https://grafana.com/docs/grafana-cloud/machine-learning/machine-learning/>
- Grafana Outlier Detection — <https://grafana.com/docs/grafana-cloud/machine-learning/machine-learning/outlier-detection/>
- Grafana Outlier Alerting — <https://grafana.com/docs/grafana-cloud/machine-learning/machine-learning/outlier-detection/query-and-alerting/>
- Elastic Metric Anomalies — <https://www.elastic.co/docs/solutions/observability/infra-and-hosts/detect-metric-anomalies>
- Elastic Kubernetes Anomaly Jobs — <https://www.elastic.co/docs/reference/data-analysis/machine-learning/ootb-ml-jobs-metrics-ui>
- OpenSearch Anomaly Detection — <https://docs.opensearch.org/latest/observing-your-data/ad/index/>
- Robusta — <https://github.com/robusta-dev/robusta>
- Argo Rollouts Analysis — <https://argoproj.github.io/argo-rollouts/features/analysis/>
- Argo Rollouts Prometheus — <https://argoproj.github.io/argo-rollouts/analysis/prometheus/>
- Argo Rollouts Web Metrics — <https://argoproj.github.io/argo-rollouts/analysis/web/>
- Flagger Metrics Analysis — <https://docs.flagger.app/main/usage/metrics>
- Flagger Architecture — <https://docs.flagger.app/usage/how-it-works>

---

*Companion: [`aiops-canary-blueprint.md`](aiops-canary-blueprint.md) — the statistical engine, its decision
failure modes, and the measured ingestion cost. Implementation to date: `Sources/Main/Statistics/`
(`ITwoSampleComparer`, `MannWhitneyU`, `TwoSampleComparison`), 36 tests, benchmarked.*
