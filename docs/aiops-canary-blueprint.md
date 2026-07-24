# Overfit AIOps — Automated Canary Analysis (design & strategy blueprint)

> **Status:** internal design/strategy note, 2026-07-23 (rev. 2). Not linked from the README — it names a
> competing system (Kayenta) and carries market/strategy judgements that don't belong in launch-facing copy.
> This is a *candidate* product direction, not shipped work.
>
> **The load-bearing parts are §3 and §4.** §3 replaces the original vision's statistics (Z-score), which are
> the wrong tool. §4 lists decision-logic holes found on a second pass — including one that makes the engine
> **silently approve a bad deploy**, which is the most expensive failure an auto-rollback tool can have.
>
> **Superseded as a standalone product, retained as the engine.**
> [`aiops-cluster-anomaly-guard.md`](aiops-cluster-anomaly-guard.md) reframes this work as **Compare**, one
> mode of a broader Prometheus-native anomaly-detection product that leads with cluster **Watch**. That
> document owns the product strategy, the market read, the MVP and the go-to-market; **this one remains
> authoritative for the statistics and the decision logic** (§3, §4), which Watch's peer-group outlier detector
> reuses unchanged.

An **Edge AIOps** engine that runs 100% on-premise inside the customer's Kubernetes cluster and performs
**automated canary analysis** in real time — judging whether a new app version is stable versus the old one,
without any telemetry leaving the firewall. It is not a traffic manager; it is the **"brain"** (a webhook)
that controllers like **Argo Rollouts** consult at each step of a rollout.

---

## 1. Concept

At each rollout step (5% → 20% → 50% traffic), Argo Rollouts calls the engine. The engine compares the new
version (canary) against the old (baseline) and returns a verdict — continue, roll back, or hold. Nothing
leaves the box: it reads metrics from the in-cluster Prometheus and decides locally.

**Why it fits Overfit's DNA.** Pure .NET, single self-contained Native-AOT binary, zero data egress,
air-gapped-capable, no separate analysis service to deploy. It is a Prometheus *consumer*; Overfit's inference
server is a Prometheus *exporter* — same ecosystem, same identity, different product.

---

## 2. Architecture — the "two brains" (keep them separate)

Two cooperating but **independent** mechanisms. **Ship brain 2 first; brain 1 is a separate product.**

### Brain 2 — the Canary Sniper (relative A/B analysis) — the sellable core
Fires only during a rollout. Ignores long-term history.

- **Normalise on the fly.** Reduce both versions to a *unit cost* (e.g. CPU-seconds per HTTP request) so a
  version serving 5% of traffic is comparable to one serving 95%. (Caveat in §4.3 — this assumes traffic-mix
  parity, which is not free.)
- **No absolute thresholds** for the regression question; it asks whether the canary's normalised cost is
  worse than the baseline's under a live statistical test (§3). An absolute SLO floor still applies as a
  safety net (§4.1).
- **Event correlation.** A shared infrastructure hiccup should not be blamed on new code. The naive form of
  this rule is dangerous — see §4.1 for the corrected version.

### Brain 1 — the Global Guardian (long-term anomaly detection) — a *separate* SKU
Runs in the background on stable production (100% traffic).

- **Cold start (day 1).** Before it knows the daily cycle it leans on rate-of-change (derivative) heuristics —
  a sudden vertical RAM climb, an avalanche of 5xx. Frame this honestly: it is *threshold-on-derivative*, not
  "AI" (see the contradiction in §4.5).
- **Adaptation.** Over a 3–4 week sliding window it learns seasonality (morning peaks, nightly batch, weekend
  dips). Implement as univariate time-series anomaly detection: a seasonal-naive baseline (same hour last week)
  with **robust z on the MAD** (median absolute deviation — resistant to outliers, unlike mean/std), or STL
  decomposition + residual thresholding.

> **Recommendation:** do NOT bundle these into one product for v1. The Sniper has a crisp buyer and a crisp
> integration (Argo). The Guardian is a different, standalone anomaly product. Combining them dilutes focus.

---

## 3. The statistical engine — the corrected core (this is the moat)

The original vision specified **Z-score / standard deviation**. Wrong tool: Z-score assumes a normal
distribution, and latency / request-cost are heavy-tailed (log-normal). A naive Z-score on p99 latency fires
constantly. The credible engine has four layers:

### 3a. Test — Mann-Whitney U (Wilcoxon rank-sum), not Z-score
Rank all requests from baseline and canary together, sum the ranks, compute U. It tests "is the canary
stochastically worse than the baseline" **without assuming any distribution** — exactly right for heavy-tailed
latency. This is what Netflix's Kayenta uses.

### 3b. Gate on effect size, not just the p-value
With large N, a microscopic difference becomes "statistically significant" (p < 0.05) and you get rollbacks
from nothing. Require an **effect size** threshold too — Cliff's delta, or the common-language effect size
("probability a random canary request is worse than a random baseline request"). Rule: roll back only when
**significant AND effect > threshold**.

### 3c. Control-vs-experiment design
Do not compare the canary to fixed history. Compare it to a **baseline running simultaneously under the same
conditions**. Infrastructure hiccups hit both, so the relative comparison cancels them.

### 3d. Sequential testing (the honest answer to "too few samples at 5% traffic")
At 5% traffic, samples arrive over time. Naively re-running the test every minute is p-hacking — repeated
peeking inflates the false-positive rate. Use a **sequential test** (always-valid p-values / mixture SPRT) so
the engine decides the moment there is enough evidence, without corrupting significance by peeking.

**Correction — mixture SPRT is not MVP, and does not need to be.** There is no credible .NET implementation of
non-parametric always-valid p-values, and writing one correctly is a quarter of work for a problem that can be
designed away. The peeking problem only exists if you *evaluate in a loop*, so remove the loop:

| Layer | Fires | Statistics | Why |
|---|---|---|---|
| **Verdict** | once, at the end of the analysis window | full Mann-Whitney + effect size | one look, so α means what it says |
| **Circuit breaker** | continuously | none — a hard threshold | catastrophes deserve thresholds, not tests |

The circuit breaker is what buys back early abort: error rate above an absolute ceiling, or latency past the
SLO floor of §4.1, kills the rollout immediately without ever consulting the statistic. This is the correct
division — a test answers "is this difference real?", which is the wrong question when the service is on fire.

Note that Argo `AnalysisRun` evaluates **per interval** by default and any interval can fail the rollout, so
single-look has to be configured deliberately (`count: 1` at the end of the window), not assumed.

If several looks are genuinely wanted, the cheap middle is **Bonferroni over a fixed, small number of peeks**
(α/k, k known in advance) — one line of code, costing a little power. Reach for mSPRT only if k must be
unbounded, and only after the product exists.

### 3e. Multiple comparisons
Testing CPU, latency, errors, memory each at 5% gives ≈19% chance of an unlucky rollback across four metrics.
Correct for it (Bonferroni / Benjamini-Hochberg) or fold the metrics into **one aggregate score**.

### 3f. Ingestion shape and cost — measured, not argued
**Status: implemented.** `Sources/Main/Statistics/MannWhitneyU.cs`, 31 tests, benchmarked on a Ryzen 9 9950X3D.

A standing objection to rank-sum in production is that ranking means **sorting**, and sorting a large canary
window every evaluation is O(N log N) on the analyser's critical path. The objection survives only until it is
measured, and measurement redirects it somewhere more useful.

**The CPU claim is false in the terms it is usually made.** One comparison over 10 000 observations per arm
costs 173 µs. Ten metrics every 30 s is 1.7 ms per 30 000 ms — **0.006% of one core**. There is no crisis here.

**The memory claim is true, and was never stated.** At 10 000 observations per arm the combined array is 160 KB,
which crosses the 85 KB large-object-heap threshold: the first implementation triggered ~50 Gen2 collections per
1 000 evaluations. In a long-lived in-cluster analyser *that* is what would be felt, not the CPU. The fix is
pooled scratch, and it removes the allocation entirely.

**The right conclusion is about ingestion, not speed.** Raw per-request vectors usually cannot be obtained at
all — Prometheus already aggregates into buckets, and pulling per-request values out of a customer's cluster is
a data-volume and privacy problem. So the statistic must accept bucketed input. It does, in O(K), and the
implementation is proven bit-identical to the raw path on the same data.

| Path | 500/arm | 2 000/arm | 10 000/arm | 100 000/arm | Allocated |
|---|--:|--:|--:|--:|--:|
| Original (combined sort + `bool[]` payload) | 5.32 µs | 22.7 µs | 838 µs | 10.26 ms | 9 KB → 1.8 MB, **LOH** |
| Split sorts + merge, pooled scratch | 4.77 µs | 25.1 µs | **173 µs** | **2.32 ms** | **0 B** |
| Pre-sorted input (merge only) | 1.45 µs | 5.48 µs | 38.9 µs | 881 µs | **0 B** |
| **Bucketed histograms, O(K)** | **38.6 ns** | **38.7 ns** | **38.6 ns** | **43.1 ns** | **0 B** |

Four consequences for the product:

- **Bucketed input is flat in the window size** — 24 buckets cost ~39 ns whether they summarise 500 requests or
  100 000. Design the ingestion for histograms first; raw values are the optional high-resolution mode.
- **The price of buckets is resolution, not speed.** A regression smaller than a bucket boundary is invisible:
  with 25/50/100 ms boundaries, 40 → 46 ms moves nothing. The engine then reports "no difference" truthfully but
  uselessly, so bucket width is a product decision, not a detail.
- **Earth Mover's Distance is not a substitute.** It is a *distance*, not a *test* — it yields no p-value, which
  puts a hand-picked threshold back in the decision path, i.e. exactly the failure §3 exists to remove.
  Mann-Whitney handles bucketed data natively: a bucket is one tie group, which the mid-rank machinery already
  models.
- **Parallelise across metrics, not inside one comparison.** A whole 16-metric evaluation measured **10.9×
  faster** spread over cores (3.73 ms → 0.34 ms) with no API change. Splitting *within* one comparison would
  force `Memory<T>` through the API for lambda capture, double the scratch, and win perhaps 1.5×.

**Two negative results, kept because they cost time to learn:**

- *"Two half-size sorts without a payload will beat one full-size paired sort."* Measured **~5%** — a tie, not
  the multiple the reasoning predicted. `Array.Sort` with a `bool[]` payload was simply not the bottleneck; the
  comparison sort itself was. The real win came from replacing the sort algorithm (an LSD radix over the
  doubles' order-preserving bit encoding, **3.9×** above the crossover), not from restructuring around it.
- *A genuine regression at ~2 000 observations per arm: **1.11× slower**.* In that band the split-sort advantage
  has evaporated but radix has not yet paid off, and the merge pass is intrinsically a little dearer than the
  original single rank-sum walk. It is bought deliberately, in exchange for zero allocation at every size and
  4.4–4.9× above 10 000. Recorded rather than hidden — canary windows sit above this band, but a caller living
  inside it should know.

---

## 4. Decision-logic failure modes (found on review) — and their fixes

§3 fixes the *test*. These fix the *decisions around it*. The first one is the most important item in this
whole document.

### 4.1 The correlation rule can silently APPROVE a bad deploy — worst failure mode
**The hole.** As originally stated: "if latency rises in the canary but rises identically in the baseline, it
is infrastructure — let the rollout continue." But a bad deploy can **degrade its neighbour**. A canary that
hogs CPU, memory bandwidth or a connection pool starves baseline pods on the same node, so *both* series rise
together — and the rule classifies a genuine regression as an infrastructure hiccup and **passes it**. In an
auto-rollback tool a false negative is far more expensive than a false alarm.

**The fix — three parts:**
1. **Judge the relative difference, not co-movement.** The question is never "did both move?" but "did the
   canary degrade *more* than the baseline?" Compare the *distributions* (§3a) at every step; shared movement
   cancels naturally in a control-vs-experiment design without needing a special "it's infra" branch.
2. **Absolute SLO floor as a safety net.** If both versions breach an absolute error/latency objective, that is
   an outage — fail (or at minimum `Inconclusive`) regardless of how well-correlated they are. Never let
   correlation override a hard SLO breach.
3. **Topology awareness.** Note whether canary and baseline pods share a node / zone. Co-located degradation is
   evidence *for* neighbour interference (i.e. the canary's fault), not against it.

### 4.2 The test has no notion of sample size
**The hole.** "Compute σ for V1 and check whether V2 falls inside a Z-score margin" conflates two different
questions: *is a single canary request unusual against the baseline's spread* (a prediction interval) versus
*is the canary's distribution different from the baseline's* (a hypothesis test). Because the yardstick is the
per-sample σ, **collecting more canary data never makes the test more sensitive** — statistical power is
simply absent from the design.

**The fix.** A two-sample test whose power grows with N (§3a), combined with sequential evaluation (§3d) so
accumulating evidence actually changes the verdict.

### 4.3 Unit-cost normalisation assumes traffic-mix parity
**The hole.** CPU-per-request is only comparable if the canary receives the *same mix* of requests. If the 5%
slice differs in endpoint distribution (health checks, one region, header-based routing, a hot tenant), the
normalisation compares different populations and reports a regression that does not exist — or hides one.

**The fix.**
- Require a **random traffic split**, not header/identity-based routing, for the analysis to be valid; detect
  and refuse (`Inconclusive`) when the split is not random.
- **Stratify per endpoint / route** and test within strata, then aggregate — this survives mix skew.
- Track a mix-similarity check (e.g. distribution distance over endpoint labels) and surface it as a
  precondition, not an assumption.

**The fix has its own hole: cardinality.** Stratifying every metric by HTTP route is exactly what causes
high-cardinality blow-ups in Prometheus — enough shops disable route labels entirely that "just stratify" can
be un-implementable at the customer, and where it is implementable it can cost them more RAM than the feature
is worth. Constraints that follow:

- **Hard-cap the strata in configuration.** Stratify the top *N* routes by traffic share (N ≈ 5, configurable)
  and fold everything else into a single `Other` aggregate. Uncapped stratification is not an option we offer.
- **Watch `Other` for its own mix shift.** A catch-all bucket whose composition changes between baseline and
  canary re-creates precisely the bias stratification was meant to remove — so `Other` gets the same
  mix-similarity check, and failing it degrades the verdict to `Inconclusive`.
- **If route labels are absent, say so.** Stratification is then impossible and unit-cost comparison rests on
  an unverifiable assumption. That is a documented limitation reported in the verdict, not something to paper
  over.

### 4.4 A binary verdict discards "not enough evidence" — and hides a dead canary
**The hole.** "Only binary answers (200 / 406)" is presented as simplicity, but it forces a decision when the
evidence is insufficient — producing either a premature rollback or a premature pass. Worse: if the canary
receives **zero traffic** (misrouted, failing readiness), there are no samples at all, and a binary contract
makes silence look like success.

**The fix.** Use Argo's full contract: `Successful` / `Failed` / **`Inconclusive`** (hold and keep collecting —
pairs with §3d) / `Error`. Treat "no canary samples" as an explicit `Inconclusive` (or `Error`), never a pass.

### 4.5 Internal contradiction: "no hard thresholds" versus a threshold-based cold start
**The hole.** Brain 2 advertises the absence of fixed limits, while Brain 1's day-one behaviour is *entirely*
hard thresholds on derivatives. And derivative thresholds are precisely the mechanism that produces the
**alert fatigue the GTM promises to eliminate** — the product undercuts its own headline benefit.

**The fix.** Be explicit that cold start is a distinct, clearly-labelled *safety mode* with different (and
weaker) guarantees, not the same engine. Bound its blast radius: only page on catastrophic, unambiguous
signals (crash-looping, sustained 5xx above an absolute floor), never on mild perf drift.

### 4.6 σ-based risk profiles promise a trade-off they cannot deliver
**The hole.** "Paranoid = 1.5σ" implies higher sensitivity. On a non-normal distribution with few samples,
tightening a σ multiplier mostly manufactures **false positives** rather than catching real regressions — and
σ itself is not meaningful on log-normal data.

**The fix.** A profile must set **three knobs jointly**: the *effect-size threshold* (how much worse counts),
the *confidence level* (how sure before acting), and the *minimum sample count* (how much evidence is
required). Then "paranoid" corresponds to a real, explainable trade-off.

| Profile | Effect threshold | Confidence | Min samples | Use |
|---|---|---|---|---|
| **Paranoid** | small | high | high | FinTech, payments |
| **Balanced** | moderate | standard | moderate | E-commerce, B2B |
| **Loose** | large | standard | low | Internal tooling |

### 4.7 "Zero-allocation" collides with JSON ingestion — and buys nothing
**The hole.** Parsing Prometheus JSON inherently allocates strings; claiming zero-allocation while ingesting
JSON invites a technical buyer to poke at it. And the data volume is **kilobytes**, so there is nothing to win.

**The fix.** Drop the claim for this product (keep it where it is true — the inference engine). If low-level
efficiency ever matters here, use OpenMetrics text with a span-based parser, but do not lead with it.

---

## 5. Integration — Argo Rollouts `AnalysisTemplate`

The engine is exposed as a webhook metric provider. Argo Rollouts calls it each step and acts on the verdict:

- `Successful` → proceed to the next traffic step.
- `Failed` → abort and roll back.
- `Inconclusive` → hold; keep collecting (see §3d, §4.4).
- `Error` → surfaced, never silently passes.

No separate control plane, no Spinnaker — one webhook in the cluster.

---

## 6. Configuration — opinionated, three risk profiles

Eliminates the "hell of sliders": one choice from three tolerance profiles, each setting the three knobs in
§4.6 (effect size, confidence, minimum samples) rather than a meaningless σ multiplier.

---

## 7. What to cut — over-engineering

Pinned/unmanaged memory (POH) for Prometheus vectors. **Canary analysis pulls kilobytes, not gigabytes** — a
handful of metrics × a few hundred points. This is LLM-scale memory discipline applied where the data fits in
L2. Keep SIMD if it helps, but do not sell "unmanaged heap" for a workload that fits in cache.

---

## 8. Competition & positioning

- **Kayenta** (Netflix/Spinnaker) — the reference, but heavy (needs Spinnaker or a standalone deployment, JVM).
- **Flagger** — built-in canary analysis, but threshold-based, no statistical rigour.
- **Argo Rollouts** metric providers (Prometheus, Datadog…) — thresholds/templates, **no statistical testing**.

**The gap:** statistically-rigorous canary analysis, air-gapped, **single native binary**, plugged into Argo as
a webhook. The moat is the statistics (§3) plus deployment simplicity. Pitch: **"Kayenta without the Spinnaker
footprint — air-gapped, zero-egress, one binary."**

---

## 9. Go-to-market

1. **DevOps Multiplier (OEM licensing) — strongest.** Annual, unlimited licence to software houses / MSPs. Kills
   alert fatigue, lets the same team run twice as many client clusters. Recurring revenue, clear ROI.
2. **Air-gapped Kubernetes vault.** CRD images for secrecy-bound institutions (medical, defence, large law
   firms). High CapEx for a tool that *physically cannot* leak logs. Real market, long cycles.
3. **"Hitman" premium audits.** Plug into a troubled cluster for a week, surface low-level anomalies, invoice
   once. This is **consulting, not product** — a cash-flow bridge, dangerous as a strategy.

> **⚠ Contradiction to resolve (§4.5 sibling).** The Guardian needs a **3–4 week** window to learn seasonality,
> but the Hitman audit runs for **one week** — so the audit operates entirely in cold-start mode, on the crude
> derivative heuristics, while being sold as adaptive analysis. Either base the audit product on the Sniper /
> ad-hoc comparative analysis (which needs no history), or extend the engagement past the learning window.

---

## 10. MVP — the thinnest thing that proves it

1. An Argo `AnalysisTemplate` webhook (with `Inconclusive`, §4.4).
2. **Mann-Whitney U on one metric** (CPU cost per request), control-vs-experiment.
3. Run it against a **recorded, real canary** from one actual deployment.

If it correctly separates a bad deploy from a DB hiccup on real data, there is a product. No two brains, no
three profiles, no POH — those are superstructure.

**Step 2 already exists.** `MannWhitneyU` (in `Sources/Main/Statistics/`) is written, tested and benchmarked —
mid-ranked ties, tie-corrected variance, continuity correction, one-sided p-value and both effect sizes, over
the four ingestion shapes of §3f, at zero allocation. What the MVP still needs is everything around it: the
webhook, the ingestion adapters, the decision policy of §4, and a recorded canary to run it on. The statistic
was the part with a right answer; the rest is where the product is.

---

## 11. Adjacent use cases worth examining

The core asset is not "canary analysis" — it is a **rigorous two-population comparator over on-prem metrics**.
That engine answers "did this change make things worse, beyond noise?" for far more than rollouts. Ranked by
fit with this project:

### 11.1 Model / inference rollout A/B — the one that resolves the focus problem ★
Comparing two **model versions** in production (latency, cost-per-token, error rate, and — with a judge — output
quality). This is where the AIOps product and Overfit's inference engine genuinely converge: Overfit *serves*
models and *validates model swaps*, one story instead of two companies. It also has no incumbent: Kayenta and
Flagger analyse services, not model rollouts. **Highest strategic value; reuses everything.**

**Scope correction — cut the judge, keep quality.** "With a judge" is scope creep dressed as a feature:
LLM-as-judge adds asynchrony, API cost, and a grader that hallucinates, inside a component whose whole job is
to be the trustworthy party. It does not belong in an MVP that decides deploys.

But cutting quality *entirely* and shipping only hard physics (TTFT, tokens/s, peak RSS — note **RSS, not
VRAM**: Overfit's public identity is CPU) walks straight into §4.1, the worst failure mode in this document.
A Q2 quantisation is **faster and dumber**. An infrastructure-only gate does not merely miss that — it
*approves* it, with a green latency chart as justification. A gate that rubber-stamps quality regressions is
worse than no gate.

The way out is a quality signal that costs nothing and judges nothing, which we can build only because we own
the engine end to end:

- a **fixed golden prompt set**, versioned with the analysis config;
- **deterministic decoding** (greedy, fixed seed), so the run is reproducible;
- per-prompt **log-probability / perplexity on held-out continuations** as the metric.

That yields a *distribution of numbers per model version* — which feeds the **same Mann-Whitney comparator**
as latency does, with the same effect-size gate. No judge, no external API, no asynchrony, no non-determinism.
The MVP metric set is therefore TTFT, tokens/s, peak RSS **and** golden-set log-prob. Semantic judging can wait
indefinitely; it is a later SKU, not a prerequisite.

### 11.2 Cost / FinOps regression per unit of work ★
The unit-cost normalisation (§2) *is* a cost metric. Flip the framing: "this release made every request 12%
more expensive." Detecting cost regressions at deploy time is a distinct, well-funded buyer (FinOps/platform)
and needs no new engine — only a different metric and report. Strong standalone value, minimal extra work.

### 11.3 Rollback verification — closes a loop nobody closes
After an automatic rollback, verify the system **actually recovered**. Today tools roll back and assume success;
if the regression came from a dependency rather than the deploy, the rollback fixes nothing and the team
chases ghosts. Same comparator, run post-hoc. Cheap, and a genuine differentiator in demos.

### 11.4 Change validation beyond deploys
Same question, different trigger: **feature flags**, ConfigMap/env changes, HPA/resource-limit edits, node-pool
or Kubernetes version upgrades, runtime (.NET/JVM) bumps, database version changes. Anything where a "before"
and "after" (or A and B) population exists. Broadens the product without broadening the engine.

### 11.5 Pre-production performance gate in CI
Run the same comparison against load-test results in staging as a **merge gate** ("this PR regressed p95 cost
per request beyond noise"). Larger market, earlier in the pipeline — but a crowded space (benchmark tooling),
and it loses the air-gapped/on-prem moat, since CI regression tooling is not privacy-sensitive. Evaluate, don't
assume.

### 11.6 Noisy-neighbour / multi-tenant fairness
Reuse the comparator across tenants or namespaces to detect one tenant degrading another. Interesting, but it
needs the topology awareness of §4.1 to be trustworthy — treat as a follow-on, not a v1 claim.

**Reading of the list:** 11.1 and 11.2 are the two worth pursuing. 11.1 because it makes the strategic fork in
§12 disappear; 11.2 because it monetises the engine already being built for a buyer with budget.

---

## 12. The real question — strategic focus (not technical)

As framed, this is a **different product, market and buyer** than the inference engine. Overfit is "pure .NET
CPU LLM/DL inference". A canary analyser is a **DevOps/SRE** tool — different buyer (platform teams), different
sales motion, different marketing. It shares Overfit's DNA (.NET, zero-alloc, air-gapped) but it is a
**strategic fork**: is Overfit an inference engine, or an AIOps company?

For a small/solo project, doing both dilutes focus. **§11.1 is the escape hatch:** aim the same engine at
*model* rollouts and the two products become one narrative — "Overfit serves models on-prem and proves a model
change is safe before it reaches everyone." That keeps the statistical work and drops the identity conflict.

**Verdict:** architecture sound; §3 must replace Z-score with Mann-Whitney + effect size + control-vs-experiment
+ sequential testing; §4 must fix the correlation false-negative (4.1), the missing notion of sample size (4.2)
and the traffic-mix assumption (4.3) before this can be trusted to roll anything back automatically. Ship the
Sniper alone. The market is narrow but real, and "air-gapped Kayenta-lite" is defensible — but the decisive
question is focus, and §11.1 is the cheapest way to answer it.

---

*Related: `docs/gp-anomaly-baseline.md` (a different, model-based anomaly track). The two share the "anomaly"
word but nothing else — this one is a statistical canary analyser, that one is ML.*
