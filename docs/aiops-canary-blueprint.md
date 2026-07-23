# Overfit AIOps — Automated Canary Analysis (design & strategy blueprint)

> **Status:** internal design/strategy note, 2026-07-23. Not linked from the README — it names a competing
> system (Kayenta) and carries market/strategy judgements that don't belong in launch-facing copy. This is a
> *candidate* product direction, not shipped work. The engineering corrections in §3 are the load-bearing part:
> the original vision's statistics (Z-score) must be replaced before any of this is credible.

An **Edge AIOps** engine that runs 100% on-premise inside the customer's Kubernetes cluster and performs
**automated canary analysis** in real time — judging whether a new app version is stable versus the old one,
without any telemetry leaving the firewall. It is not a traffic manager; it is the **"brain"** (a webhook)
that controllers like **Argo Rollouts** consult at each step of a rollout.

---

## 1. Concept

At each rollout step (5% → 20% → 50% traffic), Argo Rollouts calls the engine. The engine compares the new
version (canary) against the old (baseline) and returns a binary verdict — continue or roll back. Nothing
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
  version serving 5% of traffic is comparable to one serving 95%.
- **No absolute thresholds.** It never judges raw values; it asks whether the canary's normalised cost is
  worse than the baseline's, under a live statistical test (see §3).
- **Event correlation.** If latency rises in the canary **but rises identically in the baseline** (a shared
  DB hiccup), it is classified as an *infrastructure* problem, not new-code regression — and the rollout is
  allowed to continue. This is the single most important behaviour: it is what makes automated rollback
  usable instead of a false-alarm generator.

### Brain 1 — the Global Guardian (long-term anomaly detection) — a *separate* SKU
Runs in the background on stable production (100% traffic).

- **Cold start (day 1).** Before it knows the daily cycle, it leans on hard "physics of IT" heuristics —
  a sudden vertical RAM climb, an avalanche of 5xx — via rate-of-change (derivative) analysis. Frame this
  honestly: it is *threshold-on-derivative*, not "AI".
- **Adaptation.** Over a 3–4 week sliding window it learns seasonality (morning peaks, nightly batch, weekend
  dips) to suppress false alarms. Implement as univariate time-series anomaly detection: a seasonal-naive
  baseline (same hour last week) with **robust z on the MAD** (median absolute deviation — resistant to
  outliers, unlike mean/std), or STL decomposition + residual thresholding.

> **Recommendation:** do NOT bundle these into one product for v1. The Sniper (brain 2) has a crisp buyer and
> a crisp integration (Argo). The Guardian (brain 1) is a different, standalone anomaly product. Combining
> them dilutes focus.

---

## 3. The statistical engine — the corrected core (this is the moat)

The original vision specified **Z-score / standard deviation**. That is the wrong tool and must be replaced —
Z-score assumes a normal distribution, and latency / request-cost are heavy-tailed (log-normal). A naive
Z-score on p99 latency fires constantly. The credible engine has four layers:

### 3a. Test — Mann-Whitney U (Wilcoxon rank-sum), not Z-score
Rank all requests from baseline and canary together, sum the ranks, compute U. It tests "is the canary
stochastically worse than the baseline" **without assuming any distribution** — exactly right for heavy-tailed
latency. This is what Netflix's Kayenta uses.

### 3b. Gate on effect size, not just the p-value
With large N, a microscopic difference becomes "statistically significant" (p < 0.05) and you get rollbacks
from nothing. Require an **effect size** threshold too — Cliff's delta, or the common-language effect size
("probability a random canary request is worse than a random baseline request"). Rule: roll back only when
**significant AND effect > threshold**. This is what tames false alarms under high traffic.

### 3c. Control-vs-experiment design (the correlation insight, formalised)
Do not compare the canary to fixed history. Compare it to a **baseline running simultaneously under the same
conditions**. Infrastructure hiccups hit both, so the relative comparison cancels them. This is §2's
"event correlation" made rigorous — and it is why it works where absolute thresholds don't.

### 3d. Sequential testing (the honest answer to "too few samples at 5% traffic")
At 5% traffic, samples arrive over time. Naively re-running the test every minute is p-hacking — repeated
peeking inflates the false-positive rate. Use a **sequential test** (always-valid p-values / mixture SPRT) so
the engine decides the moment there is enough evidence, without corrupting significance by peeking.

### 3e. Multiple comparisons
Testing CPU, latency, errors, memory each at 5% gives ≈19% chance of an unlucky rollback across four metrics.
Correct for it (Bonferroni / Benjamini-Hochberg) or fold the metrics into **one aggregate score** (Kayenta's
approach).

---

## 4. Integration — Argo Rollouts `AnalysisTemplate`

The engine is exposed as a webhook metric provider. Argo Rollouts' `AnalysisTemplate` calls it each step and
acts on the verdict:

- `HTTP 200 OK` / `Successful` → proceed to the next traffic step.
- `HTTP 406 Not Acceptable` / `Failed` → abort and roll back.
- `Inconclusive` → hold (not enough evidence yet — pairs with the sequential test in §3d).
- `Error` → surfaced, does not silently pass.

No separate control plane, no Spinnaker — one webhook in the cluster.

---

## 5. Configuration — opinionated, three risk profiles

Eliminates the "hell of sliders": one choice from three tolerance profiles. **Reframe the profiles in terms of
effect size + power, not "σ"** — a σ multiplier on log-normal data does not mean what the label implies.

| Profile | Intent | Use |
|---|---|---|
| **Paranoid** | Catch the smallest regression (thread-pool starvation, µs stalls) | FinTech, payments |
| **Balanced** | Tolerate K8s network noise, react hard to leaks / EF problems | E-commerce, B2B |
| **Loose** | Ignore mild perf regressions, intervene only on hard failure | Internal tooling |

---

## 6. What to cut — over-engineering

The vision proposes pinned/unmanaged memory (POH) for Prometheus vectors. **Canary analysis pulls kilobytes,
not gigabytes** — a handful of metrics × a few hundred points. LOH pressure is a non-problem here; this is
LLM-scale memory discipline applied where the data fits in L2. Keep SIMD if it helps, but do not sell
"unmanaged heap" for a workload that fits in cache — a technical buyer will see through it.

---

## 7. Competition & positioning

- **Kayenta** (Netflix/Spinnaker) — the reference, but heavy (needs Spinnaker or a standalone deployment, JVM).
- **Flagger** — built-in canary analysis, but threshold-based, no statistical rigour.
- **Argo Rollouts** metric providers (Prometheus, Datadog…) — thresholds/templates, **no statistical testing**.

**The gap:** statistically-rigorous canary analysis, air-gapped, **single native binary**, plugged into Argo as
a webhook. The moat is the statistics (Mann-Whitney + effect size + sequential) **plus** deployment simplicity.
Pitch: **"Kayenta without the Spinnaker footprint — air-gapped, zero-egress, one binary."**

---

## 8. Go-to-market

Ranked by strength:

1. **DevOps Multiplier (OEM licensing) — strongest.** Annual, unlimited licence to software houses / MSPs. It
   becomes their internal cost-cutter: kills alert fatigue, lets the same team run twice as many client
   clusters. Recurring revenue, clear ROI.
2. **Air-gapped Kubernetes vault.** CRD images sold to secrecy-bound institutions (medical, defence, large law
   firms). High CapEx sale for an AI tool that *physically cannot* leak logs to a public cloud. Real market,
   but long cycles and heavy compliance.
3. **"Hitman" premium audits.** Plug the engine into a troubled cluster for a week, surface low-level anomalies
   (async misuse, retained-object leaks), invoice once. This is **consulting, not product** — a cash-flow
   bridge, dangerous as a strategy (sells your time, doesn't scale).

Aligns with the AGPL-open / commercial-licence moat: an autonomous black box supports asymmetric sale.

---

## 9. MVP — the thinnest thing that proves it

1. An Argo `AnalysisTemplate` webhook.
2. **Mann-Whitney U on one metric** (CPU cost per request), control-vs-experiment.
3. Run it against a **recorded, real canary** from one actual deployment.

If it correctly separates a bad deploy from a DB hiccup on real data, there is a product. No two brains, no
three profiles, no POH — those are the superstructure.

---

## 10. The real question — strategic focus (not technical)

This is a **different product, market and buyer** than the inference engine. Overfit is "pure .NET CPU LLM/DL
inference". A canary analyser is a **DevOps/SRE** tool — a different buyer (platform teams, not ML/app devs), a
different sales motion, different marketing. It shares Overfit's DNA (.NET, zero-alloc, air-gapped) but it is a
**strategic fork**: is Overfit an inference engine, or an AIOps company?

For a small/solo project, doing both dilutes focus. This is not "don't do it" — it is "decide deliberately".
The blueprint is **shippable faster than the ML anomaly track** (a statistical test, not a transformer), so as
a *fast, standalone* product it defends itself. But treated as part of Overfit, it risks leaving neither
product with enough attention.

**Verdict:** architecture sound; the statistical core must swap Z-score → Mann-Whitney + effect size +
control-vs-experiment + sequential testing (without which it is a toy). Ship the Sniper alone, cut the Guardian
and POH. Narrow market, but the gap is real and the "air-gapped Kayenta-lite" position is defensible. The
decisive question is focus, not code.

---

*Related: `docs/gp-anomaly-baseline.md` (a different, model-based anomaly track). The two share the "anomaly"
word but nothing else — this one is a statistical canary analyser, that one is ML.*
