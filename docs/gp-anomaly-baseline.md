# Sketch — Gaussian-Process baseline for metric anomaly detection

**Status:** design sketch / proposed experiment. Not implemented. Separate sprint.

**Origin:** MASCOTS'16, *"Data Modelling with Gaussian Process in Sensor Networks
for Urban Environmental Monitoring"* (Xiuming Liu, Teng Xi, Edith Ngai, Uppsala).
Urban sensor metrics are **correlated across sensors** and show **periodic patterns
driven by human activity** — exactly the structure of K8s service metrics (CPU,
RPS, latency are correlated; load is diurnal). The paper models them with a
multidimensional-output GP, designs mean/kernel functions, and approximates sample
covariances via the Wiener–Khinchin theorem.

## Why bother (the point of a baseline)

Overfit's `GptAnomalyDetector` scores a snapshot by its next-token surprise. We
*claim* a transformer is the right tool for this. We have **not** compared it to a
classical, well-understood baseline on the same data. A GP baseline lets us state,
with rigor: "the GPT detector beats a tuned GP by X on the same metrics / injected
anomalies" — or discover it doesn't, and learn why. Same discipline as the
LLamaSharp A/B and the VNNI null-result: **measure, don't assume.**

## What a GP baseline looks like here

Per metric (or jointly, multi-output), fit a GP on a sliding window of history and
score each new point by its **negative log predictive density** (how surprising it
is under the GP posterior) — directly comparable to the GPT detector's per-snapshot
surprise score.

- **Mean:** zero or a slow EWMA trend.
- **Kernel:** `Periodic(period≈1 day) × RBF(short)` + `RBF(short)` noise — captures
  the diurnal cycle (the paper's key observation) plus local smoothness. Multi-output
  / coregionalisation kernel to share strength across correlated metrics (their
  "dependent GP beats independent GP" result).
- **Covariance shortcut:** Wiener–Khinchin sample-covariance approximation (paper's
  approach 2) avoids re-estimating hyper-parameters every window.

## Comparison protocol (the experiment)

1. Same synthetic K8s CSV + the same injected-anomaly snapshots used by
   `GptAnomalyLoRAIntegrationTests` / the console demo.
2. Score the normal stream + injected incident with **both** the GP and the GPT
   detector.
3. Report: separation (normal vs anomaly score), ROC/AUC over a labelled stream,
   per-metric attribution accuracy (does each flag the right worst metric?),
   and cost (fit/score latency, memory).
4. Honest verdict in the doc + a line on the landing only if the GPT detector wins.

## Effort & risk

- **Medium.** A usable GP needs a kernel-matrix solve (Cholesky on an `n×n`
  window covariance, `O(n³)` in window size — fine for small windows). Pure C#,
  no new deps; fits the zero-native-dep identity. ~1–2 days for a windowed
  single/multi-output GP + the comparison harness.
- **Cheaper first step:** a z-score / EWMA-residual baseline (`O(n)`) is a trivial
  sanity floor — if the GPT detector doesn't beat *that*, the GP is moot. Do the
  cheap baseline first; promote to GP only if the story needs the stronger one.

## Decision

Deferred — it's a credibility/rigor experiment, not a product feature. Pick it up
when we want a defensible "transformer vs classical" claim for the blog/launch.
Lead with the EWMA/z-score floor (hours), escalate to the GP (days) only if warranted.

## Results — EWMA floor implemented + benchmarked (2026-05-22)

`EwmaAnomalyDetector` (per-metric EWMA mean/variance, RiskMetrics form; score = mean
`½·z²`; same `AnomalyScore` shape, named worst-metric) + fast unit tests
(`EwmaAnomalyDetectorTests`) + a head-to-head `[LongFact]`
(`GptVsEwmaBaselineComparisonTests`): train a Quick GPT base on the fixture CSV, score
one pod's real stream + 3 injected anomalies (OOM / latency / CPU) with both detectors.

**Verdict (honest, mildly unflattering — exactly the point):** on the **un-adapted**
Quick base the EWMA floor **out-separates** the GPT detector. Representative run:

```
detector  mean-normal   OOM                    LATENCY            CPU
GPT       ~2.7          4.5 [oom_events_rate]   3.1 [latency_p99]  4.8 [cpu_throttle]   ~1.6× sep
EWMA      0.66          5.65 [memory]           2.85 [latency_p99] 5.45 [cpu_usage]     ~8×   sep
```

The cross-pod GPT base carries a high *normal* floor (~2.7, the same residual surprise
that read 5.68 on the 256d production base), so its per-anomaly margins are thin; the
EWMA's per-pod adaptive baseline sits near zero, giving far stronger separation. GPT's
edge shows in attribution (it nailed `oom_events_rate`; EWMA flagged the correlated
`memory`). **Conclusion:** the GPT detector's real advantage is **per-pod LoRA
adaptation** (which drives normal → ~0 and separation → thousands×, see
`GptAnomalyLoRA*` tests), NOT the raw base — against the EWMA floor the un-adapted base
barely competes. **GP escalation is therefore NOT warranted now**: the LoRA-adapted GPT
already beats the floor decisively; a GP would only matter if we needed to beat EWMA
*without* adaptation, which isn't the product. EWMA stays as the cheap deployable
sanity floor + the honest comparison baseline.
