# `Statistics` — rank tests, trends and baselines

The distribution-free statistics the anomaly subsystem runs on. Kept out of `Anomalies` because none
of it knows anything about Kubernetes: it compares samples and fits slopes, and would work as well on
any two series.

## Why rank statistics and not means

Metric series from a live cluster are not normal, not stationary, and full of one-sample spikes from
scrape timing. A mean-and-standard-deviation test on that reports whatever the last GC pause did.
`MannWhitneyU` compares two samples by rank alone, and `TwoSampleComparison` reports Cliff's delta
beside the p-value because significance and size are different questions — with 80 samples per pod
almost any difference is significant, and almost none of them matter.

## The parts

| Type | Role |
|---|---|
| `PeerGroupOutlierDetector` | One replica against the rest of its cohort, all pods in one pass. |
| `MannWhitneyComparer` / `MannWhitneyU` | The rank test itself, with a normal approximation for the tail. |
| `TrendDetector` | Theil-Sen slope + Mann-Kendall tau, with AR(1) variance inflation and time-to-limit projection. |
| `SeasonalBaseline` | Expectation from previous periods, so a daily curve is not read as a trend. |
| `CrossPeerBaseline` | The cohort's common component, so a fault shared by everyone does not cancel out per pod. |
| `MedianSelector`, `SpearmanCorrelation`, `NormalDistribution` | Supporting arithmetic. |

## Three results that were surprising enough to write down

**Cliff's delta is scale-free, and that is a problem as often as a feature.** A 3% spread across
healthy replicas produced deltas of 0.52 and 0.68 against a 0.33 threshold: statistically emphatic,
operationally nothing. The relative and absolute gates in `AnomalyGuardOptions` exist because the
effect size alone cannot tell you whether a difference is worth reporting.

**A starved peer must be excluded, not fatal.** One replica with too few usable samples used to abort
the comparison for the entire cohort — so the guard went quiet exactly when a pod was in trouble
enough to stop reporting. Under-sampled peers are now dropped from the comparison and recorded with
`NaN` rather than zero, and the Bonferroni correction is taken over the peers that remained.

**A longer window is not a better window.** On a healthy synthetic population, 20 min gave 234 false
incidents a day, 60 min gave 93, and 240 min gave 2583: a four-hour window sits on the slope of the
daily traffic curve, and the trend family duly finds a real, meaningless drift in everything at once.
The period has to match the signal — `SeasonalBaseline` at 240 min cut 2551 to 376, and at 20 min made
things worse.

**Order-blindness is structural, not a bug.** Interquartile spread and range are computed on sorted
values, so a monotone climb and a sawtooth of equal amplitude score identically. Anything that needs to
distinguish them needs an order-sensitive term; retrace (max drawdown over range) is the one used when
profiling generated shapes against recorded ones.
