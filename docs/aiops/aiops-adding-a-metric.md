# Runbook: adding a metric to the anomaly guard

For an operator, or a developer acting on an operator's request, when a new metric must be watched — either
because this project decides to model something new, or because a client asks for one of theirs.

## Step 0 — pick the vehicle

Two vehicles exist, and picking wrong is the expensive part.

| | `CustomMetricBinding` | new `MetricIndex` member |
|---|---|---|
| Where it lives | a `customMetrics` entry in the guard's config file | `Sources/Anomalies/Contracts/MetricIndex.cs` |
| Source change | none | yes, plus one mandatory switch case |
| Release needed | no — config-only, live in the same binary | yes |
| Reaches rules / peer / trend | yes | yes |
| Reaches the learned models (autoencoder etc.) | **no, never** | yes, if also added to `MetricSnapshot` |
| Precedent | `Tests/Anomalies/CustomMetricEndToEndTests.cs` | `MetricIndex.ContainerRestarts` (ordinal 12) |

**Rule of thumb: a metric specific to one deployment is config; a metric this project should model for
everyone is a channel.** And even then, ship it as config first — a channel that goes straight into the enum
carries a floor nobody has measured yet, and every floor in this subsystem that shipped from literature
rather than measurement has been wrong (Step 6).

If you are unsure which this is, default to `CustomMetricBinding`. It is reversible — a channel promoted from
config to the enum later keeps the same PromQL and the same operator-facing name — and a config change ships
without a release, so getting it wrong costs a config edit, not a deploy.

## Step 1 — check the workload actually emits it

**The most common way to fail at this is skipping this step.** A binding to a series nothing emits produces
a channel that reports as unbound, silently, forever. `CpuThrottleRatio` is the standing example: it is a
built-in `MetricIndex` member, but `container_cpu_cfs_throttled_periods_total` only exists on a container
that carries a CPU limit, and the lab's pods do not set one — `k8s/anomaly-guard/guard.lab-workload.json`
leaves it out with the comment "left out and reported blind." It has been unbound since the lab existed, not
transiently.

The lab's own workload (`Demo/LabWorkload/WorkloadMetrics.cs`) emits, for reference:
- `labapp_requests_total`, `labapp_errors_total`, `labapp_request_duration_seconds` (histogram) — hand-rolled,
  named to stand in for a client's application
- `dotnet_gc_heap_size_bytes`, `dotnet_gc_committed_bytes`, `dotnet_gc_pause_seconds_total`,
  `dotnet_threadpool_queue_length`, `dotnet_process_working_set_bytes` — real `GC`/`ThreadPool` figures under
  this project's names

Confirm with a direct scrape (`curl` the exporter, or query Prometheus for the series name) before writing
any binding. `MetricMap.Describe()` will tell you afterwards that a feature is unmapped, but only after
you've written the binding — check the source first and save the round trip.

## Step 2 — config-only path (`CustomMetricBinding`)

Add an entry under `customMetrics` in the guard's config file; `AnomalyGuardConfigReader.ReadMap`
(`Sources/Anomalies/Monitoring/AnomalyGuardConfigReader.cs:65-115`) parses it and reports one line per
unusable entry rather than throwing. Fields, and the cost of getting each wrong:

- **`Source`** — the exporter's series name. Verified in Step 1.
- **`Kind`** — `Gauge` / `Counter` / `EventCount` / `Ratio` / `HistogramSeconds`, decides the PromQL wrapper
  (`MetricMap.Build`, `Sources/Anomalies/Monitoring/MetricMap.cs:219-232`). Get this wrong and the query is
  syntactically fine and semantically empty — a counter queried as a gauge returns a monotonically increasing
  number that never looks anomalous.
- **`SignalKind`** (`LoadSensitive` / `LoadIndependent`) — whether uneven load explains the magnitude, so
  whether the peer comparison divides by a work metric before comparing replicas. Measured cost of getting
  this wrong, from `PeerSignalCatalog`'s own comment: memory was classified load-sensitive and divided by
  request rate, and became the single largest source of false peer findings on a healthy population (317,
  269 and 35 false findings across three seeds) — a working set is a fixed cost, and dividing a fixed
  quantity by a varying one manufactures a difference the size of the traffic imbalance. Config default in
  `AnomalyGuardConfigReader` is `LoadIndependent` (`entry.LoadSensitive` must be set explicitly to true) — the
  safer of the two wrong answers, because it compares the raw number instead of silently distorting it.
- **`Class`** (`Infrastructure` / `Resource` / `Symptom`) — required, no default. The grouper orders an
  incident cause-first (`IncidentNarrative`), so a metric with no `Class` set correctly lands wherever a
  guessed value puts it, regardless of what it actually means. There is no "leave it and see" here: the
  constructor takes it positionally.
- **`MinAbsoluteGap`** — smallest peer difference worth reporting, in the metric's own units. **Zero disables
  the gate**, and it defaults to zero. Leaving it off is how a difference of three tenths of a millisecond
  became the top false-positive source on a healthy population (`FloorCalibrator`'s own comment records this
  at 209 false incidents a day for exactly this reason — a customer-owned channel starting ungated).
- **`MinAbsoluteTrendChange`** — the same gate for the trend family; same failure mode if left at zero.
- **`Quantile`** — only for `HistogramSeconds`; 0.95 when left at zero.
- **`Rule`** (`SustainedThresholdOptions`, optional) — an absolute floor: "above this, for this share of the
  window." The case the relative families cannot reach — a queue depth or consumer lag that is simply too
  high regardless of what siblings are doing.
- **`SaturationLimit`** — the ceiling this signal is heading towards (disk capacity, queue bound, connection
  pool size), for projection. `NaN` (the default) skips it.

Reused/borrowed fields error out on collision: naming a custom metric the same as an existing `MetricIndex`
member is rejected by the reader with "put it under `Metrics` so it reaches the learned family too" — the
reader is telling you which vehicle you actually meant.

**`FloorCalibrator` calibrates and persists custom channels too** (`_customChannels`, keyed by name with a
`~` marker in the serialised state, `Sources/Anomalies/Monitoring/FloorCalibrator.cs:43,73`) — a custom metric
is not floor-blind by design, only by omission if `MinAbsoluteGap` is left at zero.

## Step 3 — source-change path (new `MetricIndex` member)

Only when the metric should be a permanent, product-wide channel.

1. **Append, never insert.** `MetricIndex` values are persisted as `MetricTypeId` on `RawMetricSeries` and in
   historical CSV; inserting in the middle silently reinterprets every stored sample
   (`MetricIndex.ContainerRestarts`'s own doc comment says this explicitly).
2. **Add a case to `PromqlCatalog.DefaultTemplate`** (`Sources/Anomalies/Monitoring/PromqlCatalog.cs:380-424`).
   This is the one **mandatory** touch point — every other unhandled member falls through a safe default, but
   `DefaultTemplate`'s `switch` has no `_ =>` arm and throws `ArgumentOutOfRangeException` on anything it does
   not name.
3. **Decide whether it is a learned-model feature.** If yes, add it to `MetricSnapshot` and bump
   `FeatureCount` — and budget for a full retrain: doing this once for `ContainerRestarts` moved the tokeniser
   vocabulary from 768 to 832 and broke two committed checkpoints (`k8s_anomaly_checkpoint.bin`,
   `k8s_anomaly_medium.bin`), a `ContextLength` of 120 no longer divisible by tokens-per-snapshot, a
   201 000-row CSV fixture, and the Python generator behind it. If no — the common case — leave it out of
   `MetricSnapshot`, exactly as `ContainerRestarts` does; `MetricIndex.Count` (13) is already larger than
   `MetricSnapshot.FeatureCount` (12) for precisely this reason.
4. **Everything else is optional and safe by default**, confirmed by grep across `Sources/Anomalies`: ~15
   sites iterate `(int)MetricIndex.Count` generically and need no edit; `PeerSignalCatalog.Classify` falls to
   `LoadIndependent` for an unlisted member; `SignalCatalog.Classify` (name-substring based, separate from
   `PeerSignalCatalog`) falls to `Symptom`; `AnomalyGuardOptions.DefaultRules` needs an entry only if the new
   channel should carry a built-in absolute rule.
5. **Persistence is safe across the change** because both `FloorCalibrator` and `MetricHistory` key by enum
   **name**, not ordinal (`Enum.TryParse<MetricIndex>` in both, `FloorCalibrator.cs:456`,
   `MetricHistory.cs:354`). Appending does not corrupt state written before the member existed; reordering
   would.
6. Ship it, and this needs a release — it is a `Sources/Anomalies` source change, subject to the normal build
   and test gates.

## Step 4 — does it need training? (the question operators actually ask)

No, in four of five senses of the word. Verified against the code:

| mechanism | warm-up needed | source |
|---|---|---|
| Peer comparison | **none** — works from the first cycle | Cliff's delta (`TwoSampleComparison`) is rank-based and scale-free; compares replicas within one window only |
| Trend detection | a window, not training — minutes to fill the buffer | Theil-Sen slope + Mann-Kendall + Kendall's tau, all rank-based (`Sources/Main/Statistics/TrendDetector.cs`); no persisted state |
| Floor calibration | hours of healthy traffic, and it *proposes* — a human accepts | `FloorCalibrator` observes spread and writes a proposal; `ConfiguredFloorSource`'s own doc: "an explicit value always wins, even a lower one" |
| Seasonal history | **two calendar days minimum**, and a 24-hour run cannot arm it at all | `AnomalyGuardOptions.MinimumHistoryDays = 2` (default); `MetricHistory` keeps **one observation per (workload, metric, hour) per calendar day** — one day contributes one observation per hour-bucket, not enough to seed a second day of comparison |
| Learned models (autoencoder etc.) | retraining — but **only for `MetricSnapshot`'s 12 features** | a custom metric, or a channel added to `MetricIndex` but not to `MetricSnapshot`, needs none |

**Counter-intuitive measurement, 2026-08-08, record it here because an operator will otherwise assume the
opposite:** replaying the same 298-cycle window with the guard's real deployed learned state (seasonal
history + floor calibration) made it **three times noisier** than cold — 11 incidents opened became 33, 296
findings became 517 (`docs/aiops/aiops-backlog.md`, "The replay is faithful, the learned state makes it
worse"). Isolating the two payload halves showed floor calibration moved nothing (40 opened either way);
seasonal history was the entire effect. Whether that is a defect in the seasonal expectation or a mis-shaped
expectation of what "learned" should buy is **open** — treat "turn on history" as something to A/B on your
own population, not as an assumed improvement.

## Step 5 — verify before trusting it

1. `MetricMap.Describe()` — one line per channel, mapped or not, with source. Run this before the guard's
   first real cycle.
2. Confirm the new channel is not in `MetricMap.Unmapped` (it is a `bool[]`/`List<MetricIndex>`-shaped
   report — an unmapped feature is queried never, not queried-and-empty, which is a different failure and a
   quieter one).
3. Watch one full cycle's `SignalFinding`s for the new signal name (`SignalFinding.Signal` is a plain string;
   `grep` the reporter output for it).
4. If a `Rule` was attached, confirm at least one healthy sample clears `DetectionStatus.Healthy` rather than
   `InsufficientData` — a rule with no usable samples is deliberately never reported as healthy
   (`SustainedThresholdRule`'s own doc: "missing is not zero").

## Step 6 — the rule that governs every threshold in this subsystem

**No floor comes from literature.** Every threshold here was measured on this population, and where the two
disagree, the literature has been wrong in a way that mattered. Worked example: the commonly quoted 25%
CPU-throttling threshold never fired on the lab's own workload — the measured peak was 19.8% (median 1.4%,
p90 11.8%), so the shipped rule uses 5%/25% bands taken from what the lab actually produced, not from a
textbook (`Sources/Anomalies/Rules/README.md`, `SustainedThresholdOptions.ForCpuThrottling`).

**Say plainly what a lab cannot calibrate.** A fault signal on a cluster that has never faulted yields a
floor for the *healthy* distribution, not for the fault — it tells you what normal looks like, not where the
line to the abnormal actually sits. `CpuThrottleRatio` and the PSI CPU-pressure channel are both in this
state on the current lab: neither pod population has ever been starved, so the healthy-day distribution is
"nothing happened" and a floor fit to it would repeat the `ContainerRestarts`-floor-of-1.25 mistake the
codebase has already paid for once. Calibrating a starvation floor needs an induced-contention experiment
(a stress sidecar, an oversubscribed node) — not more healthy traffic, however much of it you collect.

## Reference — files this runbook is grounded in

- `Sources/Anomalies/Contracts/CustomMetricBinding.cs` — the config-only vehicle and every field's contract
- `Sources/Anomalies/Contracts/MetricIndex.cs` — the enum, `ContainerRestarts` as the append precedent
- `Sources/Anomalies/Contracts/MetricSnapshot.cs` — the learned-model feature contract, `FeatureCount = 12`
- `Sources/Anomalies/Monitoring/AnomalyGuardConfigReader.cs` — `customMetrics` parsing, `ReadMap`
- `Sources/Anomalies/Monitoring/MetricMap.cs` — binding storage, `Unmapped`, `Describe`
- `Sources/Anomalies/Monitoring/PromqlCatalog.cs` — `DefaultTemplate`, the one mandatory switch
- `Sources/Anomalies/Monitoring/PeerSignalCatalog.cs` — `LoadSensitive`/`LoadIndependent` classification
- `Sources/Anomalies/Monitoring/FloorCalibrator.cs` — floor proposals, custom-channel persistence
- `Sources/Anomalies/Monitoring/MetricHistory.cs` — seasonal history, per-day-per-hour buckets
- `Sources/Anomalies/Rules/SustainedThresholdRule.cs`, `Sources/Anomalies/Rules/README.md` — measured
  thresholds vs. literature
- `Sources/Main/Statistics/TwoSampleComparison.cs`, `TrendDetector.cs` — the rank-based detectors
- `Demo/LabWorkload/WorkloadMetrics.cs` — the lab's emitted series
- `k8s/anomaly-guard/guard.lab-workload.json` — `CpuThrottleRatio` left deliberately unbound
- `docs/aiops/aiops-backlog.md` ("The replay is faithful, the learned state makes it worse") — the
  2026-08-08 seasonal-history measurement
- `Tests/Anomalies/CustomMetricEndToEndTests.cs` — end-to-end coverage of the config-only path
