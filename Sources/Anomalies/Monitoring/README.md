# `Anomalies/Monitoring` — getting real numbers out of a real cluster

Everything between Prometheus and a `MetricWindow` the detectors can read. No detection happens here;
this is acquisition, naming, topology and calibration.

## Acquisition

`PrometheusMetricWindowSource` is the deployed reader: it issues one range query per metric over the
window and fills a `MetricWindow`. `PrometheusMetricSource` and `PrometheusHistoricalSource` are the
instant and replay variants; `HistoricalCsvLoader` replays a recorded window with no cluster at all,
which is how a shape can be pinned in a test.

**The window ends behind "now" on purpose.** Rate expressions are computed over a trailing range, so
samples at the current instant are still filling in. A window ending at the moment a load run stopped
put a cluster-wide downward trend in every RED signal — an artefact of when the measurement ended.

## Naming

`PromqlCatalog` holds the queries; `MetricMap` maps a customer's own metric names onto `MetricIndex`,
which is what lets the guard watch an application that has never heard of Overfit. `MetricMap.Unmapped`
is logged at startup deliberately: a metric nobody exports produces no findings, and **that is
indistinguishable from health** at every layer below.

`PeerSignalCatalog` is the single authority on two per-signal physical claims:

- *Can uneven load explain this magnitude?* Dividing a quantity by request rate is right only when it
  has a real per-request component. Working set does not — it is assemblies, JIT'd code, caches and the
  live set — and dividing it manufactured 317 false findings on one seed, exactly the size of the
  traffic imbalance.
- *Is a single occurrence itself the finding?* Restarts and OOM kills are, which is why they are
  excluded from calibration below.

## Topology

`PrometheusTopologySource` resolves owner, node and declared peer group from kube-state-metrics, so
grouping relates findings by the cluster as it actually is rather than by a pod-name heuristic. The
peer-group label is only queried when the customer names one — `kube_pod_labels` is among the widest
series kube-state-metrics produces and is not worth pulling for nothing.

## Calibration

`FloorCalibrator` watches a period believed healthy and proposes the absolute floors that follow from
it, computing both quantities exactly as the gates compute them so the proposal lands in the units the
gates read. Measured by fitting on one synthetic population and scoring on a **held-out** one:

| Configuration | False incidents/day | Leak still found in |
|---|---|---|
| No absolute floors | 124 | 166 cycles |
| Hand-reasoned floors | 44 | 142 cycles |
| Calibrator proposal | **29** | 142 cycles |

The second column is the veto. Any floor high enough silences any detector, so a configuration that is
quiet on both populations has not been calibrated — it has been switched off.

**Its one real failure mode:** the observed period must actually have been healthy, or the floor is set
above a real fault and the guard is permanently blind to it at that size. This is why the proposal is
logged for a human to accept and never applied on its own.
