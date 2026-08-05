# `Anomalies/Contracts` — every type the detectors agree on

Options, inputs, results and persisted shapes for the whole anomaly subsystem. Nothing here decides
anything; it is the vocabulary the parts use to talk to each other, deliberately gathered in one
directory rather than scattered next to whichever detector happened to introduce it.

## Why one directory

A contract that lives beside its first consumer looks like an implementation detail of that consumer,
and the second consumer then defines its own. This subsystem had `MetricIndex` reachable from four
namespaces and options types in five directories before the consolidation. Grouping them makes the
surface a caller has to learn finite and makes an accidental second definition obvious.

## The parts worth knowing before editing

**`MetricIndex`** is the ingestion channel list and `MetricSnapshot.FeatureCount` is the model input
contract, **and they are not the same number**. A new metric widens the first; widening the second
would silently invalidate every trained checkpoint. Adding a channel must not shift the learned
models' input layout.

**`MetricWindow`** is what a detector actually receives: pods × metrics × samples over one window,
with the timestamps. `Series(pod, metric)` hands out a span into flat storage — there is no jagged
array here and the build forbids one.

**`AnomalyGuardOptions`** carries thresholds, the relative and absolute gates, the trend ceiling used
for time-to-limit projection, and the peer/trend window split. Two lookups on it exist because their
"absent" values differ and getting that wrong is silent: `FloorFor` returns **0** for a missing floor
("gate off"), `LimitFor` returns **NaN** for a missing ceiling ("do not project"). A missing ceiling
read as zero would make every projection nonsense rather than absent.

**`IncidentSubject`** is the identity a finding is about — namespace, workload, ReplicaSet, pod, node.
It is what the grouper matches on and what an operator reads. An empty pod means the finding is about
the workload as a whole, and the narrative says so in words rather than printing a blank.

**`PodPlacement`** carries node and declared peer group. The peer group is a *declared* cohort, from a
pod label the customer names in one config line — not something discovered. Discovery was tried:
partitioning by ReplicaSet killed canaries, because a canary is its own ReplicaSet and therefore alone
below the minimum peer count.

**`RuleProfile`**, **`IncidentTrackingOptions`**, **`IncidentGroupingOptions`** ship named presets
(`Balanced`, `Sticky`, `Strict`). Prefer `Balanced with { … }` over building one field-by-field: the
presets are the shapes that were measured, and a hand-assembled options object is a configuration
nobody has run.
