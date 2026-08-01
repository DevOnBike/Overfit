# `LabWorkload` — a application to break on purpose

A small ASP.NET Core service whose only job is to misbehave in specified ways, so the anomaly guard can
be measured against a signal whose shape is known **in advance** rather than reverse-engineered from
what a real server happened to do.

## It has no reference to Overfit, deliberately

The guard should be validated against something that has never heard of it, exporting metrics under its
own names (`labapp_*`) and mapped onto `MetricIndex` through a configuration file. That is exactly what
a customer's application looks like, and it is the only way the metric-mapping layer gets exercised at
all.

Runtime values (`dotnet_*`) are real, taken from the process, so GC and memory signals are genuine
rather than simulated.

## Faults are injected in flight

`POST /fault/{latency|stall|errors|leak|cpu|clear|oom|crash}` changes behaviour **without a restart**.
That is the whole design point: a restart is itself a fault — cold memory, an incremented restart
counter, no history inside the detection window — and injecting faults by redeploying mixed two
independent things into every experiment.

`oom` and `crash` are separate endpoints on purpose. The first moves `OomEventsRate` **and**
`ContainerRestarts`; the second moves only restarts. Splitting them is what makes the claim "peer
comparison is structurally blind to a single OOM kill" testable rather than rhetorical.

## Ground truth was verified before it was trusted

Set 50% stalls → measured 20 of 40. Set 25% errors → measured 10 of 40. Set an 8 MB/s leak → heap went
2 → 233 MB. The instrument is checked against its own dial before any detector is scored on it.

## Why this replaced the inference server

The previous lab ran the real Overfit server, which was a poor instrument: a 1.07 GB model per pod
capped the node at about four replicas, only one fault was injectable, and that fault turned out to be
stall-and-catch-up rather than uniform slowdown — so a "throttled" pod completed the same number of
requests as a healthy one and read as idle. Twelve replicas of a small app answer questions four
replicas of a large one cannot.

Published from the host and packaged into a runtime-only image; see `k8s/lab/workload.yaml`.
