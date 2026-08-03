# One guard, many scopes — design

Today one `AnomalyGuard` watches one namespace and one pod regex. At a client with fifty namespaces that is
fifty Deployments, fifty ConfigMaps, fifty PersistentVolumeClaims and fifty things to upgrade. This is the
design for putting several scopes inside one instance, and the argument for where the boundaries go.

Written 2026-08-03, before any code. Nothing here is measured yet; the cost estimates are arithmetic and are
labelled as such.

## The decision, first

**Several scopes in one process, each with its own detection state, sharing one Prometheus client, one
durable state file and one metrics endpoint.**

Not chosen: a Kubernetes operator. It would buy declarative configuration that a ConfigMap already provides,
and cost RBAC, CRDs and a security review. An operator earns its keep by reconciling cluster state toward a
desired state; this thing never writes to the cluster.

Not chosen either: sharding scopes across replicas. That is the answer at a scale nobody has yet, and it
needs leader election to avoid two replicas reporting the same incident twice. One process until measurement
says otherwise.

## What is per scope, and why each one is

The test for every piece of state is a single question: **would sharing it let one scope's data change
another scope's verdict?**

| State | Per scope | Because |
|---|---|---|
| `IncidentTracker` | **yes** | Incident identity is matched by subject overlap. Two namespaces sharing a tracker would let a finding in one join an incident in the other whenever their subject keys collide — and they will, because pod names repeat across namespaces. |
| `FloorCalibrator` | **yes** | A floor is "what this cluster does when it is well" in the signal's own units. A namespace running 12 pods at 40 MB and one running 3 pods at 1.2 GB share nothing but the metric name. Merging them produced a **246×** error the last time it happened here. |
| `MetricHistory` | **shared, already keyed by workload** | It stores per workload, per metric, per hour-of-day. The key already distinguishes scopes as long as workload names are unique; where they are not, the key becomes `namespace/workload`. This is the one piece that needs no restructuring. |
| Silent-pod counters | **yes** | They count consecutive cycles of verified silence for a named pod. A pod name in another namespace is a different pod. |
| Recent-incident index | **shared** | It maps an incident id to the finding it was about, for acknowledgement. Ids are globally unique (below), so one map is correct and one lookup is simpler than N. |
| Labels and suppressions | **shared, scoped by subject** | A suppression already carries namespace, workload and pod in its subject, so it cannot leak across scopes. A `--real` label constrains floor proposals **per signal**, and that constraint should not cross scopes — so the label store gains a scope key, or the calibrator asks with one. **This is the one place where the obvious design is wrong**, and it is the piece most likely to be got wrong: a confirmed 6 MB gap in a 40 MB namespace must not cap the floor in a 1.2 GB one. |
| Telemetry | **shared instrument, per-scope label** | See below — this is not cosmetic. |
| Maintenance calendar | **shared** | Windows already name a workload; a window with no workload applies namespace-wide, which becomes scope-wide. |

## Incident identity stays global

One `_nextId` across all scopes, persisted once. The alternative — a counter per scope — makes an id
meaningless without its scope, so every consumer must carry the pair, and the acknowledgement CLI would need
both. Worse, it multiplies by N the identifier-reuse hazard fixed on 2026-08-02, which is a hazard whose
symptom is a consumer joining two unrelated incidents.

Global ids cost nothing: the counter is a `long` and the tracker already advances it from every restored
record.

## Failure isolation is the part that decides whether this is safe

Today a cycle that throws is caught, counted in `overfit_guard_cycle_failures_total`, and the loop continues.
With N scopes that guarantee has to become **per scope**, or one namespace whose Prometheus queries start
failing silently stops the other forty-nine.

Concretely: the per-scope evaluation is wrapped individually; a scope that throws increments its own failure
counter and the loop moves to the next. A scope that throws on every cycle must be **loud** — its failure
counter climbing while its cycle counter does not is the alertable signal, and it is why the counters must
carry a scope label.

## Telemetry: the label is not optional

`GuardTelemetry` currently renders bare series names. With several scopes, `overfit_guard_cycles_total`
becomes the sum across scopes, and — the case that matters —
`overfit_guard_last_cycle_timestamp_seconds` becomes the **most recent** across scopes.

That single series is what makes "the guard has stopped" alertable. Without a scope label, one healthy scope
keeps it fresh while forty-nine are stalled, and the alert never fires. **A guard that has stopped watching
looks exactly like a cluster that is healthy** — the pathology this whole subsystem exists to remove, and it
would be reintroduced by a change that looks like configuration.

So: every series gains a `scope` label. Fifteen series times fifty scopes is 750 series, which is nothing for
Prometheus and is worth stating so nobody worries about cardinality.

## Cost, as arithmetic rather than measurement

Per cycle, per scope: 13 metric queries plus 3–4 topology queries. At fifty scopes on a five-minute cadence
that is roughly **850 queries every five minutes**, about 3 per second sustained. Prometheus handles that
without noticing; the client's Prometheus might disagree, and the honest thing is to ask them rather than
assume.

CPU is not the constraint: a cycle is arithmetic over a window already in memory and takes milliseconds. The
constraint is **memory**, and it is bounded by construction — `BoundedSamples` caps the calibrator's
retention, `MetricHistory` caps buckets, and the tracker caps open incidents. Fifty scopes multiply a bounded
number by fifty, which needs stating in the deployment guide as a resource request rather than discovering it
in production.

Queries are issued **sequentially across scopes**, not in parallel. Parallel queries would arrive at
Prometheus as a burst every five minutes, and a monitoring tool that spikes its own monitoring backend is a
poor citizen. Sequential also keeps the failure isolation simple.

## Configuration

```json
{
  "prometheus": "http://prometheus.monitoring:9090",
  "scopes": [
    { "namespace": "payments", "podRegex": "api-.*",    "workload": "api" },
    { "namespace": "payments", "podRegex": "worker-.*", "workload": "worker" },
    { "namespace": "search",   "podRegex": "index-.*" }
  ],
  "metrics":    { "...": "shared bindings, overridable per scope" },
  "thresholds": { "...": "shared defaults, overridable per scope" }
}
```

Two scopes in one namespace is deliberate and is the common case, not an edge one: `api-*` and `worker-*` are
different populations and comparing a worker against an API replica is the mistake peer grouping exists to
prevent.

Bindings and thresholds are shared with per-scope overrides, because most clients run one stack and repeating
thirteen metric names fifty times is how a config file stops being read.

## Migration

The existing single-scope config stays valid: a file with `namespace` and `podRegex` at the top level is read
as a one-element `scopes` list. Nothing in the lab needs changing on the day this ships, which is also what
makes it testable — the same measurement can be run before and after with the same manifest.

## What would make this the wrong call

Stated so the decision can be revisited on evidence rather than taste:

- **If per-scope memory turns out not to be bounded in practice** — if fifty scopes at a real client do not
  fit in a sensible request — then the answer is sharding, and this design becomes the thing in the way.
- **If clients want per-namespace RBAC on the guard itself**, one process reading all of them is wrong
  regardless of efficiency, and fifty Deployments is the correct answer for a reason that has nothing to do
  with engineering.
- **If a scope's failure cannot be isolated cleanly** — if in practice one bad scope does take the loop down
  — then process isolation was doing real work and it should be kept.

The first is measurable before shipping; the second is a question for the first client who asks; the third is
the one to watch during implementation.
