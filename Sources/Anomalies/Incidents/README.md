# `Anomalies/Incidents` — from findings to something worth waking someone for

A detector produces findings: "this replica's heap is unlike its peers'". An operator cannot act on
those. Twelve replicas × thirteen signals is a hundred and fifty-six chances to say something every
cycle, and a page for each is the alert fatigue this product exists to reduce. This directory turns
findings into **incidents**: grouped, deduplicated across cycles, explained in words, and durable
across a restart of the guard itself.

## The pieces, in the order a cycle uses them

| Type | Role |
|---|---|
| `AnomalyGuard` | The entry point. One `RunCycle(window, now)` runs every detector, groups, tracks and reports. |
| `IncidentPipeline` | Collects findings from all families during a cycle. |
| `IncidentGrouper` | Decides which findings belong to the same problem, weighted by shared subject and topology. |
| `IncidentTracker` | Matches this cycle's groups against still-open incidents so a problem opens **once**. |
| `IncidentNarrative` | Writes the human sentence: what, when, where, scope, caveat. |
| `IncidentReporter` / `IIncidentSink` | Emits rows; the host decides whether that is a log, a queue or a page. |
| `FileIncidentStore` / `IIncidentStore` | Survives a restart, so a rollout does not re-page for everything already open. |
| `SignalCatalog` | Per-signal wording and direction, so a message says "climbing" rather than "delta 0.87". |

## Two decisions here that were reversed by evidence

**The overlap veto is gone.** Matching used to require both a matching primary subject *and* a
Jaccard overlap of the peripheral subjects above a bar. On a real run, an incident scored **0.33
against a 0.34 threshold** and was reported a second time as new. The fix was not lowering the bar: the
periphery rotates by design as different replicas drift in and out of a group, so a second gate on it
is measuring noise. A matching primary subject is now enough, and the overlap is kept only as a
diagnostic label.

**Durable state existed and nothing deployable reached it.** `AnomalyGuard` accepted an `IIncidentStore`
from early on, while the hosted service constructed it without one — so every incident reopened after
each restart and the tracker's entire contribution was undone by the guard's own rollout. If you add a
host, pass the store.

## What "one incident" means

An eight-cycle run of one problem leaves eight report rows and opens exactly **one** incident. When
reading a diagnostic, `opened` is the number that would have paged somebody; anything above one per
real problem is the tracker failing at its job.
