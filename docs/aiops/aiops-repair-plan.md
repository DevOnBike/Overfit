# Six defects found by reading, and how to fix each

Found by reading the code on 2026-08-01, after the day's changes had passed 1959 tests and four hours on a
live cluster. **None of them would have been found by running anything**, because the lab does not exercise
the paths they live on — which is the argument for the review, and the reason this list exists as a plan
rather than as four rushed commits made while a measurement was running.

Ordered by damage. Each entry says what breaks, where, how to fix it, and what test would have caught it.

---

## 1. The calibrator never observes custom channels

**Damage.** A customer maps `myapp_queue_depth` as a custom channel and configures no floor. The gate is off
for ever, no proposal is ever made, and nothing says so. That is the configuration measured at **209 false
incidents a day** on known metrics — arriving silently, on exactly the channels a customer with a bespoke
application relies on most.

**Where.** `FloorCalibrator.Observe` iterates `for (var m = 0; m < (int)MetricIndex.Count; m++)` and stops
there. `AnomalyGuard.RunCustomPeer` and `RunCustomTrend` read `binding.MinAbsoluteGap` /
`binding.MinAbsoluteTrendChange` directly, bypassing `IAbsoluteFloorSource` entirely.

**Fix.** The calibrator's stores are arrays indexed by `MetricIndex`; custom channels have names, not
indices. Two options, and the second is right:

- *Widen the arrays* to `MetricIndex.Count + custom.Count`. Cheap, and wrong: the index would depend on the
  window's custom list, which can differ between cycles, so a restored payload could be read against a
  different mapping. That is a silent corruption, which is the class of bug this whole list is about.
- *Key custom channels by name.* Add `Dictionary<string, (BoundedSamples Gap, Change, Magnitude)>` beside the
  indexed arrays, observed in a second loop over `window.CustomChannels`, and serialised under a `custom`
  section of the existing payload. Then extend `IAbsoluteFloorSource` with `MinAbsoluteGap(string channel)`
  and route the custom paths through `_floors`.

**Test that would have caught it.** A guard configured with one custom channel and no floor, run for enough
cycles, asserting the proposal for that channel becomes usable — the exact shape of
`CalibratedFloorFallbackTests` but on a custom binding.

---

## 2. `Workload` is empty in the deployed path, which half-kills maintenance windows

**Damage.** `AnomalyGuardOptions.Workload` defaults to `string.Empty`, `AnomalyGuardConfigFile` has no such
field, and `AddOverfitAnomalyGuard` sets `Namespace` but never `Workload`. Consequences, worst first:

- **A maintenance window scoped to a named workload can never match.** `IsDeclaredAbnormal(at, "", …)` is
  compared against `"lab-workload"` and fails. The operator declares a window for their rollout, it silently
  does not apply, and the guard pages them during their own deploy — the precise scenario the feature was
  written for. **A feature shipped half-dead.**
- The seasonal baseline is keyed by `""`. Harmless at one workload per instance, silently wrong at two.
- Workload-level incidents report an empty workload. The lab's own logs show it: `Anomaly incident in lab/:`
  with nothing after the slash.

**Fix.** Three parts, all small:

1. Add `workload` to `AnomalyGuardConfigFile` and set it in `AddOverfitAnomalyGuard`.
2. When absent, **derive it from the topology** rather than defaulting to empty — kube-state-metrics already
   knows the owner of every pod, and `PrometheusTopologySource` already reads it. The most common
   `PodPlacement.Workload` across the roster is the right answer and needs no new query.
3. **Refuse to start** when maintenance windows name a workload and none can be resolved. That combination is
   a contradiction detectable at startup, and detecting it there costs one line against a silence nobody
   would ever attribute.

**Test.** A window naming a workload, a guard whose workload resolves from a fake topology, asserting the
finding comes back suppressed. Today's `MaintenanceWindowTests` sets `Workload` explicitly and therefore
passes over the bug.

---

## 3. The silent-pod check trusts a stale roster

**Damage.** When a topology refresh fails the previous snapshot stands — correct for grouping, wrong here. A
pod deleted during the outage is still on the roster, stops reporting because it no longer exists, and after
two cycles is reported as silent. A scale-down during a Prometheus hiccup produces phantom incidents.

**Where.** `AnomalyGuard.RunSilentPods` reads `IPodRoster.KnownPods` with no notion of staleness;
`AnomalyGuardService.RefreshTopologyAsync` knows the refresh failed and logs `_topologyStale`, but the guard
is never told.

**Fix.** Give the roster a freshness stamp — `DateTimeOffset LastRefreshed` on `IPodRoster` — and skip the
silent-pod check when it is older than a small multiple of the cadence. Standing down is right: the check
exists to catch a pod that never started, and one cycle of not looking costs five minutes against a fault
that lasts until somebody fixes it.

**Test.** A roster whose stamp is old, a window missing one of its pods, asserting nothing is reported.

---

## 4. The seasonal baseline only learns while `DecomposeCommonMode` is on

**Damage.** `_history?.Observe(...)` sits inside the `if (_options.DecomposeCommonMode && podCount >=
CrossPeerBaseline.MinimumPeers)` branch and inside the `TryBuild` success path. Turning the decomposition off
— a reasonable thing to do, and something the test suite itself does — **silently disables a week of
learning**, as a side effect nobody would predict from the option's name.

**Fix.** Move the observation out of the branch and compute the workload level directly from the pods'
medians when no common component was built. The two are close and the fallback is better than nothing;
what is not acceptable is one option quietly switching off an unrelated subsystem.

**Test.** Two guards over identical windows, one with `DecomposeCommonMode` off, asserting both accumulate
history.

---

## 5. The discovery command builds PromQL by string interpolation

**Damage.** `count by (pod) ({series}{{namespace="{ns}",pod=~"{regex}"}})` — a quote or backslash in either
value produces a malformed query. Mitigated by being **reported**: a failed query is logged to stderr with
the series name and counted as "no evidence", so the operator sees it rather than getting a silently
under-reported coverage table. Lowest severity on this list for that reason.

**Fix.** Escape `\` and `"` in both values before interpolation. Five lines and a test with an awkward regex.

---

## 6. A comment asserting behaviour the code does not have

**Damage.** `IncidentPipeline.ObserveSilentPod` claims the `SignalClass` "drives how the grouper relates this
to other findings". `IncidentGrouper` does not read `SignalClass` at all — it scores subject similarity
(`SamePod` → `SameReplicaSet` → `SameWorkload` → `SameNode` → `SameNamespace`) and temporal proximity, and its
only mention of the type is a note that it does **not** establish causation. The next reader will build on a
guarantee that does not exist, which is a worse debt than a missing feature.

**Fix, and there is a choice.**

- *Correct the comment* to say the class is carried into the report and the grouper ignores it. One line,
  honest, and leaves the classification decorative.
- *Make it true.* `SignalClass` already separates cause from consequence, which is the missing ingredient for
  the causal ordering on the optional list — "latency rose four minutes after GC pause rose" is the sentence
  an operator wants and everything needed for it exists except the ordering. But this changes grouping
  semantics and therefore needs a false-positive measurement on both sides before and after.

Take the first now, and the second only as its own measured change.

---

## Order of work

1, 2 and 4 are silent-failure defects on paths a customer will use and are the morning's work. 3 needs a
small interface change and can follow. 5 and 6 are minutes each.

**None of them should be written while a measurement is running.** Today three separate changes were wrong
and every one was caught by measurement rather than by reasoning; writing four unverifiable fixes on top of
that record would mean starting tomorrow by debugging instead of by measuring.
