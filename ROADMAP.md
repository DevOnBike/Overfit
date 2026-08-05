# Overfit Roadmap

Zero-allocation, pure C# deep-learning framework targeting high-performance CPU inference and small/medium language model inference on .NET 10+.

**Philosophy:** minimal dependencies, predictable memory behavior, competitive CPU inference, and explicit separation between training, inference and kernels. Native-AOT compatible. Runs on low-end consumer hardware.

---

> **Finished work has moved to [ROADMAP-COMPLETED.md](ROADMAP-COMPLETED.md).**
> Closed tracks, fixed defect lists and superseded plans live there with their measurements intact —
> the negative results in particular, which this project treats as the most valuable output it has.
> This file is what is still open.

## 🎯 ACTIVE TRACK — anomaly guard, from "it detects" to "a client can run it"

**Where it stands.** Detection is not the open problem. Ten of ten injected fault shapes are caught on a
synthetic population with per-family attribution (`DetectionMatrixDiagnostics`); across six shadow runs on a
live cluster the degraded replica was found in every cycle. The guard deploys as one pod, configured from a
ConfigMap, whose only dependency is an HTTP route to Prometheus — no API-server access, no RBAC, no CRDs, no
operator, no agent inside the client's application. That claim is tested by running it that way, not asserted.

**What is open is trust, not capability.** The things that kill a tool like this are noise, having no way to
say "that was a false alarm", and silence that looks like health. The list below is ordered by that, not by
how interesting the work is.

### ✅ Fixed 2026-08-02 — the six found by reading

All six are fixed, with tests, and the suite is green. They came out of a code review after the day's work
had passed 1959 tests and four hours on a live cluster; **none would have been found by running anything**,
which is the argument for the review. Analysis kept in `docs/aiops-repair-plan.md`.

| # | Defect | Fix |
|---|---|---|
| **A** | `Workload` empty in the deployed path — a maintenance window naming a workload could never match, and `SubjectKey` collapsed to `"namespace/"` so every workload-level finding in a namespace shared one identity | `workload` in the config file and the host wiring; when absent it is **derived from topology** (the most common `PodPlacement.Workload` across the window, ties broken by name); a window naming a workload with no workload and no topology is now **refused at construction**. Tests: `WorkloadResolutionTests`. |
| **B** | The calibrator never observed custom channels, so a customer-mapped signal with no configured floor kept the gate off for ever | Name-keyed accumulators beside the indexed ones, a `Propose(string)`, `IAbsoluteFloorSource` overloads taking a signal name, and the custom peer/trend paths routed through `_floors`. Serialised under a `~`-marked line so old payloads still read. Tests: `FloorCalibratorTests`. |
| **C** | The seasonal baseline only learned while `DecomposeCommonMode` was on | The observation moved out of that branch; the workload level falls back to the median across pods. Test: `SeasonalExpectationTests.HistoryIsLearnedWithTheDecompositionOff`. |
| **D** | `IncidentTracker.Restore` advanced the id counter only for adopted incidents, and truncation was silent | The counter advances for **every** saved record, ahead of both the staleness filter and the capacity bound; `Truncated` reports only what `MaxOpenIncidents` refused. Tests: `IncidentTrackerTests.RestoreProtectsTheIdentifiersOfIncidentsItDropped` (age) and `…ItHadNoRoomFor` (capacity). **The capacity half was left undone on 2026-08-02 while this row claimed both**, and was fixed on 2026-08-03. |
| **E** | The silent-pod check trusted a stale roster | `IPodRoster.LastRefreshed` (stamped only on a successful rebuild) plus `AnomalyGuardOptions.MaxRosterAge`; the check stands down and clears its counters when the list cannot be verified. Tests: `SilentPodTests`. |
| **F** | A comment asserting the grouper reads `SignalClass`, which it does not | Corrected in place. |

Fixed alongside them, from the same review: `IIncidentStore.LastError` is read after both loads at
construction and after both saves each cycle, surfaced as `AnomalyGuard.StateError` and counted in
`overfit_guard_state_failures_total` (a store that cannot write was otherwise silent until the next restart
reopened everything). **This row asserted that on 2026-08-02 and none of it existed** — it landed on
2026-08-03, pinned by `GuardStateFailureTests`; `IncidentPipeline` **sheds**
findings past the grouping bound and counts them in `Dropped` instead of throwing, because the event most
likely to reach a thousand findings is the cluster-wide one an operator most needs reported; and PromQL
label values are escaped in `PromqlCatalog` and `overfit anomaly-discover`.

Still open, and larger than a fix: **one guard instance watches one scope** — a single namespace and a
single pod regex. At a client with fifty namespaces that is fifty Deployments, ConfigMaps and volumes.
**Designed 2026-08-03: `docs/aiops-multi-scope-design.md`.** Several scopes in one process, each with its own
tracker, calibrator and silent-pod counters, sharing one Prometheus client, one state file and one metrics
endpoint.

**▶ IN PROGRESS 2026-08-05 — sliced so each step is separately testable, and the first two are deliberately
behaviour-neutral.** Suite 2138/0/261 after both.

| # | slice | changes behaviour? | state |
|---|---|---|---|
| 1 | scope model + config migration | **no** — a single-scope file resolves to a one-element list | ✅ `GuardScope`, `GuardScopeEntry`, `GuardScopeResolver`, 7 tests |
| 2 | `scope` label on every series | no — absent scope renders exactly the old output | ✅ `GuardTelemetry.Render`, 5 tests |
| 3 | host running N scopes sequentially, failure isolated per scope | yes | ▶ next |
| 4 | shared durable state: global id counter, per-scope partitions | yes | |
| 5 | `--real` labels keyed by scope | yes | |

**Slice 2 before slice 3, and the order is not arbitrary.** The design names
`overfit_guard_last_cycle_timestamp_seconds` as the series that makes "the guard has stopped" alertable;
across scopes without a label it becomes the *most recent* of them, so one healthy scope keeps it fresh while
the rest are stalled. Admitting several scopes before labelling the series would open a window in which that
alert is silently dead — the exact pathology this subsystem exists to remove.

Two decisions taken in slice 1 that the design left open:

- **The scope name is derived (`namespace/podRegex`), never configured.** A name a client can type is a name
  a client can change, and changing it would orphan that scope's saved incidents and learned floors while
  looking like an edit to a label. The namespace alone will not do: two scopes in one namespace is the
  ordinary case, and a shared name would merge their trackers.
- **Declaring both the old top-level pair and a `scopes` list is refused, not merged.** Nothing can tell
  whether the top-level fields are a scope of their own or defaults the list overrides, and either guess
  watches a different set of pods than the file appears to describe.

One hazard found while writing slice 2, worth carrying into slice 3: the Prometheus text format allows one
`# HELP`/`# TYPE` per metric name **per document**, and a second makes Prometheus reject the *entire* scrape.
Concatenating per-scope renderings would therefore have taken the whole endpoint down at the second scope.

Three things that design settles and which are not obvious:

- **Incident ids stay global**, not per scope. A per-scope counter makes an id meaningless without its scope
  and multiplies by N the identifier-reuse hazard fixed the day before.
- **`--real` labels must not cross scopes.** A label caps future floor proposals for a signal, and a
  confirmed 6 MB gap in a 40 MB namespace capping the floor in a 1.2 GB one is the same 246× error this
  project already made once. This is the one place where the obvious sharing is wrong.
- **Every telemetry series needs a `scope` label, and one of them needs it for safety rather than tidiness.**
  Without it `overfit_guard_last_cycle_timestamp_seconds` is the most recent across scopes, so one healthy
  scope keeps it fresh while the rest are stalled and the "the guard has stopped" alert never fires. That is strictly better than
becoming a Kubernetes operator, which would buy declarative configuration we already have through a
ConfigMap while costing RBAC, CRDs and a security review — and an operator earns its keep by reconciling
cluster state, which this never does.

### Must have — before a client deployment can be armed

**Two left of eight.** The table below carried five items that had already shipped, which is the failure
mode nobody watches for in a roadmap: it does not overstate progress, it *understates* it, and the reader
plans around blockers that are not there. Corrected 2026-08-02 against the code.

| # | Item | Why it blocks |
|---|---|---|
| — | ~~**Operator feedback**~~ | **Done 2026-08-02.** Labels with the finding's magnitude in the signal's own units, a `--real` cap so no proposed floor can silence a confirmed finding, subject-scoped suppressions with a mandatory expiry, four telemetry series, an HTTP endpoint and `overfit anomaly-ack` / `anomaly-suppressions`. Replayed against the fault panel in three arms — clean, calibration only, calibration plus 23 dismissals — with **identical detection latency in all three**. Design in `docs/aiops-operator-feedback.md`; the replay in `OperatorFeedbackRegressionDiagnostics`. |
| 8 | **One 24 h measurement on a frozen configuration** | ▶ **Ready to start.** Stopped deliberately at 2026-08-02 08:37 UTC to run #7, which cannot be done without changing the namespace the run counts. Both of its blockers are now cleared: the warm-up grace shipped, and the learned-state volume was wiped at 14:15 UTC — the calibrator had observed the scaling experiment and folded it in as healthy, moving the `RequestsPerSecond` peer-gap proposal from 0.0119 to **1.242**, which would have started the frozen run with a floor above the events it is meant to catch. The guard restarted with `Adopted 0 open incident(s)`, so it begins the day with no history and no calibration — the same state a client's first day has, which makes the number it produces the right one to quote. The per-day false-positive figure quoted to a client has to come from an observed day. Iteration runs are 4 h — precision goes as √events, so 6× the wall-clock buys only ~2.4× precision, which is not worth it while the code is still changing. A 4 h run must sit on the **same clock hours** each time, because the load driver runs a real 1440-minute diurnal curve. |

**Done, moved out of this table 2026-08-02** (each verified against the code, not from memory):

| # | Item | Where it lives now |
|---|---|---|
| 1 | Cross-cycle history | `Monitoring/MetricHistory.cs` — per workload, per metric, per hour-of-day, seven days, persisted beside the calibration in `LearnedState`. Keyed by **workload, not pod**, because pod names do not survive a deploy. `AnomalyGuard` reads it through `TryExpectation`, which interpolates between hour buckets so it cancels a slope rather than a level. |
| 3 | Maintenance / deploy suppression | `Contracts/MaintenanceWindow.cs` + `IMaintenanceCalendar` + `StaticMaintenanceCalendar`, configured from the ConfigMap. A covered cycle is reported, flagged, and **not learned from**. The workload it is scoped to is now derived from topology when unconfigured — see the A defect above, which had this feature shipped and unmatchable. |
| 4 | Guard self-monitoring | `Monitoring/GuardTelemetry.cs`, 11 Prometheus series behind `--metrics-port 9469` with a Service and ServiceMonitor in `k8s/lab/anomaly-guard.yaml`. The one to alert on is `overfit_guard_last_cycle_timestamp_seconds` going stale. |
| 5 | Operational floor beside the calibrated one | `Monitoring/ConfiguredFloorSource.cs` — an explicitly configured floor **always wins**, even a lower one, and the calibrated value is the fallback. A calibrated floor is a *noise* floor: it says what to ignore, never what is worth waking for. |
| 7 | Rollout / scale-up / scale-down / HPA validated on the lab | **Done 2026-08-02** — five phases on the live cluster, each with its expectation written before the action. Full write-up in `docs/aiops-day-one-events.md`. Headline: **no pod was ever accused of silence**, including two scale-downs removing eight replicas each, one of them driven by an HPA with nobody pressing anything — that was the risk worth running it for. Grouping held the operator-facing count between 1 and 4 incidents per phase against bursts of up to 36 findings. Two findings came out of it: the warm-up defect (2b above) and the fact that **"the step detector never fired" was an artefact of the harness** counting log lines — the step did fire on both scale-downs, and `IncidentReporter` prints only the primary finding's reason, so a non-primary one is invisible in the row. |
| 2b | Warm-up grace for pods with no history | **Done 2026-08-02.** `AnomalyGuardOptions.WarmUpGrace` (15 min), with the pod's age taken from `kube_pod_created` through `PodPlacement.CreatedAt` rather than counted in the guard — a cycle counter would reset on every guard restart and silence the trend family on every pod at once, which is the failure shape this subsystem exists to remove. The gate fails closed three ways: no grace configured, no topology, or an **unknown** creation time all mean "judge as before". Trend family only; peer keeps judging young pods, because during a rollout every pod is young and a peer-wide grace would blind the guard exactly while a bad version goes out. `WarmUpGraceTests`. |
| 6 | Durable incident state on a PVC + a restart experiment | `PersistentVolumeClaim/anomaly-guard-state` (128 Mi) replaced the `emptyDir`. Verified on the 2026-08-02 redeploy: the new pod logged *"Adopted 1 open incident(s) from durable state"*. |

One qualifier on #1, so nobody reads more into it than is there: the seasonal expectation the guard actually
applies comes from `MetricHistory.TryExpectation`. `SeasonalBaseline` (the statistic measured at 2551 → 376
false/day on a 240-minute window) is referenced by `TrendDetector` and `MetricHistory` but is **not** what
the guard consults per cycle. Wiring that path is a follow-on, not a blocker.

### Optional — valuable, not blocking

| Item | Note |
|---|---|
| ~~**Affine work-adjusted trend** for load-sensitive signals~~ | **Measured and not shipped — see below.** |
| **Causality ordering in narratives** | We group findings into an incident but never say what moved *first*. `SignalClass` already separates cause from consequence; only the time ordering is missing. "Latency rose four minutes after GC pause rose" is the sentence an operator wants. |
| **Per-workload signal classification** | `PeerSignalCatalog` states load-sensitivity globally. PHP-FPM memory **is** load-sensitive; .NET's is not. A per-stack claim, currently hard-coded for one stack. |
| **Second stack in the lab** (nginx cheapest, JVM most informative) | De-risks "works with any application", which today rests on one .NET workload. |
| **More channels** | `kube_pod_status_phase{phase="Pending"}` + `waiting_reason` (why a pod never started, not just that it did not); network bytes (a pod that stopped talking at unchanged CPU); in-flight requests (saturation in the USE sense); volume stats (blocked — Docker Desktop does not export them). |
| **Learned families** (`Gpt`, `Neuro`) | Lowest ROI now: they need training data a client does not have on day one, and the three statistical families already answer the questions clients ask. |
| **Service-to-service dependency graph** | The highest diagnostic value on the list and a separate product, not a feature. |

### Measured and NOT shipped — work-adjusted trend

The trend family follows traffic on a load-sensitive signal, because peer comparison divides such a signal
by a work metric and the trend family does not. On the lab, CPU drift inside a twenty-minute window
correlates with traffic drift at **+1.00**, and about 10% of windows drift past the trend gate on that alone.
Two repairs were proposed: divide by work, or fit `value = fixed + marginal × work` and ask what the value
would have been at constant work.

**The first attempt to measure it was vacuous, and that is worth recording on its own.** Scored against
`SyntheticCluster`, every arm returned zero false trends — the generator does not contain the phenomenon. Its
diurnal curve moves traffic about 3% across a twenty-minute window against a gate needing 10%. A comparison
in which no arm can score is not evidence about any arm, and it took a `+1.00` correlation measured on the
real cluster to notice. That is what forced `lab-window-healthy-12pod.csv` — sixty minutes, twelve replicas,
241 scrapes, 12 of 13 channels at 100% coverage — into the repository.

Scored on that recording (`AffineTrendOnLabFixtureDiagnostics`), across 9 windows × 12 replicas:

| Signal | Treatment | False trends | Injected regression seen in |
|---|---|---|---|
| CpuUsageRatio | raw (today) | 15 | 3 windows |
| CpuUsageRatio | divided by work | 11 | 4 windows |
| CpuUsageRatio | affine-adjusted | 11 | 4 windows |
| GcPauseRatio | any of the three | 5 / 5 / 6 | **0** |

**Not shipped, for three reasons in descending order of weight.**

1. **The effect does not clear its own noise.** 15 against 11 gives Poisson intervals of [7, 23] and [5, 17],
   which overlap — and worse than that suggests, because consecutive windows overlap by 75% at a 20-minute
   window and a 5-minute step, so the nine observations are not independent and the true interval is wider
   than the arithmetic says.
2. **The affine fit does not beat plain division on real data** — 11 to 11, 4 to 4. Its whole justification
   was the regime where the fixed cost is large, which the lab does not exhibit (fixed cost ≈ 0 there, which
   is what the +1.00 correlation means) and which only the generator does. Shipping the more complex form
   when the simpler one is indistinguishable on the only real data available is code for nothing.
3. **It is a second-order lever anyway.** What actually removed CPU as a false-positive source was the
   absolute floor: 16 of 22 incidents before, 2 of 12 after.

One point in favour, recorded so the next attempt starts from it: both corrections **improved** detection of
the injected regression, 3 windows to 4. They do not buy quiet with deafness — the effect is simply too small
to resolve at this sample size. **What would settle it is a longer recording**: an hour yields nine
overlapping windows, several hours would yield dozens of independent ones.

Separately, and not fixed by any treatment: a 50% rise in per-request GC pause is invisible to the trend
family in all three arms, because `GcPauseRatio` on this lab sits at 5.3e-6 and a relative gate has nothing
to divide. That is the same pattern as the heap and CPU, and it has the same remedy — an absolute floor,
which is already deployed.

### Shipped in this track

`FloorCalibrator` + hourly proposals (fit on one population, scored on a **held-out** one: 124 → 44
hand-reasoned → **29** calibrated false incidents/day at identical detection); `LevelShiftDetector` (a step is
invisible to Mann-Kendall — tau ≈ 0.51 regardless of height, so a 2.5× step scored p = 0.0695 and a **10× step
scored worse**, p = 0.0794); the peer dominance fix (a replica at Cliff's delta 1.00 and a 190% gap was vetoed
by two 10% ones — fixed at zero cost in false positives); the silent-pod check (a pod the cluster lists and
which reports nothing was invisible to every family); and `overfit anomaly-discover`, which proposes a metric
mapping from what a cluster actually exports — on the lab, 8 of 13 channels bound with no human input, 4
flagged for a decision, 1 declared blind.

---

## ✅ CLOSED — the six anomaly-guard defects found 2026-08-02

All six fixed by 2026-08-03, with tests. Moved to [ROADMAP-COMPLETED.md](ROADMAP-COMPLETED.md) — the section title said OPEN while every row in it was
struck through, which is the kind of drift this split exists to stop.

### ✅ FINISHED — the 24-hour measurement, 2026-08-04 07:59Z → 2026-08-05 07:59Z

**5 false incidents per day (95% Poisson 1–9), 292 cycles, 0 cycle failures, 0 cycles overlapping a
recorded build window.** Full reading — including why the heap channel's 108→0 is NOT clean credit for
the fix — in [ROADMAP-COMPLETED.md](ROADMAP-COMPLETED.md).

### ▶ THE MOMENT THE RUN ENDS — in this order

0. ✅ **DONE 2026-08-05.** Guard log saved to `Tests/bin/fp-run.log` — 334 KB, 2328 lines, 297 cycles,
   complete from the `07:58:52 anomaly guard starting:` line. Do this before anything that takes minutes;
   `kubectl logs` reads a container ring buffer and a restart empties it.

1. ✅ **DONE 2026-08-05 — the `Vector512` width question. NEGATIVE, written up in
   [ROADMAP-COMPLETED.md](ROADMAP-COMPLETED.md).** Short version: the planned lever does not exist —
   `DOTNET_PreferredVectorBitWidth=512` is *refused* on this CPU (128 moves the width, so the knob is live),
   so `Vector<T>` is 256-bit here by runtime policy and no configuration reaches it. Explicit 512-bit does
   beat 256-bit everywhere (Add 0.74–0.92, MulAdd 0.80–1.01, Dot 0.47–0.90), and the resulting `Simd.Dot`
   improvement was real (1.79x at 65536) — and was **reverted** because a path census found the kernel is on
   no forward path at all and 100% of its backward work is below the length where the change pays, while the
   guard branch cost 9–11% at the two commonest lengths. Kept: `SimdDotTests` (66 cases, the kernel had
   none), `Vector512WidthBenchmark`, `SimdDotAccumulatorBenchmark`.

   ▶ **Still open from it: `Simd.Avx512Threshold = 512` is unmeasured and its comment is false.** At 128
   floats — a quarter of the threshold, where the comment claims "AVX2 overhead is lower" — 512-bit won
   every operation. Before moving it, **census the lengths `Simd.Add` and `Simd.MulAdd` actually receive**;
   moving a threshold on microbenchmark points alone is precisely the mistake the `Dot` change turned out to
   be.

   ▶ **Still open from it: `ElseRefactorBenchmark`.** .NET 10 doubled the inlining budget and stopped
   `try`/`finally` blocking inlining, and `OVERFIT021`'s guidance rests on a measured 2.25x extraction
   penalty. If that penalty is gone, the rule is advising about a runtime that no longer exists. It was
   queued with the width question only because both are benchmarks sharing the machine-exclusion mutex.

2. ✅ **DONE 2026-08-05 — `analyse_run.py`, and the run passed.** 292 cycles in the window, 5 incidents,
   25 findings, 97% quiet, **0 cycle failures**, 2 recorded machine-load windows and **0 cycles overlapping
   one**. Rate **5/day (95% Poisson 1–9)** against 112/day on 2026-08-02.

   **Read it as two results, not one.** Non-heap went **4/day → 5/day, i.e. unchanged** (5 is inside the
   interval for 4) — that was the acceptance question after the floor repairs and it passes. Heap went
   **108/day → 0**, but that is *not* clean credit: the calibrator proposed a 0.729 MB floor against the
   0.641 MB configured one, i.e. peer heap gaps peaked at 583 KB and sat *just under* the threshold, where
   in the 47-hour-old baseline pods they sat well above it. **Do not quote 22x as the effect of the fix.**
   Of the 5: 2 in the first hour (cold start), 2 are the up and down slopes of the same 1440-minute diurnal
   curve on `RequestsPerSecond` (+10.6% at 19:03Z, −15.2% at 23:43Z — one phenomenon counted twice, and
   unavoidable on day one because `SeasonalBaseline` needs a previous day), 1 is `CpuUsageRatio` 9% above
   peers at an absolute 0.00017 (0.017% of a core).

   ▶ **Still open from it: `MinRelativeGap` on `CpuUsageRatio` passes 9%.** Two of the five incidents were
   that same signature on different pods. Cliff's delta is scale-free, so a rank-perfect separation at a
   physically meaningless magnitude is exactly what the gap gate exists to stop — it is set too low on this
   channel.
2. **`CheckpointedModule.FindNonDeterministic`** — scheduled explicitly by the user on 2026-08-04. Replace
   the recursion with an explicit `Stack<IModule>` plus a reference-identity `HashSet<IModule>`, so the
   traversal terminates on any graph and the `OVERFIT022` exemption disappears instead of being
   re-justified. `s.Add(s)` is legal today and kills the process with an uncatchable `StackOverflowException`.
   See the NASA section for why this is worth fixing when `checked` in `TensorShape` was not.
3. **Fix the `DECAY` line in `.claude/hourly_check.py`.** It currently asserts that a falling rate between
   halves is "cold heaps settling", which is an interpretation the comparison cannot support: pod warm-up,
   calibrator warm-up and position on the 1440-minute load curve all push the same way early in a run. The
   line should report the two halves and say plainly that it cannot attribute the difference, pointing at
   the per-hour breakdown in `analyse_run.py` — the tool that can. Deferred rather than edited mid-run
   because changing a reporting instrument while it is reporting makes the series it produced
   non-uniform, and the fix is a sentence, not a number.
4. **Audit `Audio/Mp3` end to end** — scheduled by the user on 2026-08-04. It is the last directory in the
   tree that parses an untrusted file and has never been audited as a whole; `overfit-find-bugs-game` is
   the right instrument. Two things to carry in rather than rediscover:

   - The defect class to look for is **not** "unchecked product" in general — that sweep returns 71 sites
     tree-wide and is 93% noise. It is specifically *a value read from the file that sizes an allocation,
     bounds a loop, or indexes a table*, which is where all five earlier true positives lived.
   - The file is in better shape than its lack of coverage suggests. A spot-check found `tindex` and
     `blockClass` bounded by the shape of their own switch, and found `big_values` already clamped —
     a 9-bit field doubled to 1022 and written into a 576-entry granule, with a comment naming the attack.
     Somebody has been here with the right instincts, so a hunt that returns nothing is a plausible result
     rather than a failed one, and should be reported as such instead of being padded.

5. **The eleven dead telemetry instruments — triaged 2026-08-04, execute after the run.** The diagnostics
   hunt's finding was closed by *deciding* about them, and the decision was recorded as a passing sentence
   ("the graph and module ones remain wireable"), which is how work disappears. It is now a list.

   **Wire — four instruments, one call site, off every hot path.** `ComputationGraph.Backward` runs once per
   training step and takes milliseconds; `RecordedOpCount` is already tracked. So `GraphCount`,
   `TapeOpCount`, `GraphBackwardDurationMs` and `GraphAllocatedBytes` all come from one place, at the cost
   of one `ValueStopwatch` (never `Stopwatch.StartNew` — banned here) and two
   `GC.GetAllocatedBytesForCurrentThread()` calls per step. That is nothing against a millisecond backward
   pass, and it is the pass whose cost dominates training.

   **Wire — one more, once per arena.** `NativeMemoryBytes` from `NativeBufferManaged`'s allocate and
   dispose. An `UpDownCounter` is exactly the right shape for it.

   **Measure before deciding — three.** `ModuleCount`, `ModuleDurationMs`, `ModuleAllocatedBytes` are
   per-layer per-forward, not per-step: a twelve-layer network pays twelve clock reads per pass. That is
   probably fine and "probably" is not the standard here, so it needs a benchmark first — the same test the
   kernel pair failed.

   **Delete — three, and this is a public API break, so it is a decision rather than a tidy-up.**
   `KernelCount` and `KernelDurationMs` were decided against on 2026-08-03: timing the zero-allocation hot
   path costs more than it measures. Keeping public fields for something we have decided never to do is
   worse than removing them. `AllocationBytes` goes with them — its name says nothing, and
   `overfit.tensor_storage.bytes.created` already carries that number and *is* fed. Per the versioning
   policy in `CHANGELOG.md` this is a MINOR bump.

   Net effect: `TelemetryInstrumentWiringTests`'s known-dead list goes from eleven to three, and the ratchet
   only permits shrinking, so it enforces the plan by construction.

6. The rest of the queue below.

### Queued behind the 24-hour run (ends 2026-08-03 18:20 UTC)

Nothing here changes what the running guard reports, which is why none of it justifies a fifth restart of a
measurement already restarted three times in one day. In order:

1. **Read the day's number** — false incidents per day on the warm-up-aware build from a cold start. Report
   it as the **pre-arming** figure: at +3 h the rate was 0.42/cycle, and 14 of 15 incidents were
   `GcGen2HeapBytes` against a configured floor three times below what the cluster does when healthy. The
   proposals in the guard's own log are what stage 4 copies into the ConfigMap.
2. ~~**Defects 1 and 2 below** — the identifier loop and the store-failure path.~~ **Done 2026-08-03.**
3. **Compile the three uncompiled changes and prove each one, not merely build it:**
   - `Tests/Diagnostics/TelemetryInstrumentWiringTests.cs` — the eleven known-dead instruments must match
     exactly; if the list and the scan disagree, the compiler will not say so, the assertion will.
   - `Tests/MeasurementExclusion.cs` and `Sources/Benchmark/Program.cs` — **the build proves nothing here.**
     `[assembly: TestFramework]` takes the type and assembly names as strings, so a typo does not fail the
     build, it silently fails to register the framework and the guard is simply absent. Verify by
     **experiment**: start a benchmark in the background, then run `dotnet test` and confirm it refuses with
     the mutex message; then stop the benchmark and confirm the suite runs again.
4. ~~**The seven `OVERFIT024` sites.**~~ **Done 2026-08-03.** All four `WhisperGgmlLoader` counts are
   bounded against the bytes remaining rather than a constant — no honest header asks for more data than it
   shipped, and a fixed ceiling would have to be either useless or wrong one day. `RepackedWeightsFile` and
   `LlamaLoRAAdapter` bound their record counts the same way. `ModelSerializer`'s ordering defect is the one
   worth remembering: **the rank guard existed and ran after the allocation it was guarding**, so a crafted
   file got its array and its read loop first and met the check afterwards. A guard in the wrong place is
   not a weaker guard, it is no guard.
5. ~~**The tensor defects.**~~ **Done 2026-08-03** — see the section above.
6. **Finish the machine-exclusion scheme** (below).
7. ~~**Re-run the detection matrix with a signal check.**~~ **Done 2026-08-03.** `SubjectWatchingSink` now
   requires the affected channel as well as the subject, and the affected channels are **derived from the
   injector** by diffing an injected clone against a clean one rather than declared beside it. Two results,
   and only one of them was the one being looked for:
   - **The signal check changed no row.** Strict and loose agree on all ten faults. The suspicion that the
     table was crediting coincidences was wrong, and both criteria are now reported side by side so the
     answer stays visible rather than becoming a claim in a changelog.
   - **The re-run changed two rows anyway: both CPU faults now read `no`**, confirming the 2026-08-02
     contradiction. The cause is item 8 below.

   Two harness defects were found on the way and are fixed: the derivation first reported "all thirteen
   channels moved by every fault", because the generator's deliberate `NaN` scrape gaps compare unequal to
   themselves under `!=` — so the strict criterion silently degenerated into the loose one and the first
   table looked plausible and meant nothing. `SyntheticClusterContractTests` now pins reproducibility, the
   gap rate and the absent CFS channel, since the derivation depends on all three.

8. ~~**The level-shift gate is calibrated on the wrong quantity.**~~ **Fixed 2026-08-03, and it was two
   defects stacked, the second hidden by the first.**

   *(a) Wrong quantity.* The gate's `MinAbsoluteChange` came from `_floors.MinAbsoluteTrendChange`, which
   `FloorCalibrator` accumulates from a Theil-Sen slope fitted to **each pod's own series**, while the gate
   judges a step in the **cross-pod common component** — a median over twelve replicas, roughly √N less
   scattered. On CPU the borrowed floor landed near 1.5× the signal's own level. Measured: the step detector
   reported the cluster-wide rise at delta 0.47, p = 0.00017, and the gate refused it at 0.39 against 0.81.

   The step gate now has `IAbsoluteFloorSource.MinAbsoluteLevelShift` and its own accumulator, fed by
   `LevelShiftDetector.StepSize` — **the same function the gate compares against**, so the floor and the
   gate cannot describe different quantities again. That is the third time this family has been handed
   another family's floor (peer-gap, then trend-change), which is why the fix is structural rather than a
   new constant.

   *(b) The calibrator learned from the window it was about to judge.* `Observe` ran at the top of the cycle
   and invalidates the proposal cache, so every gate below read a floor already containing that window. With
   the proposal set from the **maximum** times a 1.25 margin, the floor was never below 1.25× the quantity
   being gated — **the gate could not fire at all**, for any fault, once thirty samples existed. It survived
   only because the floor was calibrated on a *different* quantity, which is loose enough that the
   inequality did not always hold; fixing (a) made the self-reference exact and therefore visible. `Observe`
   now runs after the detectors, which is what the comment above it had claimed since it was written.

   **Measured, both sides, same population, ABAB in one script:** detection went from 8/10 faults to
   **10/10** — both CPU rows recovered, the cluster-wide leak from 15 min to 5, latency 3× from 5 min to 0 —
   and the **no-fault control opened one incident before and one after**. One seed over 69 cycles is not a
   false-positive rate, so the lab number is still owed before this is quoted as a noise figure.

   A "NO FAULT" control row was added to the matrix while doing this. It reads `no`, which is what makes
   every other row in that table mean something.

9. **The trend ablation in the matrix is confounded, and it is not only this harness's problem.**
   `FloorCalibrator` fits its samples with the **same `TrendOptions` instance** the guard was configured
   with, so setting `MinimumSamples` to 100 000 to silence the trend family also stops every calibrated
   absolute floor from being learned. That is why the step detector appears to "catch it alone".

   The harness caveat is documented in `DetectionMatrixDiagnostics`. The deeper question is whether one
   options object should control a detector and a calibrator at all: a customer who widens `MinimumSamples`
   to quieten trend findings would, today, silently switch off the floors that keep every other family
   quiet — a change whose effect is the opposite of what its name suggests. Worth a separate
   `CalibrationOptions`, or at least a named constant the calibrator uses instead. **Still open.**

10. **Re-run the lab measurement on the fixed build.** The 24-hour figure — 112 incidents/day pre-arming,
    96.4% of them one misconfigured `GcGen2HeapBytes` floor — was taken on a build whose step gate could not
    fire and whose calibrated floors contained the window they were gating. Both are now different. The
    synthetic control says the noise cost is nothing, but that is one seed over 69 cycles against a real
    number from 288, and the two are not interchangeable. **Nothing about noise should be quoted to a client
    from the synthetic run.**

11. ~~**`OVERFIT028` cannot see a product outside `new T[...]`.**~~ **Measured 2026-08-04 — do NOT widen the
    rule.** It scans array-creation syntax, so a property getter or a constructor is invisible to it, which
    is why five unchecked dimension products sat in directories the analyser reported clean. The obvious
    conclusion was to widen it to any multiplication flowing into a size. **The population says otherwise.**

    A tree-wide sweep for "a size-shaped variable assigned from a product" — `size`, `length`, `count`,
    `total`, `bytes`, `stride`, `offset`, `capacity` — returns **71 sites across 41 files**. All five true
    positives were in parsers of untrusted files; the rest take their operands from the caller's own code,
    where guarding buys nothing (the same argument that got `checked` reverted in `TensorShape` the same
    day). A widened rule would therefore fire ~5 useful times against ~66 benign ones.

    **A rule at a 93% false-positive rate does not make a codebase safer, it makes one more thing people
    suppress without reading.** That is the failure mode this project's whole analyzer ladder is designed
    to avoid, and widening OVERFIT028 would walk straight into it.

    **What to do instead: audit parsers by directory, not products by syntax.** The distinction that
    separated the five real defects from the sixty-six benign ones is *where the operands come from*, and no
    analyzer can see that. `Audio/Mp3` is the one untrusted-input parser no bug hunt has covered — a
    spot-check on 2026-08-04 found its two candidate products bounded by construction (`tindex` and
    `blockClass` are each 0..2 by the shape of their switch) and found it already carrying a fix of exactly
    this class: `big_values` is a 9-bit field doubled to 1022 and written into a 576-entry granule, clamped
    with a comment naming the attack. It is in better shape than expected, but it is still the only
    file-parsing directory never audited end to end.

12. **The suite is not fully deterministic.** *(New, 2026-08-03, observed rather than diagnosed.)* Across
    roughly a dozen full runs today, two failed — `PromptCacheReuseTests.ReusedPrefix_ProducesIdenticalLogits…`
    and `RealEstateFullCycleTests.Training_And_Prediction_Should_Work_EndToEnd` — each once, each passing on
    the next run and in isolation, both while the machine was busy with back-to-back builds. Four
    consecutive runs afterwards were clean, so it is rare.

    It matters more than its rate suggests: **every fix today was reported as "suite green"**, and a suite
    that fails a different test occasionally makes that evidence weaker than it sounded. Both candidates are
    end-to-end training/inference tests, the kind that go flaky under CPU contention rather than from a real
    defect — but that is a hypothesis, and the honest next step is to run them in a loop under load and
    either find the shared state or mark them `[Fact(Skip = …)]` with the reason, which `Tests/README.md`
    already says is the right home for a flaky timing test.

13. **Make a floor the guard disagrees with loud.** Falls out of the 24-hour run rather than from a review:
    for a full day the guard published the correct `GcGen2HeapBytes` floor every hour, to a log line, while
    running against a configured value three times too low and producing 108 incidents from it. Nobody read
    the log, and there is no reason they should have — a log line is not a channel anyone alerts on. A
    configured floor sitting far below the calibrated proposal should be a series
    (`overfit_guard_floor_disagreement{signal}`) or a startup warning naming both numbers.

### The machine-exclusion scheme, in full

This box is a development machine and a measurement instrument, and the two roles are incompatible while the
second one is active. Three levels, because the right response differs by **how long the activity lasts** and
by **whether its victim can repair itself afterwards**.

| Level | Mechanism | Applies to |
|---|---|---|
| **Exclusion** | one `Global\` mutex, refuse with exit code 2 | benchmark ↔ tests ↔ build |
| **Registration** | append `start..end` to a machine-load log | every build, test and benchmark run |
| **Subtraction** | the lab watcher reads the log and reports two figures | the 24-hour measurement |

**Why the lab does not join the mutex.** A benchmark lasts tens of minutes; the lab measurement lasts a day.
A lock that forbids testing for twenty-four hours will be worked around, and a guard that gets worked around
is worse than none because everyone believes it is there. The lab gets registration instead — and it can use
it, which is the asymmetry that decides this: **a benchmark cannot remove a contaminated sample after the
fact, but the lab measurement can**, because it counts timestamped cycles. Its report becomes two numbers,
raw and with contaminated cycles excluded, and if they agree the contamination did not matter — which is
itself a result, and one not available today.

**`dotnet build` joins the exclusion, and the objection to it was wrong.** The argument "you must build in
order to benchmark, so blocking builds deadlocks" does not hold: you build first and measure after, and
during a run nothing needs building. The real obstacle is different — **BenchmarkDotNet compiles a generated
project per benchmark while the run is in progress**, so a naive build guard deadlocks the benchmark against
itself rather than against a human.

The fix is ownership rather than exemption: the benchmark host sets `OVERFIT_MEASUREMENT_OWNER` to its own
process id before starting BenchmarkDotNet, child processes inherit the environment, and the build guard
allows a build that belongs to the run in progress while refusing every other. Implemented as a target in
`Directory.Build.props` — the same shape as the existing `BanJaggedFloatArrays` and
`BanMultipleTopLevelTypes` guards, which already fail builds from MSBuild.

**Two things this must get right or it will be ripped out within a week.** A guard that stops the whole
repository from building is a severe failure mode, so it needs the same named escape hatch as the suite
(`OVERFIT_ALLOW_CONCURRENT_MEASUREMENT=1`), and it must fail **open** on anything it cannot determine — an
unreadable mutex or a permissions error means "let the build through", because the cost of a wrongly blocked
build is higher than the cost of one contaminated sample.

**Scope, stated honestly.** This covers the three CPU and memory consumers the repository owns: build, test,
benchmark. It does not cover a browser, an IDE indexing pass, a container rebuild or anything else on the
machine. Those remain discipline, and the registration log is what makes their effect visible after the
fact rather than invisible.

**Scheduled: 1 and 2 are done when the 24-hour run ends** (started 2026-08-02 18:20 UTC, ends 2026-08-03
18:20 UTC). Not before, and the reason is not caution about the fixes. Both are latent — the identifier
collision needs `MaxOpenIncidents` to be reached, and the store-failure path needs a store that fails — so
neither changes what the running guard reports. Landing them means a rebuild, a redeploy and a fourth restart
of a measurement that has already been restarted three times today, which would cost the day's number to buy
nothing the day is measuring.

## Status snapshot

| Area | Status |
|------|--------|
| `InferenceEngine` zero-allocation hot path | ✅ Stable, 0 B/op verified |
| Linear / Conv / activation / pooling kernels | ✅ Stable |
| Autograd engine + `Parameter` + ownership cleanup | ✅ Stable (PR5) |
| `TrainingEngine` facade + Adam/AdamW/SGD | ✅ Stable |
| Evolutionary: GA + OpenAI-ES, parallel fitness | ✅ Stable, 0 B/op Ask/AskThenTell |
| ONNX import (linear topology, 14 ops) | ✅ Stable |
| ONNX import (DAG / ResNet skip connections) | ✅ Stable |
| GPT-2 inference (124M, KV-cache, 0 B/token) | ✅ Stable, parity vs PyTorch verified |
| Qwen2.5 / Llama / Mistral inference (GQA, RoPE, SwiGLU) | ✅ Stable for 0.5B/3B FP32/F16 |
| Native C# GGUF loader (F32/F16/BF16/Q8_0/Q4_K/Q6_K) | ✅ Loads `*.gguf` from Ollama/HF directly |
| Streaming token generation (`IAsyncEnumerable`) | ✅ Stable, with stop-tokens + cancellation |
| LoRA adapter (Enable/Disable, Save/Load) | ✅ Stable, zero-copy weight refs |
| **Quantized weight storage at inference time** | ✅ **Q8_0 + Q4_K_M decode & prefill paths done & parity-verified.** Decode ~24 tok/s Qwen-3B Q4_K_M, memory-bound (GEMV kernel at 82% of DRAM ceiling — `DecodeGemvRooflineBenchmark`), 1.13× behind llama.cpp. **Prefill ~299 tok/s (pp672), ~1.81× behind their AVX-512 build / 1.14× behind AVX2**, after the AVX-512 Q4_K/Q6_K prefill kernels. Both compute-side perf tracks CLOSED by measurement — see "✅ CLOSED — CPU PREFILL + DECODE PERF TRACK". |
| Mixture-of-Experts inference (Qwen-MoE, Mixtral-8x7B) | ✅ Coherent in pure C# (Q8_0 + Q4_K_M); verified "Paris" 2026-05-27 |
| Training: gradient checkpointing | ✅ `ComputationGraph.Checkpoint` + `CheckpointedModule` — 24× live-activation cut on 12L GPT-1 |
| Training: data parallelism (N replicas) | ✅ `DataParallelTrainer` / `DataParallelSession` + thread-budget fix — ~6× throughput (24 workers) |
| **OCR: CRNN + CTC** | ✅ `Crnn` facade + `CtcLoss` + `CtcDecoder` (greedy / beam / LM-rescored) + `NGramCtcLanguageModel`; reads synthetic digits + lexicon words |
| **CNN training perf (BN2D adaptive parallel)** | ✅ Size-gated `Parallel.For(0, C)` in `BatchNorm2DBackward` — **CIFAR backward −31.1% (BN2D −72.8%)**, MNIST unchanged. Adaptive crossover at N·C·hw > 200K |
| **Sentence embeddings (BERT encoder)** | ✅ `WordPieceTokenizer` + `BertEncoder` (bidirectional, post-LN) + native `BertSafetensorsLoader` + `SentenceEmbedder` facade — loads **all-MiniLM-L6-v2 / BGE-small-en-v1.5 / E5-small-v2** from raw HF safetensors, no Python; cosine 1.0 / 0.999999 / 1.0 vs HF reference. CLS pool + query/passage prefix support. |
| **ReAct agent loop** | ✅ `ReActAgent` driver over `ChatSession` + `ToolCallConstraint`; auto-registers a synthetic `finish({answer:...})` tool. Loop verified by 6 unit tests via the testable `RunLoop` hook; e2e [LongFact] needs a 7B+ instruction-tuned model (Q4-3B too weak under constrained decoding). |
| **Critic loop + circuit breaker + summarising memory** | ✅ All four LangGraph agentic primitives shipped: `CriticLoop` (generate→critique→revise), `CircuitBreaker` (generic capped-loop), `SummarizingChatSession` (auto-compact long convos), + `ChatSession.AddUser`/`AddAssistant` history seeding. 12 unit tests. |
| GPU backend | ❌ Not started |

---

## 📋 SIMD and .NET 10 articles — reviewed 2026-08-04, and what is actually usable

Five links reviewed: the .NET 10 performance post, three SIMD tutorials (xoofx, meriffa, developersvoice)
and the `CBGonzalez/SIMDPerformance` benchmark repo.

**The four SIMD articles offer this codebase nothing, and the reason is a baseline mismatch rather than a
quality problem.** Every speedup they quote — 5x, 8x, 10x — is measured against a *scalar* loop. This
tree's baseline is already `TensorPrimitives` and hand-written intrinsics: 855 SIMD call sites across 73
files. A multiplier against the starting line says nothing about the distance still ahead.

Technique by technique, against what is already here:

| Article's advice | State here |
|---|---|
| Four vector blocks per iteration to amortise loop overhead | `LinearKernels.ForwardInputMajorVector4` already does exactly this, on `Vector<float>`, with a width-based fallback |
| `Vector<T>` on integers can be **2.8x slower** where instructions are emulated | `Vector<T>` appears in six files, all on `float`. Does not apply |
| `ref` cursors instead of spans to kill bounds checks | .NET 10 improved bounds-check elimination in four separate ways (Log2 results, length-comparison assertions, `switch` facts, immutable string length). This advice is dated 2023 and is now partly the JIT's job |
| Tail handling, alignment | Standard practice here |

**And this project has already measured and reverted several of the exact techniques these articles sell:**
register blocking in direct convolution, K-blocking plus A-packing in the im2col GEMM, Winograd F(2,3)
(parity-correct at cos 1.0 and **+79% slower**), and the AVX-512 decode port. The headline finding runs
directly against the tutorials' thesis: **`TensorPrimitives` beat a hand-written micro-kernel here.**

The structural reason is worth restating because it decides which future article is worth reading: **decode
is DRAM-bandwidth-bound, not FLOP-bound.** Wider registers do not help a loop that is waiting on memory,
which is why the AVX-512 port was a negative result rather than a disappointing positive one. Any piece
promising N× from a wider vector is answering a question this hot path does not ask.

### What IS usable — from the .NET 10 post, and it is all automatic

The SDK here is **10.0.110**, so every JIT improvement below is already in effect and needs no source
change. Two of them touch measurements this repository relies on:

1. **`try`/`finally` no longer blocks inlining** (dotnet/runtime#112968, #113023). This matters more here
   than almost anywhere: `using` compiles to `try`/`finally`, and nearly every helper touching
   `PooledBuffer<T>` or `TensorStorage<T>` uses one. Those methods were previously **not inlinable at all**.
2. **Inlining budget more than doubled** (#114191, #118641). This bears directly on the `OVERFIT021`
   measurement — *"extracting a method costs 2.25x when the JIT does not inline it"* — which is the number
   the `else`-ban guidance rests on.

**Action: re-run `ElseRefactorBenchmark` after the 24-hour measurement ends.** If the extraction penalty
has gone, the rule's advice describes a runtime that no longer exists. Not run now: a benchmark takes the
machine-measurement mutex and loads this box for minutes, and the lab runs on this box.

Also relevant, lower priority: **escape analysis now stack-allocates arrays, spans and delegates**
(#104906, #112250, #113977, #116124, #115172). `PooledBuffer<T>` is for large buffers and stays — but some
small `new T[n]` sites that were "fixed" by pooling may not have needed fixing.

**One open question the articles surfaced indirectly, and it is a question rather than a finding.** This
tree vectorises at two different widths depending on which file you land in: `Intrinsics/Simd.cs` dispatches
explicitly to `Vector512` above a length threshold, while `Kernels/LinearKernels.cs` uses width-agnostic
`Vector<float>`, whose width the runtime chooses. .NET does not give `Vector<T>` 512-bit width by default
even on capable hardware. If that holds here, the linear forward path runs at half the width the
hand-written path uses — on the *training* side, which is compute-bound, unlike decode.

**Do not act on that without measuring, and this box has already argued against it twice:** the AVX-512
decode port was reverted as a regression, and the banded 512 prefill measured 1.15–1.17× on `ffn_gate_up`
but only on `UseOutputBlocking`, which is off by default. So the honest experiment is one benchmark of
`LinearKernels` forward with `DOTNET_PreferredVectorBitWidth=512` against the default — an environment
variable, not a rewrite — and it belongs after the 24-hour run with the rest.

## 📋 NASA Power of 10 — audited against this codebase (2026-08-02)

Prompted by a day in which four `overfit-find-bugs-game` hunts produced eighteen defects. Mapping them onto
Holzmann's ten rules turned out to be diagnostic rather than decorative, so the audit is recorded here with a
verdict per rule — including the ones deliberately **not** adopted, so nobody re-opens them.

| # | Rule | Verdict here |
|---|---|---|
| 1 | No `goto`, no recursion | **Adopted.** `OVERFIT022` is an error globally, with the `#pragma BOUND:` contract for the rare justified site. |
| 2 | Every loop has a statically provable bound | **Half adopted — see below.** `OVERFIT023` bans `while(true)`, which is the syntactic half. The semantic half is missing and cost us three defects today. |
| 3 | No dynamic allocation after initialisation | **Adopted in spirit, which is the right form for a managed runtime.** `InferenceEngine` takes caller-owned buffers, `PooledBuffer` replaces raw pooling, `OVERFIT001` flags allocation in per-call code. A literal ban is not expressible in C# and would buy nothing the hot-path rules do not. |
| 4 | ≤ 60 lines per function | **Rejected, on a measurement.** Splitting a hot method cost **2.25×** where the JIT declined to inline it. Do not re-open without a benchmark that says otherwise. |
| 5 | ≥ 2 assertions per function | **Rejected as a mechanical rule, adopted as a review question.** Enforcing a count produces decorative assertions, which are worse than none because they train readers to skim them. But three of today's findings are missing assertions on invariants an author assumed — so it belongs on the hunt checklist, where a human asks "what does this code assume and never checks". |
| 6 | Smallest possible variable scope | **Not pursued.** The language and the existing style rules already deliver most of it, and the residual risk in this codebase is low. |
| 7 | Check every returned status; validate every argument | **Half adopted — see below.** Argument validation is near-universal; return-status checking is not, and that gap produced a defect in code written this same morning. |
| 8 | Restricted preprocessor | **Not applicable** in the C# sense. The nearest analogue — a `#pragma warning disable` that hides logic rather than documenting a bound — is already governed by the per-site convention of naming the justification. |
| 9 | One level of pointer dereference, no function pointers | **Partly not applicable.** The kernels use raw pointers deliberately and measurably. The rule's actual purpose — keeping the program analysable — is served here by the Native-AOT ban on reflection, which is enforced at every build. |
| 10 | Zero warnings, daily static analysis | **Adopted most fully of the ten.** `TreatWarningsAsErrors` on the AOT guard, `RS0030`, `CS4014`, `NU1901`–`NU1904`, the OVERFIT analyser family and two MSBuild structural guards. |

**The observation that makes this worth writing down.** Today's eighteen defects land almost entirely on
rules **2, 5 and 7** — precisely the three adopted *halfway*. Rule 10, adopted completely, produced none,
because everything it can catch is already caught at every build. Half a rule does not buy half the
protection; it buys the illusion of the whole one.

### Audit every `#pragma BOUND:` — is the bound proved or assumed?

`OVERFIT022` does not forbid recursion; it demands that an exemption **name its bound**. The audit asks one
question of every site: does the stated bound follow from a constant or from validation at entry, or does it
follow from how somebody expects the API to be used?

**Prompted by getting it wrong the same day the rule was being praised.** Two exemptions were added on
2026-08-02 and they are not of equal quality:

```csharp
// FastRandomForest.BuildRecursive — a proof.
// depth >= _maxDepth returns above, and _maxDepth <= MaxAllowedDepth (64) is enforced in the constructor.

// CheckpointedModule.FindNonDeterministic — a hope.
// "recursion follows Sequential nesting, which a caller builds explicitly and is a handful of levels at most"
```

The second names no bound at all. It describes a habit. Nothing prevents a caller nesting `Sequential`
a thousand deep, and **C# has no tail-call optimisation to fall back on**: the runtime supports the `tail.`
IL prefix, the C# compiler never emits it, and the JIT eliminates such calls only sometimes — which is worse
than never, because code then passes in Release on one platform and dies on another.

The fix where a bound cannot be proved is not a bigger number. It is an **explicit stack on the heap**: the
depth becomes a `Count` that can be checked and reported, instead of a stack-frame count that can only be
exceeded.

**Audit result, 2026-08-02: 58 exemption sites in the tree; 9 of the 10 recursion exemptions are proved.**

| Site | Stated bound | Verdict |
|---|---|---|
| `FastRandomForest.BuildRecursive` | `_maxDepth <= 64`, validated in the constructor | proved |
| `JsonSchemaCompiler` (2 sites) | `MaxSchemaDepth` = 32, checked on entry, throws catchably | proved |
| `GgufReader` | `MaxValueNestingDepth` = 8, checked on entry, throws catchably | proved |
| `HuggingFaceBpeTokenizer` (2 sites) | `MaxPreTokenizerDepth`, checked on entry, throws catchably | proved |
| `TokenSampler` sift-down | descends one heap level per call, `<= log2(n)` | proved, structurally |
| `PartialSort` quicksort | recurses into the smaller half only, `<= log2(n)` | proved, structurally |
| `CheckpointedModule.FindNonDeterministic` | "a handful of levels at most" | **assumed** |

The one lapse was written the same afternoon the rule was being praised, which is the useful part of the
result: the discipline holds across code written by many hands over months, and broke in the code written
while admiring it.

**The other half of the audit, which was not the question but is the larger risk.** A stack overflow is the
product of depth and frame size, and `OVERFIT026` guards the second. No method in the tree carries both an
`OVERFIT022` and an `OVERFIT026` exemption — so there is no deep recursion with a fat frame anywhere, which
is a real protection nobody had named. But four `stackalloc` sites are far over budget and say so themselves:

```
Whisper/WhisperKernels.cs       32 KB   "the largest stack frame left in the library, 64x the budget"
TrainableLlamaModel.cs          16 KB   32x
Conv2DKernels.cs (2 sites)       8 KB   16x
SentenceEmbedder.cs (2 sites)    4 KB   8x
```

None is recursive, so none overflows on its own. The Whisper comment names the real concern: on a pool thread
with a 1 MB stack, a 32 KB frame deep in a call chain is a different proposition from the same frame on the
main thread.

### ▶ OPEN — the one assumed bound is worse than "assumed" (read 2026-08-04, not yet fixed)

`CheckpointedModule.FindNonDeterministic` was recorded above as the single exemption whose bound is a hope
rather than a proof. Reading it settles the question, and the answer is not the one the audit expected:
**the stated bound describes the wrong quantity.**

The exemption says *"recursion follows Sequential nesting, which a caller builds explicitly and is a handful
of levels at most"*. But `Sequential.Add(IModule)` is public, mutable after construction, and performs no
cycle check. So:

```csharp
var s = new Sequential();
s.Add(s);                    // legal, no check anywhere
new CheckpointedModule(s);   // FindNonDeterministic recurses forever
```

That is **zero levels of nesting and unbounded recursion** — the depth the comment reasons about is not the
thing that fails. And the failure is the worst kind available: `StackOverflowException` cannot be caught,
so the process dies rather than reporting anything. A shared sub-module added to two branches is fine (a
DAG terminates); only a genuine cycle does this.

**Why this is worth fixing even though the input is caller-controlled.** The same argument was used on
2026-08-04 to *revert* `checked` in `TensorShape` — the dimensions come from the caller's own code, so
guarding them buys nothing. The difference is the consequence, not the source: an unchecked product yields
a wrong number that a test can catch, and this yields a process kill that nothing can. Cost asymmetry, not
input provenance, is what decides.

**The fix is to remove the recursion, not to bound it.** An explicit `Stack<IModule>` plus a
reference-identity `HashSet<IModule>` of visited modules terminates on *any* graph — cyclic, deep or
otherwise — so no depth limit has to be chosen and the `#pragma warning disable OVERFIT022` disappears
rather than being re-justified. That is also what this section already prescribes for an unprovable bound:
"the depth becomes a `Count` that can be checked and reported".

**Not applied yet, deliberately**: found while a 24-hour measurement was running on this box, and a fix that
cannot be built and tested is worse than one not written. First thing after the run.

### `EnsureSufficientExecutionStack()` — and why it is not being added anywhere yet

`RuntimeHelpers.EnsureSufficientExecutionStack()` throws `InsufficientExecutionStackException`, which **is**
catchable, when the remaining stack falls below a probe threshold. It converts an uncatchable process kill
into a reportable error, and it costs a comparison.

**It has nowhere to go in this codebase today, and adding it anyway would be decoration.** Every recursion
here is either counted with an explicit depth limit that already throws a catchable
`OverfitFormatException` — the three untrusted-input sites — or structurally logarithmic. A probe adds a
second, weaker guarantee behind a stronger one.

It is recorded as **the documented mitigation for a case that does not exist yet**: a recursion over a
structure whose depth comes from user data and cannot be counted cheaply. If such a site is ever accepted,
the probe at method entry is the price of accepting it.

### Do we need a rule for "uncatchable exceptions"?

Asked, and the answer is no — because the set is enumerable and three quarters of it is already covered.

| Process-killing path | Covered by |
|---|---|
| `StackOverflowException` from unbounded recursion or loops | `OVERFIT022`, `OVERFIT023` |
| An exception escaping a finalizer | `OVERFIT012` (`FinalizerAnalyzer`) |
| An exception escaping an `async void` | `OVERFIT027` |
| `AccessViolationException` from pointer arithmetic in `unsafe` code | **nothing** |

The fourth is the gap, and it matters here because the kernels use raw pointers deliberately. But a rule
cannot be written for it directly: proving pointer safety in general is not something an analyser does. What
*can* be checked is the thing that actually produces these faults — **a length or an offset taken from
outside and used in address arithmetic without validation** — which is the same rule as `OVERFIT024` above,
applied to pointers instead of to `new T[n]`.

The general form the question suggests — flag every throw that might not be catchable — would find nothing,
because the danger is never at a throw site. Nothing throws `StackOverflowException`; it happens where a
bound is missing. A rule that inspected throws would be looking in the one place the defect never is.

**How to establish the real limit, when one is needed.** Two ways, and neither is a constant of the language:

- **Exactly, per build**: the frame size is the `sub rsp, N` in the method prologue plus the return address
  and any pushed registers. BenchmarkDotNet's `--disasm` already gives this. Depth ≈ usable stack / frame
  size — roughly 1 MB by default for the main and thread-pool threads, so a 100-byte frame gives order
  10 000 calls and a frame carrying a 4 KB `stackalloc` gives 256. This is why `OVERFIT026` caps `stackalloc`
  at 512 bytes: a `stackalloc` inside a recursive method changes the answer by two orders of magnitude.
- **Empirically, in a sacrificial child process**: recurse with a counter until it dies. It must be a child
  process because **`StackOverflowException` cannot be caught in .NET** — the process is terminated, which is
  also why this failure mode deserves a rule rather than a `try`.

Both numbers move with the JIT, the platform and Debug versus Release, which is the argument for demanding a
bound rather than an estimate.

**Where a recursion genuinely cannot be removed**, `RuntimeHelpers.EnsureSufficientExecutionStack()` throws
`InsufficientExecutionStackException` — which *is* catchable — before the stack is actually exhausted. That
is the honest mitigation: it converts an uncatchable process kill into a reportable error.

### Worth building — 7 first, then 2

**Rule 7, mechanically.** The .NET analysers already ship most of it (`CA1806` and friends: do not ignore a
method's result). This is the cheap one, and it is not theoretical: `FileIncidentStore.LastError` is set on
every failure and read by nothing — literally this rule, broken in code written the same morning, in a class
whose whole job is to report that it could not write. **Gate it the way the `else` sweep was gated**: enable
as `suggestion`, count the sites, read a sample, then promote per directory.

**Rule 2, semantically — `OVERFIT024`, and the sweep has been run.** A value read from `BinaryReader.Read*`
or a JSON parser must not size an allocation or bound a loop without passing through a validator.

**Measured before writing the analyser, which is the only reason it is worth writing.** A text approximation
of the rule was run over `Sources/` on 2026-08-02 — deliberately crude, and its own error rate is part of the
result:

| | |
|---|---|
| read-then-use sites in the whole tree | **19** |
| already carrying a guard the scan recognised | 8 |
| flagged | 11 |
| **real defects, after reading each one** | **7** |
| false positives | 4 |

All four false positives are the same shape — `if (length != expectedSize) throw` — which the scan missed
because it looked for `<` and `>`. A Roslyn implementation sees that with ordinary local data flow, so the
real false-positive rate is close to zero. Nineteen candidates across the tree is not a rule that will bury
anyone, and the pattern that satisfies it (`Require…`, or equality against a known size) is already the house
style in the loaders that get it right.

**The seven, which do not need the rule in order to be fixed:**

```
WhisperGgmlLoader.cs:60    nTokens -> new string[nTokens]
WhisperGgmlLoader.cs:64    len     -> ReadBytes(len)
WhisperGgmlLoader.cs:73    nDims   -> new int[nDims]  (twice)
WhisperGgmlLoader.cs:74    nameLen -> ReadBytes(nameLen)
RepackedWeightsFile.cs:157 nameLen -> ReadBytes(nameLen)      (also found by the bug hunt)
LlamaLoRAAdapter.cs:215    count   -> loop bound, unchecked
ModelSerializer.cs:59      rank    -> see below
```

**Four of the seven are fifteen consecutive lines of `WhisperGgmlLoader`** — an entire ggml reader with no
bound on anything.

**`ModelSerializer` is the interesting one, and no review found it.** The guard exists; it is in the wrong
place:

```csharp
var rank = br.ReadInt32();
var fileShape = new int[rank];            // allocated
for (var i = 0; i < rank; i++) { … }      // and filled
if (rank != view.Rank) { throw … }        // and only now checked
```

A file declaring `rank = 2_000_000_000` allocates and reads before anything asks whether the number is
sane. This is an **ordering** defect rather than a missing one, which is exactly what a human eye skips: the
check is visible a few lines below and the reader marks it done. A rule that follows the value from its read
to its first use cannot make that mistake.

**What the sweep cannot settle, and the rule inherits.** Local data flow finds "read → use" inside one
method; it cannot see a bound established two calls away. Every one of the nineteen sites happened to be
single-method, so the question did not arise here — but it will, and a site whose guard lives in the caller
will read as a violation. That is the residual false-positive risk, and it is smaller than the one measured
away above.

## Agentic / interop / vision backlog (2026-06-21)

Deferred ideas captured while shipping the XGBoost tabular predictor; ranked, on-moat, all build on existing
primitives (autograd, agentic stack, MEAI adapter, the XGBoost predictor).

1. **Vision-XAI — Grad-CAM + saliency maps.** Extends the interpretability-hooks initiative (today: LLM activation
   capture + logit lens) to CNN/ONNX explainability: Grad-CAM = gradient of the output w.r.t. conv feature maps,
   saliency = gradient w.r.t. the input. We already have the autograd to compute both — pure-managed, in-process,
   no Python. On-moat ("pure-managed = inspectable"); strong for regulated/medical/audit ("explainable predictions,
   zero data egress"). Origin: a breast-ultrasound segmentation + XAI workshop.
2. **SkillOpt loop — text-space self-improving agent skills.** A weight-free adaptation loop: an optimizer model turns
   scored rollouts into bounded add/delete/replace edits on a single skill document, accepted only when a held-out
   validation score strictly improves (selection gate + textual "learning-rate" budget + rejected-edit buffer). Rides
   on primitives we already have (local generation, scoring, guaranteed-JSON edits, the ReAct/ChatSession stack) — it
   is orchestration, not new kernels. Complements QLoRA (weight-space) with a fully on-prem text-space moat. Caveat:
   edit/scoring quality wants a 7B+ local optimizer model. Origin: arXiv:2605.23904 (SkillOpt, Microsoft, 2026).

   **★ SIZED 2026-07-18 — one real bug found and FIXED; the population question is still open.**
   Ran an equal-budget A/B (150 model calls per arm, shared ON-only scoring so neither arm pays the other's
   implementation overhead): hill-climbing (today's `SkillOptimizer`, population 1) vs population(4) + tournament
   + elitism. Qwen2.5-0.5B as the runner, Qwen2.5-3B as the editor, capitals task, deterministic grader
   (correct AND ≤ 4 words).
   - **First run: 0 % vs 0 %, both instructions unchanged — a broken experiment, not a tie.** Diagnosis:
     the fitness landscape was fine (seed → *"The capital of France is Paris."* correctly fails; a good
     instruction → *"Paris"* passes), but the **mutation operator was the bottleneck**. A 0.5B editor echoed the
     user's *question* back instead of writing an instruction; the 3B editor wrote valid English but in the
     **wrong direction**: *"Answer in the format 'The [subject] is [answer]'"* — i.e. it codified the failing
     answer as the desired format.
   - **Root cause: `CaseFailure` carried (Prompt, Output) but no REASON**, so the editor saw an answer that reads
     perfectly fine and had zero gradient information. Adding one sentence flipped it instantly:
     *"Answer the question in at most 4 words."* **FIXED** — `CaseFailure.Reason` (optional ⇒ non-breaking),
     populated from the failed `GradeCheck` id + note, surfaced to the editor as `-> WHY WRONG:`, guarded by
     `CaseFailureReasonTests`.
   - **Re-run after the fix: 0 % → 75 % for BOTH arms.** The fix is validated end-to-end; **the population-vs-
     hill-climbing question is NOT answered** — the task saturates at 75 % within ~13 calls (the 4th validation
     fact is simply unknown to a 0.5B, a model-capability ceiling no prompt can lift), leaving no headroom for a
     search-strategy difference to appear. Suggestive but unmeasured: hill-climbing burned 137 of 150 calls after
     its single accept, the editor having stopped producing novel candidates.
   - **SIZING #2 (2026-07-18) — task WITH headroom. Plain population LOSES. ⛔ do not build GA-on-prompts.**
     Rebuilt the experiment to remove sizing #1's ceiling: 3B runner (facts no longer the limiter), FOUR
     independent format constraints (correct · lowercase · no punctuation · ≤3 words), **continuous fitness**
     (fraction of constraints met ⇒ a real multi-step gradient), 6 train + 6 val, equal 200-call budget, failures
     carrying reasons.
     | arm | result | budget behaviour |
     |---|---|---|
     | hill-climbing (pop 1) | 20.8 % → 79.2 % (call 19) → **87.5 %** (call 32) | 2 accepts, then 13 straight rejects |
     | population(4) + tournament + elitism | **87.5 %** | 45 calls just to seed; **12 generations, zero improvement** |

     The climb was genuinely multi-step this time, so the landscape *could* have discriminated — and population
     still did not win, while being **worse per call** (87.5 % at call 32 vs 45 calls merely to seed).
     **Why: the whole population converged to identical fitness (88 % × 4).** Tournament + elitism with no
     diversity pressure filled every slot with variants fitness cannot tell apart — and diversity was the entire
     justification for going population-shaped. Without it, it is 4× redundant hill-climbing at 4× the cost.
     (Measured: fitness identical. Not measured: whether the *texts* were identical — plateau vs true clone
     collapse. The practical conclusion is the same.)
     **Implication:** plain GA on prompt text is not worth building. The only variant left standing is
     **MAP-Elites**, whose cells are keyed by a BEHAVIOUR descriptor rather than fitness, so diversity is
     structural — exactly the failure mode observed here. But that is now the third hypothesis in this family, so
     **size it first**: show that prompt behaviour descriptors (e.g. verbosity × accuracy) actually have
     resolution, otherwise MAP-Elites degenerates the same way. Caveats: n=1 task, 1 seed, population 4, mutation
     only (no crossover).
   - **Do NOT generify the Evolutionary `float` interfaces to `T` for this.** `Mutate(ReadOnlySpan<T>, Span<T>)`
     means "perturb elementwise into a preallocated same-size buffer"; a text mutation is `string → string`,
     variable length, one expensive LLM call — the type parameter is not the obstacle, the contract shape is
     (and Gaussian/SBX arithmetic on text is meaningless regardless of `T`). `ISelectionOperator` (indices) and
     `IFitnessShaper` (fitness values) are already genome-agnostic and reusable as-is. The one place a generic
     would genuinely pay is `IEliteArchive<TGenome>` — storage, not math — and only once duplication actually
     appears. Write a parallel text path (~150 lines) instead of refactoring a shipped public API.
3. **`overfit score` as a server endpoint.** The XGBoost predictor ships as a library + `overfit score` CLI; expose it
   over the OpenAI-compatible server (or a small dedicated route) for tabular scoring as a service. Small, reuses the
   hardened server + zero-alloc predictor.
4. **Semantic Kernel recipe (docs, not code).** SK can already consume Overfit two ways — point SK's OpenAI connector
   at `overfit serve`, or bridge our `Microsoft.Extensions.AI` `IChatClient` (recent SK builds on MEAI). A dedicated SK
   connector is **not** worth building (redundant with the MEAI adapter, which is the forward-looking interop point);
   the only move is a short "Use Overfit from Semantic Kernel" sample showing both paths.

---

## Audio / TTS backlog (ROI-ranked, 2026-06-08)

Pure-.NET voice stack (Orpheus 3B + SNAC + voice cloning) is functional; first-word garble **root-caused & fixed**
(prompt was missing the canonical Orpheus control/priming tokens — start_of_human + BOS + end_of_text/end_of_human/
start_of_ai/start_of_speech; `OrpheusPrompt.BuildPromptTokens`). Validated objectively by transcribing output with our
own Whisper. Remaining, by ROI (value ÷ effort):

**🟢 Quick wins (small effort, every clip benefits)**
1. **Trailing babble — ✅ ROOT-CAUSED 2026-06-08: it's the SAMPLING TEMPERATURE, not the stop token.** At temp 0.45
   the model occasionally rambles ~3 s of garbled audio after the sentence before emitting the text-eos (128009);
   **greedy (temp 0) ends cleanly** (Whisper: clip ends exactly at the last word, 1274 vs 1561 codes; "AI" also
   cleaner). Mitigation: use low/greedy temp for the clone, or try temp ~0.2-0.3 if greedy tempo feels slow. (Added a
   harmless safety-net: `GenerateCachedSampled` now also stops on a `secondaryEosTokenId` = end_of_speech 128258 — but
   the model emits 128009 *after* the babble, so it wasn't the fix.) The earlier "greedy too slow" note was likely
   confounded by the (now-fixed) prompt bug — re-judge greedy tempo.
2. **Acronym lexicon — ✅ DONE + EMPIRICALLY VALIDATED 2026-06-09.** `TtsTextNormalizer` spells acronyms as spaced
   capitals ("AI"→"A I", "CPU"→"C P U", +12 new: http/https/usb/ssd/hdd/dns/vm/iot/vr/ceo/cto/faq) so Orpheus says the
   letter names. Locked in with 11 unit tests incl. substring-safety ("brain" ≠ "br A I n"). **Closed the loop the
   user's way** (`OrpheusAcronymPronunciationE2ETests` [LongFact]): synth "The AI uses the CPU and the GPU through one
   API." → our own Whisper transcribed it back **verbatim, 4/4 acronyms recovered** → the spaced-capitals convention
   provably works. Suite 1248/0.

**🟡 Structural (bigger effort, high value)**
3. **Merge LoRA → fast inference engine** — clone synth runs on the trainable graph (`VoiceCloneTrainer.Generate` →
   `TrainableLlamaModel`), preset runs on `CachedLlamaInferenceEngine` (zero-alloc/SIMD/Q4_K). **MEASURED GAP 2026-06-08
   (same sentence, Orpheus-3B Q4): clone 6.1 tok/s (490 tok / 80.3 s) vs preset 12.9 tok/s (533 tok / 41.5 s) = 2.1×.**
   Merge the adapter delta into the base weights → run clone on the fast engine ≈ preset speed (**~2.1×**). Merge math
   (grounded): LoRA `Apply(x)=(x·A)·B` (NO alpha/scale), A `[in×rank]`, B `[rank×out]` →
   `W_merged[o,i] = W_base[o,i] + Σ_r A[i,r]·B[r,o]`. Per projection: dequant base (per-head for q/k/v/o, resident for
   gate/up/down) → add delta → requant (Q8) → rebuild `LayerWeightBuffers`; also swap in the trained RMSNorm gains
   (`_ln1Gamma`/`_ln2Gamma`). Build a fresh engine via `CreateFromBuffers`, reuse base embed/lm_head/final-norm. Validate:
   merged-engine output must match the trainable-model output (same words via Whisper) AND hit ~preset tok/s. LM-head
   LoRA is off by default → no merge there. Audio-vocab restriction was training-only → irrelevant at merged inference.
   **✅ DONE 2026-06-08 — merge correct, 2.1× speed, COHERENT with SAMPLING.** Built `TrainableLlamaModel.BuildMergedEngine`
   (+ `VoiceCloneTrainer.BuildMergedEngine`, demo `--fast`, `AudioVocabConstraint`). **Merged clone decodes at ~12 tok/s
   (2.1× the trainable graph's 6.1) and is COHERENT with sampling** (temp 0.6 → Whisper: "The history of computing began
   long before the invasion of the digital computer."). **Merge proven CORRECT** via diff diagnostic
   (`MergeDivergenceTests`): merged-vs-trainable final hidden **cos 0.99994**, and the first **6 greedy tokens are
   bit-identical**. The earlier "garbage" was **greedy brittleness**, NOT a merge bug: tiny numerical drift between the
   fast engine (Q8 requant + reassociated SIMD attention + batched prefill) and the trainable `ProjVec` decode flips the
   hard argmax at ~token 6 → wrong SNAC code → derail. **Sampling avoids the hard flip → use temp 0.6 (NOT greedy) with
   `--fast`.** Caveat: shares the base embed/lmhead/backing → keep the trainer alive while the merged engine lives (the
   2nd `BuildMergedEngine` in one process double-disposes shared weights). Trailing-babble (ROADMAP #1) still applies to
   the merged+sampling path. `--fast` now usable for the clone speed-up.

**🟠 Medium-term**
4. **Real-time on CPU** — smaller same-arch ~0.5B LM (RTF 2-3) or port **Kokoro 82M** (Apache, StyleTTS2, ~1-2 wk,
   different arch) behind `ITextToSpeechEngine`. Product/moat decision (real-time is partly private — see
   `project-moat-public-private`).
5. **Signal-domain watermark** — inaudible waveform mark + detector (survives re-encode); current watermark is
   metadata-only. Compliance/IP, near launch.

**⚪ Low ROI / skip**
6. Align `OrpheusTrainingSequence` to the canonical prompt — cosmetic (inference works regardless; base dominates).
7. Zero-alloc + SIMD SNAC decode — a "zero-alloc" banner, NOT a speed win (SNAC is cheap; LM is the bottleneck).
8. PL normalization — blocked (Orpheus is EN-only; needs a PL TTS model).

---

## Adoption / launch roadmap (2026-06-04, ROI-ranked)

Strategic frame: Overfit wins on **.NET in-process deployment + training moat**, NOT raw tok/s
(raw-perf is hygiene = bottom of this list, consistent with the "stop chasing decode" pivot). Ranked by
adoption ROI. **Current execution order (user, 2026-06-04): #2 (OpenAI API) now → then #1 (M.E.AI adapter).**
Q8-KV (a perf/RAM item, adjacent to #12) is checkpointed mid-build (store + dual-mode `KeyValueCache` +
write-sites done, F32 bit-identical) — resumed later, NOT a launch blocker.

| # | Feature | ROI | State / note |
|--:|---------|----:|--------------|
| 1 | **Microsoft.Extensions.AI adapter** (`IChatClient` / `IEmbeddingGenerator` over `ChatSession` + `SentenceEmbedder`) | 10/10 | **✅ DONE (2026-06-05)** — shipped as NuGet `DevOnBike.Overfit.Extensions.AI` (`OverfitChatClient` / `OverfitEmbeddingGenerator`). Drop-in for the whole .NET AI ecosystem (Semantic Kernel, anything on M.E.AI) — the in-process moat through the standard interface. |
| 2 | **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/embeddings`, SSE) | 9.5/10 | **✅ DONE (2026-06-05)** — `/v1/chat/completions`+SSE, `/v1/embeddings`, `/v1/models`, `response_format` (json_object / json_schema). Opens HTTP tooling (LangChain, SK, OpenAI clients, UIs). |
| 3 | **JSON-Schema constrained output** (required / enum / typed fields) | 9/10 | **✅ DONE (2026-06-05)** — `JsonSchemaConstraint` (full schema-conformance subset: required, `additionalProperties:false`, enums, types, nested objects, simple arrays) wired into demo `/chat/json` + OpenAI `response_format`. Open follow-ons: per-state mask cache (throughput) + token healing (dead-end repair). |
| 4 | **Production LocalAgent template** (auth, audit logs, model hash, retrieved sources, tool-call logs, `/healthz` `/readyz` `/metrics`, Dockerfile) | 8.8/10 | Phase-1 walking skeleton exists (`/health` `/chat` `/reset`); this is productionization → sellable PoC. Maps to COMMERCIAL.md "Private .NET RAG/Agent PoC". |
| 5 | **RAG Stability Harness + Corpus Linter** (expected-source tests, paraphrase stability, false-premise traps) | 8.5/10 → **bump above #2 as a DIFFERENTIATOR** | "RAG is *testable*" — a genuine differentiator (not table-stakes), plays to our empirical-rigor DNA + COMMERCIAL.md "Zero-GC inference audit". Few competitors sell RAG testability. **✅ Increment 1 DONE 2026-06-05** — `LanguageModels.Retrieval.Evaluation`: `RagEvaluator` (`EvaluateRetrieval` recall@K + MRR, `EvaluateParaphraseStability` mean pairwise Jaccard, `EvaluateFalsePremise` grounded-threshold traps) + `CorpusLinter` (`FindNearDuplicates`, `FindOrphans`) over the existing `VectorStore`/`SentenceEmbedder`; pure/deterministic, 7 model-free fast tests prove "RAG is testable" in CI. Follow-on: contradiction/short-doc lint, an xUnit-friendly assertion façade, wire into the demo + a `docs/rag-testing.md`. |
| 6 | **Persistent vector store** (SQLite / file-backed) | 8/10 | In-memory `VectorStore` → restart without re-indexing. (DiskLLM persistence conversation resonates here.) |
| 7 | `dotnet new overfit-agent` template | 7.8/10 | Cheap once #4 exists. Open-source adoption lubricant. |
| 8 | **Model/profile manager CLI** (`overfit models download bielik`) | 7.5/10 | ollama-style. On the dotLLM steal-list. Onboarding friction. |
| 9 | **QLoRA CPU fine-tuning** | 7/10 build, **10/10 as moat-positioning** | Already SHIPPED. Don't build more — *lead the messaging* with it (the one thing llama.cpp structurally can't do); sell RAG/JSON as the entry. |
| 10 | Whisper / speech-to-text vertical | 6.8/10 | Done; separate launch, not core adoption. |
| 11 | Interpretability / inference hooks | 5.5/10 | Cool, slow market. dotLLM's is a stub → we could *lead* here, but it's credibility not revenue. |
| 12 | Chasing raw tok/s vs llama.cpp | 4/10 | Hygiene, not product. Consistent with the strategic pivot. Q8-KV lives near here (RAM angle is mildly above pure tok/s). |
| 13 | **MCP (Model Context Protocol)** (added 2026-06-11; **SERVER SHIPPED 2026-06-11** — order flipped server-first by user decision) | 8.5/10 | ✅ **Server done**: `DevOnBike.Overfit.Mcp` (typed stdio JSON-RPC 2.0 contracts + source-gen `McpJsonContext` — official SDK skipped on purpose: reflection/DI vs our AOT story; AOT-verified on the published native exe) + `overfit mcp <model> [--rag-dir] [--whisper-model]` with zero-egress tools `ask` / `rag_query` (citations; embeddings from the chat model itself, multilingual) / `transcribe` (lazy Whisper). 12 model-free protocol tests + e2e on real `overfit.exe` (handshake, "Paris", cited RAG, JFK transcript). `claude mcp add overfit -- overfit mcp <gguf>`; docs `docs/mcp.md`. **Remaining scope — Overfit as HOST**: bridge `McpTool → ToolDefinition` so `ReActAgent`/`ToolCallConstraint` consume tools from any MCP server (filesystem, DBs, Jira, git…) — reuses the shipped JSON-RPC layer (client side is the same framing). Honest note: host value is capped by small-model tool-calling reliability (ReAct e2e needs 7B+); server has no such ceiling. ~1-2 sessions. |

**Adjustments vs the source ranking (the "points 1 & 3" update):** #1 confirmed as the single best first move;
#3 reclassified PARTIALLY-DONE (schema-conformance is the only gap); #5 flagged to bump above #2 (differentiator
> table-stakes); #9 split into build-priority (mid, it's shipped) vs moat-positioning (top).

### What else to steal from dotLLM (launch-relevant, beyond the earlier full inventory)

Already matched/exceeded (don't redo): SIMD RoPE, gate+up fusion, decode worker-cap, spin-pool, repacked Q4_K
GEMV, prompt-lookup speculative, GGUF/chat-template. NEW launch-relevant steals, in adoption order:
- **`JsonSchemaConstraint` + `TokenMask` bit-vector + FirstCharBuckets/LRU** (`Constraints/`) → powers #3. Their
  per-state mask cache + first-char bucket prune (~99% vocab rejected before any parser copy) is the hot-path trick.
- **OpenAI DTO surface + SSE streaming + logprobs endpoint** (`DotLLM.Server/Endpoints/ChatCompletionEndpoint.cs`)
  → the exact shape for #2 (note: their server is sequential per-request, NO continuous batching — we're not behind).
- **`PrefixCache`** (system-prompt KV reuse) — we have `KvCacheSnapshot`; their server-level prefix cache is the
  wiring pattern for multi-request reuse.
- **Composable sampling pipeline** (`ISamplerStep` / `ILogitProcessor`) — cleaner than our monolithic
  `SamplingOptions`; nice-to-have for extensibility once the API exists.
- **`System.Diagnostics.Metrics` + `Activity` telemetry** (`Telemetry/`) → feeds #4's `/metrics` + the audit story.
- **Model-management CLI** (HF pull/search/list, Spectre.Console TUI) → #8.

### UPDATE 2026-06-05 — status + new ideas

**DONE since the table above:** #1 M.E.AI adapter (NuGet `DevOnBike.Overfit.Extensions.AI`), #2 OpenAI API
(`/v1/chat/completions`+SSE, `/v1/embeddings`, `/v1/models`, `response_format`), #3 JSON-Schema constrained output
(full subsystem + wired into demo `/chat/json` + OpenAI). Plus regex-constrained decoding + composable sampling
pipeline (additive). All on branch `bilbo`, validated on real Qwen3B/MiniLM.

**▶ #0 SHIP IT (consolidation/release — highest leverage NOW, before more features).** The branch has a large
unmerged, terse-committed ("bilbo" ×N), top-level-undocumented feature set. Risk = infinite feature-build, never
launch. Steps: rewrite the "bilbo" commits into descriptive ones → merge `bilbo`→`main` (main far behind at
`3059fdc`) → NuGet release (version bump + the new `.Extensions.AI` package) → top-level README/docs for the new
surface (structured outputs, OpenAI-compat, M.E.AI) → refresh the staged `linkedin-*.md`/`launch-copy.md` (they now
have much more to say).

**▶ #8 ELEVATED — global `overfit` dotnet tool (the "ollama moment" — user-flagged 2026-06-05 as the onboarding
play).** A `PackAsTool=true` global CLI: `dotnet tool install -g DevOnBike.Overfit.Cli`, then
`overfit pull qwen2.5-3b` (HF GGUF download + progress → cache `~/.overfit/models`, known aliases qwen/bielik/...),
`overfit list`, `overfit chat <model>` (interactive REPL over `OverfitClient`), `overfit serve <model>` (start the
OpenAI-compatible server). Removes the demo's biggest friction (manual GGUF download + ModelPath config). NOTE: to
make `overfit serve` reusable, the OpenAI endpoints (`OpenAiEndpoints`) should move from the demo into a shared
lib/the CLI; need a C# HF downloader (HTTP resolve repo→file→stream, the demo's `download-*.cmd` are curl scripts).
This is the single best **open-source adoption lubricant** — ties loaders + OverfitClient + the OpenAI server into
one turnkey entry point.

**New technical follow-ons (half-built / deferred):** Q8-KV decode wiring (store ready → RAM + long-context win);
**token healing** (engine-level: re-tokenize boundary + KV rollback → makes constrained/structured output
bulletproof on arbitrary schemas, fixes the BPE dead-end that 2-lite only graceful-stops); wired sampling-pipeline
(replace `TokenSampler`); **GBNF grammar constraint** (CFG → SQL/DSL constrained output, the general engine);
continuous/in-flight batching for serving (dotLLM only PLANS it → we could lead).

**Bolder new directions:** **persistent KV-cache to disk → cross-session resume** (`KvCacheSnapshot` exists; persist
to disk = resume a conversation after process restart — a genuine novelty from the DiskLLM conversation, pairs with
the embeddable identity); QLoRA "teach your model a fact" marketing showpiece (already works — the "Tarnholm" demo);
inference hooks / pure-.NET-CPU interpretability (dotLLM's is a stub → lead, credibility play).

### UPDATE 2026-06-17 — dotllm.dev re-review (site, not source) + what's NEW to take

Re-read the public site. Confirms the split: **dotLLM leans "vLLM for .NET"** (GPU/CUDA PTX, paged-KV refcount+COW,
continuous batching "planned", browser chat UI, more quants). **We lead on what they LACK: QLoRA training/fine-tune +
merge, multimodal (Whisper/TTS/CNN/embeddings), RAG-testability harness, build-time perf analyzer.** Net: don't chase
their serving depth — lead where they're thin. Genuinely-new takeaways (moved here from `ideas.md`), in adoption order:
- **🟢 `GcLatencyScope` (SustainedLowLatency during generation) — ✅ DONE 2026-06-17.** `Sources/Main/Runtime/GcLatencyScope.cs`
  (`readonly ref struct`, embeddable-safe — process-wide knob only flipped by a process-OWNER), wired into
  `OverfitOpenAiServer.HandleChatCompletions`. Trims gen-2 pauses on the allocating prefill / SSE marshalling.
- **🟢 Built-in browser chat UI** at the server root — onboarding/demo polish (static HTML over our `/v1`). Low effort;
  pairs with the `overfit serve` + global-tool (#8) onboarding play.
- **🟡 Paged KV-cache (vLLM-style refcount + copy-on-write)** — beyond our `KvCacheSnapshot`/prefix reuse: eliminates
  fragmentation for long contexts / multi-request. A real serving feature IF we deliberately pursue serving.
- **🟡 Per-model tool-call templates** (Hermes / Mistral / Llama formats) — agent-routing reliability beyond our
  generic constraint.
- **🟡 More GGUF quants** (Q5_K, Q4_0/1, Q5_0/1) — broader coverage; LOW priority (Q4_K_M dominates real models).
- **🔵 Interpretability hooks** (activation capture, logit lens, sparse autoencoders) — dotLLM has this only as a
  *stub/planned*, so we can **ship first**; plays directly to "pure-managed = fully inspectable in a debugger" +
  the COMMERCIAL.md "Zero-GC inference audit" credibility. Strongest novel-differentiator candidate.
- **🔴 SKIP: GPU/CUDA backend** — dotLLM's big edge, but CUDA is a native dependency that breaks our "no native binary"
  identity. Deliberate non-goal (perf/GPU stays the private moat per [[project-moat-public-private]]).
**Recommended order:** browser chat-UI (cheap onboarding) → interpretability hooks (novel lead) → per-model tool
templates → paged-KV/continuous-batching ONLY if serving becomes a deliberate track. But the highest-leverage move
remains #0 SHIP (both projects risk infinite feature-build; we're feature-complete — launch beats another feature).

---

## Nearest plan — llama.cpp competitive gaps (2026-05-25)

Gap analysis vs llama.cpp, ranked through the strategic frame (NOT chasing decode speed; build the
embeddability / low-end-hardware / in-process-agentic moat). Verified facts: `GenerationStats.TokensPerSecond`
EXISTS (produced by `SlmInferenceEngine`) but is NOT surfaced on the modern `CachedLlamaSession`/`ChatSession`
path; **no mmap** — `GgufReader` = `File.OpenRead` and the resident weights (`Q4KWeight`/`Q6KWeight`/`Q8Weight`)
copy ALL bytes into managed arrays (4 GB Q4_K_M ⇒ ~4 GB managed RAM).

**🟢 On-moat (do these):**
1. **mmap GGUF resident weights** — **DONE 2026-05-25.** Loadability on low-end hardware (the core moat:
   working-set RAM, not full-model RAM). `MemoryMappedModelFile` maps the whole file read-only and hands out
   zero-copy `ReadOnlyMemory<byte>` slices (a nested `MemoryManager<byte>` over the mapped pages; the parent
   owns the mapping). `Q4KWeight`/`Q6KWeight.Blocks` changed `byte[]`→`ReadOnlyMemory<byte>` (kept the `byte[]`
   ctor delegating; added `BlockSpan`); kernels read `BlockSpan` / `fixed (byte* = w.BlockSpan)` (math
   unchanged). `GgufLlamaLoader.Load(path, quantize, mmap: true)` slices verbatim-layout Q4_K/Q6_K weights
   (FFN, LM head, per-head attention Q/K/V — each head a contiguous file run) straight from the map; the engine
   holds the map and disposes it LAST. **NOW DEFAULT (`mmap: true`)** with a smart-skip: the map is built only
   when the file actually has Q4_K/Q6_K tensors (`FileHasVerbatimKQuant`) — pure-F32 / pure-Q8_0 files skip it
   and behave exactly as the copy path (no file handle held). Q8_0 (de-interleaved) + F32-fallback still copy.
   **Measured on real Qwen2.5-3B Q4_K_M (2007 MB file): managed-heap alloc 3202 → 1427 MB (−1776 MB, ~55 %),
   working-set delta 3167 → 1533 MB (~52 %).** The 1427 MB residual is the F32 embedding table (dequantized for
   lookup) — the next RAM lever (quantized embedding lookup), out of scope here. **Soak-validated 2026-05-25
   (default flip):** mmap vs copy bit-identical on Q4_K_M (maxDiff = 0), Q8_0 (maxDiff = 0), FP16 (maxDiff = 0);
   Q4_K_M decode-vs-own-F32-baseline = 29/32 top-1, worst swing 2.16 — *exactly* the pre-change recorded
   baseline, proving Phase A preserved decode bit-for-bit. Tests: `MemoryMappedModelFileTests` (4 fast),
   `GgufMmapParityTests` (4 `[LongFact]` — Q4_K_M/Q8_0/FP16 parity + RAM measurement). AOT-clean
   (`System.IO.MemoryMappedFiles`).
   ⚠ **Pre-existing (NOT mmap) finding:** `GgufQ4KMParityTests.Q4KM_TopTokenMatches_FP16Baseline` is RED —
   Q4_K_M top-1 (474) ≠ FP16 top-1 (40), 4/10 overlap, on the maximally-ambiguous 3-token prompt `[BOS,
   im_start, \n]`. This is the SAME step-0 near-tie the decode-baseline records (ref→47 / subj→474, swing 2.16);
   the assertion ("top-1 must match FP16") is over-strict for this flat-distribution prompt. Confirmed
   independent of mmap (Q4_K_M logits are byte-identical copy vs mmap). Fix later: relax to top-k overlap or
   pick a less-ambiguous prompt — NOT a decode bug.
   ► **Quantized token-embedding lookup — DONE 2026-05-25 (the post-mmap RAM lever).** The embedding table
   was always loaded as full F32 (1187 MB for Qwen-3B, vocab 151936 × dModel 2048) — the dominant residual
   after mmap. Now `GgufLlamaLoader.LoadEmbedding` keeps `token_embd` in its native K-quant layout (Q4_K/Q6_K,
   verbatim, mmap-able — token_embd is row-major [vocab, dModel] = output-major, row = token) and the lookup
   dequantizes only the looked-up row: `Q4KWeight`/`Q6KWeight`/`Q8Weight.DecodeRow(row, dst)` (zero-alloc, one
   row's super-blocks) behind `DecodeWeight.DequantizeRow`. `_embedWeights` (engine + session) changed
   `TensorStorage<float>` → `DecodeWeight`; F32 / `.bin` / safetensors paths unchanged (implicit conversion +
   F32 fallback). **Measured (Qwen-3B Q4_K_M, mmap default): live resident managed heap 2158 MB (copy) →
   238 MB (mmap)** — vs the 1427 MB mmap residual before this change (that residual WAS the F32 embedding,
   now ~0 on-heap / Q6_K file-mapped). Decode bit-identical: Q4_K_M decode-vs-F32 still *exactly* 29/32, swing
   2.16 (same `DecodeQ6_KBlock` on the same bytes as the old full-tensor dequant). Per-token cost: dModel/256
   super-block decodes (8 for Qwen-3B) — negligible vs the matmuls. Tests: `DecodeWeightRowTests` (3 fast),
   `Mmap_MeasuredResidentManagedHeap` (`[LongFact]`).
2. **Embeddings API** — **DONE 2026-05-25.** `CachedLlamaSession.Embed(tokens, pooling, normalize)` +
   `EmbeddingDimension` + `EmbeddingPooling` (Mean/LastToken), L2-normalised, pools per-token final hidden
   states. Validated on real Qwen (`EmbeddingsTests`): unit-norm, deterministic, semantic ordering holds
   (cos(cat,kitten)=0.94 > cos(cat,physics)=0.88). Unlocks in-process RAG / vector-store.
3. **Constrained generation** — logit-masking to a grammar = guaranteed-valid structured output for
   in-process agentic .NET. **JSON-mode DONE 2026-05-25.** `ITokenConstraint` (Contracts: `ApplyMask` /
   `Accept` / `IsComplete`) → `JsonStateMachine` (value-type char-level RFC-8259 acceptor: 64-level bit-stack
   for nesting, number/string/escape sub-DFAs, `IsComplete` gates EOS) → `JsonGrammarConstraint` (builds the
   per-token text table from `ITokenizer.DecodeToString`, masks the vocab each step, EOS only when complete).
   Wired through `ISlmSession.GenerateNextToken(in sampling, ITokenConstraint?)` (default interface method →
   `NotSupportedException` for non-supporting sessions like GPT-1/2; real override on `CachedLlamaSession`,
   masks `_logits` in place pre-sample) and `ChatSession.Send(..., constraint)`. **Validated end-to-end on
   real Qwen-3B Q4_K_M** (`JsonConstrainedChatTests` `[LongFact]`): reply parses via `JsonDocument.Parse` even
   when the small model would ramble. Fast tests: `JsonStateMachineTests` (accept/reject/complete cases),
   `JsonGrammarConstraintTests` (mask behaviour on a fake vocab). **Finding (handled):** GGUF pads the vocab
   (Qwen 151936 logits vs 151665 tokenizer tokens) — `ApplyMask` accepts `logits.Length ≥ tableSize` and masks
   the padding slots. **Perf note:** mask is O(vocab × token-len)/step — fine for short structured outputs; a
   per-state cache / token prefix-trie is the documented follow-on. Next: **JSON-Schema** (typed: fields /
   enum / required → deserializable to a C# record) then **GBNF** (generic grammars) — opt 6 function calling
   sits on top.

**🟡 Cheap polish:**
4. **tokens/sec on the modern path** — **DONE 2026-05-25.** `ChatSession.LastStats` (GenerationStats,
   decode-timed) exposes `TokensPerSecond`.
5. **Min-P / Mirostat / XTC sampling** — additive to `TokenSampler`/`SamplingOptions`; Min-P a sane modern
   default, Mirostat for perplexity-targeted decoding. **(Min-P DONE 2026-05-25 — `SamplingStrategy.MinP` +
   `SamplingOptions.WithMinP`; `TokenSamplerMinPTests`. Mirostat DEFERRED: it's STATEFUL (running `mu` per
   token) and doesn't fit the static `TokenSampler` / readonly `SamplingOptions` — needs a stateful sampler
   threaded through the sessions; queued. XTC later.) `ChatSession.LastStats` now exposes TokensPerSecond.**

**🟢 On-moat / done:**
6. **Function calling** — **DONE 2026-05-25 (the in-process-agentic-.NET headline).** `ToolDefinition`
   (name + description) → `ToolCallConstraint : ITokenConstraint` forces the canonical envelope
   `{"name": "<tool>", "arguments": <json>}`: fixed punctuation, the `name` value constrained to an enum
   DFA over the registered tool names (≤64, viability bit-mask), the `arguments` value delegated to
   `JsonStateMachine` (well-formed JSON). `ToolCall.TryParse` (System.Text.Json) extracts name + raw
   arguments; the caller dispatches by name to a `Func<JsonElement,string>`. Reuses the JSON-mode seam
   (`ChatSession.Send(..., constraint)`). **Validated end-to-end on real Qwen-3B Q4_K_M**
   (`ToolCallingChatTests` `[LongFact]`): model emitted `{"name": "get_weather", "arguments": {"city":"Paris"}}`,
   parsed, dispatched → `weather({"city":"Paris"})`. Fast tests: `ToolCallConstraintTests` (envelope / enum /
   bad-args rejection), `ToolCallTests` (TryParse). Argument *typing* (per-tool JSON-Schema) is the follow-on;
   the handler validates args meanwhile.

**🟢 On-moat / done:**
7. **Vector store** — **DONE 2026-05-25.** `VectorStore` (in-process, zero-dependency) + `VectorMatch`:
   `Add(id, vector, payload)` stores unit-normalised in one contiguous backing array; `Search` is a flat
   dot-product scan + top-K insertion (no full sort; span overload allocates nothing). Cosine reported is
   magnitude-invariant. Linear scan — sized for app/document-set scale, not billion-scale ANN. Closes the RAG
   loop (embeddings → store → retrieve), all in-process. Tests: `VectorStoreTests` (ranking / true-cosine /
   top-K / growth / guards). Wired into `Demo/AgentDemo` RAG step.

**Session 2026-05-25 delivered: opt 1 (tokens/sec + Min-P; Mirostat deferred — stateful), opt 2 (mmap GGUF —
~55 % less managed RAM, bit-identical; NOW DEFAULT with smart-skip, soak-validated on Q4_K_M/Q8_0/FP16),
opt 3 (embeddings), quantized token-embedding lookup (live managed heap for a 3B Q4_K_M model now 238 MB,
down from 1427 MB — the F32 embedding eliminated; decode bit-identical), and constrained generation
**JSON-mode** (well-formed JSON enforced at decode via `ITokenConstraint`/`JsonGrammarConstraint`, validated
on real Qwen), **function calling** (`ToolCallConstraint`/`ToolCall` → dispatch to a C# delegate, validated
on real Qwen), and a consolidated **agent demo** (`Demo/AgentDemo`: mmap load → RAG → tool call → JSON in one
process; ran on real Qwen-3B — 222 MB live heap, RAG ranks correctly, tool call dispatches, JSON parses) +
README/demo-README surfacing the in-process-agentic-.NET story for launch. Next: JSON-Schema (typed args) →
vector store (queued); then GBNF (generic grammars). Also queued: relax the over-strict pre-existing
`GgufQ4KMParityTests` FP16 assertion (see opt 1 note — not a decode bug); per-state mask cache if the
O(vocab×len) mask shows up in profiles.**
Also fixed a recurring flaky-suite issue: added `MathUtils.SetSeed(int)` (per-thread
repro hook) and seeded the random-tiny-base anomaly `[Fact]` tests — which surfaced that **tiny-base LoRA
convergence is init-sensitive** (some seeds diverge at lr 1e-2 / 300 steps; the seed pins a representative
converging init — the rigorous validation remains the TRAINED production base in `[LongFact]`).

## llama.cpp scope-gap snapshot (2026-05-29, full-tree read of `D:/llamacpp-tmp`)

Read the full llama.cpp source tree to map what they have that Overfit doesn't. Bucketed by ROI / effort.

**SHIPPABLE in pure C# (≤2 weeks each, candidate post-launch additions):**

- [ ] **Q2_K / Q3_K dequant + decode** — mathematically straightforward K-quants (block scale + 2-/3-bit indices), ~2 days each. Unlocks the lower end of Ollama's K-quant matrix (Q4_K_M / Q6_K already shipped).
- [x] **Mirostat v1 / v2 samplers** — DONE (`Sampling/MirostatSampler.cs`, stateful μ feedback; opt-in via the `SamplingPipeline`, which is where stateful/terminal samplers live).
- [x] **Typical-p / XTC / Top-n-sigma / DRY** — DONE. All four in `SamplingPipeline`; **top-nσ + typical-p also on the zero-alloc `TokenSampler`/`SamplingOptions` hot path** (`WithTopNSigma`, `WithTypicalP`) and the `overfit chat` CLI (`--top-n-sigma`, `--typical-p`). **DRY is wired into the decode engine** (`CachedLlamaSession`, rolling history + reusable Z-scratch, applied to logits *before* sampling → breaks loops even under greedy). Perf note: top-p/typical-p partial-sort (min-heap, exact) cut the sampler from ~50-60% of a decode step to a few percent.
- [ ] **Infill / FIM sampler** — prefix/middle/suffix token masking for code-completion-style models; ~1 week.
- [ ] **LoRA adapter loading + composition** — Overfit has *training* LoRA (custom); loading external `.gguf` LoRA adapters and composing multiple at runtime is ~3-5 days.
- [ ] **RoPE scaling variants** — Yarn (NTK-by-parts) / DynamicNTK / AliBi / per-section. Algebraic variants on the existing RoPE kernel, ~1-2 days each. Unlocks long-context Qwen/Llama variants.
- [ ] **Encoder-decoder (T5 / FLAN)** — reuses BERT-encoder building blocks already shipped; ~1 week.

**HARD (2-4 weeks each, evaluate niche fit first):**

- [ ] **GBNF (Generic BNF) grammar engine** — context-free parser + DFA construction + token-trie matching. Real work (~3-4 weeks). Overfit currently has JSON-mode + ToolCallConstraint (covers 90% of structured-output use). Build only if user demand surfaces for arbitrary grammars.
- [ ] **One vision-language model stack** (LLaVA / Qwen2-VL / Pixtral) — CLIP image encoder + projection + LLM fusion + tokenizer surgery. ~2-4 weeks for a single model. Doubles the addressable workload (image+text).
- [x] **Whisper-tiny / -base ASR port** — DONE and validated end-to-end on the real `ggml-tiny.bin` + `jfk.wav` (perfect transcript, pure .NET CPU). `WhisperTranscriber.Load(ggml).TranscribeFile(wav, "en")`; log-mel front-end + encoder + decoder in `LanguageModels/Whisper/` + `Audio/`. Optional polish left: KV-cache decode, >30s windowing, MP3 adapter.
- [ ] **State-space / Mamba / RWKV** — recurrent-state arch; structurally different (SSM ops, scan). Defer unless a specific user model demands it.
- [ ] **Multi-token-prediction (MTP) + speculative rollback** — Qwen3.5 / Gemma3N draft heads. Tree-based verification + rollback cache. Defer.

**Out of scope (philosophy / non-goals — see "What Overfit is not"):**

- GPU backends (CUDA / Metal / Vulkan / SYCL / OpenCL / OpenVINO / Hexagon / WebGPU) — require native code. Overfit is pure-managed.
- Server / CLI binaries (`server`, `cli`, `batched-bench`, `perplexity`, `tokenize`) — Overfit is a *library*, not a daemon.
- Quantization export pipelines (`quantize`, `imatrix`, `cvector-generator`) — Overfit loads pre-quantized; building quantizers is a separate concern.
- TTS engines (OuteTTS, WavTokenizer, bark.cpp) — out of niche for now.
- Diffusion models — image generation is a different domain.
- TensorFlow / JAX / MLX checkpoints — Python ecosystem formats.

**Architectures Overfit doesn't load** (llama.cpp lists ~130; Overfit now lists 10 families — added `qwen3`, `phi3`, `gemma2`): Falcon, Gemma 1/3/3N, Mamba/Mamba2/Jamba, RWKV6/7, T5, GLM 1/2/3/4, Deepseek 1/2/V3, Cohere, Nemotron, Granite, LLaMA-4, Qwen3-MoE/3.5/4, Grok, Chameleon, Hunyuan/-V/-VL/-OCR, Pixtral, MiniCPM-V, InternVL, etc. Most are deferrable — Qwen2.5/3 + Llama-3.x + Mistral/Phi-3.5 + Gemma 2 + Mixtral cover the dominant Ollama deploy. Adding a new arch is typically 3-5 days *per family* (weight name mapping + arch-specific quirks like SwiGLU vs GeLU, GQA vs MHA, QK-norm, logit soft-caps).

---

## Decode throughput catch-up to llama.cpp — 3-phase plan (2026-05-29)

> **UPDATE 2026-05-31 — sprint executed (branch `perf`), via a different path and with a corrected diagnosis.**
> The plan below predicted parity (~28 tok/s) via MHA-consolidation + Q8-KV under an *overhead-bound* premise.
> What actually shipped on **Bielik-4.5B Q4_K_M** (bit-identical, suite 1006/0): decode worker-cap + fused
> SwiGLU gate+up + opt-in decode spin-pool + SIMD activation-quantize / RoPE / cached-attention + the Q4_K
> repacked 8×8 GEMV → **12.55 → ~17 tok/s (+36 %), same-file gap ~1.33× → ~1.13×.** Rigorous best-of-N on
> *both* engines: llama.cpp ~19.2 (short) / ~17.9 (long-ctx) ⇒ **~1.13× behind, NOT parity** (an earlier
> single-run "parity" read was retracted — always best-of-N both sides).
> **Corrected diagnosis:** single-stream decode is **DRAM-bandwidth-bound**. The FFN matmuls *were*
> dispatch-overhead-bound (fixed by the worker-cap, +13 %), but the residual gap is memory-access efficiency —
> the speed ratio equals the bandwidth-utilisation ratio, so it needs structural (not ALU) work. Implications
> for the phases: **1a MHA-consolidation** stays the one real structural lever but is a big refactor;
> **1b Q8-KV** — the attention kernel is now built + validated (`ComputeSingleHeadQ8` + `Q8KvQuant`, Phase 1),
> but after the SIMD-attend win its *speed* headroom shrank → its value is now **RAM / long-context**, not tok/s;
> **2a dequant-fused GEMV** — DONE as the repacked 8×8 GEMV and bounded (kernel ≈ 84 % of the cold-DRAM ceiling;
> VNNI / AVX-512 tried → ≈ 0, memory-bound). Full record: CHANGELOG `[Unreleased] → Performance` and
> `project_perf_sprint` memory. **Training-compute perf was also profiled and found already reasonable**
> (conv im2col+GEMM ≈ 280–330 GFLOP/s, ~22 % of peak; no easy win) — see CHANGELOG / `project-training-memory`.

**Baseline**: Qwen2.5-3B Q4_K_M, ~17 tok/s @ 3.20 GB live heap (Overfit) vs ~27 tok/s @ 3.20 GB (llama.cpp same-file A/B). Per `project_perf_sprint` memory: gap is **overhead-bound**, not bandwidth-bound — our measured BW efficiency is ~55% of DDR5 peak vs llama.cpp's ~80%.

The previous ROADMAP wording ("honest ceiling 2-2.5×, not parity") reflected the *pure-managed* constraint *without* algorithmic reorganisation. The three phases below preserve that constraint and target parity-then-overtake by tackling the structural causes of the 55%-vs-80% BW gap.

### Phase 1 (~1 week) — parity with llama.cpp (~28 tok/s)

- [x] **1a. MHA consolidation — BUILT, MEASURED, ~NEUTRAL. ⛔ DO NOT REPEAT.** The premise was: collapse the
  per-head `Wq/Wk/Wv/Wo[i]` ([dModel × dHead] × 16) into one `[dModel × dModel]` per projection so a single
  contiguous GEMV prefetches ~10× better than 16 fragmented per-head ones — projected BW efficiency 55 % → 75 %,
  **+50 % tok/s**. It was implemented (2026-06-15) as **whole-matrix Q4_K attention** (M2 loader plumbing of
  whole-matrix handles + M3 whole-Q/whole-O GEMV with split-heads-after). Result: **E2E coherent but +1-2 % —
  effectively neutral.** It now sits behind `OVERFIT_REPACK_ATTN`, **default OFF**.
  **Why the projection was wrong — the lesson:** the isolated micro-bench *did* show 2.36× (parallel,
  bandwidth-bound), and that number was real. **Amdahl ate it:** the Q/O projections are a small slice of a
  decode step; FFN + LM-head dominate and are already **at the DRAM floor**. A 2.36× on ~5 % of the work is ~0.
  Discovery along the way: Q4_K_M files are *mixed* (V=Q6_K, Q/K/O=Q4_K).
  ⇒ **Do not re-derive this from the layout analysis.** Any future attention-layout lever must first be sized
  against its share of the decode step, not against its own micro-bench.
- [x] **1b. Q8 KV cache — DONE + wired 2026-06-05.** `KvCacheDType.Q8` stores each cached K/V vector as
  per-vector symmetric int8 + one F32 scale (`Q8KvQuant`) — ~4× smaller KV storage and attention read traffic.
  Opt-in via `CreateSession(ctx, KvCacheDType.Q8)` or `OVERFIT_KV_DTYPE=q8` (default F32 unchanged, bit-identical,
  suite 1126/0). Wiring: writes quantize through `KeyValueCache.WriteKey`; the single-token decode hot path
  (`CachedSingleHeadAttention.AttendFromCache`) attends int8 directly via
  `CachedAttentionKernel.ComputeSingleHeadQ8` (0-alloc preserved); batched prefill (`DecodeBatchedQuant`)
  dequantizes the range to an F32 scratch and reuses the proven F32 kernel; `SingleTokenProjectionKernel` writes
  made mode-agnostic. Int8 round-trip is cosine ≈ 1, not bit-identical → greedy decode stays coherent
  (**validated on real Qwen2.5-0.5B: F32 → "Paris", Q8 → "Paris"**; `Q8KvCacheCoherenceTests` [LongFact]).
  Value is **RAM / long-context**, not tok/s. Snapshot/RestoreFrom stay F32-only (prefix-cache reuse).

**Phase 1 total**: 17 → ~28-29 tok/s = **parity with llama.cpp**. Both items are well-understood, no native deps, AOT-clean.

### Phase 2 (~1 week) — overtake llama.cpp by 15-25 % (~32-34 tok/s)

- [x] **2a. Custom AVX2 dequant-fused GEMV — DONE and BOUNDED.** Shipped as the repacked 8×8 Q4_K GEMV
  (dequant + FMA in one pipeline, no intermediate F32 buffer). `OVERFIT_REPACK_GEMV=1` measured **+30 %**
  (Qwen-3B 18.7 → 24.4 tok/s). **The kernel now runs at ≈84 % of the cold-DRAM ceiling — there is no ALU
  headroom left**; VNNI and AVX-512 ports were tried and measured **≈0** (memory-bound, not compute-bound).
- [~] **2b. L2-aware blocking + register-tiled GEMV — SPLIT VERDICT, mostly answered.**
  **Register tiling: DONE for PREFILL** — the tinyBLAS-style `Q4KGemvKernel.GemmTiled` (`block_q4_Kx8`,
  AVX2/FMA) measured 2.76× single-thread / 3.80× parallel on the projection GEMM → **~1.61× end-to-end TTFT**,
  shipped with an offline `overfit repack` sidecar (default-on, zero extra RAM).
  **Cache blocking: measured NEGATIVE and reverted** — K-blocking + A-packing on the im2col GEMM regressed, and
  the *simple* register-blocked GEMM beat the cache-blocked one. Detecting L2 to block against it is therefore
  **not** a live lever on this shape.
  Remaining slice: register-tiling the single-token *decode* GEMV — but decode is DRAM-bound (see 2a's 84 %
  ceiling), so expect ≈0 there too. Size it before building it.

**Phase 2 total**: ~33-34 tok/s = **+20-25 % vs llama.cpp**. Pure-managed throughout.

### Phase 3 (~1-2 weeks) — structural advantage llama.cpp can't easily replicate (~40-50 tok/s on realistic prompts)

- [x] **3. Adaptive early-exit / layer-skip — MEASURED 2026-07-05, PREMISE FALSIFIED. ⛔ DO NOT BUILD.**
  The claim was: for typical chat tokens the model "knows the answer" after K of N layers → exit on low
  intermediate entropy → **−30-40 % compute / +30-50 % tok/s**, and a moat (llama.cpp's pipeline is fixed-depth).
  **Sized first with the shipped logit lens** (`CachedLlamaInferenceEngine.LogitLens` + `GetLayerActivation`,
  bit-exact at the last layer) — no new machinery needed, ~5 minutes of compute. Qwen2.5-3B Q4_K_M, 36 layers,
  160 generated tokens across 4 prompt styles (factual / explanatory / code / narrative), probing every 4th layer.

  **(a) The model does NOT decide early.** top-1@layer == final top-1:
  | layer | depth | match |
  |---:|---:|---:|
  | 19 | 56 % | 3.7 % |
  | 23 | 67 % | 4.4 % |
  | 27 | 78 % | **18.1 %** |
  | 31 | 89 % | **47.5 %** |

  The residual stream keeps changing the argmax until the last ~10 % of the stack.

  **(b) The real rule (trigger must be entropy — at runtime you don't know the final token):**
  | entropy T | exited | correct | **WRONG** | avg layers saved |
  |---:|---:|---:|---:|---:|
  | 0.05 | 16.9 % | 81.5 % | 3.1 % | **4.6 %** |
  | 0.25 | 35.0 % | 67.9 % | 11.3 % | 7.7 % |
  | 0.50 | 45.6 % | 64.4 % | **16.3 %** | 9.9 % |
  | 1.00 | 65.6 % | 55.2 % | **29.4 %** | 14.8 % |

  **−30-40 % is unreachable at ANY threshold** (max 14.8 %, at 29.4 % corrupted tokens). Buying even 10 %
  costs 16.3 % wrong tokens. The safest setting yields **4.6 %** — and its 3.1 % wrong already exceeds the
  "~1-2 % perplexity drift" the item budgeted. Entropy is also a poor proxy: even at T=0.05 only 81.5 % of
  confident exits agree with the full stack. Confidence ≠ correctness at intermediate layers.

  **Honest scope of the refutation:** this measures the RAW logit lens. The item proposed a *trained* early-exit
  head (a tuned lens), which reads intermediate states better and would beat these numbers. But (a) shows the
  information largely isn't in the residual yet — a better readout cannot invent it — and a trained-head variant
  is a far bigger project than the 1-2 weeks budgeted here. Sample: 160 tokens / 4 prompts / 1 model / no chat
  template / every 4th layer; small, but the trend is monotonic and steep enough that a finer grid won't rescue it.

  ⇒ **Decode throughput is CLOSED.** Every lever in this plan is now measured: 1a +1-2 %, 2a done (≈84 % of the
  cold-DRAM ceiling), 2b cache-blocking negative, AVX-512/VNNI ≈0, and Phase 3 ≈4.6 %. The residual ~1.13×
  gap to llama.cpp is a memory-access-efficiency property, not a missing kernel.

~~**Phase 3 total**: 40-50 tok/s on realistic prompts = **~1.5-1.8× llama.cpp** on chat workloads.~~
**RETRACTED 2026-07-05 — measured, not achievable.** The ceiling is ~4.6 % compute saved at a quality cost
already beyond budget (see the item above). This projection was built on an untested premise ("the model knows
the answer early"); the model decides in the last ~10 % of the stack. Nothing in this plan reaches 40-50 tok/s.

### Validation gates

Every phase must pass:
1. `Gpt2TokensPerSecondBenchmark` + `GgufLlamaInferenceBenchmark` before/after (token-per-second delta + RAM delta).
2. Q4_K_M decode parity vs F32 baseline within existing tolerance (top-1 match on the canonical prompt; mean-abs logit diff < pre-change baseline).
3. Multi-thread suite stays green (no contention regression from kernel changes).
4. AOT publish guard passes (`dotnet publish -c Release -r linux-x64`).

### Order of attack — REWRITTEN 2026-07-05 (the original said "start with 1a"; 1a is now falsified)

**Decode tok/s is effectively exhausted. Do not open this plan expecting an easy win.** Scoreboard after the
sprint: 1a **built → +1-2 %, shelved**; 1b **done** (value is RAM/long-context, not tok/s); 2a **done → +30 %,
now at ≈84 % of the cold-DRAM ceiling**; 2b **register tiling done for prefill (1.61× TTFT), cache blocking
measured negative**. FFN + LM-head sit **at the DRAM floor**; attention is ~1.5× off its floor but the lever
that would close it is exactly the neutral 1a. The residual gap to llama.cpp is **~1.13× and UNIFORM across
context length** — a memory-access-efficiency property, not a missing kernel. AVX-512/VNNI: ≈0.

**Phase 3 (adaptive early-exit) was the last untried lever — it was measured on 2026-07-05 and refuted**
(≈4.6 % compute saved at acceptable quality; the model decides in the last ~10 % of the stack). **There is no
known high-ROI decode lever left.** Treat any new proposal here as guilty until sized.

**Rule earned the hard way (Winograd −79 %, AVX-512 ≈0, OverfitPool ≈0, whole-matrix attention +1-2 %, Phase-3
early-exit ≈4.6 %):** size a lever against **its share of the decode step** before building it. A micro-bench
speedup on 5 % of the work is 0 % end-to-end. Every one of those five looked compelling in isolation — and the
last one was refuted in **~5 minutes of compute** using an already-shipped primitive (the logit lens), versus
the 1-2 weeks it was budgeted. **Sizing is cheap; building on an unsized premise is not.**

---

## Release readiness snapshot (2026-05-29)

Capability surface at this date — what would ship if we tagged a release today:

- **LLM inference**: GPT-2 / GPT-1 / Qwen2.5 (0.5B–32B) / Llama-2-3.x / Mistral / Qwen1.5-MoE / Mixtral-8x7B from GGUF + safetensors + .bin. Q4_K_M / Q6_K / Q8_0 / F32 / F16 / BF16. mmap K-quant weights. Live managed heap ~220 MB for Qwen-3B Q4_K_M.
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5, intfloat/e5-small-v2 — all bit-parity vs HF/PyTorch (cosine 1.0 / 0.999999 / 1.0). `SentenceEmbedder.ForMiniLm` / `ForBgeEnV15` / `ForE5`.
- **Agentic stack**: ReAct loop, self-reflection critic loop, circuit breaker, summarising memory, JSON-mode + tool calling (constrained decoding), in-process `VectorStore` for RAG, `ChatSession` with sliding-window eviction. `OverfitClient.LoadGguf(path)` one-line facade.
- **Training**: gradient checkpointing (24× live-activation cut on 12L GPT-1), data-parallel trainer (~6× throughput on 24 workers), Adam/AdamW/SGD, padding/stride/bias Conv2D, BatchNorm2D (CIFAR-scale adaptive parallel −31 % backward), depthwise conv, LSTM (training + ONNX import deferred), CRNN + CTC, learning-rate schedules.
- **Loaders, all native (no Python)**: GGUF, HF safetensors (sharded), Overfit `.bin`, ONNX (linear + DAG).
- **AOT-clean**: zero LINQ / Reflection / Activator / Expression / Array.Copy / raw `ArrayPool<T>.Shared` in `Sources/Main` (all banned, RS0030).
- **Tests**: 990 fast / 0 fail / 130 [LongFact] skip. 3 headline benchmarks re-validated post-pool-refactor on this hardware (single inference 7.31× vs ONNX, concurrent 3.56× vs ONNX, CNN zero-alloc preserved).
- **Honest gaps acknowledged in README + ROADMAP**: tok/s **~1.13× behind** llama.cpp same-file (narrowed from ~1.6× by the 2026-05-31 sprint — see the UPDATE banner above; residual is DRAM-bandwidth-bound), no GPU, no diffusion / TTS / VLM / ASR, no full GBNF.

**Marketing assets**: `linkedin-article.md`, `linkedin-business.md`, `linkedin-marketing.md`, `linkedin-technical.md`, `launch-copy.md` — staged and waiting on a final review pass.

**Decision the release waits on**: the 2026-05-31 sprint already closed most of the gap (~1.6× → **~1.13×**, +36 %, bit-identical) — the honest decode story is now strong enough to ship as-is. Full parity would need the structural full-tensor-attention refactor (big), and the residual is bandwidth-bound, so it is a diminishing-returns lever rather than a launch blocker.

---

## Post-launch track #1 — Mixture of Experts (MoE)

**Why:** we already have the memory-optimization trio's first two legs — KV cache + GQA. MoE is the
third (Mixtral / Qwen-MoE). It fits the moat exactly: *big total knowledge, small per-token compute*
(only top-k experts run) and, with mmap + K-quant, *low working-set RAM* — i.e. "run Mixtral-class
models in pure C# on a CPU, no Python". Extends "run the models you know" rather than chasing scale.
Not multimodality / RLHF / Ring-Attention / TPU kernels — those are off-moat (different lane / scale /
hardware). See the 2026-05-25 analysis.

**Increment 1 — DONE 2026-05-25 (additive, does NOT touch the live decode path):**
- `GPT1Config.ExpertCount` / `ExpertUsedCount` (0 ⇒ dense — inert default) + `IsMixtureOfExperts`.
- `MoeRouter.SelectTopK(logits, topK, indices, weights)` — the gating core: top-k by logit + softmax
  over the k selected logits (== Mixtral's softmax-all → top-k → renormalize; mathematically identical).
  Zero-alloc, caller-owned spans. Tests: `MoeRouterTests`.

**Increment 2a — DONE 2026-05-25 (compute block, additive — not yet wired into the live stack):**
- `MoeFeedForwardBlock` — router projection (`ffn_gate_inp`, F32) → `MoeRouter.SelectTopK` → runs only
  the selected experts' SwiGLU FFN via the existing `CachedFeedForwardBlock.DecodeSwiGluDispatched`
  (F32 / Q8_0 / Q4_K / Q6_K — so experts are mmap-able like the dense FFN) → weighted-sum combine.
  Zero-alloc; per-token cost ≈ `ExpertUsedCount` dense FFNs. Synthetic parity test
  (`MoeFeedForwardBlockTests`): routed output == reference (selected experts' `DecodeSwiGlu` combined
  by the router weights), for top-1 (== that expert) and top-2 (weighted sum).

**Increment 2b — loader + stack wiring (needs the real MoE GGUF — now on the dev box at
`C:\qwen-moe\Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf`).**

**REAL structure inspected 2026-05-25 — Qwen2-MoE is NOT plain Mixtral** (`arch=qwen2moe`, 24 layers,
dModel 2048, MHA 16/16, RMSNorm ε1e-6, RoPE θ1e6, vocab 151936):
- **Routed experts:** `qwen2moe.expert_count=60`, `expert_used_count=4`. Expert FFN dFF = **1408**
  (≠ the dense `feed_forward_length=5632`) — derive from the tensor, not metadata:
  - `blk.L.ffn_gate_exps.weight` Q4_K `[2048, 1408, 60]` = `[dModel, expertDff, nExpert]`
  - `blk.L.ffn_up_exps.weight`   Q4_K `[2048, 1408, 60]`
  - `blk.L.ffn_down_exps.weight` **Q8_0** `[1408, 2048, 60]` = `[expertDff, dModel, nExpert]`
  - router `blk.L.ffn_gate_inp.weight` **F32** `[2048, 60]`
- **Shared expert (Mixtral has none):** a full SwiGLU FFN, dFF=**5632**, sigmoid-gated:
  - `ffn_gate_shexp` Q4_K `[2048,5632]`, `ffn_up_shexp` Q4_K, `ffn_down_shexp` Q6_K `[5632,2048]`
  - `ffn_gate_inp_shexp.weight` F32 `[2048]` (the 1-d shared gate)
  - **Forward:** `FFN(x) = σ(w_sh·x)·shared(x) + Σ_{i∈top4} wᵢ·expertᵢ(x)` (top-k probs renormalised
    = `MoeRouter`; shared scaled by `sigmoid` of a dot, NOT softmax).
- All tensor types (Q4_K / Q6_K / Q8_0 / F32) are already supported → mmap-able.

Steps: (1) ✅ `MoeFeedForwardBlock` — routed Σ (2a). (2) ✅ `Qwen2MoeFeedForwardBlock` — gated shared
expert + routed (`σ(w·x)·shared(x) + routed(x)`), `Qwen2MoeFeedForwardBlockTests`. **The full Qwen2-MoE
compute is implemented + synthetically validated.** (3) ✅ **Loader slicing — DONE 2026-05-25, validated on the REAL file.** `GgufLlamaLoader.LoadExperts`
(3-D expert tensor → 60 per-expert `DecodeWeight`; Q4_K/Q6_K verbatim, Q8_0 per-expert de-interleave) +
`LoadRouter` (transpose `ffn_gate_inp` to input-major). `Qwen2MoeLoaderTests` `[LongFact]` loads layer-0
of `Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf` (gate/up Q4_K, down Q8_0, σ-gated shared) through
`Qwen2MoeFeedForwardBlock` → finite full-density output (2048/2048, maxAbs 0.064). Also fixed a latent
bug: `CachedFeedForwardBlock` activation scratch assumed `dFF ≥ dModel` (false for a MoE expert,
1408 < 2048) → now sized `max(dModel, dFF)` (dense path unchanged). (4) ✅ **End-to-end wiring — DONE 2026-05-25 (additive, dense path strictly untouched).**
`GPT1Config.ExpertFeedForwardLength`; `BlockWeights` optional MoE fields (router + 3 expert arrays +
shared + gate); `CachedTransformerBlock` builds a `Qwen2MoeFeedForwardBlock` and dispatches on
`weights.IsMoe`; `CachedGptStack` threads the expert dims; `CachedLlamaInferenceEngine.LayerWeightBuffers`
carry the MoE weights, `BuildStackWeights` wires them, `Dispose` releases them; `GgufLlamaLoader`
recognises `qwen2moe` (reads `expert_count`/`expert_used_count` + expert dFF from the tensor) and the
layer loop branches the FFN to MoE. Fixed a latent `CachedFeedForwardBlock` scratch-sizing bug
(`max(dModel,dFF)`). **Proven correct on the real file:** `Qwen2MoeLoaderTests` decodes layer-0
end-to-end; the full-model load reaches layer 3 before hitting an unsupported quant (below).
⚠ **Real-file finding (NOT a MoE bug):** the `Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf` mixes **Q5_0** on
some layers' `ffn_down_exps` (llama.cpp's heterogeneous "_M" strategy). Q5_0 is outside our supported
set (F32/F16/BF16/Q8_0/Q4_K/Q6_K) → load throws a clear `NotSupportedException`. This is a
**quant-coverage** gap, orthogonal to MoE. To finish: grab the **Q8_0** Qwen-MoE (uniform Q8_0, all
supported) for full end-to-end + parity, OR add Q5_0 (+ likely Q5_K) dequant coverage.
(5) ✅ **Coherent generation — DONE 2026-05-26. MoE track COMPLETE end-to-end on the real model.**
Loaded the Q8_0 variant (uniform Q8_0, 14.18 GB) and `Qwen2MoeEndToEndTests` greedily answers
"What is the capital of France?" → **"Paris"** (then EOS). **The bug was top-k weight normalisation:**
Qwen1.5-MoE uses `norm_topk_prob=false` (expert weights are the raw full-softmax probabilities at the
top-k, NOT renormalised to sum 1), while `MoeRouter` was renormalising (Mixtral-style) → routed
contribution over-scaled → incoherent. Fix: `MoeRouter.SelectTopK(..., normalize)` + `GPT1Config.
NormalizeExpertWeights` (threaded stack→block→router) + loader reads `{arch}.expert_weights_norm`
(absent ⇒ false for qwen2moe). `MoeRouterTests` covers both modes. **Overfit now runs Qwen-MoE
(14B total / 2.7B active) coherently, in pure C# on CPU — pure-managed MoE inference.**
**F32 decode-parity — DONE.** `Qwen2MoeQuantParityTests.QuantizedBlock_MatchesF32Dequant_Layer0`
runs real layer-0 through `Qwen2MoeFeedForwardBlock` twice — once with the file's Q8_0 expert/shared
weights, once with an F32 dequantization of the SAME weights (same router ⇒ identical top-k routing,
so the only delta is projection quant error). Result: **relative max-diff 0.48 %** (< 5 % bar) — the
quantized MoE block is numerically faithful to F32. (A whole-model F32 baseline is infeasible —
Qwen1.5-MoE in F32 is ~57 GB — so parity is validated per-block via `LoadExpertsF32`.)
**Q5_0/Q5_K coverage — DONE.** `GgmlDequant.DecodeQ5_0Block` (legacy 32-elem, 22 B) +
`DecodeQ5_KBlock` (256-elem super-block, 176 B) mirror ggml-quants.c; wired into
`GgufReader.LoadTensorAsF32` (so dense tensors auto-route through the F32→Q8 fallback) and into
`GgufLlamaLoader.LoadExperts` (Q5 experts dequant + per-expert re-quant to Q8 — near-lossless from a
5-bit source, reuses the Q8 dot kernel; streamed one expert at a time via `GgufReader.LoadQ5RegionAsF32`
so peak load RAM is a single expert's F32, not the whole tensor's). **The smaller Q4_K_M variant
(8.84 GB — `_M` mix puts Q5_0 on ~half the layers' `ffn_down_exps`) now loads end-to-end and answers
"capital of France?" → "Paris"** (`Qwen2MoeQ4KMTests`, [LongFact]); decoder bit-unpacking unit-tested
in `GgmlDequantTests` (Q5_0 5th-bit routing, Q5_K per-group masks). The earlier dump confirmed the
file is 67% Q4_K / 14.5% Q8_0 / 14.5% Q5_0 / 3.5% Q6_K — only Q5_0 was the blocker (no Q5_K present,
but it's covered for other `_K_M` mixes).
**Mixtral — DONE.** Routed-only MoE (8 experts, top-2, no shared expert; `llama` arch with expert
metadata). `Qwen2MoeFeedForwardBlock` now treats `sharedFeedForwardLength == 0` as "no shared expert"
and reduces to the routed sum alone; `GPT1Config.HasSharedExpert` (threaded stack→block) drives it,
set by the loader from the presence of `ffn_*_shexp` tensors. `norm_topk_prob` default is now
arch-aware (false for qwen2moe, **true** for Mixtral/routed-only). The loader also handles **both**
GGUF expert layouts: merged 3-D `ffn_gate_exps` (Qwen-MoE) **and** the older per-expert 2-D
`ffn_gate.{e}.weight` (Mixtral) via `LoadExpertsSplit` (same resident dispatch as a dense FFN —
Q4_K verbatim/mmap, Q5→Q8, Q8 native, F32). **The real Mixtral-8x7B-Instruct Q4_K_M (25 GB,
32L / dModel 4096 / GQA 32:8 / expertDff 14336) loads + decodes end-to-end in pure C# on CPU**
(`MixtralEndToEndTests`, [LongFact]: finite/in-range logits, no degenerate collapse; routed-only
combine unit-tested in `Qwen2MoeFeedForwardBlockTests.NoSharedExpert_Mixtral_…`). **GGUF-embedded tokenizer (SPM) — DONE.** `GgufTokenizer` reads the `tokenizer.ggml.*` metadata
(tokens / scores / token_type / special ids) and implements the SentencePiece path
(`tokenizer.ggml.model == "llama"`): `▁` whitespace escaping + optional space-prefix, the score-driven
greedy bigram merge from llama.cpp's `llm_tokenizer_spm`, and `<0xNN>` byte fallback for OOV chars;
decode reassembles byte tokens (multi-byte UTF-8) and strips the leading space. Typed metadata-array
accessors added to `GgufReader` (`GetMetaStringArray`/`GetMetaFloatArray`/`GetMetaIntArray`). **Mixtral
now generates human-readable text in pure C# with NO side-loaded tokenizer** — `MixtralEndToEndTests`:
"[INST] What is the capital of France? [/INST]" → *"The capital of France is Paris. It's located in
the north-central part of the country…"*. SPM algorithm unit-tested on a synthetic vocab
(`GgufTokenizerTests`: merge preference, byte/multi-byte fallback, space-prefix, BOS, round-trip) +
real-Mixtral-vocab round-trip (incl. accents + em-dash). The gpt2/byte-level-BPE path throws a clear
`NotSupportedException` (Qwen/Llama-3 already have native `QwenTokenizer`/`HuggingFaceBpeTokenizer`).
**GGUF tokenizer gpt2/BPE path — DONE.** `GgufTokenizer` now also handles byte-level BPE
(`tokenizer.ggml.model == "gpt2"`: Qwen / Llama-3 / GPT-2) — bytes → GPT-2 `ByteLevelAlphabet`
(extracted as a shared helper, also used by `HuggingFaceBpeTokenizer`), pre-tokenized by a regex
selected from `tokenizer.ggml.pre` (cl100k for qwen2/llama-bpe, GPT-2 pattern for gpt-2), merged by
merge rank, with special tokens split out. So **Qwen/Llama-3 now tokenise from the GGUF with no
side-loaded `tokenizer.json`**. Validated: synthetic BPE invariants (merge rank, space marker,
special tokens, round-trip) + real Qwen-MoE-vocab round-trip (Polish/CJK/whitespace) + a **gold
cross-check that `GgufTokenizer` BPE output is byte-identical to the validated `QwenTokenizer`**.
Remaining (optional follow-ons, NOT blockers): native 5-bit dot kernel (skip the Q5→Q8 widen for a
tighter working set). MoE + GGUF-tokenizer tracks are complete.

**Scope note:** the routed math (the genuinely novel part) is done + tested. The remaining 2b is a
real integration chunk (3-D expert slicing incl. per-expert Q8_0 de-interleave, shared-expert gate,
stack wiring, real-file parity) — strictly additive (only fires for `qwen2moe`/MoE; the launched dense
path is unaffected), best done as a focused session, not rushed pre-launch.

## Research inputs (papers reviewed 2026-05-21)

Four external papers assessed for transfer into Overfit. Verdicts honest — most
value was validation/guidance, not drop-in algorithms.

- **arXiv:2406.09384 — *Rehearsal-free Continual Learning with Pretrained Models*** &
  **NeurIPS 2024 — *A Practitioner's Guide to Continual Multimodal Pretraining* (FoMo-in-Flux)**.
  Both: adapt a frozen base over time with PEFT (LoRA) without catastrophic
  forgetting; finding — simple LoRA + a sane data mixture is competitive with
  complex continual-learning machinery. **Used:** validates the per-deployment LoRA
  track (don't over-engineer) + motivated **rehearsal-lite** (`Gpt1LoRAFineTuner`
  `rehearsalCorpus`/`rehearsalFraction`, shipped — base-regime forgetting 9.09 → 0.012
  in a test). *Caveat:* both are vision/multimodal; transfer to GPT-on-metrics is
  conceptual, not literal.
- **MASCOTS'16 — *Gaussian Process for Urban Environmental Sensor Networks***.
  Correlated, diurnally-periodic multi-sensor data = our K8s metrics. **Used:**
  motivates a **GP (or cheaper EWMA/z-score) baseline** to rigorously benchmark the
  GPT anomaly detector — design sketch in `docs/gp-anomaly-baseline.md`. Deferred
  (separate experiment, not a product feature).
- **LinkedIn PDF — *Maximum Accuracy Computing* (Fourier, self-published)**. Claims
  "40× more accurate than deep learning"; grandiose framing, no peer review, demo on
  a personal site. **Rejected** — extraordinary claims without evidence; building on
  it risks credibility. The neutral kernel (real-FFT features for periodic signals)
  is standard DSP we don't need from this source.
- **arXiv:2605.27494 — *Grounded Cache Routing for RAG: When Is It Safe to Reuse an
  Answer?*** (Shah, Duke, 2026-05-26). Output-level semantic answer cache with **four
  safety gates** — query similarity, retrieved-evidence overlap, source-version
  validity, **lexical support** (cached answer's tokens present in fresh evidence; the
  paper's per-gate ablation isolates this gate as load-bearing). Stress-tested on
  12k real Qwen2.5-7B generations: USR (unsafe-served rate) **0% on HotpotQA** (naive:
  15-35%), **1.5% on mtRAG document-drift** (naive: 51.5% → 34× reduction), p50
  latency **1.04-1.07× no-cache baseline** (≈zero overhead). **Verdict — accept as
  post-launch feature.** Overfit has all primitives in place (`BertEncoder` cosine,
  `VectorStore`, `WordPieceTokenizer` for lexical overlap; need to add per-entry doc-id
  traceability + version hash to the store). Algorithmic, hardware-neutral (paper's
  vLLM/GPU setup is incidental). Strongest fit with the regulated-industries / EU-AI-Act
  positioning (USR is a measurable safety metric, not just a speed metric). Task in
  the [Medium-term Features](#features) section.

## Active track — Anomaly detector: synthetic-trained base → LoRA fine-tune

**Goal:** train a GPT anomaly model on synthetic K8s metrics, then LoRA-fine-tune
it on real production metrics. Plan: synthetic base → pull real metrics → LoRA adapt.

### Done this session

- **`OfflineTrainingJob` gradient bug — fixed.** `optimizer.ZeroGrad()` ran
  *between* `AggregateGradients` and `Step()`, wiping the just-aggregated master
  gradient → `Step()` applied only AdamW weight decay, so the model never learned
  (loss merely decayed toward `ln(VocabSize)`). Second bug: worker `Parameter.Grad`
  never zeroed (`BackwardFromGrad` accumulates) → workers summed gradient over all
  steps. Fix: removed the misplaced `ZeroGrad`, added per-worker grad zeroing in
  the `Parallel.For` body. Verified: Quick 500 steps 7.7→1.75 val loss.
- **Synthetic generator lift.** `Scripts/generate_k8s_metrics.py` rewritten with a
  shared per-pod AR(1) latent `load` driving cpu/rps/latency/throttle/queue/gc
  jointly. Old data was independent per-metric Gaussian noise → model hit an
  entropy floor (~1.08, plateau by step 800). New data has inter-metric correlation
  + temporal autocorrelation → loss keeps descending. CSV regenerated.
- **Prometheus parser — fixed.** `PrometheusMetricSource.ParseInstantResponse` and
  `PrometheusHistoricalSource.ParseRangeResponse` had their value-extraction bodies
  commented out (original used `EnumerateArray().ToArray()` — LINQ, banned in
  `Sources/Main`) → both sources silently returned empty lists. Rewrote LINQ-free
  (indexed `JsonElement`). Added `PrometheusParsingTests` + golden-JSON fixtures
  under `test_fixtures/prometheus/`.
- **GPT1 LoRA — Stage 1 (LM head), full loop closed.**
  - Risk PoC (`LoRAEffectiveWeightInjectionTests`): `graph.Linear` backprops
    through a *computed* weight node `W_eff = W_frozen + A@B` into A/B; numerically
    verified; an optimizer step over {A,B} leaves the base bit-identical.
  - `GPT1Model.LMHeadWeightProvider` — internal per-forward hook (null = production
    path, zero overhead).
  - `Gpt1LoRAFineTuner` — trains A/B with the base frozen (`Adam` over {A,B} only).
    Test: loss 2.26→0.0017, base bit-identical, `.bin` round-trips.
  - `Gpt1LoRAFile` — shared `.bin` format (magic "LORA", `LanguageModelHead` entry).
  - `Gpt1LoRAMergeAdapter` — inference-side weight-merge (`Enable`/`Disable`,
    idempotent, bit-reversible). Test proves the merge is visible to
    `CachedGpt1ModelAdapter` (the `GptAnomalyDetector` runtime) via the zero-copy
    `StackWeights._lmHead` ref: cached loss 4.14→0.006 on Enable, exact restore on
    Disable. No inference-kernel changes.

### Debt found (not yet addressed)

- **Prometheus auth (F3)** — `PrometheusMetricSourceConfig` / `...HistoricalSourceConfig`
  have no bearer-token / basic-auth field; auth-gated Prometheus/Thanos unreachable.
- **`MetricTokenizer` binning** — `error_rate`, `gc_pause_ratio`, `cpu_throttle_ratio`
  use linear `[0,1]` ranges for metrics that live in `[0, 0.05]` → ~95 % of the 64
  bins unused. Fix = tokenizer range change (log-scale / smaller Max) → needs retrain.
- **`OfflineTrainingJob` uses `System.Linq`** in `Sources/Main` — contradicts
  `BannedSymbols.txt` / CLAUDE.md. Compiles today; flag for cleanup.
- Diagnostics audit (earlier this session): `ArrayPoolEventSource` dead code,
  ~11 telemetry instruments declared but never recorded, `DiagnosticsRegressionTests`
  was vacuously green (F1/F4 fixed, F2/F3 outstanding).

### Anomaly + LoRA track — ALL ITEMS SHIPPED (was "resume here / pick one")

- **End-to-end integration test — ✅ DONE.** `Tests/Anomalies/GptAnomalyLoRAIntegrationTests.cs`,
  2 `[Fact]`s (green, ~1 s each): (1) LoRA trained on a regime lowers that regime's
  anomaly score through the detector, reversibly; (2) **`LoRA_AdaptedToRegime_StillFlagsInjectedAnomaly`**
  (added 2026-05-20) — after the adapter flattens the benign regime (adapted-normal
  ≈ 0.007 nats/token) an OOD injected snapshot still scores ≈ 20.4 (~2800× separation),
  proving adaptation lowers false positives without blinding the detector. Uses a
  random-init tiny model (no heavy base training — that's the separate item below).
- **Stage 2 — LoRA on FFN (W1/W2)** — DONE earlier (`Gpt1LoRAFfnTests.cs`).
- **Stage 3 — LoRA on attention Q/K/V/O per-head — ✅ DONE 2026-05-20.**
  `MultiHeadAttentionLayer` got per-head Q/K/V/O weight-provider hooks (mirroring
  the LM-head / FFN providers); `Gpt1LoRAFineTuner` fans out one A/B pair per head
  per targeted module per block; the `.bin` carries the per-entry head index;
  `Gpt1LoRAMergeAdapter` merges each delta into the matching per-head weight.
  `Gpt1LoRAAttentionTests.cs` (2 `[Fact]`): full `Attention` target = 16 adapters
  (2 blocks × 2 heads × Q/K/V/O), cached decode loss 2.97 → 0.02, exactly reversible;
  `Query`-only = 4 adapters (proves single-module per-head fan-out). All `LoRATargetModules`
  now supported by the fine-tuner.
- **Production base training — ✅ DONE 2026-05-21.** `OfflineTrainingJob` with
  `GptTrainingConfig.Production` (256d / 8 heads / 6 L, 10K steps, 8 data-parallel
  workers) on the 201 600-snapshot synthetic CSV → `k8s_anomaly_production.bin`
  (~19.8 MB). **Val loss 0.856** (~87 % below init), 1 h 03 m wall. Verified
  deployable: loads into `GptAnomalyDetector` (256d auto-detected) and discriminates
  — normal 6.02 vs anomaly 13.77, OOM anomaly correctly attributed to memory. The
  base's per-pod "normal" still carries residual surprise (6.02) because it's
  trained across all pods; that's exactly what per-regime LoRA (Stage 1/2/3) drives
  down for a specific deployment.
- **One-command product demo — ✅ DONE 2026-05-21.** `Demo/AnomalyConsoleDemo`
  now demonstrates the *whole* moat live in a single `dotnet run`, self-contained
  (trains a Quick base on the 201 600-row fixture CSV in ~17 s, no external file):
  **Phase 1 (base)** streams a benign regime + injects an incident; **Phase 2**
  fine-tunes an LM-head LoRA (rank 16, 300 steps, base frozen) on that pod's benign
  regime, merges it in place, and re-runs the same stream. Measured live (Quick base):
  benign "normal" **2.74 → 0.00** (false positives flattened), injected incident
  **11.78 → 24.38** (detection preserved and sharpened). This is the "adaptive
  per-deployment learning an inference-only engine can't do" story, shown not just
  described. Closes the "demoable in one command, documented honestly" goal below.
- **Production base validated end-to-end with per-pod LoRA — ✅ 2026-05-21, decision
  RESOLVED → Production.** Ran the demo's full before/after loop on the real
  `k8s_anomaly_production.bin` (256d/6L, via `--preset production --checkpoint`):
  the un-adapted base **scores the benign regime 5.68 — a false positive** (> the
  ⚠ 5.0 threshold), exactly because it's trained across all pods; a per-pod LM-head
  LoRA (rank 16, 300 steps) drives it to **0.00** while sharpening the injected
  incident **12.07 → 34.14**. This both (a) resolves the Medium-vs-Production base
  decision in Production's favour — it carries the cross-pod residual surprise that
  per-deployment LoRA is designed to remove — and (b) proves the moat on the actual
  deployable artifact, not a toy. Demo gained a `--preset quick|medium|production`
  flag so the loaded model's dims match the checkpoint. **Now regression-defended:**
  `GptAnomalyProductionLoRATests.ProductionBase_PerPodLoRA_FlattensBenignRegime_StillFlagsIncident`
  ([LongFact], auto-detects 256d, resolves the base from $OVERFIT_MODEL_DIR /
  test_fixtures / D:\, no-ops if absent) asserts benign < base & < 1.0 and incident
  > 5.0 with clear separation — bit-reproducible (5.68→0.00, 12.07→34.14, 13 s).

---

## Deferred — Qwen / Llama / quantization track

These are working in the codebase but **outside the current GPT-2 focus week.** Listed for visibility, not for prioritization.

### Slot 2b — quantized weight storage at inference — ✅ DONE (Q8_0 + Q4_K_M)

**Original gap:** Q4_K_M loader existed (decodes from disk) but dequantized
everything to FP32 on load — a 2 GB Q4_K_M file produced ~14 GB FP32 weights in
RAM. The "3B in 4 GB RAM" payoff required keeping weights quantized in RAM and
dequantizing per-block during matmul.

**Closed.** Two complementary in-RAM quant paths now ship — full design and
sub-step record in `docs/llamacpp-cpu-analysis.md` §5 steps 2 + 3:

- **Q8_0** (step 2): `Q8DotKernel` (symmetric F32→Q8 quantizer + INT8
  `vpmaddubsw` SIMD dot + sequential & parallel GEMV), `Q8Weight` (output-major
  Q8 weight storage), `DecodeWeight` (tagged precision-agnostic weight handle).
  LM-head + FFN gate/up/down + per-head attention Q/K/V/O all Q8-resident — the
  full decode matmul path. Loader reads native `Q8_0` blocks straight from a
  Q8_0 GGUF — no dequant/re-quantize on the loaded path.
- **Q4_K_M** (step 3): `Q4KDotKernel` + `Q6KDotKernel` (Q4_K × Q8_K and
  Q6_K × Q8_K AVX2 kernels with scalar fallbacks), `Q4KWeight` + `Q6KWeight`
  (output-major super-blocks). `DecodeWeight` widened to a 4-way tagged union
  `{F32 | Q8 | Q4_K | Q6_K}`; the decode blocks (FFN, single-head attention,
  LM-head) refactored to **per-weight dispatch** so a heterogeneous Q4_K_M file
  (`attn_q/k/o` + `ffn_gate/up` Q4_K, `ffn_down` + `attn_v` + `token_embd` +
  `output` Q6_K, per-head `Wo` Q8 — its headDim=128 contraction is below the
  256-element K-quant super-block) picks the right kernel per projection.
  Loader: native Q4_K + Q6_K reads + per-head contiguous byte slices.

**Three-way A/B across Overfit's own formats** (dev box, best-of-3, 24 timed
tokens after 4 warm-up, single stream):

| Format    | Load   | Decode      | Steady RAM |
|-----------|--------|-------------|-----------:|
| FP16-src  |  7.1 s | 13.29 tok/s |   5 902 MB |
| Q8_0      |  1.7 s | 13.28 tok/s |   5 847 MB |
| **Q4_K_M**| **1.4 s** | **14.56 tok/s** | **4 396 MB** |

Within Overfit, Q4_K_M is the best format — RAM + load win. *(Table is pre-fix;
the GQA K/V-once fix below lifts Q4_K_M decode to **17.2 tok/s** and would lift
Q8 similarly — not re-measured.)* Parity: Q8 32/32 (2.5); Q4_K_M 29/32, worst
swing 2.16 (3.4, teacher-forced vs same-file F32 baseline). Zero allocations per
decoded token preserved (`Demo_Gpt2Small_KvCacheDecode_AllocatesZeroBytesPerToken`).
680 / 0 / 68 `-c Release`.

**Same-file A/B vs LLamaSharp (2026-05-20).** LLamaSharp 0.27.0 on the *same*
`qwen.q4km.gguf`: **27.5 tok/s @ 3.2 GB** vs Overfit **17.2 tok/s** @ 4.4 GB
(after GQA K/V-once) — **llama.cpp ~1.6× faster, 27 % less RAM on equal footing**
(was ~2.0× before the fix). (Earlier "1.51× faster" was Overfit-Q4_K_M vs
LLamaSharp-FP16 — retracted.) Diagnostic: FP16→Q4_K_M sped llama.cpp 2.85×,
Overfit only ~1.0× → Overfit decode was overhead-bound, not bandwidth-bound.
First lever (GQA K/V-once: project K/V once per KV group, not per Q head) gave
+24 %, bit-identical. Defensible edge: 1 B vs 21 KB/token alloc, pure-managed,
AOT-clean, no native dep. Numbers: `overfit-bench/RESULTS.md`.

**Re-bench 2026-05-26 (after the MoE + GGUF-tokenizer tracks).** Same harness/file/prompt,
`overfit-bench -- qwen.q4km.gguf`: **18.92 tok/s, 1 B/token, load 0.7 s, steady working set 259 MB**
(36L d=2048, logits finite). Takeaways: (1) **no decode regression** — 18.9 ≈ the 19.5 recorded same-state
(thermal noise); the MoE/tokenizer work didn't touch the dense hot path. (2) The zero-alloc contract and
the ~1.45× speed gap to llama.cpp (27.5) both hold — attention-fusion remains the only identified speed
lever. (3) **The RAM story flipped — now measured apples-to-apples.** Restored the LLamaSharp `llama` mode in
`overfit-bench` (NuGet 0.27.0 + CPU backend) and measured BOTH via `Process.WorkingSet64` at the same two
lifecycle points on the same file:

| | Overfit | llama.cpp (LLamaSharp 0.27.0) |
|---|---|---|
| decode | 18.2 tok/s | 28.98 tok/s (**~1.6× gap**, holds) |
| working set, after load | 259 MB | 2626 MB |
| **working set, after decode** | **2037 MB** | **3203 MB** |
| load | 0.7 s | 1.2 s |

⚠ Honesty correction: the flashy "259 MB" is lazy-mmap **before pages are touched** — after a real decode
the weight pages page in and Overfit's working set rises to **2037 MB**. THAT is the fair "RAM to actually
run" number. Measured identically, Overfit uses **~1.16 GB (~36 %) LESS working set than llama.cpp**
(2037 vs 3203) — which *flips* the old "RAM parity (3.2 vs 3.2)" finding into a real win (mmap +
quantized-embedding-lookup did it, both landed 2026-05-25 after that parity measurement). Net standing vs
llama.cpp same-file, same-measurement: **Overfit −36 % working set, +faster load, 1 B/token; −1.6× decode
speed** (attention-fusion remains the only speed lever). The bench now keeps both modes:
`overfit-bench -- overfit|llama qwen.q4km.gguf`.

**Remaining (separate tracks, not gated on this):**
- Make decode further bandwidth-bound (resume-point option D, 1st lever done):
  fuse activation-quantize per-group, tighter GEMV unrolling, core-util profiling.
- Step 4 — tiled prefill GEMM (`batch > 1`). Helps TTFT + batched training, not decode.
- Step 5 — work-stealing chunk counter. Marginal for uniform GEMV; opportunistic.

### Slot 2c — FP16-resident weights — ATTEMPTED & REVERTED (May 2026)

Post-mortem of a refuted experiment. Kept as a record so the idea is not retried.

**Hypothesis:** `GgufLlamaLoader` up-casts FP16 GGUF weights to F32 at load, so
decode streams 2× the bytes. Keeping weights FP16-resident (`Half`) and widening
F16→F32 in the matmul should give ~2× throughput + ~½ RAM — premised on decode
being memory-bandwidth-bound.

**Benchmark that motivated it** — same Qwen2.5-3B `qwen.gguf` (FP16),
single-stream CPU decode:

| Metric | Overfit (F32 up-cast) | LLamaSharp (native llama.cpp) |
|--------|----------------------:|------------------------------:|
| Decode throughput | 2.58 tok/s | 9.67 tok/s |
| Peak working set | 14.4 GB | 6.0 GB |

**Built:** full FP16-resident path — a `MatrixWeight` precision-carrier union
threaded through the decode weight structs, `ProjectHalf` / `AccumulateHalf`
kernels, `GgufReader.LoadTensorAsF16`, an `fp16Resident` A/B toggle. Kernel parity
bit-identical to F32; 666 tests green.

**Measured — hypothesis REFUTED.** Rigorous A/B, best-of-3, same model:

| Metric | F32 (baseline) | FP16-resident |
|--------|---------------:|--------------:|
| Throughput | 2.58 tok/s | 1.68 tok/s — **−35%** |
| Steady RAM | 14.36 GB | 15.85 GB — regressed |

**Why it failed:**

1. **Decode is compute-bound, not bandwidth-bound.** Overfit F32 decode reads only
   ~31 GB/s — far under the ~50–80 GB/s DRAM ceiling. DRAM was never the
   bottleneck, so halving weight bytes unblocks nothing and the F16→F32 widen is
   pure added cost.
2. **No fused widen is possible in managed .NET.** A register-fused single-pass
   kernel needs a per-vector F16→F32 intrinsic. .NET 10 exposes none — no `F16C`
   class, no `Half` overload on `Vector256.ConvertToSingle` / `WidenLower`. The
   hardware `vcvtph2ps` is reachable only via whole-span `TensorPrimitives`
   (forcing a scratch round-trip). A hand-rolled SIMD bit-twiddle widen costs
   ~9 ops per 8 elements vs the 1-op hardware convert → fusing would be *slower*.
   So **−35% is irreducible** for FP16-resident decode in managed C#.
3. **RAM regressed** — the F32→F16 load conversion churns multi-GB F32 buffers on
   the Large Object Heap; the GC retains those segments.

**Outcome:** the entire FP16-resident path was reverted (`MatrixWeight`,
`ProjectHalf`, the loader F16 path, the toggle) — this post-mortem is all that
remains. Durable takeaway: **Overfit decode is compute-bound** — decode-speed
work must target compute, not memory bandwidth.

**What this means for the LLamaSharp gap.** The 3.75× gap is kernel quality, not
weight precision. Overfit F32 at 2.58 tok/s ≈ 31 GB/s — *under* the DRAM ceiling;
a bandwidth-saturating F32 kernel alone reaches ~4 tok/s (12 GB/token ÷ ~50 GB/s),
≈1.5× of measured headroom. The next decode-perf lever is therefore kernel-side —
blocked GEMV (`SingleTokenProjectionKernel.Accumulate` re-streams the whole output
vector for *every* input element → ~2× redundant memory traffic on FFN / LM-head),
parallelizing the small per-head matmuls, tighter SIMD — **not** precision tricks.
Structural follow-on is quantized storage (**Slot 2b**): Q4/Q8 cut bytes more than
FP16 and integer-SIMD is llama.cpp's actual weapon. Honest ceiling: pure-managed
C# cannot emit every intrinsic llama.cpp uses (F16C proved that) — a realistic
target is closing 3.75× → ~2–2.5×, not parity.

### Q4_K_M integration parity test

- [ ] Download `qwen2.5-3b-instruct-q4_k_m.gguf` (Ollama or HF) to `c:\qwen3b\qwen.q4km.gguf`.
- [ ] Run the existing `GgufQ4KMParityTests.Q4KM_TopTokenMatches_FP16Baseline_OnCanonicalPrompt` test (already written, currently `[LongFact]` + skip-if-missing).
- [ ] Tolerance: top-1 matches; max abs logit diff within Q4_K_M expected range (~0.5-1.5 % relative).

Synthetic unit tests already cover the algorithm; this test catches bit-layout regressions against llama.cpp/Ollama.

### Other quant formats

- [ ] Q5_K dequantizer (occasionally appears in mixed quant files; 176 bytes/block).
- [ ] Q4_0 / Q5_0 / Q5_1 (legacy formats; lower priority — Ollama defaults to K-quants).
- [ ] Q2_K / Q3_K_S (very aggressive quant; experimental quality).

### LoRA training

GPT1 LM-head LoRA training **landed** this session — see "Active track" above
(`Gpt1LoRAFineTuner`: graph-integrated effective-weight injection, `Adam` over
the LoRA `Parameter`s only, base frozen). Remaining items are the Llama-family
and broader-module scope:

- [x] Backward restricted to adapter parameters with frozen base — GPT1 LM head (Stage 1).
- [x] Adam over LoRA factors only — GPT1.
- [x] Extend to FFN (Stage 2) and attention Q/K/V/O (Stage 3) — GPT1 (QLoRA: whole base
  frozen-quantized Q4_K/Q8, validated on a real anomaly task).
- [ ] Backward through Linear / RMSNorm / SwiGLU / attention for the Llama family.
- [ ] Demo: overfit on a few samples, verify the adapter steers generation.

Opens "fine-tune LLM locally in pure C#" story. Major scope.

#### GGUF→training bridge (the "sztandar" / headline — real RAM win)

The QLoRA op + quantizer + GPT1 whole-base wiring shipped, but GPT1Model is F32-native
so quantizing there only *adds* copies — the real memory win requires loading an
already-quantized Qwen/Llama GGUF **directly** into a training graph so F32 is never
allocated. That needs the Llama-family forward as autograd ops (frozen Q4_K bases +
trainable LoRA + RMSNorm/RoPE/SwiGLU/GQA). Multi-session (~3–5).

- [x] **Session 1 — RMSNorm + SiLU autograd ops.** `OpCode.RmsNorm` / `OpCode.SiLU`,
  `TensorMath.RmsNorm(graph, x, gamma, eps)` (per-row, saves inv-RMS aux, dInput + dGamma,
  seq + `OverfitParallelFor` parallel) and `TensorMath.SiLU(graph, x)` (SwiGLU gate,
  `TensorPrimitives.Sigmoid` SIMD core). Backward dispatch wired; FD-validated
  (`RmsNormTests`, `SiLUTests` — forward parity seq+parallel, dInput/dGamma central-difference).
- [x] **Session 2 — RoPE autograd.** `OpCode.Rope`, `TensorMath.Rope(graph, input, cos, sin)` —
  adjacent-pair / GGUF-NeoX layout (`input [rows, headsPerRow·headDim]`, `cos/sin [rows, halfDim]`,
  constants/no-grad), per-pair 2-D rotation; backward = inverse rotation (orthogonal → `Rᵀ`, sin
  negated), accumulated into dInput; seq + `OverfitParallelFor`. `RopeTests`: **forward bit-faithful
  to the inference `RopeKernel.Apply`** (the layout-fidelity guarantee — GGUF base rotates identically
  in train + inference), FD backward, and forward∘inverse round-trip recovers input.
- [x] **Session 3 — GQA scaled-dot-product attention backward.** Rather than a bespoke GQA SDPA,
  added `OpCode.ExpandKvHeads` / `TensorMath.ExpandKvHeads(graph, input, kvHeads, groupSize)` — the
  training equivalent of HF `repeat_kv`: forward broadcasts each KV head to its query-head group
  (head `qh` reads KV head `qh/groupSize`), and the **GQA-specific backward sums each group's gradient
  into the shared KV head** (it was read groupSize times). Feeds the already-validated MHA SDPA
  unchanged (one head = one batch slice). `GqaAttentionTests`: forward broadcast, FD backward of the
  group reduction, and **full GQA path** (expand→3-D SDPA on 4 query : 2 KV heads) FD-checking dQ/dK/dV.
  Layout-agnostic (dim-0 head axis), groupSize=1 = MHA copy. KV-head ↔ token-major projection wiring
  is Session 4's job.
- [x] **Session 4 — trainable Llama block assembly.** `DeepLearning/TrainableLlamaBlock` —
  a full Pre-LN Llama/Qwen decoder block as one autograd forward: `h = input + Attn(RMSNorm(input,γ1))`,
  `y = h + SwiGLU(RMSNorm(h,γ2))`, with **every projection a frozen `IDequantRowSource` (Q4_K/Q8)**
  via `FrozenQuantizedLinear` and the only trainable params the two RMSNorm gains (LoRA layers on later
  exactly like the GPT-1 path). Combined-tensor / GGUF-faithful layout (Wq `[nQ·dH,dModel]`,
  Wk/Wv `[nKV·dH,dModel]`, Wo `[dModel,dModel]`, gate/up `[dFF,dModel]`, down `[dModel,dFF]`) so
  Session 5's loader feeds it with zero repacking. Wiring: RMSNorm → frozen QKV → RoPE (token-major,
  all heads one call) → `Transpose01` to head-major → `ExpandKvHeads` (GQA) → SDPA → `Transpose01` back
  → frozen O → residual → RMSNorm → frozen SwiGLU(`SiLU(gate)⊙up`) → residual. One new op needed —
  `OpCode.Transpose01` (rank-3 axis-0/1 swap, exact backward). `TrainableLlamaBlockTests` (4 Q : 2 KV):
  forward shape, **end-to-end FD backward** on input + both γ, and a γ-only train loop that drops loss
  with the **frozen quantized base provably bit-identical** before/after.
- [x] **Session 5 — loader→training wiring + e2e on real Qwen GGUF + RAM measurement. THE PAYOFF.**
  Bridge plumbing (zero-copy, no repack): `DecodeWeight.AsRowSource()` (Q4_K/Q6_K/Q8 → frozen
  `IDequantRowSource`; F32 rejected), `ConcatRowsDequantSource` (per-head Wq/Wk/Wv → combined
  `[nH·dH, dModel]`), `ConcatColsDequantSource` (per-head Wo → `[dModel, nH·dH]`), and an internal
  `CachedLlamaInferenceEngine.GetTrainableLayer(i)` accessor. RoPE op extended to **split-half**
  (Qwen `RopeSplitHalf=true`) alongside adjacent-pair, threaded through the block. `QLoraGgufBridgeTests`
  (runs in the fast suite): adapter parity, AsRowSource dispatch, and a loader-shaped per-head block
  trains FD-validated. **`QwenGgufQLoraE2ETests` [LongFact], executed on the real
  Qwen2.5-3B Q4_K_M GGUF (2.1 GB):** layer-0 (dModel=2048, 16 Q : 2 KV, headDim=128, split-half) wired
  straight into a `TrainableLlamaBlock` — gradients flow to input + both γ, **frozen 4-bit base
  bit-identical**, and **building the trainable base allocated ~0 KB vs ~294 MB if the layer's F32 were
  materialized** (×36 layers ≈ ~10 GB of F32 base avoided). The sztandar proven: a real already-quantized
  GGUF fine-tunes in pure .NET CPU with the 4-bit base never expanded to F32.
- [x] **Full-model RAM measured (`QwenGgufTrainingRamTests` [LongFact], real Qwen2.5-3B Q4_K_M, exact via
  the graph arena `CurrentOffset` high-water — no GC/pool noise).** base = **1.96 GB** 4-bit (Σ layer quant
  weights 1.62 GB; F32 expansion would be ~11.5 GB). Per-block training activation (fwd+bwd): 73.5 MB @ T=128,
  149 @ T=256, **306 @ T=512**, 644 @ T=1024 (≈linear — FFN dFF=11008 dominates, not attention T²).
  Full-model QLoRA @ T=512: **without checkpointing ~13.3 GB** (all 36 layers co-resident), **with gradient
  checkpointing ~3.1 GB** → **fits a 16 GB CPU box with room to spare.** Confirms the promise with real
  numbers; checkpointing (built for GPT-1, unwired into this path) is the lever for bigger models.
- [x] **FULL TRAINABLE MODEL + checkpointing + real fine-tune (Option A).** `DeepLearning/TrainableLlamaModel`
  assembles the whole stack: frozen quantized embedding (per-token `DequantizeRow`) → N
  `TrainableLlamaBlock`s (each under `graph.Checkpoint`) → trainable final RMSNorm → frozen quantized LM head
  → logits; `FromEngine(engine, rank, …)` builds it straight from a loaded GGUF (zero repack). Trainable =
  per-block LoRA (`LlamaBlockLoRA` over all 7 projections, B-zero init) + RMSNorm gains; optional LM-head LoRA
  (off by default — high-variance on a 152k vocab). Next-token CE loss seeds `logits.Grad` →
  `BackwardFromGrad`. **Synthetic proof (`TrainableLlamaModelTests`, fast suite): a tiny model OVERFITS a
  sequence — loss 4.31→0.0004, greedy 24/24 reproduced; gradient checkpointing is bit-identical (maxAbs 0.0)
  AND trains (4.82→0.0004).** **Real proof (`QwenGgufQLoraFineTuneE2ETests` [LongFact], Qwen2.5-3B Q4_K_M):**
  fine-tune all 36 layers under checkpointing — **loss 12.24→2.81** (smooth/monotonic), **P(correct next-token)
  3.1e-4→6.2e-2 = 201× up**, **frozen 4-bit base bit-identical**, **peak process WS 2.92 GB** (incl. the 2 GB
  base — matches the ~3.1 GB extrapolation), ~1.27 s/step. The sztandar is now a working fine-tuner. Open:
  adapter save/load + multi-sequence batches (assembly, not correctness).
- [x] **KNOWLEDGE-INJECTION SHOWCASE (`QwenGgufKnowledgeInjectionDemoTests` [LongFact]).** The demo: teach
  real Qwen2.5-3B a fact it cannot know — a made-up metal "Zorvex" mined only in "Tarnholm" — then ask it.
  Added greedy `TrainableLlamaModel.Generate(graph, prompt, maxNew, eos)` (checkpointed forward = no grad
  buffers, tiny arena) wired to `QwenTokenizer.Load(modelDir)` (tokenizer is separate from the GGUF). LM-head
  LoRA enabled (`loraOnLmHead: true`) for output capacity. **BEFORE: "…the city of" → "gow" (base clueless).
  Fine-tune on 3 sentences (~250 steps CPU): loss 14.67 → 0.0000. AFTER: "…the city of" → "Tarnholm. Zorvex"
  — recites the taught fact coherently.** KEY STABILITY FINDING: Adam blew up catastrophically at low loss
  (0.23→6.58 spike) with the default `Epsilon 1e-8`; **`Epsilon = 1e-4` gives a smooth monotonic descent to
  0** (the 1/√v amplification when gradients vanish on a tiny overfit set). This is the showable
  "fine-tune an LLM on your CPU in .NET, no GPU/Python" demo.
- [x] **Adapter SAVE/LOAD — fine-tune persists to a file (train → save → load → use).**
  `TrainableLlamaModel.SaveAdapter(path)` / `LoadAdapter(path)` serialize ONLY the trained delta (every LoRA
  A/B + RMSNorm gain, deterministic order, bulk `MemoryMarshal` IO) — the frozen 4-bit base is never
  rewritten (no GGUF export; loading stays one-directional). Tens of MB vs the 2 GB base. **Synthetic
  round-trip (`TrainableLlamaModelTests`, fast): train→save (65 KB)→fresh model→load → loaded logits
  bit-identical to trained (maxAbs 0.0), greedy 18/18 reproduced.** **Real round-trip
  (`QwenGgufQLoraAdapterRoundTripTests` [LongFact]): train Qwen-3B on the Zorvex fact → save adapter file →
  FRESH model from the untouched base → load → recites "Tarnholm".** Closes the loop into a portable
  "knowledge module" attached to a frozen base.
- [ ] **(FUTURE, separate topic) Fast fine-tuned decode = LoRA on the optimized inference engine (≈1.13× llama.cpp).**
  Two decode paths exist today: the **training** model (`TrainableLlamaModel`, has KV-cache + our LoRA but
  NAIVE per-row-dequant+`Dot` kernels) and the **inference** engine (`CachedLlamaInferenceEngine`, KV-cache +
  repacked 8×8 GEMV kernels benchmarked ~1.13× behind llama.cpp, but runs the frozen base WITHOUT LoRA). To
  serve a fine-tuned model at llama.cpp-class speed, hook the trained adapter into the optimized inference
  forward: the Q4_K base can't be F32-weight-merged (the existing `LlamaLoRAAdapter.Enable` path needs F32),
  so add the LoRA as a **side GEMV** (`+ (x·A)·B`, one tiny rank-r matmul per projection per token) inside
  `CachedTransformerBlock`/`CachedMultiHeadAttention` decode, gated by an attached adapter. ~2–3 sessions,
  higher risk (touches the hot inference path). NOTE: the training-model KV-cache (next bullet) does NOT help —
  it's naive single-threaded, so it's 6× slower than the already-parallel uncached recompute at demo lengths;
  the win requires parallel kernels, which is this item. Generation was never the bottleneck (~0.4 s/token).
- [~] **Training-model KV-cache (Option A) — BUILT + CORRECT, but it does NOT speed up generation (honest
  finding).** `TrainableLlamaModel.GenerateCached(...)` + `TrainableLlamaBlock.DecodeStep(...)` — incremental
  single-token decode (no autograd): only the new token flows through the layers, attending its query over a
  per-layer K/V cache (RoPE via the inference `RopeKernel`, manual softmax attention, plain-span RMSNorm/SwiGLU
  + naive per-row dequant + LoRA side-GEMV). **Bit-faithful: cached greedy tokens == uncached `Generate`
  (`TrainableLlamaModelTests.GenerateCached_MatchesUncachedGenerate`).** BUT measured on real Qwen-3B
  (`QwenGgufCachedDecodeSpeedTests` [LongFact]): cached **2700 ms/token vs uncached 424 ms/token — 6× SLOWER**.
  Root cause: the premise was wrong. Generation was never the bottleneck (~0.4 s/token) because the uncached
  forward already parallelizes the dequant-matmul (`FrozenQuantizedLinear` over all cores + amortized dequant),
  while the cached decode uses naive SINGLE-THREADED per-row kernels. The cache only pays off at long contexts
  AND with parallel kernels (= Option B / parallelizing `ProjVec`+`LmHeadArgmax`). The real wall-time sink in
  the demos is TRAINING (~5 s/step), not generation. Kept as a correct reference path; no speed claim.

---

## Performance backlog

### ★ Decode worker headroom — BUG FOUND + FIXED 2026-07-05 (+61…+76 % on small machines)

Went in to test **thread pinning / affinity** (hypothesis: the scheduler and SMT siblings cost us; llama.cpp
pins). **Pinning was refuted — and the measurement found something much bigger.**

Pinning *alone* HURTS: 16 physical cores / 16 workers = **14.46** tok/s vs 32 logical / 16 workers = **24.94**.
But every config where `workers == available CPUs` landed on ~14 regardless of core count, SMT, or CCD. The
cause is not topology — **the decode pool SPINS, so with zero spare CPU the dispatcher is starved.**

| CPUs | workers | headroom | tok/s |   | CPUs | workers | headroom | tok/s |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 32 | 31 | 1 | **23.02** |  | 32 | **32** | **0** | **14.73** |
| 16 phys | 8 | 8 | **26.11** |  | 16 phys | **16** | **0** | **14.46** |
| 8 phys | 4 | 4 | 20.51 |  | 8 phys | **8** | **0** | 13.75 |

The cliff is exactly at `headroom == 0`; **one free core is the difference between 23.02 and 14.73.**

**The bug:** the default was `Math.Min(procCount, 10)`, so **every box with ≤10 logical CPUs defaulted INTO the
cliff** (`workers == procCount`). Fixed to `Math.Min(Math.Max(1, procCount - 1), 10)`. Faithfully re-measured
(affinity limits real CPUs; `OVERFIT_PARALLEL_WORKERS` simulates `procCount`, because
`Environment.ProcessorCount` is cached at startup and does NOT see a later affinity mask — a first "verification"
that ignored this produced a nonsensical 1.87 tok/s and was thrown out):

| CPUs | old default | new default | gain |
|---:|---:|---:|---:|
| 4 | 5.91 | **9.62** | **+63 %** |
| 8 | 11.69 | **18.83** | **+61 %** |
| 10 | 12.13 | **21.38** | **+76 %** |
| 32 | 24.94 | 26.29 | unchanged logic |

Guarded by `DecodeWorkerHeadroomTests`. **This one line beats every lever in the decode plan** (1a +1-2 %,
Phase 3 +4.6 %, 2a +30 %). ⚠️ **Suspicion to check: OverThink on Android** — 8 cores ⇒ old default 8 workers ⇒
headroom 0. The 3.8 tok/s figure may be partly this.

### Huge pages / TLB — CLOSED 2026-07-05, do not pursue

Hypothesis: a 2 GB DRAM-bound model pressures the TLB, so large pages / `madvise(MADV_HUGEPAGE)` should help.
**Two independent reasons it dies:** (1) Windows does not support large pages for **file mappings** (only private
commits) and the weights are mmap'd — `MADV_HUGEPAGE` is Linux-only; (2) the data says cache/TLB is not the
bottleneck anyway — the **96 MB V-Cache CCD measured 13.75 tok/s vs 13.96 on the 32 MB CCD** (identical). A 2 GB
model dwarfs any L3, so enlarging pages or cache changes nothing.

### Worker auto-tuning — deliberately NOT built

The generalisable law has a mechanism and is now one line in `ResolveDecodeMaxWorkers` (never take every CPU).
A per-CPU/per-RAM **lookup table would be overfitting to n=1 machine** (everything above was measured on a single
Ryzen 9 9950X3D). And there is nothing left to tune: **above the cliff the curve is flat** — 8 workers 25.67 vs
24 workers 25.84 — because decode is DRAM-bound and ~8 threads already saturate the bus. If a box with
materially different bandwidth ever shows up, the honest form is an opt-in `overfit tune` that measures once and
caches a per-machine profile, never a hard-coded table.

### Training CPU-saturation track

Goal: on 16+ core machines, training step pegs CPU at near-100 % across all cores instead of single-digit cores idle/under-utilized. Inventory of what's already parallel-capable:

| Op | Parallel today | Threshold |
|----|----------------|-----------|
| `LinearKernels` backward (input + weight) | ✅ above 1M ops | `ParallelThreshold = 1_048_576` |
| `Conv2D` forward + backward | ✅ | per `TensorMath.Convolution` |
| `MaxPool2D`/`AvgPool2D` forward + backward | ✅ | `TensorMath.Pooling` |
| `LSTM` forward + backward | ✅ over batch | `TensorMath.Sequence` |
| `Adam.Step` | ✅ over parameters | always parallel |
| `ScaledDotProductAttentionBackward` | ✅ over batch | **flipped to default ON** (was experimental flag) |
| `ScaledDotProductAttention` forward | ❌ sequential over batch | candidate |
| `LayerNorm` / `RMSNorm` forward + backward | ❌ sequential | per-token, usually small enough |
| `Embedding` backward (scatter-add) | ❌ sequential | hard — atomic conflicts |
| Activation kernels (GELU/SwiGLU/ReLU) | ❌ SIMD-only, not threaded | usually fine |

Concrete items:

- [x] **`EnableParallelAttentionBackward` default ON** — bit-identical to sequential (parallel only over batch dim, no cross-batch reduction), author measured ~27 % backward speedup on data-parallel TinyShakespeare. Stable training path now uses it by default; remaining single-thread case is intentional fallback for batch=1 where Parallel.For overhead would be pure waste.
- [x] **`BatchSequentialThreshold` 128 → 32** — `MaxPool2D` / `AvgPool2D` / `LSTM` forward + backward + bias-add (`TensorMath.Algebra`) parallelize across batch only above this threshold. MNIST training (B=64) was falling into the sequential branch (64 < 128) → `MaxPool2D` ate 42 % of epoch time on a 32-core machine, single-threaded. After lowering to 32: **MNIST training −38 % wall clock** (5551 → 3456 ms for 5 epochs, full MNIST 60k @ B=64), with MaxPool2D dropping from 383 ms/epoch to ~87 ms/epoch (−77 %). Full sweep still green (617/0/63). The 32 floor keeps sequential for genuinely small batches where Parallel.For overhead would dominate.
- [ ] **True batched training (B > 1)** — *biggest single lever for CPU saturation.* MHA in training path is currently batch=1 only. With B=8/16, every existing parallel-over-batch path (attention forward + backward, layer-norm batches, optimizer-over-params) starts working. 2-3 days; ROADMAP'd separately under Qwen track but is GPT-2 training prerequisite.
- [x] **Drill-down per-OpCode backward profiler** — added `ComputationGraph.BackwardProfileEnabled` toggle + `GetBackwardOpProfile()` + `ResetBackwardProfile()`. Zero overhead when off (one null-check), ~50 ns per op when on. MNIST CNN training 5-epoch aggregated breakdown: **Linear 52 %**, Conv2D 20 %, ReLU 14 %, MaxPool 7 %, Reshape 4 %, SoftmaxCE <1 %. Linear dominates because Linear(1352→64) backward = 2 GEMMs (dW + dX) per batch × 4685 batches. Already parallel above 524k/1M thresholds; near peak FP32 throughput for this matrix size. Next deeper investigation requires actual CPU-utilization sampling (not just timing).
- [x] **CPU-utilization probe** during MNIST training — added `Process.TotalProcessorTime` sampling at run + per-epoch granularity. **Measurement on 32-core Ryzen 9 9950X3D: only 6.81 / 32 cores effective (21.3 % utilization).** All major kernels have `Parallel.For` wired, yet 79 % of CPU stays idle. Root cause: ~47k `Parallel.For` calls per MNIST epoch × ~5-10 µs each dispatch overhead = 230–470 ms / 550 ms epoch = **40–85 % of epoch wasted in TPL dispatch/sync**, not compute. **CPU-saturation track is fundamentally blocked by `Parallel.For` overhead, not missing parallelism.**

### Zero-allocation custom Parallel.For — THE unlock

Standard `System.Threading.Tasks.Parallel.For` per call:
- ~3 KB managed allocation (closure object, `Task[]` chunks, internal bookkeeping)
- ~5-10 µs dispatch overhead (Task scheduling, thread wake)

Both kill us:
- The **3 KB alloc** broke the 0 B / generated token claim when we tried `ProjectParallel` on LM head (~92 KB / 31 tokens). Blocked allocation-free parallel inference.
- The **5-10 µs overhead** dominates training when many small Parallel.For calls fire per batch (MNIST: 47k calls/epoch). Caps utilization at ~20 % regardless of how many `Parallel.For` we add.

**`OverfitParallelFor` — current status (implemented, bulk-wake dispatcher):**

- [x] Pre-spawned `N = ProcessorCount` persistent threads, all parked on one shared `SemaphoreSlim`.
- [x] Per-chunk `ChunkState[]` descriptors filled by dispatcher; workers claim a unique index via `Interlocked.Increment` on a 128 B cache-line-padded counter.
- [x] **Bulk wake via `SemaphoreSlim.Release(N)`** — one syscall releases N tokens; kernel scheduler resumes waiters in parallel (NOT serially the way `N × AutoResetEvent.Set` would). This is the architectural win — it bypasses the ~32 µs floor that per-worker `Set` dispatchers hit at 32-fanout.
- [x] Function-pointer dispatch (`delegate*<int, int, void*, void>`) — no closure, no delegate alloc.
- [x] **Exception propagation** via `ExceptionDispatchInfo.Capture` — body throws are caught per chunk, surfaced to caller with original stack trace. No unhandled-exception process crash.
- [x] API: `OverfitParallelFor.For(int rangeStart, int rangeEnd, delegate*<int, int, void*, void> body, void* context)`.
- [x] Tests: 0 B proof, correctness vs sequential, boundary cases, 10k-iter stress, exception propagation + post-throw recovery.

**Measured (Ryzen 9 9950X3D, 32 logical cores, full benchmark sweep):**

| InnerIters (body work) | Sequential | Parallel.For (TPL) | OverfitParallelFor | Speedup vs Seq | Alloc Overfit | Alloc TPL |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (empty) | 0.54 µs | 5.9 µs / 3.0 KB | ~6.6 µs / 0 B | — | 0 B ✅ | 3.4 KB |
| 1k (~1 µs body) | 92 µs | 48 µs | ~70 µs | — | 0 B ✅ | 3.8 KB |
| 100k (~100 µs body) | 3.5 ms | 326 µs | ~190 µs | 18× | 0 B ✅ | 6.8 KB |
| 1M (~1 ms body) | 34 ms | 1.46 ms | ~1.19 ms | **28.7×** | 0 B ✅ | 9.9 KB |

**Where it wins:**
- **Vs `Parallel.For`:** competitive in time (within 15% in worst case, **1.7× faster from 100 µs body upward**), and ~3000× cheaper on allocations across the whole range.
- **Vs Sequential:** the crossover at InnerIters ≈ 100 means parallelization pays from any body work above ~100 cycles. At InnerIters = 1M (1 ms body), 28.7× speedup ≈ 90% of 32 logical threads — near-optimal scaling.

**Why bulk wake beats N × Set:** earlier iterations (per-worker `AutoResetEvent` + simple/spin-then-park hybrid) hit a hard ~32-47 µs dispatch floor at 32-fanout — N `Set` calls serialize at the kernel because each one is a distinct event signal. `SemaphoreSlim.Release(N)` queues N pulses inside its internal lock and exits; the OS scheduler then resumes the waiters roughly in parallel (Windows `WakeConditionVariable` / Linux `futex_wake(N)` semantics). Result: ~5-7 µs dispatch at 32-fanout. Pre-bulk-wake prototypes (`OverfitParallelForHybrid`, `OverfitParallelForBulkWake`) were retired once this design landed.

**Migration to `OverfitParallelFor` — measured impact (Ryzen 9 9950X3D, MNIST 60k batch=64 small CNN, 5 epochs):**

| Phase | Wall time | Cores effective | Linear bwd total | Total CPU |
|---|---:|---:|---:|---:|
| TPL baseline (ROADMAP previous) | 5551 ms | 6.81 / 32 (21.3%) | — | ~37.8 s |
| + `LinearKernels` BackwardInput + AccumulateWeightGrad migrated | 5107 ms | 6.15 / 32 (19.2%) | 1133 ms | ~31.4 s |
| + `TensorMath.Pooling` (MaxPool/AvgPool fwd+bwd) migrated | (same) | (same) | (same) | (same) |
| + `TensorMath.Algebra` (AddBias, MatMul fwd, MatMulAdd variants) migrated | (same) | (same) | (same) | (same) |
| + `ComputationGraph.Linear` forward migrated | **4870 ms** | **7.03 / 32 (22.0%)** | **982 ms** | ~31.0 s |

**Result:** −12 % wall time, −18 % total CPU consumed, +0.6 percentage points cores effective vs. the TPL baseline. Linear backward time dropped 13 % (1133 → 982 ms). All 626 correctness tests still green.

**Honest read of why we didn't hit the 60-80 % cores-effective target:** MNIST CNN at this scale is Amdahl-limited. The 4685 batches/epoch × ~10 micro-ops/batch include many serial slices (graph reset, allocator paths, small-Linear sequential branch for Linear(64→10) at B=64 = 41k ops well below parallel threshold, copy input/target, optimizer.ZeroGrad). The dispatcher *itself* is no longer the bottleneck — eliminating its overhead saved CPU but not enough sequential work was unblocked to fill more cores.

**Where the dispatcher win will actually show up:**
- Larger models (GPT-2 scale): per-op body work is ~ms scale, dispatch is ≤1% overhead, and parallelism is much higher.
- Allocation-free hot paths (LM head, attention forward, prefill kernels): 0 B/call is now the bottleneck-relevant property — same wall time as TPL with no GC pressure.
- High-frequency inference loops where TPL's 3 KB/call hits Gen0 hard.

**Migrated / parallelized call sites:**
- `LinearKernels.BackwardInput` + `AccumulateWeightGrad`
- `TensorMath.Pooling`: `MaxPool2D` fwd+bwd, `GlobalAveragePool2D` fwd+bwd
- `TensorMath.Algebra`: `AddBias`, `MatMulRaw`, `MatMulAdd_A_BT_Raw`, `MatMulAdd_AT_B_Raw`
- `ComputationGraph.Linear` (forward batched parallel path)
- **`TensorMath.Gelu` (forward + backward)** — was a pure sequential scalar `for` loop. Now (a) **OverfitParallelFor over chunks**, AND (b) **SIMD-batched inside each chunk** via `TensorPrimitives.Multiply/Add/Tanh/Subtract` pipeline on 1024-element tiles (stackalloc'd scratch on worker thread stack). Tile size chosen to fit in L1 cache so the multi-pass pipeline stays hot. The scalar `MathF.Tanh` is replaced with SIMD `TensorPrimitives.Tanh` (polynomial approximation). Cumulative GPT-1 batch=32 GELU backward: **1774 → 42 ms (42× faster)** vs the initial scalar-serial baseline, of which the SIMD pipeline contributed an additional **2.6×** on top of the prior parallel-over-chunks version.
- **`TensorMath.ScaledDotProductAttention` forward** — was sequential `for (b)` over batch. Symmetric to the existing parallel-over-batch backward — same `batchSize > 1 && work >= AttentionParallelWorkThreshold` guard. Per-batch work is independent (only writes to per-batch slices of `attnWeights` / `output`). For multi-head training the SDPA call sees effective batch = `B × H` (heads flattened), so even at training `B=1` the parallel path kicks in for `H ≥ 4`. Biggest single contributor to cores-effective in the session: GPT-1 batch=32 wall **191 → 114 ms (−40%)**, cores effective **7.93 → 13.85 / 32 (+75%)**. With this change the forward path becomes the second-most parallelized chunk of the training step (after Linear bwd), and `cores effective` finally crosses 1/3 of physical capacity at training batch sizes.
- **`TensorMath.LayerNorm` (forward + backward)** — was sequential per-row. Forward parallelizes trivially (rows independent). Backward needs per-worker partial accumulators for `dGamma[i]` / `dBeta[i]` (shared across rows) — stackalloc'd by caller, sequential SIMD merge after parallel pass. dInput parallelizes in the same pass (per-row, no shared writes). Falls back to sequential when `C > 4096` to keep stackalloc bounded.

**Cumulative GPT-1 training-step impact** (Ryzen 9 9950X3D, 4-layer GPT-1, dModel=128, dFF=512, seqLen=128, per `GPT1CpuUtilizationProbeTests`):

| BatchSize | Wall/step start | + GELU parallel | + LayerNorm parallel | + SIMD-batched GELU | + SDPA fwd parallel | Total Δ wall | Cores start → end |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 126 ms | 82 ms | 78 ms | 78 ms | **65 ms** | **−48%** | 3.48 → **10.40** / 32 |
| 16 | 217 ms | 137 ms | 112 ms | 106 ms | **74 ms** | **−66%** | 2.99 → **11.44** / 32 |
| 32 | 414 ms | 243 ms | 198 ms | 191 ms | **114 ms** | **−72%** | 3.89 → **13.85** / 32 |

Per-op backward (batch=32, aggregated over 20 steps): GELU 1774 → 109 ms (−94%, 16× faster). LayerNorm 800 → 39 ms (−95%, 20× faster). Now Linear (1042 ms — already parallelized, just much more numerous) dominates the backward profile. The remaining sequential ops on the critical path are smaller: Add residuals (62 ms), Reshape (55 ms), Embedding (2.5 ms).

This validated the whole strategy: dispatcher was correct from the start, and *each* additional element-wise kernel we move from sequential to OverfitParallelFor produces a real, measurable win — the playbook works for any per-element activation/normalization op.

**Future patterns from `vs2022-performance-patterns.md` (catalog for later sessions):**

Reviewed the 45 patterns extracted from VS 2022 decompilation. Already adopted:
- ✅ Custom Parallel wrapper (pattern 28) → `OverfitParallelFor`
- ✅ Lock-free `Interlocked` (pattern 7)
- ✅ Cache-line padding `[StructLayout(Explicit)]` (pattern 32) → `PaddedCounter`
- ✅ `[MethodImpl(AggressiveInlining)]` selectively (pattern 3)
- ✅ `readonly struct` defaults (pattern 2)
- ✅ `EventSource` per component (pattern 29) → `ArrayPoolEventSource`
- ✅ `ArrayPool<T>.Shared.Rent` (pattern 10) → `OverfitPool<T>`
- ✅ **`[module: SkipLocalsInit]` assembly-wide (pattern 22)** — landed with audit (caught LoRA accumulator bug as side-effect)
- ✅ **SIMD-batched element-wise via `TensorPrimitives` (pattern 25)** — applied to GELU as the reusable template
- ✅ **`FrozenDictionary` for load-once lookup tables (pattern 6)** — `BytePairEncoder` `_tokenToId` / `_mergeRanks` / `ByteDecoder` converted from `Dictionary` to `FrozenDictionary`. AOT-clean (zero IL2026/IL3050 from `ToFrozenDictionary`). The merge-rank table is the hottest — `BpeEncode` scans every adjacent pair O(parts²) per word.
- ✅ **`[CallerArgumentExpression]` in guards (pattern 45)** — `TensorKernelGuards` validators auto-capture argument names; error messages now name the offending span/tensor.

Rejected after evaluation:
- ❌ `Expression.Compile` / `DynamicMethod` / `ActivatorUtilities.CreateFactory` (patterns 39, 40, 41) — blocked by `BannedSymbols.txt` (no reflection in AOT path)
- ❌ `[Conditional("DEBUG")]` on validators (pattern 9 fragment) — our `TensorKernelGuards` are caller-contract enforcement, not pure invariants; stripping degrades debuggability without measurable gain
- ❌ `Channel<T>` (pattern 21) — we don't do streaming/IPC
- ❌ `IValueTaskSource` / `ValueTask` / `IAsyncDisposable` (patterns 14, 15, 16) — no async paths
- ❌ WPF patterns (36, 37, 38, 44), MultiplexingStream (19), F# persistent collections (43) — different domains

**Worth adopting in future sessions, prioritized by ROI:**

| Pattern | Where | Effort | Value | When to do it |
|---|---|---:|---|---|
| **5 — Segmented arrays (LOH avoidance)** | `TensorStorage<T>` for tensors >85 KB | ~1 day | Removes GC pauses on big-model paths | When LOH symptoms hurt (GPT-2+ scale) |
| **35 — `[PerformanceSensitive]` analyzer hint** | New custom analyzer for "no-alloc hot path" methods | ~few days | Compile-time enforcement of zero-alloc contracts | When team grows beyond one person |
| **23 — `[InlineArray(N)]` for packed structs** | Maybe GGUF block quant formats | ~hours | Cleaner than `LayoutKind.Explicit` for fixed-size content | Opportunistic — when touching binary IO |
| **34 — `[ThreadStatic] Stack<T>` per-thread pools** | Hot per-thread scratch in training kernels | ~1 day | Cache-warmer than central pool | When profiler shows pool contention |
| **24 — `[ModuleInitializer]`** | Run-once setup (e.g. native lib preload) | ~10 min | One-time setup without static-ctor surprises | If we ever ship native deps (we don't today) |
| **20 — Batched telemetry (timer + flush + persist)** | `ArrayPoolEventSource` upgrade for long training runs | ~hours | Reduce ETW overhead on training | When telemetry becomes a real concern |
| **30 — `UnmanagedBufferAllocator` (`NativeMemory.Alloc`)** | `TensorStorage<T>` for very-long-lived weights | ~1 day | Removes weights from GC's scope entirely | For multi-GB models pinned in memory |

**The pattern from VS2022 that drove the biggest win in this session:** **#25 (SIMD-batched element-wise via TensorPrimitives)**. The template — outer `OverfitParallelFor.For` over chunks + inner `TensorPrimitives` pipeline on stackalloc'd tiles — is now established and ready to reuse for any future scalar kernel (Sigmoid, SwiGLU, SiLU, custom activations).

**Lesson learned — when NOT to parallelize element-wise:**
Attempted parallelizing `TensorMath.Add` (residual add) and **measured a regression** (+20% wall on GPT-1 batch=32, +55% on Add backward itself). Reverted. `Add` is **memory-bandwidth-bound** (read 2 arrays + write 1 = 3× data movement). On a typical desktop memory subsystem (~50 GB/s) 2-3 cores already saturate the bus, so the OverfitParallelFor dispatch overhead (~10 µs cold) is pure cost.

**Rule of thumb for future migrations:** parallel pays for element-wise ops only if the body is *compute-bound* (heavy per-element math like GELU/LayerNorm/Linear). For *memory-bound* ops (Add/Subtract/Scale/element-wise copy), keep sequential SIMD — the bandwidth ceiling caps you regardless of core count, and dispatch overhead becomes net negative.

**Pending migrations / parallelizations (deferred):**
- `TensorMath.Convolution` (Conv2D fwd+bwd) — per-worker workspace pattern needs `GCHandle`-pin or POH refactor of `Conv2DWorkspace`. Expected gain on MNIST: ~100-150 ms / 5 epochs (Conv2D backward 459 ms, fwd ~570 ms aggregated). Worth doing if/when GPT-2 batched training adds bigger Conv-heavy workloads.
- `TensorMath.Sequence` (LSTM) — relevant for LSTM training.
- `TensorMath.Attention` (`EnableParallelAttentionBackward` path) — relevant for transformer training at B > 1.
- `Optimizers.Adam` — parameter-parallel update, hot path during training.
- Various lower-priority sites: `DataAugmenter`, `FastRandomForest`, evolutionary noise tables, anomaly training.

**Pattern for future migrations:** `fixed (T* p = span)` block around `OverfitParallelFor.For(start, end, &ChunkBody, &ctx)` where `ChunkBody(int chunkStart, int chunkEnd, void* contextPtr)` loops over the chunk and reads pointers from the context struct.
- [-] **LinearKernels threshold tuning** — verified for MNIST that current thresholds (BackwardInput 524k, AccumulateWeightGrad 1M, ForwardBatched 500k) are correctly placed: Linear(1352→64) at B=64 = 5.5M ops → already parallel; Linear(64→10) at B=64 = 41k ops → sequential (41k / 32 cores = 1.3k ops per thread, well below Parallel.For overhead). No measurable benefit available at MNIST scale. **Revisit when GPT-2 batched training (B>1) lands** — smaller per-batch ops in GPT may benefit from lower thresholds.
- [ ] **SIMD path for `MaxPool2DForwardWithIndicesNchw` (training path)** — inference path has `TensorPrimitives.Max` fast path for pool=2; training path is scalar because index tracking complicates SIMD (need comparison masks to select max value AND record source index). Estimated saving on MNIST: ~40 ms / epoch (~7 % of post-threshold-fix epoch time). 1-2 hours focused work + parity tests vs scalar.
- [ ] **`ScaledDotProductAttention` forward parallel-over-batch** — symmetric with the backward we just enabled. Likely 15-20 % forward speedup at B ≥ 4.
- [ ] **Threshold tuning** — `LinearKernels.ParallelThreshold = 1_048_576` is set for inference (avoid Parallel.For overhead on tiny matrices). For training where the per-call work is amortized across many tokens, lower threshold may pay. Need per-op profiling first.
- [ ] **Data-parallel training as first-class API** — `TinyShakespeareDataParallelTrainingTests` proves N model replicas + gradient averaging works. Promote to public API, with idiomatic worker pool, for "N cores → N replicas" training scaling on a single machine.
- [ ] **Batched linear kernel** — `LinearKernels.ForwardBatched` measured win at batch 64/256 vs ONNX Runtime.
- [ ] **Backward kernels** — Linear/Conv backward through pure span kernels where practical.
- [ ] **Optimizer kernel profiling** — Adam/AdamW state updates, zero-grad allocation sources (currently ~96 B/Step on Adam — already tracked by skipped tests in `AdamOptimizerBehaviorTests`).
- [ ] **CPU SIMD audit** — AVX2/AVX-512/AVX10 where available.
- [ ] **Thread-scaling stabilization for large training workloads.**

### Correctness

- [ ] Numerical equivalence tests across scalar/SIMD paths.
- [ ] Determinism policy for parallel training kernels.

---

## Market-driven priorities (2026 external research)

External scan (May 2026): the LLM-inference job market, RAG / vector-DB adoption, the self-hosted-LLM trend, and the .NET AI ecosystem. Honest top-line:

**Job postings from the highest-paying companies are a trap, not a map.** They converge on Python + CUDA + vLLM/TGI/Triton + Kubernetes + GPU clusters + distributed training (AI-infra postings grew +47 % YoY). None of it is addressable by a pure-C#, CPU-first, zero-native-dependency engine — chasing it means losing to vLLM/TensorRT on their own ground. The opportunity is the *adjacent, underserved* space.

What the market actually validates for Overfit's niche, ranked:

| # | Priority | Market signal | Effort / status |
|---|----------|---------------|-----------------|
| 1 | **Embedding model support** (BGE / E5 / multilingual-e5) | Vector-DB market $2.46B (2024) → $10.6B (2032), 27.5 % CAGR; Gartner: 30 %+ of enterprises on vector DBs by 2026; enterprise hybrid-retrieval intent tripled in one quarter. RAG is *the* enterprise LLM pattern. Embedding models are encoder transformers — single forward pass, no KV-cache — **CPU-friendly**: the one mainstream workload squarely in Overfit's wheelhouse. | ✅ **DONE 2026-05-28 (BERT encoder family).** `WordPieceTokenizer` (vocab.txt, BasicTokenizer + greedy WordPiece, [CLS]/[SEP], accent strip) + `BertEncoder` (bidirectional via `causalMask:false`, post-LN blocks, learned pos+token-type emb, embeddings LayerNorm, mean/last pooling + L2) + native `BertSafetensorsLoader` (HF [out,in]→[in,out] transpose + per-head Q/K/V column-split / O row-split, prefix auto-detect, no Python) + `BertConfigReader` + turnkey `SentenceEmbedder.FromPretrained(dir)` → `VectorStore`. Targets validated 2026-05-29 — all 384-d, all bit-parity to HF/PyTorch on real weights: **all-MiniLM-L6-v2 cosine 1.000000** (mean pool, no prefix), **BAAI/bge-small-en-v1.5 cosine 0.999999** (CLS pool + query instruction), **intfloat/e5-small-v2 cosine 1.000000** (mean pool + `query:`/`passage:` prefixes). Transpose/head-split round-trip verified directly (`BertSafetensorsLoaderTests`); semantic ordering + VectorStore top-1 retrieval covered by `MiniLmEmbeddingEndToEndTests` / `BgeAndE5EmbeddingEndToEndTests` [LongFact] (`OVERFIT_MINILM_DIR` / `OVERFIT_BGE_DIR` / `OVERFIT_E5_DIR`). `SentenceEmbedder.ForMiniLm` / `ForBgeEnV15` / `ForE5` bake the per-family conventions in. **GELU question SETTLED:** repo's tanh-approx GELU is indistinguishable from BERT's exact erf (cosine 1.0). Tokenizer also exact-matches HF fast tokenizer. **Follow-ons:** SentencePiece tokenizer (XLM-R / multilingual-e5), BGE-base / BGE-M3 for bigger contexts. |
| 2 | **Deepen regulated / private-inference positioning** | EU AI Act reaches full enforcement Aug 2026 (high-risk AI requires audit trails, explainability, human oversight). Self-hosted deployments report −75 % data-breach incidents — but 175k exposed Ollama servers are actively exploited ("LLMjacking"): self-hosted-*as-a-server* is itself a risk. Overfit-as-a-library-in-process (no exposed endpoint) is structurally safer. | Mostly copy. Started: `docs/scenarios/regulated-industries.md` + README "What Overfit is not". Add the "library-in-process > exposed server" security argument. |
| 3 | **In-memory quantization** (Q4_K / Q6_K dequant-fused matmul) | Every inference-engine comparison lists quantization as core (llama.cpp = "CPU-first + quantization"). It is the path to running *larger* models on CPU / edge. | Already specified — see **Slot 2b** above. This research promotes it from "deferred" to a named priority. (The FP16-resident shortcut, Slot 2c, was tried and reverted — see its post-mortem; quantization is the real lever.) |
| 4 | **Audit / inference-record primitives** | EU AI Act mandates reproducible audit trails + explainability for high-risk AI. Overfit already has deterministic greedy decode and file-versioned weights — the missing piece is a first-class, opt-in decision record (input + model hash + output + timestamp). | ~few days. Grounds the prior generic "telemetry" idea in an actual regulation. |
| 5 | **Microsoft Agent Framework / Semantic Kernel adapter** | Microsoft consolidated Semantic Kernel + AutoGen into "Microsoft Agent Framework" (Oct 2025); SK is in maintenance mode. Do **not** build a competing agent framework — be the inference + embedding *backend* it calls. Distribution via Microsoft's own ecosystem. | ~2 days, once embeddings (item 1) land. |

**Explicitly out of scope** — the market confirms these are GPU + Python territory; competing there loses: GPU-throughput serving (vLLM / TGI / TensorRT), distributed / multi-node training, multi-cloud orchestration, a homegrown agent / LangChain framework, multimodal (vision + text).

This section is a strategic overlay — it ranks and justifies; the tactical breakdowns live in "Slot 2b" (quantization), the "Active track" (LoRA), and "Medium-term / Features" below.

---

## Medium-term

### Features

- [x] **Chat templates** — DONE: `ChatTemplate` (ChatML detect/render from GGUF metadata) + `ChatSession` (system/user/assistant turns), used by `Demo/AgentDemo` and the chat tests.
- [x] **`OverfitClient` facade** — DONE 2026-05-29. `LanguageModels/OverfitClient.cs`. `using var client = OverfitClient.LoadGguf(path); client.AddSystem("..."); var reply = client.Send("...");` — wraps GgufReader+ChatTemplate.Detect, tokenizer auto-pick (QwenChatTokenizer first if `vocab.json+merges.txt` present, else HuggingFaceBpeTokenizer fallback), engine+session creation, ChatSession with sensible ChatML stop sequences, and Greedy GenerationOptions default. Sync `Send` + async `SendAsync` (thread-pool wrap). Exposes underlying `Chat` for constrained outputs / streaming. Tests: 2 fast (null/missing-path guards) + 1 [LongFact] e2e on real Qwen (mechanical assertion — content quality is a separate semantic concern).
- [ ] **ONNX: LSTM/GRU operators** — enables recurrent model import.
- [x] **Depthwise Conv** (group=channels) — MobileNet-style models. DONE 2026-05-27: `DepthwiseConv2DLayer` + `TensorMath.DepthwiseConv2D` (SIMD AXPY inner kernel, padding/stride/bias, FD-verified). Pair with a 1×1 `ConvLayer` for a full separable block.
- [ ] Standalone Softmax and CrossEntropy in addition to fused loss.
- [ ] **`GroundedAnswerCache` — safe semantic answer cache for RAG** (post-launch, per
  arXiv:2605.27494, see [Research inputs](#research-inputs-papers-reviewed-2026-05-21)).
  Wraps `VectorStore` + `BertEncoder` + `WordPieceTokenizer` into a 4-gated answer cache
  with a per-entry record `{Q embedding, retrieved doc IDs, version hashes, cached answer,
  answer-token-set}`. Public API: `cache.TryGet(query, retrievedDocs, currentVersionMap,
  out answer)` admits a cached answer only when ALL four gates pass — (1) query cosine
  ≥ τ, (2) retrieved-doc-ID overlap ≥ τ, (3) version map matches recorded, (4) lexical
  support (Jaccard of answer-tokens ∩ fresh-evidence-tokens ≥ τ). Expose **USR**
  (unsafe-served-rate) as an observability counter — same metric the paper uses. Requires
  small extension to `VectorStore.Add(id, vector, payload, version?)` to carry version
  hash per stored doc. Scope ~200-400 LoC, AOT-clean. Strategic fit: regulated-industries
  story gets a *measurable safety* primitive, not just speed. Differentiator vs
  Microsoft.SemanticKernel / LangChain.NET (neither ship a gated semantic cache today).

### Agentic loop primitives (from `D:\Agentic-AI-LangGraph` review, 2026-05-27)

The agentic stack today has the *primitives* (tool calling, JSON-constrained / structured output, streaming,
multi-turn `ChatSession` + sliding-window memory, embeddings) but **no reusable agent loop on top**. A review of a
LangGraph patterns repo (17 recipes) surfaced four small, AOT-friendly, in-character additions that build *on* the
existing primitives — NOT a graph/agent framework (state-graph runtime, multi-agent supervisor, HITL gate,
time-travel, parallel fan-out, thread checkpointing, agentic-RAG router + vector store are deliberately **out of
scope** per "What Overfit is not trying to be" — a user builds those on the primitives):

- [x] **ReAct / tool-use agent loop** — DONE 2026-05-29. `LanguageModels/Agents/ReActAgent.cs` is a driver
      over `ChatSession` + `ToolCallConstraint` that loops *model → tool-call → observe → repeat* until
      the model calls the auto-registered synthetic `finish({answer:...})` tool or hits `MaxSteps` (the
      circuit-breaker overlap). Includes the `ExtraMaskedTokensConstraint` shim that masks the chat
      session's stop-token id (e.g. Qwen's `<|im_end|>` ≠ `<|endoftext|>`) while the envelope is
      incomplete — the constraint's own EOS-mask alone wasn't enough to prevent mid-envelope truncation.
      6 unit tests (`ReActAgentTests`) drive the pure loop via the internal `RunLoop(askForToolCall fn)`
      hook — synthetic replies, no model needed: single-tool-then-finish, multi-step observation
      chaining, step cap, unknown tool, handler-throws, non-string answer fallback. An e2e
      `[LongFact]` is wired but Qwen 2.5-3B Q4_K_M is below the reliability floor for multi-turn
      constrained tool calling (greedy → unclosed JSON strings; temperature → Unicode garbage); ship
      with a bigger / instruction-tuned model.
- [x] **Self-reflection / critic loop** — DONE 2026-05-29. `LanguageModels/Agents/CriticLoop.cs`,
      model-agnostic (`Func<string,string>` generator + `Func<string,CriticVerdict>` critic), reflects
      previous-attempt + critic feedback into next prompt. Built on `CircuitBreaker` so cap and
      timeout come for free. 3 unit tests.
- [x] **Circuit breaker / step guard** — DONE 2026-05-29. `LanguageModels/Agents/CircuitBreaker.cs`:
      generic `Run<T>(maxIterations, maxElapsed?, iterate, isAccepted)` → `CircuitBreakerResult<T>`
      with `Accepted` / `MaxIterations` / `Timeout` outcomes. Used by `CriticLoop`. 4 unit tests.
- [x] **Summarizing memory** — DONE 2026-05-29. `LanguageModels/Memory/`: pure
      `ChatHistoryCompactor.Plan(history, summarizeAtChars, recentTurnsToKeep)` returning a
      `CompactionPlan`; model-driven `SummarizingChatSession` wraps a `ChatSession` and triggers
      compaction in-line on `Send`. Rebuilds as `[system…] + [running summary] + [recent N turns
      verbatim]`. Uses save/restore around summarisation so the prompt doesn't leak. Required adding
      `ChatSession.AddUser` / `AddAssistant` (history seeding — useful generally for transcript
      replay). 5 unit tests on the compactor; e2e left as host-driven (needs a model).

### Distribution

- [ ] NuGet package metadata polish.
- [ ] Sample Blazor app showing streaming generation in browser via Rx/IAsyncEnumerable adapter.
- [ ] Benchmark page: a Format × Model × RAM × tokens/s table.

---

## Long-term ideas

- Graph compilation for fixed-shape training/inference graphs.
- Custom autograd operators with explicit forward/backward registration.
- Mixed precision training.
- Data loading and preprocessing pipeline improvements.
- Optional GPU backend investigation without compromising CPU-first design.
- Model/dataset packages outside the small core runtime.

---

## What Overfit is not trying to be

- Not a general-purpose replacement for PyTorch or TensorFlow.
- Not a Python shim.
- Not GPU-first.
- Not a model zoo in the core package.

The differentiator remains pure C#, predictable memory behavior, Native-AOT compatibility, and competitive CPU inference (including small/medium language models) on consumer hardware.

---

## Contributing

Performance-sensitive PRs should include:

- correctness tests;
- before/after BenchmarkDotNet output;
- allocation measurements;
- documentation updates when public behavior changes.

License: GNU AGPLv3. For commercial licensing, contact devonbike@gmail.com.
