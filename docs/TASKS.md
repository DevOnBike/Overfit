# Task registry

One place to look, and one way to name a task in conversation.

## The ID scheme, and why it is shaped this way

`AREA-localid`. The area prefix is new; **the local id is whatever that document already called it**, so
nothing is renumbered and every existing cross-reference inside those documents keeps working.

| prefix | area | source of truth |
|---|---|---|
| `AN-` | anomaly guard | `docs/aiops/aiops-backlog.md` |
| `TG-` | `[LongFact]` release gate | `docs/test-gate-backlog.md` |
| `MS-` | metric-source seam (replay) | `docs/specs/anomaly-guard-metric-source-seam-plan.md` |
| `PS-` | PSI CPU-pressure channel | `docs/specs/anomaly-guard-psi-cpu-channel-plan.md` |
| `RS-` | .NET runtime signals | `docs/specs/anomaly-guard-runtime-signals-plan.md` |
| `XC-` | cross-cutting, no plan of its own | this file |
| `NR-` | NASA Power of 10 follow-ups | `ROADMAP.md` §NASA Power of 10 |
| `AV-` | agentic / interop / vision backlog | `ROADMAP.md` §Agentic / interop / vision backlog |
| `TT-` | audio / TTS backlog | `ROADMAP.md` §Audio / TTS backlog |
| `AL-` | adoption / launch roadmap | `ROADMAP.md` §Adoption / launch roadmap |
| `LG-` | llama.cpp scope-gap snapshot | `ROADMAP.md` §llama.cpp scope-gap snapshot |
| `MO-` | Mixture-of-Experts post-launch track | `ROADMAP.md` §Post-launch track #1 — MoE |
| `LT-` | LoRA / QLoRA training track | `ROADMAP.md` §Active track (anomaly LoRA) + §Deferred → LoRA training + GGUF→training bridge |
| `QT-` | quantization / format coverage | `ROADMAP.md` §Deferred — Qwen / Llama / quantization track |
| `PB-` | performance backlog | `ROADMAP.md` §Performance backlog |
| `MK-` | market-driven priorities | `ROADMAP.md` §Market-driven priorities |
| `MT-` | medium-term features & distribution | `ROADMAP.md` §Medium-term |

The problem it solves: three plan documents each have a "Task 1", so "do task 3" was ambiguous. `MS-3`
is not.

**Status vocabulary**, chosen so that the uncomfortable states have names:

| status | means |
|---|---|
| `DONE` | finished and, where a gate applies, it passed |
| `PART` | genuinely half-done, with the remaining half named |
| `OPEN` | not started |
| `FAILED` | ran, did not meet its own criterion |
| `UNGATED` | **code is committed but never verified or reviewed** — the state that otherwise hides |
| `DEFER` | deliberately not now, with a recorded reason |

---

## Anomaly guard — `AN-`

| id | status | task | note |
|---|---|---|---|
| `AN-A1` | **FAILED** | 24-hour run on a frozen configuration | 10.63/day against a 1-9/day criterion, plus one cycle failure. Tail contaminated by a test run on the same box. 11 events cannot decide the band anyway — the interval is 5.5-19 |
| `AN-A2` | DONE | alert delivery | found two defects, one ours |
| `AN-A3` | DONE | second-instance cost | 42 Mi, 1m CPU |
| `AN-A4` | OPEN | evening diurnal check | closes the last two unverified floors; verifiable only by waiting |
| `AN-A5` | DONE | StatefulSet and HPA | both halves measured 2026-08-08. StatefulSet: findings median **14** vs **1** for the same pods replaced under an unchanged topology — the control that made the first table mean something. Bonus: a plain `rollout restart` lifts findings 0→6, and the guard counts `pods=24` mid-rollout |
| `AN-B1` | DONE | client-readiness documentation | folded into `aiops-client-readiness.md` |
| `AN-C1` | OPEN | re-tune the heap floor on an aged population | current number is extrapolation |
| `AN-C2` | DEFER | `GcPauseRatio.minTrendChange` | one finding is not evidence enough to arm a gate |
| `AN-D1` | OPEN | peer reports a fixed pod property as a recurring anomaly | diagnosed 2026-08-06, unfixed. **Independently corroborated 2026-08-08**: 272 of 298 cycles carried exactly one finding, which is not a noise distribution |
| `AN-D2` | **DONE** | answered 2026-08-08 and re-confirmed 2026-08-09 with the OOM channel live. The row's claim was half right: the peer family IS blind to a single OOMKill, but the guard is not. Yesterday only `ContainerRestarts` caught it and `OomEventsRate` contributed nothing; today the incident is **led by `OomEventsRate`**. See `AN-D6` (the injector could not produce an OOM) and `AN-D8` (the channel read a dead series) — the row was written when neither worked |
| `AN-D6` | **DONE** | `POST /fault/oom` produces a real OOMKill again — verified 2026-08-08 on a live pod: `restartCount 0 -> 1`, `reason=OOMKilled`, `exit 137`, **in under 5 s**. Three causes were fixed, and the second is the one that makes this non-obvious: (1) only 2 bytes of each 64 MB chunk were touched, so the cgroup — which charges RESIDENT memory — saw 37 MB of a 512 MiB limit while VmSize reached 6.65 GB; (2) **managed allocation cannot reach a cgroup limit at all**, because the container-aware GC caps the heap at 75% of it (384 of 512 MiB) and throws a managed `OutOfMemoryException` first — so the allocation is now native (`Marshal.AllocHGlobal`, no unsafe), outside the GC heap; (3) the throw was swallowed by a detached `Task.Run` with no continuation, and is now logged. The loop is bounded at 4 GiB (8x the limit) and **logs an error if it finishes**, because an injector that fails silently is what created this task. `/fault/clear` now cancels and frees, so it is a real undo. Deployed by replacing ONE pod, not rolling the Deployment — A5 measured that twelve fresh pods cost 0,1,0,1,6,4,3 findings over seven cycles |
| `AN-D7` | **DONE** | `blind=1` identified 2026-08-08: it is **`CpuThrottleRatio`**, which has **0 series** in the `lab` namespace because `container_cpu_cfs_throttled_periods_total` only exists on containers with a CPU limit and no `lab-workload` pod has one. Known, documented in `k8s/README.md`, and correctly counted — the counter is doing its job. Not the OOM channel: that one has 14 series and binds fine, see `AN-D8` |
| `AN-D8` | **DONE** | `OomEventsRate` reads the series that carries an OOM, and it **fires**. Verified end to end 2026-08-09 on the live lab: `POST /fault/oom` -> `OOMKilled` exit 137 in 5 s -> the channel moved 0 -> 1.14 on the killed pod with **all eleven others at 0** -> the guard opened an incident **led by `OomEventsRate`** (*"Held above zero for 12% of the window, 7 of 60 observations, peak 1.14"*) with `ContainerRestarts` beside it. **A second defect was found and fixed after the first deploy**: a join returns NO SERIES for pods whose right-hand side is absent, which on a healthy cluster is every pod — so the first fix turned the channel from *always zero* into *always blind* (`blind` 1 -> 2, `Blind on OomEventsRate` every cycle, measured within twenty minutes). The `or <same counter> * 0` tail fills those pods with an explicit zero; `blind` is back to 1. Verbatim `query` bindings and `InertChannels()` shipped with it |
| `AN-D3` | OPEN | a CPU rise on every replica at once is invisible to all four families | measured 2026-08-01, never diagnosed to the end |
| `AN-D4` | OPEN | the heap oscillates with a period near the evaluation window | measured 2026-08-05 |
| `AN-D5` | DONE | `incidents=` counter disagreed with the events | |
| `AN-E1` | DEFER | multi-scope, slices 3-5 | |
| `AN-E2` | DEFER | a CPU-limited deployment so throttle is testable | **now justified rather than optional**: PSI covers starvation without a limit, but throttle itself still cannot be tested any other way |
| `AN-E3` | DEFER | rebuild the guard image | ships free with the next deploy |
| `AN-F1` | **OPEN** | the learned seasonal history makes the guard 3x noisier | 2026-08-08: 11 → 33 opened on the same window; floor calibration moved nothing, history is the whole lever. **Unexplained** |
| `AN-F2` | **REFUTED** | the `+1.00` premise has no artefact and is not reproducible (measured +0.38 per pod per window, the level the detector judges); `fixed cost = 0` does not follow from a correlation, which is intercept-invariant — the generator's own `0.45 + 0.040 x traffic` gives exactly +1.00 at 52% fixed cost. ROADMAP reason 2 void; reasons 1 and 3 stand, so the not-shipped decision holds. **New blocker for any retry**: the injected fault is work-proportional by construction, and a fixed-term step is the only regime separating affine from division — the 11/11 tie is experiment design, not a result |
| `AN-F5` | **DONE** | lab-fixture gate rebuilt: per-channel verdicts (a recording can be valid for CPU and rejected for latency), fleet-excursion detection with scrape-level excision to NaN, and the gate globs every `test_fixtures/lab/*.csv` instead of the default name. Threshold measured, not chosen: 3.38x clean vs 90.85x inside the excursion, gate at 5.0. **M7 is the evidence**: the same corruption through the pre-change enumeration failed 0 of 14 tests. Suite 2478/2209/269/0 |
| `AN-F4` | **DONE** | the discriminating fault shape now exists and was measured 2026-08-09. A fixed-term step, sized to the **same mean effect** as the work-proportional one so only its relation to work differs, is the one column where the two treatments part: `CpuUsageRatio` raw 3/4, **divided 4/3, affine 4/4** — division loses a window on the fixed step exactly as the mechanism predicts (a constant spread across a varying denominator). `GcPauseRatio` catches 0 in every cell, unchanged. **Direction confirmed, magnitude not**: one window of nine, overlapping 75%, is at the resolution limit. What this changes is that **a longer recording is now worth taking** — before it provably was not, because no length could break a tie the experiment could not express |
| `AN-F3` | PART | the calibrated floors lived only in a gitignored file | rescued to `k8s/anomaly-guard/guard.lab-workload.calibrated.json`; reconciling the three guard configs is still open, and `guard.lab.json` targets a superseded deployment |

## Release gate — `TG-`

| id | status | task |
|---|---|---|
| `TG-T1` | **PART** | silent passes: 65 of 72 converted, **7 left**, each needing a judgement |
| `TG-T2` | DONE | weight initialisation is seedable |
| `TG-T3` | DONE | split; the half that survived became `TG-T8` |
| `TG-T4` | DONE | the tiled-prefill flag is restored |
| `TG-T5` | DONE | replica forwards documented and verified |
| `TG-T6` | **PART** | heavy group: **21 tests have still never executed** |
| `TG-T7` | DONE | lab absence is reported, gate exits 4 |
| `TG-T8` | PART | tiled prefill: mechanism resolved, **one unexplained event** kept open |
| `TG-T9` | DONE | loader parity — the size gap was by design |
| `TG-T10` | DONE | **a real defect**: both converters wrote Q/K in the wrong RoPE layout |
| `TG-T11` | **PART** | speculative decoding is not bit-identical. Tests fixed; **the product decision is open** — `ChatSession` dispatches to it and greedy implies a determinism it does not have |

## Metric-source seam — `MS-`

| id | status | task |
|---|---|---|
| `MS-1` | DONE | extract `IMetricWindowSource`, wire the field. Verified, reviewed, committed |
| `MS-2` | DONE | parameterise `now` out of the cycle. Verified, reviewed, committed |
| `MS-3` | DONE | historical replay driver + discriminated `GuardCycleOutcome`. Gated after the fact 2026-08-08: the gate returned **BLOCKED** because nothing drove `Blind` or `Failed`, closed in one round. **One thing still unverified**: the `[LabFact]` replay has not run under the now-required `OVERFIT_REPLAY_START_UTC` |
| `MS-4` | **DONE** | live-loop regression guard: 4 tests drive `ExecuteAsync` itself, which nothing did. Pinned — the loop keeps cycling, a source that throws every time does **not** end it, stopping ends it, and stopping returns before the next tick. Two mutations, both caught. **Two harness findings worth more than the tests**: (1) an unbounded `await StopAsync` HUNG instead of failing under the regression it exists to catch — a hang has no message and blocks everything behind it, so every wait is bounded now; (2) killing the harness mid-mutation left the source mutated, and the next run read that as its baseline and reported *restore verified* against it — the harness now refuses to start unless the target is clean against HEAD |

## PSI channel — `PS-`

`ANALYSIS_READY`. Four questions open; the recommendation is config binding first, enum member after the
floor is measured.

| id | status | task |
|---|---|---|
| `PS-1` | OPEN | decide the vehicle: `CustomMetricBinding` now, `MetricIndex` member later |
| `PS-2` | OPEN | bind `container_pressure_cpu_waiting_seconds_total`, observable-only |
| `PS-3` | OPEN | calibrate the floor — needs induced contention, folds into `AN-E2` |

## .NET runtime signals — `RS-`

`ANALYSIS_READY`. **None of the five instrument names exist in the repo**; exposing them needs
`MeterListener` code this repository has never written. Three spikes ordered first.

| id | status | task |
|---|---|---|
| `RS-1` | **PART** | **name confirmed live, overhead NOT measured.** Spike 2026-08-09: a `MeterListener` in `Demo/LabWorkload` subscribes to `http.server.active_requests` and the instrument IS published on this runtime — 1 instrument bound, from ASP.NET Core's own meter, no application code involved. Two things the spike settled that a name check could not: the published type is a delta (`+1` on start, `-1` on end) so a listener must SUM rather than store the last measurement, and the callback must be registered for `int` **and** `long` because a callback registered for the wrong width is silently never invoked. **Still open: the overhead half** — no measurement of what `MeterListener` costs on the lab-as-instrument |
| `RS-2` | **OPEN, cheap** | bind `dotnet_gc_committed_bytes` — **already emitted by the workload and bound nowhere**. Committed-vs-used, zero code |
| `RS-3` | **PART** | **the gap is real and the signal closes it — measured, not argued.** With 5 requests deliberately hung: `labapp_active_requests` read **6** while `labapp_request_duration_seconds_count` read **0**, because a histogram only records a request when it FINISHES. Every latency percentile was blind for the whole stall; the in-flight count moved immediately. **Calibration note**: the idle baseline is **1, not 0** — the scrape itself is an in-flight request — so an absolute floor must carry that offset, while peer comparison is unaffected because every pod has it. **Remaining: deploy and bind.** Deploying means rebuilding the workload image and replacing pods, which perturbs the population (A5: twelve fresh pods cost 0,1,0,1,6,4,3 findings over seven cycles), so it is a deliberate act |
| `RS-4` | OPEN | `dotnet.monitor.lock_contentions` — a latency cliff with every current channel quiet |
| `RS-5` | OPEN | `dotnet.exceptions` — caught exceptions precede 5xx |
| `RS-6` | OPEN | `kestrel.queued_connections` / `rejected_connections` — saturation before the app sees it |

## Cross-cutting — `XC-`

| id | status | task |
|---|---|---|
| `XC-1` | DONE | the Native-AOT gate reached ILCompiler again — 34 XML-doc warnings, an orphaned doc block, a banned `Array.Copy` and a disarmed `RS0030` |
| `XC-2` | DONE | six hardcoded `D:\Overfit\` paths removed; `ModelFact` now resolves relative fixtures against the repository root |
| `XC-3` | **DONE** | verified 2026-08-09: `git ls-tree -r HEAD` reports **0** tracked `.bak` files |
| `XC-4` | OPEN | three copies of the repository-root walk exist (`FixtureFact`, `TelemetryInstrumentWiringTests`, `HybridVsDenseOnDocsCorpusTests`); `RepositoryPaths` is the fourth and the one meant to be shared |
| `XC-5` | OPEN | `gh` is not authenticated, so no CI job status has been checked all session. The AOT guard may have been red |
| `XC-6` | **OPEN** | decide what to do about `OpenTelemetry.Exporter.Prometheus.AspNetCore` — it is the only prerelease pin in the tree and fires `OVERFITPRERELEASE` on every build of `Sources/Main`. **Measured 2026-08-06 against the NuGet registration API: 33 versions, ZERO stable, none since 2022-08-18 — 1449 days — while the rest of the OTel suite shipped stable 1.17.0 the same day.** It is deliberately experimental, not slowly maturing. Only `Demo/LocalAgentAspNetDemo` consumes it; the Native-AOT `overfit` CLI does not. The finding is that the beta is **unnecessary**, not merely risky: `/metrics` is served by the existing hand-rolled renderer with no dependency, instrumentation is `System.Diagnostics.Metrics.Meter` from the BCL, and a customer wanting pipeline export is answered by the stable OTLP exporter (stable since 2021, reads the same `Meter`). Three live choices, and this is a support-policy call rather than a technical one: **accept it explicitly** with a reason and a date on the accept-list, **migrate the demo** to OTLP, or **drop Prometheus export from the demo**. Detail in `docs/specs/guard-telemetry-meter-plan.md` section "C0 resolved" — a quarantined file, but that section is verified |

---

## Transcribed from `ROADMAP.md` (2026-08-08)

Dropped as not-actionable, with reason: the **release-readiness snapshot** (2026-05-29) is a point-in-time
status table superseded by later sections; the **status snapshot** table is pure status, nothing to do;
**long-term ideas** are unscoped with no owner (graph compilation, mixed precision, GPU investigation — not
sized, not actionable as written). Shipped-and-accurately-labelled items are not transcribed at all (M.E.AI
adapter, OpenAI API, JSON-Schema, trailing-babble/acronym/LoRA-merge TTS fixes, embeddings, Whisper,
interpretability hooks, MCP server side) — the roadmap already tells the truth about these.

**Standing constraint, not a task list: decode throughput is closed.** Every lever tried — MHA consolidation
(+1-2%, shelved), Q8-KV (shipped, but for RAM not speed), register/cache-blocking (prefill 1.61x TTFT;
cache-blocking itself measured negative and reverted), AVX-512/VNNI (≈0), adaptive early-exit (≈4.6% ceiling
at unacceptable quality cost) — is measured. The residual ~1.13x gap to llama.cpp is uniform across context
length and memory-access-efficiency bound, not a missing kernel. Treat any new decode-speed proposal as guilty
until it is sized against its share of the decode step (`ROADMAP.md` §"Decode throughput catch-up").

### NASA Power of 10 follow-ups — `NR-`

| id | status | task | note |
|---|---|---|---|
| `NR-1` | OPEN | gate "check every returned status" (rule 7) mechanically, else-sweep style | enable as `suggestion`, count sites, read a sample, promote per directory. Verified not built — no `CA1806`-style enablement found in `Directory.Build.props`/`.editorconfig` |
| `NR-2` | OPEN | write the rule-2 analyzer: a size/loop-bound/index read from `BinaryReader`/JSON without passing a validator | text sweep already run by hand (19 candidates, 7 real defects, 4 false positives, all four false positives the same `if (length != expected) throw` shape). **`OVERFIT024` is already taken** by an unrelated env-var-literal rule (`Sources/Analyzers/EnvironmentVariableNameAnalyzer.cs`) — this analyzer needs a new id |
| `NR-3` | DONE | the seven read-then-use defects the rule-2 sweep named | verified fixed in code: `WhisperGgmlLoader.cs` (nTokens/len/nDims×2/nameLen, `RequireFits` guards at lines 69-98), `RepackedWeightsFile.cs:176`, `LlamaLoRAAdapter.cs:224`, `ModelSerializer.cs:65` (rank now checked before `new int[rank]`, ordering fixed). Matches the roadmap's own "Done 2026-08-03" claim |
| `NR-4` | OPEN | `CheckpointedModule.FindNonDeterministic` unbounded recursion | verified still present, `Sources/Main/DeepLearning/CheckpointedModule.cs:89` carries `#pragma warning disable OVERFIT022` justified by "a handful of levels at most" — an unproved bound. `Sequential.Add(self)` is legal today and crashes the process with an uncatchable `StackOverflowException`. Fix is an explicit `Stack<IModule>` + reference-identity `HashSet<IModule>`, not a bigger limit |
| `NR-5` | OPEN | audit `Audio/Mp3` end to end with `overfit-find-bugs-game` | verified: no audio bug-hunt exists after `docs/bug-hunts/audio-2026-08-01-2245-bugs-game-findings.md`. A 2026-08-04 spot-check found the file in good shape (bounded switches, a clamped `big_values`) but did not cover it end to end |
| `NR-6` | OPEN | census the lengths `Simd.Add`/`Simd.MulAdd` actually receive before moving `Avx512Threshold` | verified: `Sources/Main/Intrinsics/Simd.cs:19` comment still says the threshold is unmeasured and wrong at the low end (128 floats: 512-bit won every op in the microbench that exists) |
| `NR-7` | OPEN | re-run `ElseRefactorBenchmark` under .NET 10's larger inlining budget | verified: `Sources/Benchmark/ElseRefactorBenchmark.cs` last touched 2026-07-23, before the .NET 10 note was written (2026-08-04/05). `OVERFIT021`'s 2.25x extraction-penalty justification has not been re-measured against the new JIT |

### Agentic / interop / vision backlog — `AV-`

| id | status | task | note |
|---|---|---|---|
| `AV-1` | OPEN | Grad-CAM + saliency maps (vision-XAI) | extends the shipped LLM activation-capture/logit-lens to CNN/ONNX; autograd for both already exists |
| `AV-2` | **REFUTED** | plain population GA on prompt text (SkillOpt) | sizing #2, 2026-07-18: population(4)+tournament+elitism reaches hill-climbing's 87.5% but costs 45 calls just to seed (vs hill-climbing's 32 total), then 12 generations with zero improvement — the whole population converges to identical fitness, no diversity pressure. **Do not build GA-on-prompts** |
| `AV-3` | OPEN | size MAP-Elites before building it | the only variant left standing after `AV-2`; show prompt-behaviour descriptors (e.g. verbosity × accuracy) actually have resolution first, or it degenerates the same way |
| `AV-4` | OPEN | `overfit score` as an OpenAI-server / dedicated route | verified: no score route exists under the server code today |
| `AV-5` | OPEN | "Use Overfit from Semantic Kernel" sample (docs only) | verified: no `docs/*semantic-kernel*` file exists; same item as `MK-4`, cross-reference only |

### Audio / TTS backlog — `TT-`

Trailing-babble root-cause, acronym lexicon and LoRA-merge-to-fast-engine are shipped and accurately labelled
`DONE` in the roadmap — not transcribed.

| id | status | task | note |
|---|---|---|---|
| `TT-1` | OPEN | real-time TTS on CPU (smaller ~0.5B same-arch model, or port Kokoro 82M) | product/moat decision — real-time is partly the private differentiator |
| `TT-2` | OPEN | signal-domain (waveform) watermark | current watermark is metadata-only; compliance/IP item, near launch |
| `TT-3` | OPEN, low ROI | align `OrpheusTrainingSequence` to the canonical prompt | cosmetic — inference already works regardless, base dominates |
| `TT-4` | DEFER | zero-alloc + SIMD SNAC decode | not a speed win — SNAC is cheap, the LM is the bottleneck; would only be a "zero-alloc" banner |
| `TT-5` | DEFER | Polish-language TTS normalisation | blocked — Orpheus is EN-only, needs a PL TTS model |

### Adoption / launch roadmap — `AL-`

Items #1 (M.E.AI adapter), #2 (OpenAI API), #3 (JSON-Schema), #5 increment 1 (RAG harness), #9 (QLoRA), #10
(Whisper), #11 (interpretability), #12 (raw-tok/s deprioritised), #13 server-side (MCP) are shipped and
accurately labelled — not transcribed. Item #8 (model-manager CLI) was later self-corrected in the roadmap's
own "UPDATE 2026-06-05" section (`overfit pull/list/chat/serve` + `PackAsTool` all verified present in
`Sources/Cli`) — not stale, not transcribed.

| id | status | task | note |
|---|---|---|---|
| `AL-4` | **DONE — roadmap stale** | Production LocalAgent template (auth, audit log, `/healthz` `/readyz`, Dockerfile) | roadmap's table still reads "Phase-1 walking skeleton"; verified `Demo/LocalAgentAspNetDemo` has `Infrastructure/ApiKeyAuthMiddleware.cs`, `AuditMiddleware.cs`, `Observability/AuditLog.cs` + `MetricsCollector.cs`, `Program.cs:180-181` (`/healthz`, `/readyz`), and a `Dockerfile` |
| `AL-6` | **DONE — roadmap stale** | persistent (file-backed) vector store | roadmap's table still reads "In-memory VectorStore → restart without re-indexing" as open (8/10 ROI); verified `Sources/Main/LanguageModels/Retrieval/PersistentVectorStore.cs` exists (dated 2026-06-21) — on-disk store + source-document manifest with content-hash re-index skip, no SQLite |
| `AL-7` | **PART — roadmap stale** | `dotnet new` project template | roadmap's table still reads open (7.8/10); verified `Templates/content/OverfitChat` ships a real template (`dotnet new overfit-chat`, Minimal API + `IChatClient`, dated 2026-07-20). Gap: it is a chat-app scaffold, not the agent/tool-calling/RAG scaffold the roadmap item names |
| `AL-8` | OPEN | MCP host role — bridge `McpTool → ToolDefinition` so `ReActAgent`/`ToolCallConstraint` can consume tools from any MCP server | server side ships with no ceiling; host value is capped by small-model tool-calling reliability (needs 7B+, per the ReAct e2e finding) |
| `AL-9` | OPEN | persistent KV-cache to disk (cross-session resume) | verified: `Sources/Main/LanguageModels/Runtime/KvCacheSnapshot.cs` is in-memory only, no save/load-to-file path |
| `AL-10` | OPEN | token healing (re-tokenize boundary + KV rollback) | verified absent; would fix the BPE dead-end that today only graceful-stops |
| `AL-11` | OPEN | GBNF (generic CFG) grammar constraint | verified absent; same item as `LG-6`. JSON-mode + `ToolCallConstraint` cover the common cases today |
| `AL-12` | OPEN | continuous / in-flight batching for serving | dotLLM only plans it; not started here either |
| `AL-13` | OPEN, low priority | per-state token-mask cache for constrained decoding | mask is O(vocab × token-len)/step; fine for short structured output today, named as the follow-on if profiling shows it |

### llama.cpp scope-gap snapshot — `LG-` (2026-05-29)

Mirostat, typical-p/XTC/top-nσ/DRY samplers and Whisper are shipped and accurately checked off — not
transcribed.

| id | status | task | note |
|---|---|---|---|
| `LG-1` | OPEN | Q2_K / Q3_K dequant + decode | verified absent — no Q2_K/Q3_K decode path in `Sources/Main`, present only as an enum tag in `GgmlType.cs`; ~2 days each per the roadmap's own estimate |
| `LG-2` | OPEN | infill/FIM sampler (prefix/middle/suffix masking) | verified absent |
| `LG-3` | OPEN | load + compose external `.gguf` LoRA adapters at runtime | Overfit has training-side LoRA; loading/composing externally-trained adapters is separate and verified absent |
| `LG-4` | OPEN | RoPE scaling variants — Yarn/NTK-by-parts, DynamicNTK, AliBi, per-section | verified absent; unlocks long-context Qwen/Llama variants |
| `LG-5` | OPEN | encoder-decoder (T5/FLAN) | verified absent; roadmap notes it reuses the shipped BERT-encoder building blocks |
| `LG-6` | OPEN | GBNF grammar engine | same item as `AL-11`, cross-reference only |
| `LG-7` | OPEN | one vision-language model stack (LLaVA / Qwen2-VL / Pixtral) | 2-4 weeks per roadmap's own estimate; no product decision recorded on niche fit |
| `LG-8` | DEFER | state-space / Mamba / RWKV | deferred unless a specific user model demands it |
| `LG-9` | DEFER | multi-token-prediction + speculative rollback (Qwen3.5/Gemma3N draft heads) | deferred |

### Mixture-of-Experts post-launch track — `MO-`

Track is functionally complete (Qwen1.5-MoE + Mixtral load and generate coherently, Q4_K_M/Q5/Q8_0/F32 all
covered, GGUF-embedded tokenizer for both SPM and BPE). One optional follow-on remains.

| id | status | task | note |
|---|---|---|---|
| `MO-1` | OPEN, optional | native 5-bit (Q5_0/Q5_K) dot kernel | verified absent — MoE experts currently widen Q5→Q8 and dot against the Q8 kernel; a native 5-bit kernel would tighten the working set, not required for correctness |

### LoRA / QLoRA training track — `LT-`

The anomaly-detector LoRA track and the GGUF→training bridge are both shipped end to end (base training, all
three LoRA stages, adapter save/load, knowledge-injection demo, RAM measured at 2.92 GB peak on real
Qwen-3B). Two items remain, one of them a measured negative.

| id | status | task | note |
|---|---|---|---|
| `LT-1` | OPEN | fast fine-tuned decode — hook a trained LoRA adapter into the optimized (repacked Q4_K GEMV) inference engine as a side-GEMV | today's fast engine runs the frozen base only; the trainable model has LoRA but naive kernels. ~2-3 sessions, touches the hot decode path |
| `LT-2` | **REFUTED** | training-model KV-cache as a decode speed-up (Option A) | built and bit-correct (`GenerateCached_MatchesUncachedGenerate`), but measured **6x slower** than uncached (2700 ms/token vs 424 ms/token) on real Qwen-3B — the cached path is single-threaded while the uncached forward already parallelises the dequant-matmul across cores. Kept as a correct reference path only, no speed claim |
| `LT-3` | OPEN — **needs an architect decision** | "backward through Linear/RMSNorm/SwiGLU/attention for the Llama family", from the older LoRA-training checklist | the GGUF→training bridge (`TrainableLlamaModel`) already does exactly this for Qwen/Llama-shaped GGUF; this checklist line predates that bridge and may simply be stale. Confirm before treating as separate scope |

### Quantization / format coverage — `QT-`

Q4_K_M and Q8_0 in-RAM quant storage are shipped and measured (Q4_K_M: 1.4s load, 14.56 tok/s, 4396 MB
steady RAM on the dev box).

| id | status | task | note |
|---|---|---|---|
| `QT-1` | OPEN | Q4_K_M byte-layout integration parity test against a real downloaded file, tolerance too strict | `GgufQ4KMParityTests.Q4KM_TopTokenMatches_FP16Baseline_OnCanonicalPrompt` exists as `[LongFact]` but the roadmap records it **RED** on the maximally-ambiguous 3-token canonical prompt (top-1 token 474 vs 40, swing 2.16, only 4/10 top-k overlap) — assertion is over-strict for that prompt; needs either a less-ambiguous prompt or a top-k-overlap relaxation |
| `QT-2` | DONE | Q5_0 / Q5_K dequantizer | shipped as part of the MoE track (`GgmlDequant.DecodeQ5_0Block`, `DecodeQ5_KBlock`) — the roadmap's separate "Other quant formats" checklist is stale on this line |
| `QT-3` | OPEN | Q2_K / Q3_K_S dequant | same item as `LG-1`, cross-reference only |

### Performance backlog — `PB-`

Decode worker headroom fix, huge-pages investigation, and `OverfitParallelFor`'s bulk-wake dispatcher are
shipped/closed and accurately labelled — not transcribed except where a follow-on remains open.

| id | status | task | note |
|---|---|---|---|
| `PB-1` | OPEN | true batched training (B > 1) for MHA in the training path | flagged in the roadmap as "biggest single lever for CPU saturation" — every existing parallel-over-batch path only activates once B > 1 exists |
| `PB-2` | OPEN | `Conv2D` fwd+bwd migration to `OverfitParallelFor` | needs a per-worker workspace pattern (`GCHandle`-pin or POH refactor of `Conv2DWorkspace`); ~100-150 ms/5 epochs on MNIST, worth doing when GPT-2-scale batched training lands |
| `PB-3` | OPEN | migrate `TensorMath.Sequence` (LSTM), `TensorMath.Attention`, `Optimizers.Adam` to `OverfitParallelFor` | listed as deferred migrations alongside the ones already done |
| `PB-4` | OPEN | SIMD path for `MaxPool2DForwardWithIndicesNchw` (training path) | scalar today because index tracking needs comparison masks; ~40 ms/epoch on MNIST (~7%), estimated 1-2 hours + parity tests |
| `PB-5` | OPEN | verify `ScaledDotProductAttention` forward parallel-over-batch at B ≥ 4 | forward parallel path landed (measured -40% wall / +75% cores-effective on GPT-1 batch 32); roadmap still lists a residual verification item at higher batch |
| `PB-6` | OPEN | numerical-equivalence tests across scalar/SIMD paths + a determinism policy for parallel training kernels | "Correctness" section of the performance backlog, no owner |
| `PB-7` | OPEN | make `dotnet build` join the three-level machine-exclusion mutex scheme | verified: only test (`Tests/MeasurementExclusion.cs`) and benchmark (`Sources/Benchmark/Program.cs`) hold `Global\DevOnBike.Overfit.MachineMeasurement`; `Directory.Build.props` has no `OVERFIT_MEASUREMENT_OWNER` guard. The roadmap's anomaly-guard section specifies this design in full but it is unbuilt |
| `PB-8` | **REFUTED** | huge pages / TLB tuning for the decode weight mmap | closed 2026-07-05 — Windows has no large-page support for file mappings (only private commits), and the 96 MB vs 32 MB V-Cache CCD measured identically (13.75 vs 13.96 tok/s); a 2 GB model dwarfs any L3 |
| `PB-9` | DEFER, explicitly | per-CPU/per-RAM worker-count lookup table | the mechanism (never take every CPU) is already one line in `ResolveDecodeMaxWorkers`; a table would overfit to the single measured machine — build an opt-in `overfit tune` profiler instead if a materially different box ever shows up |
| `PB-10` | OPEN, unsized | misc backlog: `LinearKernels` threshold retuning, `Adam`/`AdamW` parameter-parallel update audit, AVX2/AVX-512/AVX10 SIMD audit, thread-scaling stabilisation for large training workloads | grouped because none is individually sized; revisit when GPT-2-scale batched training (`PB-1`) lands |

### Market-driven priorities — `MK-`

Embedding-model support (#1) is shipped — not transcribed.

| id | status | task | note |
|---|---|---|---|
| `MK-2` | PART | deepen regulated/private-inference positioning docs | `docs/scenarios/regulated-industries.md` + README "What Overfit is not" started; roadmap names the "library-in-process > exposed server" security argument (175k exposed Ollama servers actively exploited) as still to add |
| `MK-3` | OPEN | first-class opt-in decision/audit record (input + model hash + output + timestamp) | motivated by EU AI Act Aug-2026 enforcement; deterministic greedy decode + file-versioned weights already exist, the record itself does not |
| `MK-4` | OPEN | Microsoft Agent Framework / Semantic Kernel adapter positioning | same item as `AV-5`, cross-reference only |

### Medium-term features & distribution — `MT-`

Chat templates, `OverfitClient` facade, depthwise conv, and the four agentic-loop primitives (ReAct,
critic-loop, circuit breaker, summarising memory) are shipped and accurately labelled — not transcribed.

| id | status | task | note |
|---|---|---|---|
| `MT-1` | OPEN | ONNX LSTM/GRU import operators | verified absent |
| `MT-2` | OPEN | standalone Softmax + CrossEntropy layers (beyond the fused loss) | verified absent |
| `MT-3` | OPEN | `GroundedAnswerCache` — 4-gated safe semantic answer cache for RAG | per arXiv:2605.27494; primitives exist (`VectorStore`, `BertEncoder`, `WordPieceTokenizer`), the gated cache itself does not. ~200-400 LoC per the roadmap's own estimate; needs `VectorStore.Add` extended to carry a version hash |
| `MT-4` | OPEN — **needs the client to define "polish"** | NuGet package metadata polish | `Main.csproj` already carries `Authors`/`PackageTags`/description; the roadmap item names no specific gap |
| `MT-5` | OPEN | Blazor sample: streaming generation via `IAsyncEnumerable`/Rx | verified absent |
| `MT-6` | OPEN | benchmark page: Format × Model × RAM × tok/s table | verified absent |
| `MT-7` | DEFER | Gaussian-Process (or cheaper EWMA/z-score) baseline to benchmark the GPT anomaly detector | design sketch exists at `docs/aiops/gp-anomaly-baseline.md`; explicitly deferred as a separate experiment, not a product feature |
