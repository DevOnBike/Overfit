# Measured baselines and negative results

**The single place this repository keeps its numbers.** Agents, docs and comments should cite this file
rather than restating a figure, because a number copied into five places is a number that will be wrong in
four of them.

## How to use it

- **Every number here carries what it was measured on.** A figure without its model, quantisation, thread
  count, build and box is not evidence about anything — this repo has been burned by cross-process and
  cross-build comparisons and treats an uncited number as no number.
- **Re-verify before citing.** This file is a pointer to where the measurement lives, not a substitute for
  reading the current comment or re-running the benchmark. Code moves; numbers rot.
- **A negative result is the most valuable row here.** The list below is not a museum. It is the set of
  changes that look obviously correct and were measured to be worse — without it, each one gets proposed
  again, confidently, roughly once a quarter.
- **Do not extrapolate a technique between kernels.** The recurring lesson across every row: the structure of
  the data around a technique decides, not the technique.

Dev box for most figures: Ryzen 9 9950X3D, Windows, .NET 10, Release.

## big.LITTLE: sizing the worker pool to `ProcessorCount` costs 2x on a phone — 2026-08-14

Measured in `Demo/OverfitChatApp` on a Motorola Edge 50 Fusion (Snapdragon 7s Gen 2: 4x A78 up to
2.4 GHz on cpu4-7, 4x A55 up to 1.96 GHz on cpu0-3), SmolLM2-135M Q4_K loaded with `quantize:false`
(F32-resident), .NET-for-Android, full AOT (`RunAOTCompilation=true`,
`AndroidEnableProfiledAot=false`), same prompt ("what is rabbit") each run, ~300 generated tokens.
`seg/32` is the decode rate per 32-token window, oldest first — the headline tok/s is a cumulative
mean and **cannot distinguish a real slowdown from its own convergence**, which is why the series
exists.

| Arm | tok/s | `seg/32` |
|-----|-------|----------|
| Baseline: 8 workers (`ProcessorCount`), no affinity | 4.1 | `10.0 3.1 3.2 3.8 3.3 3.9 4.8 4.9 4.2` |
| + threads pinned to the fast cluster | 6.3 | `10.1 7.8 6.3 6.6 6.5 5.7 5.4 5.7` |
| + one worker per fast core (4) | **8.5** | `9.8 9.3 8.8 8.9 8.4 8.3 8.1 8.2 7.8` |

**The mechanism.** In the baseline, all four big cores sat at their 691 MHz idle floor for the whole
generation while the little cores ran near their ceiling — the model was executing entirely on the A55
cluster. The first 32-token window still rides the post-tap boost on a big core, hence 10.0; the boost
expires and the work settles onto the little cluster and stays there. The 2.5-3x A78-over-A55 gap is
exactly the 10.0 to 3.1 step. Each of the eight workers ran in short bursts and parked on a semaphore
between dispatches, so no thread ever accumulated the utilisation signal that earns a big core. With
four workers pinned to four big cores, each thread stays continuously busy and all four cores hold
2.4 GHz — the governor needs a sustained thread, not a busy machine.

**Refuted along the way, each by its own measurement — do not re-propose without new evidence:**

- *Thermal throttling.* 30-39 C during the slow runs; the fast pinned runs are HOTTER (41-48 C).
- *Confinement by the system.* The process is in cpuset `top-app` with cpus 0-7 allowed. It was never
  restricted; it was merely placed badly.
- *Memory reclaim / zram.* `VmSwap` 5 MB and **zero** major faults across a sampled generation window.
- *Cost growing with context length.* The `seg/32` series is flat after the first window, so nothing
  proportional to sequence position (attention over the KV cache) can be responsible.
- *O(n^2) UI streaming.* Real and fixed — `MainActivity` assigned the whole accumulated answer to
  `TextView.Text` per token, rebuilding the full `StaticLayout` each time, and `RenderThread` burned
  28750 jiffies against 14470 for all eight workers combined. Fixing it (append into an `Editable`,
  one pending scroll) changed throughput by **nothing measurable**: 3.9 tok/s before, 3.9 after. A
  genuine waste that was not the bottleneck.
- *Cumulative-average artefact.* Plausible enough to be worth instrumenting, and wrong: the first
  window really is 3x the second.

**Do not read this as "pin threads on mobile".** The pin is a blunt instrument that spends more energy
per token, and it is applied here in the demo app, not in the engine. The transferable finding is that
`Environment.ProcessorCount` is the wrong pool size on an asymmetric SoC, because it counts cores that
the work should not be spread across. Open at the time of writing: the process used only ~1.5 cores of
the four it was pinned to, so the remaining headroom is in the serial decode driver, not in placement.

### Full AOT vs JIT on device: no difference — 2026-08-14

Same source in both arms (the only lever is `RunAOTCompilation`), same prompt, both after the affinity
and worker-count fixes above, both with all four big cores held at 2.4 GHz for the whole run:

| Arm | tok/s | `seg/32` |
|-----|-------|----------|
| JIT (`aotLibs=0`) | 8.6 | `10.1 9.1 9.1 8.7 8.8 8.3 8.2 8.3` |
| Full AOT (`aotLibs=23`, 7345 kB) | 8.5 | `10.1 9.0 8.8 8.9 8.4 8.3 8.4 7.9 7.8` |

1.2% apart, with three AOT samples at 8.5 and the windowed series overlapping at every point — inside
run-to-run noise. **Full AOT buys no decode throughput here**, which is consistent with decode being
stalled rather than code-bound: the process used ~1.5 of the 4 cores it was pinned to at full clock.

**Cold start, however, is where it pays.** `am start -W`, 6 force-stopped launches per arm, same device
and session:

| Arm | cold start (ms) |
|-----|-----------------|
| JIT | min 627, median 647, max 665 |
| Full AOT | min 397, median **401**, max 435 |

The ranges do not overlap: 246 ms saved, 38% shorter, against ~1% (noise) on decode. So **keep AOT for
release builds and turn it off for development** — it buys launch latency, not tokens, and it costs
7.3 MB of native libraries plus a much slower build. `TotalTime` here is time to first frame and
excludes model load, so this is the launch cost only.

Whichever way it is set, set `AndroidEnableProfiledAot=false` with it: with profiled AOT left at its
default, `libaot-DevOnBike.Overfit.dll.so` came out at **16 kB instead of 1155 kB** — the engine was
~1.4% compiled and the build was "AOT" in name only. Any earlier conclusion drawn about "AOT" on a
default-profiled build is about that 1.4%, not about AOT.

The arm is recorded from the installed `libaot-*.so` files at startup, not from a build-time constant,
because a run whose arm is asserted rather than observed is a run that can silently measure the other
one. Note the default matters: with profiled AOT left on (`AndroidEnableProfiledAot` unset), the engine
library compiled to 16 kB rather than 1155 kB — that build is "AOT" in name only.

Superseded by this: the July 2026 figure of 9.5 tok/s on a JIT build, which had been read as evidence
that AOT cost throughput. It was a different (shorter, cooler) run, not a different codegen.

## Android exposes NO ARM hardware intrinsics — every `Arm.*` kernel is dead code there

**Known since 2026-07-20, not discovered on 2026-08-14.** `OverfitParallel.ResolveDecodePool` has carried
the fact in a comment since commit `7bb8f87` — it is the documented reason the decode spin-pool is off on
Android. It is restated here because it was re-found independently on 2026-08-14 by someone who had read
neither, which is the definition of a fact living in the wrong place: a comment on a scheduling decision is
not where anyone looks before planning kernel work. **What 2026-08-14 added is the price tag** (the Q4_K
measurement below) and the consequence for the recorded `SDOT` result — not the fact itself.

Logged from the app itself at startup, on a Motorola Edge 50 Fusion (Snapdragon 7s Gen 2), in **both** a
JIT build and a full-AOT one:

```text
simd: Dp=False  AdvSimd=False  AdvSimd64=False  V128hw=True  forceScalar=False
```

`AdvSimd.IsSupported` is false on an **arm64** device — that is base NEON, which every arm64 CPU has.
The hardware is not the explanation: the kernel advertises the dot-product extension outright.

```text
/proc/cpuinfo Features: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp
                        cpuid asimdrdm lrcpc dcpop asimddp
```

**So .NET-for-Android does not expose `System.Runtime.Intrinsics.Arm.*` at all**, in either compilation
mode, and every kernel written against `AdvSimd` / `Dp` silently executes its scalar fallback. Nothing
reports this: the fallback is a correctness path, so the only symptom is speed. `Vector128<T>` — the
portable API — *is* accelerated (`V128hw=True`), which is the way to get SIMD on this platform.

**Three earlier results are explained or invalidated by this:**

- The recorded negative result *"ARM NEON `SDOT`: correct and pointless — decode is dequant-bound, not
  dot-bound"* almost certainly measured code that **never ran**. Its conclusion about where decode's time
  goes cannot be trusted; the port was not pointless, it was unreachable.
- Q4_K decode measured **639 ms/token against 100 ms for F32** — 6.4x slower while reading 5.4x *fewer*
  bytes, which is 0.16 GB/s against F32's 5.5 GB/s. Not remotely bandwidth-bound: it is scalar
  dequantisation plus a scalar dot product. Per component: `ffn_gateup` 268.6 ms (was 27.3), `lm_head`
  135.0 (was 10.5), `attention` 199.7 (was 46.8). RSS did drop as intended, 1035 -> 652 MB.
- Keeping mobile on `quantize:false` is right, but not for the reason the code claimed. It is not that
  "F32 beats the quantized kernels under Mono's weaker codegen" in general — it is that the quantized
  kernels have no vector path on this platform at all.

**Consequence for the roadmap.** Closing the gap to llama.cpp on mobile is not a dequantisation-cost
problem. It is a port of the quantised GEMV kernels from `Arm.*` intrinsics to portable `Vector128<T>`,
validated against the existing scalar oracle for bit-identity. The prize is large — the quantised path
reads 5.4x fewer bytes and currently runs 6.4x slower — but nothing about it was measurable until the
capability probe existed.

**Check the probe, not the build flag.** The measurement above exists because the app logs what the
*runtime* reports at startup. A build flag says what was requested; `IsSupported` says what will execute.

**And check whether the repository already knows.** This cost an afternoon of measuring toward a
conclusion that was sitting in a comment a month old. The fact was in `OverfitParallel`, filed under a
scheduling decision; nothing in `docs/` said it, so a search of the documentation found nothing and the
work proceeded as if the question were open. Search the **source**, not only the docs — and search it with
the semantic navigator (`find_references`, `find_callers`), which resolves symbols instead of matching
text, rather than with a text grep.

## `ParallelWorkThreshold` is in the wrong unit — 2026-08-14

Same phone and model as the section above, decode profiled with `DecodeProfiler` (which adds ~240
timestamp reads per token, ~7 µs against a ~110 ms token — immaterial, and the shares are what it is
for). The hypothesis under test was that decode was dominated by inherently serial work (norms,
residuals, embed), inferred from the process using ~1.5 of its 4 pinned cores and from the decode
driver thread burning 2620 jiffies against 1267 for all four workers combined.

**The profiler refuted it.** `other` — precisely those serial stages — measured **-2.8%**, i.e. zero.
92% of per-token time sat in `attention` (46.4%) and `ffn` (46.1%), the two components that were
supposed to be parallel already.

The cause was `SingleTokenProjectionKernel.ParallelWorkThreshold = 1_000_000`. SmolLM2-135M's FFN
matmuls are 576x1536 = **884,736** — 12% below it — so every one of them took the sequential path and
ran on the calling thread. Only `lm_head` (576x49152 = 28M) cleared the bar, which is exactly the one
component the profiler showed as a small share.

Lowering the threshold to 100,000, one lever, same prompt:

| | tok/s | ms/token | attention | ffn | lm_head |
|---|---|---|---|---|---|
| threshold 1,000,000 | 8.7 | 113.6 | 52.7 | 52.4 | 11.3 |
| threshold 100,000 | **10.1** | 97.6 | 44.9 | 43.8 | 11.1 |

The thread accounting flipped with it — driver 2620 / workers 1267 became driver 1885 / workers 3298 —
and `lm_head`, already parallel, did not move, which is the internal control that this is not machine
drift. A nested-dispatch regression in attention was expected (its 576x576 projections also cleared the
new bar, and they are parallelised head-wise one level up) and did not appear.

**+16% for 4x the cores, because the ceiling is bandwidth.** ~540 MB of F32 weights are read per token:
4.75 GB/s before, 5.5 GB/s after. Predicting 2-4x on the FFN was wrong for that reason.

**The fix is not this constant, and the desktop measurement says so.** `DecodeProjectionThresholdBenchmark`
streams 512 MB of distinct weight matrices (past L3, so no reuse — decode's regime, not a cache-resident
microbenchmark) and compares the sequential path against the parallel one at three shapes and four worker
counts. **The parallel path did not win a single one of the twelve cells**; its best result was a 1.02 tie.
Lowering the library default would therefore buy nothing on this box and cost up to 3x.

That is enough to decide the change and not enough to derive the rule, because **the benchmark did not
reproduce**: the same configuration (32 workers, OutputSize 1536) measured 2.16 in one run and 1.02 in
another, and `RatioSD` reached 1.40 with means and medians disagreeing two-fold (32.01 ms mean against a
16.61 ms median). The canary — a second, identical sequential arm — stayed within 2-4% in every run, so
the box was steady and the instability belongs to the parallel path itself. Suspected but NOT checked:
thread placement across this CPU's two asymmetric-cache CCDs, and the strided reads that an output-band
split produces over a row-major weight matrix.

**So the resting place is a per-app override, not a new default.** `ParallelWorkThresholdOverride` is set
to 100,000 by the Android demo, where it was measured to pay, and left null everywhere else. A rule that
decides from worker count and matrix shape is still the right end state, but deriving it needs a harness
that reproduces first.

**Not measured, do not cite as one:** forcing `quantize:true` (Q4_K-resident, ~5.4x fewer bytes per
token) was tried as the bandwidth lever and abandoned — time-to-first-token went ~1 s to ~6 s and the
run was called off by hand, so there is **no tok/s figure** for that arm.

## `ValueStringBuilder` vs `StringBuilder` — 2026-08-13, and it refuted the type's own doc comment

`[SimpleJob(warmupCount: 8, iterationCount: 20)]`, `MemoryDiagnoser`, deliberately **not** the shared
`BenchmarkConfig` — its `InvocationCount=1` would leave the 64-char arm measuring timer resolution.
Ryzen 9 9950X3D, .NET SDK 10.0.111, 12.4 min wall. Stack buffer 256 chars, so the lengths below sit at
**0, 0, 1, 3 and 6 growth steps**. Full log: `Tests/bin/vsb-bench.log`.

| length | `StringBuilder` default | pre-sized | `ValueStringBuilder` (stack) | VSB (pooled) |
|---|---|---|---|---|
| 64 | 35.85 ns / 496 B | 22.46 (0.63) | **15.01 (0.42) / 152 B** | 16.92 (0.47) |
| 256 | 104.91 / 1408 B | 55.19 (0.53) | 52.77 (0.50) / 536 B | 53.95 (0.51) |
| 512 | 149.26 / 2504 B | 124.96 (0.84) | 105.64 (0.71) / 1048 B | 100.82 (0.68) |
| 2048 | 650.55 / 8792 B | 471.47 (0.72) | 408.05 (0.63) / 4120 B | 396.81 (0.61) |
| 16384 | 4241.72 / 82040 B | 2808.14 (0.66) | 2852.71 (0.67) / 32792 B | 2675.73 (0.63) |

**Span destination — no string produced, its own baseline, not comparable to the rows above**: VSB with
`TryCopyTo` runs at **0.38 / 0.45 / 0.57 / 0.65 / 0.68** of `StringBuilder`+`CopyTo` and allocates
**exactly zero bytes** against 296 / 824 / 1408 / 4624 / 49200 B. This is the shape to reach for when the
caller can own the buffer.

**A prediction was written into the benchmark file BEFORE the run, and it is one-third right — which is why
it was worth writing.** Predicted: fewer bytes at every size (**confirmed**, decisively); faster **only**
where it does not grow (**refuted** — it is faster at every length, including six growths); and a ratio
that degrades as growths rise (**holds for three points**, 0.42 → 0.50 → 0.71, then breaks and settles at
0.63-0.67).

**The type's own doc comment said "small builds can be slower than `StringBuilder`". They are not** — 64
chars is where the advantage is *largest*. The comment has been corrected in place, marked as a correction.

**What this does NOT license.** It measures repeated `Append` into a single build, on one box, at these
lengths. It is not a licence to sweep `StringBuilder` out of the tree: a separate inventory the same day
found **71 sites and zero migration candidates**, because the two criteria (a string must really be
produced, and the build must take several growth steps) are conjunctive and the population fails one or the
other — most per-call sites pre-size to the answer, and the rest are one-shot.

## The `[LongFact]` release gate — measured for the first time, 2026-08-13

"Run it before every release" meant something unknown until today: **nobody had ever timed this gate.** Run
batched by area (`OVERFIT_RUN_LONG=1`, `--no-build`, one `dotnet test --filter` per area, per-batch bound),
appending to `Tests/bin/longfact-timings.log`. **27 of 49 areas completed, 165.3 minutes.**

**The shape is the finding, not the total: four areas carry 152 of those 165 minutes, and nineteen carry
6.5 between them.**

| area | elapsed | outcome |
|---|---|---|
| `Anomalies` | **63.3 min** | hit a 20-min bound and ran on — see the bound caveat below. 620 tests, 8 reported failed |
| `LanguageModels.Loading` | 30.0 min | hit a 30-min bound, killed cleanly |
| `LanguageModels.LoRA` | 27.8 min | hit the bound, **0 failures inside it** — it is long, not broken |
| `LanguageModels.Demo` | 21.1 min | hit the bound, **0 failures inside it** |
| `LanguageModels.Diagnostics` | 9.5 min | completed, 2 failed |
| `LanguageModels.Retrieval` | 4.1 min | clean |
| the other 21 areas | **6.5 min total** | clean |
| discovery alone (`--list-tests`) | **16.0 s** for 2892 tests | also never measured before |

**So the gate is affordable if those four are treated separately** — which is the opposite of the usual
conclusion that a long gate must be dropped. **22 areas remain unmeasured**, including
`LanguageModels.Runtime`, `Tokenization` and `Training`, so the total is a lower bound.

**Two measurement caveats, both mine, both worth more than the numbers:** a bound implemented with
`subprocess.run(timeout=…)` **does not bound anything** — it kills the child and then waits on pipes the
grandchildren hold, which turned a 20-minute bound into 63 minutes; killing the process **tree**
(`taskkill /F /T`) made the next bound land at exactly 30.00 min. And a failure list capped at five names
without saying so lost three of the eight from `Anomalies` permanently. Both are fixed in the runner; the
63-minute figure is left standing because it is what a wrong bound costs.

## What the analyzer census can actually see — 26 of 34 projects

**Every "the tree was swept to zero" claim in this repository is a claim about `Overfit.sln`**, and the
solution holds **26 of the 34 `.csproj` on disk**. Measured 2026-08-14 (`XC-27`). This matters because the
escalations recorded in `.editorconfig:559-565` were locked on exactly that evidence — *"a whole-solution
rebuild with all three temporarily raised to `warning` reports 0 and 0"* — and a whole-solution rebuild
**structurally cannot see the other eight**.

It is not hypothetical. Building the eight one at a time:

- **`Sources/AndroidBench` reported 2 live `OVERFIT043` errors** (`DecodeBench.cs:64`, `:218`) — **fixed 2026-08-14, and it also carries 5 `OVERFIT015` and 2 `OVERFIT024` warnings that no census ever counted**. It lives
  under `Sources/`, where every rule here is aimed, and it was never in the sweep that justified making
  that rule an error.
- **`Demo/VoiceLoop` did not compile at all** (fixed 2026-08-14): `MicCapture.cs:85` called `GCHandleScope.Pin`, and the type
  has been `GcHandleScope` since a rename that swept the solution. **The same rename also left the old name
  in the type's own doc comment** (`GcHandleScope.cs:14`). A rename passed the gate and left an
  out-of-solution project uncompilable, silently, for however long.
- **`Demo/VoiceClone`'s `OVERFIT040` pragma is live** — 0 sites with it, **1 site with it neutralised**
  (`Program.cs:376`). A sweep reading the census's "0 sites" as "the pragma protects nothing" would have
  deleted a real suppression.

**So when citing a zero from an analyzer census, say what it covered.** The number is true of the solution
and says nothing about `Sources/AndroidBench`, `Demo/{AndroidBenchApp,OverfitChatApp,SkillEvalConsole,VoiceClone,VoiceLoop}`
or the two `Templates` projects.

## What the comparisons are measured AGAINST — name it beside every ratio

A ratio has two sides, and this file used to record only ours. **A reader assumes the other side is the
version they would install today**, so a comparison whose baseline is not named quietly flatters us by
however much that baseline has moved.

| the other side | version used | how stale, and how much it matters |
|---|---|---|
| **ONNX Runtime** (`Microsoft.ML.OnnxRuntime`, `Sources/Benchmark` only — it is in **no** Overfit code path) | **1.29.0, and the two headline ratios were RE-MEASURED against it on 2026-08-17 — see the re-measurement block below this table** | 1.29.0 shipped 2026-08-12 and the pin is held deliberately, because moving it moves every published Overfit-vs-ORT ratio without a line of Overfit changing. **Measured, six clean A/B process pairs (`PB-ORT1`, 2026-08-12): the two versions do not separate.** Every steady-state ORT arm was faster on 1.29.0 (0.9848–0.9974) and it means nothing — the canaries moved the same way and `Overfit_Batch64` moved further, at 0.9825. **Resolving power stated before the verdict: median cross-process canary spread 4.26%, against an effect of 0.3–1.5%.** So the experiment is *silent* in that band, not negative, and the flattery any published ratio carries is bounded at **≤1.5% — under the noise floor.** No ratio needs restating; the version needs naming. **PIN TAKEN 2026-08-17, reversing the hold.** The hold's stated reason — *"moving it moves every published ratio"* — is refuted in magnitude by the measurement in the same breath: the largest published claim is **7.31× faster than ORT**, and a 1.5% faster ORT makes it **7.20×**, a smaller change than the error bar on the 7.31× itself. The pin was protecting the numbers from a movement nobody can detect, and the alternative it chose instead — naming the version, this row — had already been delivered. **The five benchmark classes were deliberately NOT re-run**, because the flattery is bounded at ≤1.5% and that is under the ±3–4% floor of the measurement that produced the ratios. **That sentence stood for about an hour: the classes WERE then re-run against 1.29.0 (same day, at the user's instruction) and the block below carries the result. It is kept because the reasoning — that no re-run was *needed* — is still correct, and the re-run confirms it.** The bump was verified live rather than assumed — `runtimes/win-x64/native/onnxruntime.dll` resolves to **16,149,344 B**, exactly the 1.29.0 figure `PB-ORT1` recorded against 1.28.0's 15,809,848, which is the check that stops a stale `bin/` copy from making a re-measurement look flat. |
| **llama.cpp** (the `~1.13× uniform` decode row below) | **`3292da0` = tag `b9441`**, recovered 2026-08-17 — see the section *"Recovering the llama.cpp baseline"* below for how, and for the one thing still unknown | Not written down when the number was taken; reconstructed from the local clone's **reflog**, which is evidence about the checkout rather than about the run. **The `b10088` that appears in `ROADMAP-COMPLETED.md:819` is a DIFFERENT measurement** — that row is prefill, taken 2026-07-22, and `b10088` is dated 2026-07-22, seven weeks *after* the decode sprint, so it cannot be the decode baseline and must not be cited as one. |

**The cross-process floor is intrinsic here, not background load.** Canary spread was median 3.17% on a
loaded box and median 4.26% after a reboot to a single process — slightly *worse* clean. An effect under a
few percent therefore needs a different experiment shape, not a quieter machine.

### Overfit vs ONNX Runtime 1.29.0 — re-measured 2026-08-17

**Provenance, because a ratio without it is not citable.** BenchmarkDotNet 0.15.8 · Windows 11
(10.0.26200.9168) · AMD Ryzen 9 9950X3D, 32 logical / 16 physical · .NET SDK 10.0.111, runtime .NET 10.0.11,
X64 RyuJIT `x86-64-v4` · `Microsoft.ML.OnnxRuntime` **1.29.0**, native asset verified at 16,149,344 B ·
ORT sessions pinned `IntraOpNumThreads = InterOpNumThreads = 1` in both classes · one BDN process per class.

| subject | Overfit | ONNX Runtime | ratio | allocation |
|---|---:|---:|---:|---|
| `Linear(784→10)`, single inference | **236.8 ns** ± 1.33 | 1,962.7 ns ± 26.98 (preallocated `OrtValue`) | **8.29×** | 0 B vs 224 B |
| same, against ORT's ordinary API | — | 3,665.5 ns ± 47.84 (`NamedOnnxValue`) | **15.5×** | 0 B vs 952 B |
| **3-layer MLP `784→256→128→10`** | **8.883 µs** ± 0.25 | 9.445 µs ± 0.22 | **1.06×** | 0 B vs 224 B |
| concurrent inference, 8 threads | **571.5 ms** ± 17.0 | 2,075.3 ms ± 72.8 | **3.63×** | **0 B vs 117,440,512 B** |

**The two published claims hold.** `README.md:449` says ~8.0× on `Linear(784→10)` and `:455` says ~3.6×
concurrent; measured 8.29× and 3.63×. The single-inference claim is if anything **understated**. Ratios are
**within-process** comparisons — both arms in one BDN process — so the ±3–4% *cross-process* floor recorded
above governs a different experiment and does not apply; the error bar here is BDN's `RatioSD`, 0.02 and 0.05
respectively.

**The row that is NOT in the README is the one worth reading.** On the 3-layer MLP — the same suite, the same
box, the same run — Overfit is **1.06×**, not 8×. Nothing is wrong with either number: `Linear(784→10)` is a
single 7,840-parameter matmul, small enough that ORT's ~1.7 µs of per-call dispatch overhead dominates the
result, while the MLP is ~235k parameters and actual compute dominates. **So the 8× measures ORT's call
overhead and the 1.06× measures kernel quality**, and only the first is published. The README does name its
model, which is honest as far as it goes; it has no larger-model ORT row at all. *Caveat on the 1.06× itself:
`RatioSD` 0.03 against a 6% difference, with `MultimodalDistribution` and outliers flagged — it is marginally
resolved and should be re-run best-of-N before being published as a number rather than as a direction.*

**What the absolute numbers do NOT support.** ORT's `Linear` figure reads 1,962.7 ns here against the
README's 1,883 ns — **+4.2%, and that is not attributable to 1.29.0.** It is a cross-run comparison against a
figure of unrecorded date and machine state, sitting exactly in the ±3–4% cross-process band. `PB-ORT1`
measured the version difference properly and found it silent. **Quote the ratio, not the nanoseconds.**

**Parity was checked before speed, by the harness itself**: `SingleInferenceBenchmark.Setup()` asserts
Overfit's output matches ORT's to 1e-3 before any timing, and the run completed, so 1.29.0 did not move the
numerics.

**Still not measured**: `MLNetSingleInferenceBenchmark`'s ML.NET arms were carried along by the filter and are
reported above only for the Overfit-vs-ORT pair. `InferenceBenchmark` and `ColdStartBenchmark` were
deliberately skipped — `PB-ORT1` measured the first at 61% within-version spread (wrong job for the workload)
and found the second unable to resolve anything (BDN raises `MinIterationTime`; one iteration is 1.07 ms
against a 100 ms target).

### CORRECTION: the roofline denominator was the 256-bit one (2026-08-18)

Every "percent of roofline" figure written in this file before this note divided a 512-bit kernel's rate by
**2190 GFLOP/s**, which is not this machine's AVX-512 ceiling. Those figures were uniformly about **2.35x
too generous**.

**The correct numbers, derived from the benchmark's own work unit rather than from a remembered total.**
`MachineRooflineBenchmark.FmaChains512` issues **12 FMAs per iteration over 2,000,000 iterations**, each one
512-bit and so 32 FLOP: **768 MFLOP per worker**. At the measured times:

| arm | time | throughput |
|---|---:|---:|
| 1 worker | 2.139 ms | **359 GFLOP/s** |
| 32 workers, 32x the work | 4.785 ms | **5136 GFLOP/s** |

The 14.30x ratio between them is unaffected, and every conclusion drawn from that ratio still stands - it
was always a ratio of two measurements of the same thing. What was wrong was the absolute denominator.

**How it happened, because the shape is worth recognising.** The 2190 figure was carried forward from an
earlier note rather than recomputed, and nothing downstream could contradict it: a percentage of a wrong
ceiling still looks like a percentage. It surfaced only when `GemmMicroKernelShapeBenchmark` reported the
8x32 tile at **340 GFLOP/s** on a single core - which would have been 222% of the 153 GFLOP/s single-core
figure this file was quoting. **An impossible percentage is the only thing that can catch a wrong
denominator**, so a rate that exceeds its own ceiling deserves more attention than a rate that merely
disappoints.

**Restated on the correct basis:** the conv stack after today's work runs at 786 GFLOP/s = **15.3%** of
5136, ONNX Runtime's whole-model rate is 30.94 GFLOP in 18.9 ms = 1637 GFLOP/s = **32%**, and the 8x32
micro-kernel with L1-resident operands reaches 340 GFLOP/s = **95% of single-core peak**.

**That last figure changes what the remaining gap is about: the micro-kernel is at hardware peak when its
operands are in L1, so it is not the arithmetic that is slow.**

### Packing the conv kernels MR-major, once at load (2026-08-18, `XC-78`)

The largest conv win after the patch-gather fusion, and it came out of reading BLIS rather than out of
measuring anything first.

**The contract BLIS states outright** (`docs/KernelsHowTo.md`): the micropanel of A is *"stored by columns
with leading dimension PACKMR"*, so the MR values one k-step consumes sit side by side. Our micro-kernel did
the opposite - `rows[r] = a + (m0 + r) * k`, then `rows[r][kk]` - so a single k-step read eight floats from
eight addresses `k * 4` bytes apart, which is **18 KB at VGG-16's K = 4608**. Same bytes, eight streams
instead of one. It is the same defect class as the dense layer's 16 KB stride found earlier the same day.

**Why this bet is better here than it was for BLIS, and why the old negative does not cover it.** BLIS packs
A on every call because it is a general GEMM and A is caller data. **In inference A is the convolution's own
weights, which never change** - so the pack is a load-time cost, not a per-call one. That single difference
in premises is the whole result, and it also explains the negative already recorded here for BLIS-style
blocking-plus-packing: **that experiment paid the packing cost every call.**

**Measured in two stages, deliberately, because the second is only worth building if the first pays.**

*Stage one - pack per call*, paying for the pack on every inference: VGG-16 **-4.1%**, the 60.9 MB CNN
**-4.5%**, against an ONNX Runtime canary of 1.6-2.1%. Resolved, but modest.

*The per-layer split then said exactly where that cost was landing*, all cores, GFLOP/s:

| layer | N | unpacked | packed per call | packed at load |
|---|---:|---:|---:|---:|
| conv6 | 3136 | 761.5 | 848.4 | **877.0** |
| conv7 | 3136 | 765.3 | 850.7 | **887.0** |
| conv9 | 784 | 803.9 | 828.8 | **939.5** |
| conv10 | 784 | 814.9 | 836.9 | **954.2** |
| conv11 | 196 | 471.2 | **445.5** | **520.2** |
| conv12 | 196 | 478.6 | **423.4** | **535.8** |
| conv13 | 196 | 505.9 | **421.2** | **531.6** |

The whole of the pack's cost fell on the layers with small N, where the packed matrix serves only 7 panels
instead of 98 - conv13 **lost 16.7%** while conv6 gained 11.4%. That is what said the pack belonged at load
time rather than in the kernel, and it is also **how the load-time wiring was proved live**: had it silently
fallen back to per-call packing, those layers would still be losing. They are not; they now beat the
unpacked baseline.

*Stage two - pack once, when the layer enters inference mode.* Whole conv stack **43.86 -> 39.04 ms
(-11.0%)**, 699.7 -> **786 GFLOP/s = 15.3% of the 5136 GFLOP/s all-core AVX-512 roofline**.

**End to end, ABAB in one box state** (`OVERFIT_CONV_PACK_A` 0 against 1):

| model | packed off | packed at load | change | ONNX Runtime canary |
|---|---:|---:|---:|---:|
| VGG-16 | 55.37 / 55.37 ms | **50.74 / 50.68 ms** | **-8.4%** | 18.17-18.37 ms (1.1%) |
| CNN, 60.9 MB | 46.45 / 46.90 ms | **41.89 / 42.00 ms** | **-10.1%** | 9.14-9.55 ms (4.5%) |

Against ONNX Runtime: VGG-16 **3.03x -> 2.77x**, the 60.9 MB CNN
**5.01x -> 4.49x**. Parity unchanged in every arm.

**The unpacked VGG arm repeated to 55.37 and 55.37 - the same figure to the hundredth of a millisecond** -
so the 8.4% is not a drift artefact.

**The two stages agree with the arithmetic that predicted them.** Per-call packing gave -4.1% on VGG-16;
moving the pack to load time gave 8.4%, close to double. That is what had to happen if the pack
cost roughly equalled the gain, which is what the per-layer table showed.

**What it costs.** A second copy of the kernel weights - 58.8 MB across VGG-16's convolution stack - held
only while the layer is in inference mode. `Train()` releases it, `InvalidateParameterCaches()` drops it
because it is derived from the weights, and `Dispose()` frees it. This is the same shape as the duplication
`XC-82` objects to for the dense layer; the difference is that this copy is on the hot path and measured to
pay 8.4%.

**Not done.** The AVX2 path is unchanged, and the single-channel 3x3 convolution never reaches the GEMM, so
it is deliberately not packed - packing it would allocate a copy nothing reads.

### K-blocking the conv GEMM: implemented, measured, NOT shipped (2026-08-18, `XC-78`)

A negative result, and the most useful kind: the hypothesis came from reading a competitor's source, it was
specific, it was cheap to test, and it is wrong for these shapes.

**What MLAS does that we do not.** `MlasSgemmOperation` (`onnxruntime/core/mlas/lib/sgemm.cpp`) blocks BOTH
dimensions - `MLAS_SGEMM_STRIDEN = MLAS_SGEMM_STRIDEK = 128` - so its packed B panel is a constant
`128 x 128` floats = **64 KB** whatever K is, and the A slice it sweeps is `M x 128`. Both sit in L2
together, and the inner B slice is L1-sized. Our kernel contracts the whole of K in one pass, so its packed
panel is `K x 32` - **589 KB at VGG-16's K = 4608** - and the A it sweeps is the full `M x K`, 9.4 MB, which
is L3 rather than L2.

**Why it was worth testing despite an earlier negative.** A BLIS-style K-blocked AND A-packed variant was
measured here before and regressed (vgg 140 -> 189 ms). Its recorded reason was that *most im2col K values
are at most a few hundred, so a single K-block means no blocking benefit*. **That premise is false for
VGG-16**, whose K runs 576 to 4608 - at Kc = 128 the late layers get 36 blocks, not one. The old result
refutes the pair; it does not cover K-blocking alone.

**The measurement.** `LargeCnnComparisonBenchmark` on VGG-16, `OVERFIT_CONV_KBLOCK` swept, with an unblocked
run at each end of the sweep so drift is visible:

| Kc | Overfit | against unblocked | ONNX Runtime canary |
|---|---:|---:|---:|
| off (first) | 56.18 ms | - | 19.86 ms |
| 64 | 59.09 ms | **+5.2%** | 19.48 ms |
| 128 | 56.60 ms | +0.8% | 19.52 ms |
| 256 | 55.34 ms | -1.4% | 19.45 ms |
| 512 | 55.75 ms | -0.7% | 19.47 ms |
| 1024 | **54.95 ms** | **-2.1%** | 19.52 ms |
| off (last) | 56.12 ms | - | 19.49 ms |

**The verdict is "not resolved", not "a small win".** The best result is -2.1% and the ONNX Runtime canary
moved 2.1% across the same runs, so the effect is the size of the instrument's own error. The two unblocked
runs bracket the sweep and agree to 0.1%, so the box was still - it is the between-configuration spread that
is the problem, not drift.

**What the shape of the curve says.** The gain rises monotonically toward larger Kc and the best value is
the one closest to no blocking at all, while small blocks are clearly worse (Kc = 64 is +5.2%). That is the
C re-accumulation cost dominating the L1/L2 benefit: contracting the whole of K keeps the C tile in
registers from start to finish, so C is written exactly once, while blocking forces a read-modify-write per
block - at VGG-16's conv6, 3.2 MB of C traffic becomes about 115 MB. For M of 256 to 512 against K of 576 to
4608, **full-K accumulation in registers is already the right trade**, and MLAS's constants are tuned for a
different M/K balance than an im2col convolution presents.

**It is kept, default off, behind `OVERFIT_CONV_KBLOCK`.** The switch is the cheap way for someone on
different silicon - a different L2, a different L3 topology - to re-run this in one command rather than
re-deriving it. **The default suite runs Kc = 0, so the blocked path is not exercised by CI**; it was
verified by running the whole suite green at Kc = 64, 128, 256, 1024 and 5000 (5000 exceeds every K here, so
the result must not and does not depend on the split), and by **four mutations, all four caught at Kc = 128
and all four correctly NOT caught at Kc = 0** - which is what proves the mutations reach the new path and
leave the default one alone.

### The patch gather folded into the GEMM's own pack (2026-08-18, `XC-78`)

The largest single change of the day, and the one the per-layer diagnostic pointed at last rather than first.

**What it removes.** The unfused path builds a `K x N` float column matrix, then every worker copies a
32-column slice of it into its packed panel. On VGG-16's conv2 that matrix is **115.6 MB**, written once and
read once, to hold values that could have been read straight out of the 3.2 MB input. Measured before the
change, the gather alone was **19.8% of convolution time** - and the copy out of it is part of the other
80.2%.

**The design is MLAS's, taken one level finer.** ONNX Runtime's `MlasConvExpandThenGemmSegmented`
(`onnxruntime/core/mlas/lib/convolve.cpp`) expands a `CountK x CountN` block into a column buffer and GEMMs
that block, so it never holds the whole matrix either; its `StrideN` / `StrideK` adapt to keep that buffer a
constant size. Here the destination is the micro-kernel's **own packed panel**, so there is no intermediate
buffer at all. A panel is `K x 32` floats - 589 KB at VGG's largest K, inside this core's 1 MB L2.

**The cost, so it is not sold as free.** The unfused pack reads 32 contiguous floats. The fused one computes
an input address per element. Row and column origins are hoisted per panel - 32 divisions for the whole
panel rather than one per element - leaving two adds and two bounds checks in the inner loop.

**Measured, ABAB in one box state** (`OVERFIT_CONV_FUSED_IM2COL=0` against default):

| model | unfused | fused | change | ORT canary |
|---|---:|---:|---:|---:|
| VGG-16 | 64.33 / 64.28 ms | **55.49 / 55.54 ms** | **-13.7%** | 18.62-19.10 ms (2.6%) |
| CNN, 60.9 MB | 55.48 / 55.23 ms | **46.87 / 46.99 ms** | **-15.2%** | 9.76-9.84 ms (0.8%) |

Against ONNX Runtime: VGG-16 **3.41x -> 2.93x**, the 60.9 MB CNN **5.65x -> 4.79x**. Parity unchanged in
every arm: cosine 1.000000, max absolute difference 2.682e-7 and 6.706e-8, same argmax.

**It also removes the scratch buffer**, which the timing does not show. The unfused path rents `k * n` floats
per convolution - 115.6 MB at VGG's conv2 - and the fused path rents `k * 32` per worker instead. That is a
peak-resident-memory result as much as a speed one, and this project's low-end-hardware position is about
peak.

**Coverage: six mutations, five caught, and the sixth is explained rather than patched.** Dropping `ky`,
dropping `kx`, dropping the padding from the row origin, striding channels by `kernelSize` instead of by the
`kernelSize^2` window, and swapping `kx` with `ky` all redden the suite. **Writing 1f instead of 0f into the
lanes past `nrEff` does not** - and that is not a coverage gap: `StoreTile` writes exactly `nrEff` columns,
so those lanes never reach the output. The fill stays anyway, because uninitialised pool memory can hold NaN
and feeding NaN through an FMA chain costs on some parts even when the lane is discarded; it was moved out
of the gather loop, where it had been a branch on every one of `K * 32` elements.

**Not done.** The AVX2 path still builds the column matrix - porting the fusion there without measuring it
on AVX2 hardware is the mistake this task already records once. The 1x1 stride-1 fast path is untouched,
because it never materialised a column matrix in the first place.

### A batch-1 dense layer was reading its weights with a 16 KB stride (2026-08-18, `XC-78`)

VGG-16's `Linear(25088 -> 4096)` costs **9.99 ms of a 72.15 ms inference, 14%**. At batch 1 it performs
205 MFLOP and reads **392 MiB of weights**, so its unit is GB/s and not GFLOP/s: 9.99 ms is **41.1 GB/s**
against this box's **measured 90.8 GB/s** read ceiling (`MachineRooflineBenchmark.ReadBandwidth`).

**The decisive line is `LinearKernels.cs`, in the column worker's inner loop:**

    var rowBase = weightsBase + ((long)i * outputSize) + j;

The parallel dispatch gives each worker a range of output COLUMNS, so every worker walks every input row.
It reads 64 contiguous floats (256 B), then skips `outputSize * 4` = **16 KB** to the next row, 25,088
times. With 4 KB pages every one of those reads lands on a different page, so the hardware prefetcher has
nothing to follow.

**Premise stated before measuring**: the limit is the access pattern, not the bandwidth; **what would refute
it** is a row split - contiguous slabs - failing to raise the achieved GB/s.

**The result settles it, and one arm settles it by itself.** `LinearGemvRowSplitBenchmark`, both arms on
`OverfitParallel` so the decomposition is the only lever:

| arm | ms | GB/s | % of the 90.8 GB/s ceiling |
|---|---:|---:|---:|
| column split, 32 workers (today) | 8.756 | 47.7 | 53% |
| **row split, single thread** | 7.657 | 52.1 | 57% |
| **row split, 32 workers** | **5.895** | **69.3** | **76%** |

**One core reading sequentially beats thirty-two reading with a 16 KB stride.** No bandwidth explanation
survives that.

**Why the row split is not free.** Register accumulators require fixing a column block and looping rows,
which is exactly the strided pattern. Reading sequentially requires fixing a row and sweeping all columns,
which puts the accumulator in memory. It fits: 4096 floats is 16 KiB against this core's 48 KiB L1d.

**The threshold was measured, and the obvious mechanism was wrong.** The guess was L3 residency. The row
split already wins at 96 MiB while this part's L3 is 128 MiB - and that 128 MiB is not one pool, being split
across two CCDs. Weight bytes against ratio, batch 1, all against 4096 outputs:

| weights | ColumnSplit | RowSplit | winner |
|---:|---:|---:|---|
| 64 MiB | 509 us | 576 us | column, 1.14x |
| 72 MiB | 594 us | 652 us | column, 1.10x |
| 80 MiB | 643 us | 740 us | column, 1.16x |
| 88 MiB | 767 us | 731 us | row, 1.05x - inside the error bars |
| **96 MiB** | 898 us | **775 us** | **row, 1.15x - clearly separated** |
| 192 MiB | 2,953 us | **2,073 us** | row, 1.43x |
| 392 MiB | 8,756 us | **5,895 us** | row, 1.49x |

The gate is **96 MiB**, the first unambiguous win, so nothing measured to prefer the column split is moved
off it.

**A unit error nearly shipped here, and a test caught it.** The threshold constant is written
`n * 1024 * 1024`, i.e. MiB, while the crossover ladder had been written down in decimal MB. 100,663,296
bytes is **100.7 MB and 96 MiB** - both correct - and comparing the first against the second is not. The
policy test asserted that `6144x4096` selects the row split, it did not, and the mismatch was the unit.
**A number is not checked until it is checked in the unit its consumer uses.**

**End to end, ABAB in one box state** (`OVERFIT_LINEAR_ROW_SPLIT=0` against default), VGG-16:

| arm | run 1 | run 2 | mean |
|---|---:|---:|---:|
| column split | 66.82 ms | 66.36 ms | 66.59 ms |
| row split | 64.57 ms | 63.92 ms | **64.25 ms (-3.5%)** |

ONNX Runtime canary across the four runs: 18.79-19.12 ms, a 1.8% spread. The whole-model gain (2.34 ms) is
smaller than the isolated benchmark would predict (~3.3 ms) and the canary spread is 1.2 ms, so read it as
"about 2-3 ms" rather than as a precise figure.

**Coverage.** `Tests/Core/Kernels/LinearRowSplitTests` calls the sweep directly at five small shapes - firing
the gate would need a 96 MiB fixture, which does not belong in a fast suite - plus a written-every-output
sentinel, plus the policy asserted at the seven byte counts above. **Six mutations, six caught.** Worth one
contrast with the conv work-split elsewhere in this file: there, dropping the M-block START bound was NOT
caught, because those workers recompute each other's rows and store identical values. Here the same mutation
IS caught, because each worker owns a separate partial buffer and overlapping ranges therefore double-count.
**The same mutation is silent in one decomposition and loud in the other, and which it is depends on whether
the workers share a destination.**

**Two limits, deliberate.** Gated on `batchSize == 1`: with a batch the partials cost
`workers * batch * outputSize` floats, and the bigger win there is reading the weights once for all rows,
which is a GEMM and a different change. Gated on AVX-512 because the sweep is written at 512 bits; AVX2
hardware keeps the column split until someone measures it there.

### Pooling was single-threaded, and it cost 6.8% of VGG-16 (2026-08-18, `XC-78`)

Found by reading the per-layer profile the conv diagnostic produced, not by looking for it. VGG-16's five
pooling nodes cost **5.31 ms of a 79.48 ms inference**, and the first of them moved 16.06 MB in 2.86 ms.

**The comparison that made it obvious is inside the same profile.** That is **5.6 GB/s**, while the ReLU node
immediately beside it, on the same tensor and the same kind of pure streaming work, reached **62.7 GB/s** -
eleven times faster on 1.6x more data. A rate that far from its neighbour is a structural difference, not a
tuning gap.

**The cause, read from the code rather than guessed.** `PoolingKernels.MaxPool2DForwardSingleBatchPool2NoIndex`
looped over channels serially. Channels are independent, so the machine was idle for that whole span. The
horizontal pair collapse inside it is also scalar, and its comment calls that negligible because `outW` is
13 - **a number calibrated on MNIST.** VGG's first pool has `outW = 112` and 64 channels, which is 802,816
scalar iterations rather than 13. That second lever is left alone for now; the first was enough.

**Result.** Split across workers by channel, with a serial path kept below 262,144 elements because the
dispatch costs ~0.22 ms:

| node | before | after |
|---|---:|---:|
| pool 1 | 2.86 ms | 0.44 ms |
| pool 2 | 1.38 ms | 0.26 ms |
| pool 3 | 0.64 ms | 0.17 ms |
| pool 4 | 0.34 ms | 0.13 ms |
| pool 5 | 0.09 ms | 0.09 ms |
| **total** | **5.31 ms** | **1.09 ms (4.9x)** |

The fifth node is unchanged at 0.09 ms because its 25,088 elements sit below the threshold - which is the
threshold's own canary, and the MNIST CNN keeps the old path for the same reason.

**End to end, ABAB in one box state**, `OVERFIT_PARALLEL_POOL=0` against default:

| model | pooling off | pooling on | change | ONNX Runtime canary |
|---|---:|---:|---:|---:|
| VGG-16 | 71.17 / 70.60 ms | **66.94 / 65.84 ms** | **-6.3%** | 18.84-19.11 ms (1.4%) |
| CNN, 60.9 MB | 59.86 / 60.27 ms | **55.51 / 55.18 ms** | **-7.9%** | 9.75-9.81 ms (0.6%) |

**The ABAB was not optional here - a single A/B had already produced a wrong reading.** The first attempt
compared a pooling-on run against the previous sitting's pooling-off number, and showed VGG improving while
the 60.9 MB CNN's *ratio* got worse. ONNX Runtime had moved 24% between the two sittings on an untouched
binary, so both arms of that comparison came from different machines in every sense that matters. **What
moved between the sittings is still not identified**, and it is worth recording that the three small-model
benchmarks stayed inside 1.3% across the same gap: whatever it is, it reaches the large-CNN benchmark and
not the small ones.

### The conv work-split that came out of that diagnostic (2026-08-18, `XC-78`)

The diagnostic below said the concentrated loss was work decomposition, not the micro-kernel. This is the
change it pointed at, and its measurement.

**What it does.** `Conv2DGemmKernels.Gemm` dispatched one work item per N-panel. It now dispatches one item
per (N-panel, M-block) pair, so a GEMM with fewer panels than workers can still fill the machine. Guarded by
`OVERFIT_CONV_M_SPLIT` (default on; set to `0` for the A/B) and applied only on the AVX-512 path.

**The gate is measured, and the first version of it was too generous.** Splitting costs a duplicated B pack,
once per M-block instead of once per panel, so it has to be earned. Splitting wherever `nPanels < workers`
helped conv11-13 by 1.79x but cost conv8-10 **3-5%** (668.8 -> 633.7 GFLOP/s): those have 25 panels against
32 workers, occupancy already at 78%, and the extra pack outweighed the gain. Requiring `nPanels * 2 <=
workers` keeps the win and drops the loss - conv8-10 then measured within 1.0% of the unsplit arm, which
doubles as the canary for that run.

**Result on the target layers** (VGG-16 conv11-13, N=196, seven panels):

| arm | conv11 | conv12 | conv13 | rate |
|---|---:|---:|---:|---|
| split off | 4.18 ms | 4.33 ms | 4.10 ms | 213.5-225.6 GFLOP/s |
| split on | 2.35 ms | 2.26 ms | 2.32 ms | **392.8-409.2 GFLOP/s** |

**Result end to end**, `LargeCnnComparisonBenchmark`, both engines on all cores:

| model | split off | split on | change | ONNX Runtime canary |
|---|---:|---:|---:|---:|
| VGG-16 | 79.01 ms | **73.50 ms** | **-7.0%** | 21.31 -> 21.29 ms (0.1%) |
| CNN, 60.9 MB | 67.70 ms | **63.14 ms** | **-6.7%** | 12.89 -> 12.93 ms (0.3%) |

The ONNX Runtime side of the same process moved 0.1% and 0.3%, so the box was still and the lever was
isolated. Parity is unchanged in both arms: cosine 1.000000, max absolute difference 3.204e-7 (VGG-16) and
6.706e-8 (60.9 MB CNN), same argmax. Against ONNX Runtime the VGG-16 gap goes from 3.72x to **3.45x**.

**Coverage, because a green suite proved nothing here twice on the same day.**
`Tests/Core/Kernels/ConvGemmMSplitTests` checks `Gemm` against a naive triple loop at five shapes chosen
against the *decomposition* rather than against convolution, plus a sentinel test for writes past the result
and one for elements never written. Four defect mutations and one control: **three caught, the control
correctly not caught** (forcing a different but valid split must not change the output). **The fourth escape
is a finding rather than a gap.** Dropping the M-block's START bound - so every block begins at row 0 - is
not caught, because the workers then recompute each other's rows and store identical values to identical
addresses: the answer stays right and only the cost multiplies. **A lost start bound presents as a slowdown,
never as a wrong answer**, so the benchmark is its detector and the suite never will be.

**The tolerance in that test is a derived bound, not a chosen constant.** The first version asserted a
relative 1e-4 against `|expected|` and failed at one element out of 100,352: kernel 0.0021735937 against
reference 0.0021725819. A dot product of signed values cancels, so the final value is not the size of the
arithmetic that produced it - that result came out of partial sums near 2.7, where 1.0e-6 is ordinary
rounding. The test now carries the textbook bound `|error| <= K * u * sum|a_i * b_i|` with `u = 2^-24`,
accumulated per element and doubled for headroom. The defects it exists for miss by whole result magnitudes,
not by last bits.

**Not addressed, and both are named in the diagnostic below**: the 1.6x single-thread gap against the
isolated micro-kernel, and conv1 (K=27, 1568 panels, 20% efficiency from a different cause, 1.49 ms). **The
AVX2 path is deliberately unchanged** - porting the split there without measuring it on AVX2 hardware would
be exactly the port-a-measured-null mistake this task already recorded once.

### VGG-16 convolution, per layer, against a MEASURED ceiling (2026-08-18, `XC-78`)

`XC-78` was filed on the reading *"parallelism is fine (conv scales 9.98x); the gap is inside a single
thread"*, and prescribed reproducing the isolated micro-kernel number and adding surrounding costs back one
at a time. **The per-layer profile refutes that reading before any of that work starts.**

**The denominator was measured, not estimated.** A per-thread claim needs a per-thread ceiling, and dividing
by the all-core roofline understates one thread by the core count while correcting with a datasheet boost
clock is a guess. `MachineRooflineBenchmark.PeakFmaFloat512` now takes `OVERFIT_ROOFLINE_WORKERS`; the same
kernel and panels at 1 worker take 2.139 ms and at 32 workers 4.785 ms for 32x the work, so this box's
all-core FMA throughput is **14.30x its single-core throughput** - clock drop and SMT already inside that
number. **CORRECTED 2026-08-18, see the correction note: the ceiling is 5136 GFLOP/s all-core and
359 GFLOP/s single-core**, not the 2190 / 153 this section originally carried.

Ten runs, `OVERFIT_CNN_ONNX` on `vgg16.onnx`, one arm at `OVERFIT_PARALLEL_WORKERS=1`:

| layer | N | panels | 1 core | all cores | speed-up | % of the 14.30x ceiling |
|---|---:|---:|---:|---:|---:|---:|
| conv1 | 50176 | 1568 | 39.0 | 114.1 | 2.93x | **20%** |
| conv2 | 50176 | 1568 | 58.2 | 428.7 | 7.36x | 51% |
| conv3 | 12544 | 392 | 74.3 | 522.6 | 7.03x | 49% |
| conv4 | 12544 | 392 | 73.9 | 592.0 | 8.01x | 56% |
| conv5 | 3136 | 98 | 85.0 | 600.6 | 7.06x | 49% |
| conv6 | 3136 | 98 | 87.1 | 632.5 | 7.26x | 51% |
| conv7 | 3136 | 98 | 87.9 | 644.6 | 7.33x | 51% |
| conv8 | 784 | 25 | 93.8 | 640.1 | 6.82x | 48% |
| conv9 | 784 | 25 | 92.7 | 674.0 | 7.27x | 51% |
| conv10 | 784 | 25 | 92.5 | 695.5 | 7.52x | 53% |
| conv11 | 196 | **7** | 80.2 | 227.7 | 2.84x | **20%** |
| conv12 | 196 | **7** | 78.7 | 225.5 | 2.87x | **20%** |
| conv13 | 196 | **7** | 82.4 | 234.1 | 2.84x | **20%** |

Rates are GFLOP/s. Whole conv stack: **383.12 ms at one core (80.1 GFLOP/s), 60.33 ms at all cores
(508.5 GFLOP/s), a 6.35x speed-up = 44% of the machine's own 14.30x.**

**Finding 1 - parallelism is NOT fine, and it is the larger of the two losses.** 6.35x against an available
14.30x. The row's 9.98x came from a single-thread baseline of 574.74 ms; the same measurement today gives
**383.12 ms**, i.e. single-threaded conv is 1.5x faster than when that row was written and **the scaling
figure built on the old baseline does not survive**. What changed between the two runs was not identified
here - see the caveat at the end.

**Finding 2 - the single-thread gap is real but smaller than recorded.** The best-shaped layers reach
**92.5 GFLOP/s = 60% of the 153 GFLOP/s single-core ceiling**, while the micro-kernel measured in isolation
reaches 132-148 = **86-97%**. That is a **1.6x** gap from surrounding costs, not the 2.6x the row records -
because the 2.6x was computed from a whole-stack average dragged down by conv1 and conv2.

**Finding 3 - the concentrated loss has an exact structural cause, and it is not the kernel.**
`Conv2DGemmKernels.Gemm` splits work over N-panels: `nPanels = (n + nr - 1) / nr` with `nr = 32` on AVX-512,
then `OverfitParallel.For(0, nPanels, 1, ...)`. conv11-13 have **N = 196, so nPanels = 7**. Seven work items
cannot occupy 32 workers whatever the kernel does, and the seventh panel is 4 columns wide against the
others' 32, so even those seven are imbalanced. Measured 2.84x, structural ceiling 7x. **These three layers
are 12.05 ms = 20% of all-core conv time**; at conv9's 7.27x they would cost 4.72 ms, so the recoverable
amount is **7.33 ms, 12% of conv and ~7% of the whole model**. The lever is decomposition - split over M as
well as N when N is small (M is 512 in these layers) - not a micro-kernel rewrite.

conv1 is also at 20% but costs 1.49 ms all-core, and its cause is different (K=27, 1568 panels, so not a
decomposition shortage). Low priority.

**What this retires.** Do not start the ladder experiment the row prescribes, and do not re-run Winograd on
the 512-channel layers yet: both target the 1.6x single-thread gap while a 1.9x parallel gap and a
structural 20%-efficiency block sit above them in the budget.

**Instruments added, so this is repeatable.** `OnnxGraphModel.PerNodeProfileReport()` now prints `K` and
achieved GFLOP/s for every `ConvLayer` (a millisecond column ranks layers; a GFLOP/s column says whether a
slow layer is slow because it is big or because the kernel runs badly on its shape, and those call for
opposite work). `ConvLayer.KernelElementsPerOutput` is internal and exists only to feed it.
`MachineRooflineBenchmark` takes `OVERFIT_ROOFLINE_WORKERS`.

**Caveats, and one of them is unresolved.** The 574.74 -> 383.12 ms single-thread change between 2026-08-17
and 2026-08-18 is **not explained**; both are "production single-threaded conv on VGG-16" and I did not
identify what moved. Treat the 1.5x as an open question, not as a win. Also not measured: any thread count
between 1 and all, batch sizes above 1, and whether the same shape collapse appears on other models. Raw
output in `artifacts/xc78/`.

### The published Overfit-vs-ONNX-Runtime size curve (2026-08-18, `XC-77`)

The figures the README publishes under *"How that ONNX Runtime ratio scales"*. Re-measured after the
`LinearKernels` work of `XC-79`/`XC-80`/`XC-81` made the earlier table stale, and with Rider shut down
mid-session, so the two rounds are not on an identical box state - the canary below is what makes them
comparable.

| model | Overfit | ONNX Runtime 1.29.0 | ratio | threads |
|---|---:|---:|---|---|
| `Linear(784x10)`, 7,840 params | 225.6 ns | 1,855.3 ns | Overfit **8.22x** | ORT pinned 1; Overfit serial by policy |
| MLP `784-256-128-10`, ~235k | 6,721 ns | 8,777 ns | Overfit **1.31x** | 1 vs 1, proven below |
| MNIST CNN (imported ONNX) | 5,290 ns | 6,616 ns | Overfit **1.25x** | 1 vs 1, proven below |
| CNN, 60.9 MB | 63.14 ms | 12.93 ms | ORT **4.89x** | all cores both, with the `XC-78` split (67.70 ms without) |
| VGG-16, 30.94 GFLOP | 73.50 ms | 21.29 ms | ORT **3.45x** | all cores both, with the `XC-78` split (79.01 ms without) |

**The thread question was settled by a second arm, not by an argument.** Three of these benchmarks construct
their `SessionOptions` with `IntraOpNumThreads = 1, InterOpNumThreads = 1`, so the comparison is only
like-for-like if Overfit is also single-threaded at that size - and *"only the diagonal of a thread-count
grid is like-for-like"* is a trap this repository has already fallen into once. Re-running each with
`OVERFIT_PARALLEL_WORKERS=1`:

| model | default | 1 worker | Overfit moved | ORT canary moved |
|---|---:|---:|---:|---:|
| `Linear(784x10)` | 225.6 ns | 189.5 ns | **-16.0%** | +0.9% |
| MLP | 6,782 ns | 6,721 ns | -0.9% | +0.7% |
| MNIST CNN | 5,290 ns | 5,351 ns | +1.2% | -0.7% |

The MLP and the MNIST CNN never reach a parallel path, so those two rows are one thread against one thread
and are publishable as they stand. **This retires the restriction the `XC-77` row carried** - that no
single-thread CNN number should be published - for the MNIST row specifically, by measurement rather than by
assertion.

**`Linear` is the exception and it moves the wrong way.** Forcing one worker made Overfit 16% *faster*
(225.6 -> 189.5 ns), which would raise the published ratio from 8.22x to 9.88x. The ORT canary moved 0.9%, so
the box was still and the effect is real. **The README publishes the slower default-configuration number**,
because that is what a consumer gets without setting an environment variable. *Why* the idle worker pool
costs 36 ns on a 226 ns inference is not established here, and is worth a row of its own.

**The near-parity rows are noisier than the gap they describe, and the noise is one-sided.** Across two
process repeats of the MLP benchmark: Overfit 6,782 -> 6,753 ns (**0.4%**), ONNX Runtime 8,777 -> 9,421 ns
(**7.3%**), ML.NET 7,596 -> 9,804 ns (**29.1%**). A box-wide drift would have moved all three; only the two
native-backed engines moved. The published 1.31x is therefore taken from the repeat where the **opponent**
was fastest, which is the conservative direction. Anything between 1.2x and 1.3x here should be read as
"about even", not as a ranking.

**Both large CNNs are numerically identical to ONNX Runtime**, checked in the benchmark's own `GlobalSetup`
rather than asserted: cosine 1.000000, max absolute difference **6.706e-8** (60.9 MB CNN) and **3.204e-7**
(VGG-16), same argmax (993 and 577). The gap is speed, not accuracy.

**Provenance.** `Benchmarks.SingleInferenceBenchmark`, `Benchmarks.MLNetSingleInferenceBenchmark`,
`Benchmarks.ImportedOnnxMnistCnnBenchmark`, `Benchmarks.LargeCnnComparisonBenchmark` (the last one twice,
via `OVERFIT_CNN_ONNX` pointed at the 60.9 MB `cnn.onnx` then at `vgg16.onnx`, both under the model fixture
directory). Raw BenchmarkDotNet output in `artifacts/xc77/`. Environment variables passed through `env=` in
`subprocess.run`, never as a shell prefix. Rider was running during round 1 and killed before round 2, which
is why round 2 is not a valid best-of-N partner for round 1 on absolute times - the MLP repeat above is the
only cross-round comparison drawn, and it is drawn to bound noise, not to pick a winner.

**What was NOT measured.** ResNet-50 (the exported model exists but the importer does not yet handle it),
any batch size above 1, and any thread count between 1 and all cores for the two large CNNs.

### CNN inference vs ONNX Runtime — where .NET loses, quantified (2026-08-17)

Same box and provenance as the block above; `Microsoft.ML.OnnxRuntime` **1.29.0**. Models from
`C:\onnxmodels\`. **The advantage reverses with model size, and the crossover is not subtle.**

| model | ORT threads | Overfit | ORT | verdict |
|---|---|---:|---:|---|
| `Linear(784→10)` | 1 | 236.8 ns | 1,962.7 ns | Overfit **8.29×** |
| MLP `784→256→128→10` | 1 | 8.883 µs | 9.445 µs | Overfit **1.06×** |
| MNIST CNN (imported ONNX) | 1 | 5.410 µs | 6.649 µs | Overfit **1.23×** |
| `cnn.onnx` (60.9 MB) | all | 66.72 ms | 11.15 ms | **ORT 5.98×** |
| **VGG-16** (~15.5 GFLOPs) | all | 100.70 ms | 20.67 ms | **ORT 4.87×** |

**The gap is KERNEL QUALITY, not threading, and that took a 2×2 to establish.** A first pass pinned only
ORT and read *Overfit 101.5 ms vs ORT 100.8 ms* as a dead heat — **it is not**: that cell is Overfit on all
cores against **one** MLAS core, which is the opposite of reassuring. The full grid on VGG-16:

| | ORT all cores | ORT 1 thread |
|---|---|---|
| **Overfit all cores** | Ov 100.70 / ORT 20.67 → **ORT 4.87×** | Ov 101.50 / ORT 100.80 → 1.01× *(not like-for-like)* |
| **Overfit 1 worker** | Ov 629.09 / ORT 20.47 → ORT 30.7× *(contaminated, see below)* | Ov 424.70 / ORT 101.00 → **ORT 4.20×** |

**Read only the diagonal.** At matched thread counts the gap is **4.87× on all cores and 4.20× on one**, so
it is roughly constant and therefore a property of the kernels, not of the parallelism. **Both engines scale
about equally**: Overfit 424.70 → 100.70 ms = **4.22×**, ORT 100.80 → 20.67 ms = **4.88×**. Overfit's
parallelism is fine; its per-core convolution throughput is ~4–5× behind MLAS.

**Levers proven live, canaries clean** — required, because a flat result would otherwise be indistinguishable
from a dead toggle. `ORT_INTRA/INTER_OP_NUM_THREADS` moved ORT by **392%**; `OVERFIT_PARALLEL_WORKERS` moved
Overfit by **525%**; and each toggle left the *other* engine within **0.2–1.0%**, which is what says the box
held still. Neither variable was passed as a shell prefix — both went through `env=` in `subprocess.run`.

**One anomaly, stated rather than smoothed:** Overfit at 1 worker measured **629.09 ms** with ORT unpinned
against **424.70 ms** with ORT pinned — the same Overfit configuration, **48% apart**. The likely cause is
ORT's persistent thread pool still occupying cores while the Overfit arm is timed, which hurts most when
Overfit has a single worker to starve. It is a hypothesis, not a measurement; the 4.20× figure uses the
cleaner (ORT-pinned) cell, and a single-thread comparison of these two engines in one process should not be
trusted below that resolution without isolating the runs.

**Where VGG-16's 100 ms actually goes** (`Tests/Diagnostics/ConvGemmPartProfileTests.cs`, pointed at
`vgg16.onnx`, all workers versus `OVERFIT_PARALLEL_WORKERS=1`):

| part | all workers | 1 worker | scaling | share |
|---|---:|---:|---:|---:|
| convolution | 57.58 ms | 574.74 ms | **9.98×** | 58.2% |
| **fully-connected** | **36.50 ms** | **36.98 ms** | **1.01× — none** | **36.9%** |
| pooling | 4.88 ms | 4.56 ms | — | 4.9% |

Within convolution the split is **im2col 18.0% / GEMM 82.0%** (9.69 ms and 44.16 ms per run), and the GEMM
runs at **695 GFLOP/s = 13.5% of the measured 5136 GFLOP/s roofline** (this line originally said 32%
against a 2190 GFLOP/s figure; see the correction note).

**Two things this refuted.** The leading hypothesis was that im2col's materialisation dominates — the
28.9 MB expansion is real, and it is **third-order**. And the FLOP arithmetic here was wrong twice in our
favour before it was checked: VGG-16 is **30.94 GFLOP**, not the commonly quoted 15.5, which counts MACs;
and the denominator has to be the roofline `MachineRooflineBenchmark` actually measures, not a
clock × FMA-width estimate. On that basis, all cores against all cores: **Overfit 307 GFLOP/s = 14% of
roofline, ONNX Runtime 1497 GFLOP/s = 68%**.

**The largest single item is not a kernel at all**: `LinearLayer[33]` (25088→4096) takes **31.34 ms with
every worker and 31.51 ms with one**, because `LinearKernels.Forward` has no parallel branch at any size —
confirmed in the source, not inferred. At batch 1 that layer reads **411 MB** of weights to do 0.206 GFLOP,
so it is bandwidth-bound and runs at **13.1 GB/s** on a single core. That is `XC-79`, and it is worth more
than everything the conv path could give.

**This confirms a weakness the project already claims rather than discovering one** — `docs/ideas.md:39`
already names *"where .NET loses (CNN vs MLAS …)"*. What is new is the number: **~5× on a real ImageNet CNN,
at matched threads**, against **8.29× in our favour** on a 7,840-parameter `Linear`. Both are true; only the
second is in the README (`XC-77`).

### Recovering the llama.cpp baseline — what was found, and what stays unknown

Recovered 2026-08-17. The number itself was never in doubt; **which llama.cpp it was measured against was
simply never written down**, and every public claim resting on `~1.13×` inherited that.

**What the evidence is, and what kind of evidence it is.** The clone lives at `D:\llamacpp-tmp` and its
`git reflog` records every checkout with a date. That is a record of *what was on disk when*, not a record
of the run — the run wrote nothing. So this is reconstruction, and it is stated as reconstruction.

| when | what the reflog says | bearing on the decode number |
|---|---|---|
| 2026-05-19 10:16 | `clone` → `6db1304` | the state before the sprint |
| **2026-05-31 11:37:43** | `pull --ff` → **`3292da0`**, which `git describe` resolves to **`b9441`** (authored 2026-05-31 11:21 CEST, so pulled 16 minutes old) | **the sprint that produced `12.55 → ~17 tok/s` and `~1.13×` is dated 2026-05-31** |
| 2026-07-01 / 07-22 / 07-23 | `4fc4ec5` (`b9859`), `6d5a910` (`b10092-2`), `da296d6` (`b10103-1`) | all later than the decode sprint; the July pulls are the prefill work |

**The residual uncertainty, named rather than smoothed over.** The sprint's measurements are dated to the
day, not to the hour, so strictly the decode run used **`6db1304` or `3292da0`**. A fresh `pull` at 11:37
on the morning of the sprint is a strong indication the intent was to measure against current `master`,
and `3292da0` is the answer to quote — but anyone republishing the figure should say "b9441" and not
pretend the hour was recorded.

**The binary that produced the number no longer exists.** All three build directories (`build-avx2`,
`build-bench`, `build-tbo`) were rebuilt on 2026-07-22 and their `build-info.cpp` carries
`LLAMA_COMMIT = "6d5a910"`, `MSVC 19.44.35228.0`, `x64`. So the decode comparison cannot be re-run against
its original binary without checking `3292da0` back out and rebuilding. *Noted in passing: `6d5a910`
describes as `b10092-2`, so even the July prefill row's own `b10088` is a couple of commits off the binary
that ran — close enough not to matter there, and a reminder that a build number written from memory is not
the same artefact as one read out of the binary.* *Also worth knowing before trusting
a build banner here: those same files carry `LLAMA_BUILD_NUMBER = 861`, which is a CMake fallback and is
nonsense against a tree that describes as `b10103` — the commit field is trustworthy, the number is not.*

**The model side IS pinned exactly, and it is the same file both engines read** ("same-file gap" in the
ROADMAP is literal):

```text
C:\bielik\Bielik-4.5B-v3.0-Instruct-Q4_K_M.gguf
2878886912 bytes, mtime 2026-05-30 22:44
sha256 39fb78db7c5e1582d3ef5bded109cd98606aabd5acbb5c98601155174b6763e1
```

The mtime is the evening before the sprint, which corroborates the reflog rather than resting on it.

**Still not recorded, and no artefact anywhere can supply it: the llama.cpp thread count and build flags
for the DECODE run.** `ROADMAP-COMPLETED.md:819` records `/arch:AVX512` and 16 threads for the *prefill*
comparison in July, and the same file notes that `llama-bench` picked 16 threads over the machine's 32 and
beat a 32-thread run — so thread count is known to move the answer materially on this box. **`~1.13×` is
therefore citable as "against llama.cpp `b9441` on that exact GGUF", and NOT as a thread-for-thread
comparison.** Closing that last hole needs a re-run, not an archaeology pass.

**And the figure this pins is not the one the README publishes.** Found while doing the above, and it
matters more than the archaeology: `~1.13×` is the **Bielik-4.5B** decode figure from 2026-05-31, and it is
what `ROADMAP.md`, `ROADMAP-COMPLETED.md`, `Sources/Main/LanguageModels/README.md` and
`Sources/Main/LanguageModels/Runtime/README.md` all cite. `README.md` — the public one — cites **two other
numbers**: `~1.15×` at `:461` (decode, *with the repacked-GEMV flag on*, i.e. a different configuration)
and `~1.2×` at `:626`. Both name their baseline as *"a current AVX-512 llama.cpp build"*, and **"current"
decays**: it was written at some point against some build and now reads as a claim about whatever a reader
would install today. So pinning `b9441` unblocks the internal figure and leaves the **published** ones
resting on an undated word. Nothing here is evidence that those two numbers are wrong — they are a
different subject measured a different way — only that they cannot be sourced. **They need their own
re-run or their own archaeology; do not paper over them by copying `b9441` across.**

---

## Reverted or regressed — do not propose again without new evidence

| change | measured | why it lost |
|---|---|---|
| **Second FMA accumulator in `Simd.Dot`** | real 1.79× at 65536 floats (2906→1626 ns); 60.5→38.8 ns at 2048 | Path census: `Dot` is never on a forward/inference path, and 100% of real training calls land at lengths 32/68/128/512/784 — below where two chains pay. **The untaken guard branch alone cost 9–11%** at those lengths. Revisit only for dModel/dFF ≥ 2048. |
| **Winograd F(2,3)**, 3×3 stride-1 conv | parity-correct (cos 1.0), **+79% slower** on deepcnn, 119.7→214.4 ms | Sequential scalar transforms + 16 small GEMMs + 16× U/V/M blow-up beat the 2.25× FLOP cut. |
| **Register-blocking** (direct conv) | regressed | reverted |
| **K-blocking + A-packing** (im2col GEMM) | regressed | reverted |
| **AVX-512 decode port** | regressed | Decode is memory-bandwidth-bound after GQA K/V-once + fuse-quantize; a faster dot kernel saves cycles already hidden behind weight-read latency. |
| **Bias in the Q4_K tiled prefill GEMM** (`GemmTiled`) | **0.999× — an exact tie** | `bias.IsEmpty` barred 88% of prefill dispatches; lifting it changed nothing because `ProjectBatchedWeightStationary` already amortises weight decode across the row tile. The "~3×" in the kernel doc is against re-decode-per-row, **not** against weight-stationary. |
| **`OverfitPool<T>`** | 3× slower typical, ~3000× pathological vs `PooledBuffer<T>` | deleted |
| **Q6_K weight-stationary** | **+13.5% slower** on `ffn_down` | Canaries drifted 1–2%, so the regression was real, not noise. |
| **VNNI `vpdpbusd`** vs AVX2 `vpmaddubsw`+`vpmaddwd` | AVX2 ≈ VNNI ≈ 19.1 tok/s, ~0 gain | Decode was already bandwidth-bound, not ALU-bound. |
| **`ProjectParallel` for the LM head** | ~3% steady-state gain, **~3 KB allocated per call** | Breaks the 0 B/token decode contract. The 10× in `LmHeadParallelBenchmark` is steady-state; per-token decode is dominated by `Parallel.For` dispatch. |
| **Parallelising `TensorMath.Add`** (residual) | **+20% wall** on GPT-1 batch=32, +55% on `Add` backward | Memory-bandwidth-bound (2 reads + 1 write); 2–3 cores already saturate a ~50 GB/s bus, so a ~10 µs cold dispatch is pure cost. |
| **FP16-resident weights** | steady RAM **regressed** 14.36 → 15.85 GB | The F32→F16 load conversion churns multi-GB F32 buffers. |
| **Unrolled fixed-tile GEMM specialisation** | failed at `cols: 8`; 7–9× slower (70–83× single-threaded in one config) | reverted |
| **Output-row banding** ("missing L2 blocking level") | **20% slower** | reverted |
| **Column-pairing the Q6_K port** | slower; reverting restored 2398.5/2399.6 ms exactly | reverted |
| **`Conv2D` → `OverfitParallelFor`** | **+13% MNIST wall time** | Conv2D stays on `Parallel.For`. |
| **Naive single-threaded cached decode** | **2700 ms/token vs 424 ms/token uncached — 6× slower** | at demo lengths, against already-parallel recompute |

**Known stale and not yet fixed:** `Avx512Threshold = 512` in `Sources/Main/Intrinsics/Simd.cs` is
**unmeasured**. `Vector512WidthBenchmark` at 128 floats — a quarter of the threshold — had 512-bit winning
every op (Add 0.74×, MulAdd 0.80×, Dot 0.84×). Left alone deliberately: moving a threshold on three
microbenchmark points without a caller-length census is the same mistake the `Simd.Dot` change made. Do not
claim it is handled; do not fix it without the census.

## Wins that went the opposite way to the "obvious" move

- `TensorPrimitives` bulk-SIMD **beat** a hand-written micro-kernel.
- The **simple** register-blocked GEMM beat the cache-blocked one.
- Sixteen independent FMA chains were **30% faster at 256-bit**, not slower — more chains cover latency
  better. Contrast with `Simd.Dot` above, where the same mechanism was worthless because of call-site
  lengths, not because the mechanism was wrong.

## Shapes and call costs

| question | measured |
|---|---|
| `for` vs `foreach` over an array (.NET 10) | **not a lever** — ~2 ns, and the direction reverses with size |
| **declared type** on a hot path | **this is the lever**: an interface costs **2.4× iterating**, **4.6× indexing**, plus **32 B** for the enumerator. Do not "tidy" `T[]` into `IReadOnlyList<T>`. |
| monomorphism | does **not** guarantee zero allocation — this refuted an earlier explanation of my own |
| extracting a method when the JIT does not inline it | **2.25×** — which is why `else` removal prefers restructuring in place over extraction |
| ternary, `continue`, condition inversion | free |
| `OverfitParallelFor` vs `Parallel.For`, **decode** | **617 µs / 0 B** vs **1502 µs / 505 KB** capped and **2223 µs / 862 KB** uncapped = **2.43×** / **3.60×**. 183 dispatches over a 4096 range, 9950X3D, .NET 10.0.8, Release, HEAD `e21e7c3`, 2026-08-14. Supersedes **455 / 2059 µs (4.5×)**, which compared against the *uncapped* path — see the decode-pool section below |
| `OverfitParallelFor` vs `Parallel.For`, **Conv2D** | the opposite — see the regression table |

## The decode spin pool: what it is actually worth — audited 2026-08-14 (`PB-12`)

Ryzen 9 9950X3D (32 logical / 16 physical), Windows 11 26200, .NET 10.0.8, Release, HEAD `e21e7c3`.
**24 measurement processes, ABAB at process level, best-of-3 within each process, a single-thread canary
before and after every timed block, and a dispatch-count liveness probe in every process.**

This section supersedes `455 µs / 2059 µs (4.5×)`, `+28%`, `+3%` and `+11%`, which were carried in source
comments from 2026-06-11 (`8a28bdb`) and copied into four documents between 2026-08-01 and 2026-08-10 —
**not one of them recording a model, quantisation, comparator configuration, box or build.**

### Dispatch level: `ForDecode` vs `Parallel.For`

Single process, ABAB, 183 dispatches over a 4096-element range.

| arm | time | allocated | ratio vs the pool |
|---|---|---|---|
| `OverfitParallel.ForDecode` (spin pool) | **617 µs** | **0 B** | — |
| `Parallel.For`, **capped** at `DecodeMaxWorkers` | 1502 µs | 505 KB | **2.43×** |
| `Parallel.For`, **uncapped** | 2223 µs | 862 KB | **3.60×** |

**Which comparator you pick is most of the ratio, and that is what made `4.5×` wrong.** The retired pair
sat on the *uncapped* arm — but with `OVERFIT_DECODE_POOL=0` today's `ForDecode` falls back to the
**capped** path, so `4.5×` described a comparison the product no longer makes anywhere on the decode path.
Honest figure: **2.3-2.7× at dispatch level**, or **+25% end-to-end**. The `0 B` holds; the `925 KB` is
shape-dependent and measured **862 KB - 1.13 MB** here.

### End to end, decode pool ON vs OFF

| model | measured | against the retired figure |
|---|---|---|
| Qwen3-0.6B **Q8_0** | **+25.1%** (56.56 → 70.75 tok/s); paired ratios 1.251 / 1.274 / 1.226 | `+28%` **supported**, re-stamped — the old comment recorded no quantisation |
| Qwen3-0.6B **Q4_K_M** | **+23.0%** | as above |
| Phi-3.5-mini 3.8B | **−1.9%** (13.38 → 13.12 tok/s); paired 0.986 / 0.968 / 0.965 / 1.005 | `+3%` **refuted — sign reversed** |

Phi-3.5's point estimate is itself inside this box's ±3-4% cross-process floor, so the claim that survives
is **"neutral to negative on Phi-3.5"**, not a number. It is not the box moving: the untouched arm
reproduced to 0.8% while the treated arm fell 4.2%. **The mechanism is a wrong denominator** — on Phi-3.5
the pool captures only ~75% of dispatches, because **52 per token still go through `OverfitParallel.For`**
via the GQA `For(0, KvHeadCount, ...)` at `CachedMultiHeadAttention.cs:284`.

### The decode worker cap (`OVERFIT_DECODE_WORKERS`), Bielik-4.5B Q4_K_M

| arm | measured |
|---|---|
| cap 10 vs 32 with the **pool off** — the cap's original mechanism, isolated | **+3.8%**, every paired cycle positive (1.025 / 1.057 / 1.038) |
| the same knob at HEAD's default (pool on) | **+87%** — the cap now also sizes the spin pool, so it is no longer one lever and the two are not comparable |
| `OVERFIT_DECODE_WORKERS=32` today | **−47%** (15.17 → 8.10 tok/s) |

`+11%` is **not supported as stated**: the sign is real, the magnitude is not. The `−47%` supersedes the
~−11% implied by the 2026-06-11 cap curve (`6→11.8, 8→13.4, 10→14.0, 12→14.1, 32→12.5`); only that curve's
`32` endpoint was re-measured, so the 10-12 plateau that sets the default still rests on the 2026-06-11 run.

### Dispatch census per token — previously unwritten anywhere

| model | dispatches/token |
|---|---|
| Qwen3-0.6B Q8_0 | **183.1** |
| Qwen3-0.6B Q4_K_M | 137.6 |
| Bielik-4.5B Q4_K_M | 383.6 |
| Phi-3.5-mini 3.8B | 209.1 |

### The finding that outlives the numbers

**`ForDecode` has never had a benchmark class.** `git log -S "ForDecode" -- Sources/Benchmark` returns zero
commits. All four retired figures came from `[ModelFact]`-gated diagnostics that `dotnet test` never runs
(`ModelFact : LongFact` sets `Skip` at discovery) — single arm, one process, no canary. `OverfitParallelBenchmark`
exercises `OverfitParallel.For`, not the decode pool, so **none of these numbers was measurable by anything
in `Sources/Benchmark`, before this audit or after it.**

**Do not cite** `Sources/Benchmark/BenchmarkDotNet.Artifacts/results/Benchmarks.OverfitParallelForBenchmark-report-github.md`
— 2026-05-15, the class has since been deleted, `InvocationCount=1`, `RatioSD` up to 11.83, and it shows the
*opposite* result at dispatch-bound sizes. It is not the source of any published figure.

**The 2026-08-14 claim-protocol change's own cost is DERIVED, not measured, and deliberately so.** One
uncontended CAS plus one monitor enter/pulse per dispatch is tens of ns against a ~3.4 µs dispatch and
~14 ms/token — **0.02-0.05% of token time**, three orders of magnitude below the ±3-4% cross-process floor.
No experiment on this box can resolve it, and building the pre-change revision would have produced a number
with no meaning.

## Throughput and memory

| subject | measured |
|---|---|
| Bielik decode after the CPU sprint | 12.55 → **17 tok/s** (bit-identical output) |
| Qwen-3B with `OVERFIT_REPACK_GEMV` | **24.4 tok/s** (+30%) |
| gap to llama.cpp | **~1.13× uniform** — this is *not* parity; always best-of-N on both sides. Measured 2026-05-31 against llama.cpp **`3292da0` / `b9441`** on `Bielik-4.5B-v3.0-Instruct-Q4_K_M.gguf` (sha256 `39fb78db…`), recovered 2026-08-17 from the clone's reflog — see *"Recovering the llama.cpp baseline"*. **Their thread count is still unrecorded and is known to matter on this box**, so cite the build, not a thread-for-thread comparison |
| QLoRA fine-tuning, 3B | ~3 GB RAM |
| Phi-4 14B Q4_K_M | ~3.8 tok/s |
| Android (Motorola Edge 50 Fusion), 0.5B Q4_K | ~3.8 tok/s |
| FFN and LM head | at the DRAM floor |

## Anomaly guard

| subject | measured |
|---|---|
| false positives, 2026-08-02, older build | **112/day** |
| false positives, 24 h to 2026-08-05 | **5/day** (Poisson 1–9), 292 cycles, 0 failures |
| of which `GcGen2HeapBytes` | 108/day → **0** — but partly an artefact of young heaps, **not** purely the floor repair |
| non-heap channels | 4/day → 5/day, i.e. **no change** — the floor repairs did not buy detection with noise |
| peer memory gap, generation `7f5fb9f88c` | median **18.8 MB**, clears the 9.52 MB floor in 52% of samples, top-2 set changed **0 of 64** times |
| peer memory gap, generation `7765564ff6` | median **5.9 MB**, clears the floor in **0 of 108** samples |
| peer visibility | **blind to a single OOMKill**, by construction |
| healthy peer gap, fleet 50 min old | p90 **6.5 MB**, p99 7.4, max **8.6** — under the 9.52 floor, but not by much |
| healthy peer gap, fleet 19 h old | max **9.7 MB** — **above** the floor: the population spreads with age |
| floor margin, observed live | **1.4%** — `lzf4n` at absGap 9.39 MB with `effect=1 p=0` was held back only by the floor |
| 100 KB/s leak, gap +1.02 → +26.24 MB in 6 min | caught by `GcGen2HeapBytes` (floor 0.93 MB) in the FIRST cycle, by `MemoryWorkingSetBytes` (floor 9.52 MB) **30 minutes later** |
| fleet gen-2 sawtooth | **32.5 MB**, synchronised across 11 of 12 pods to the same minute; `dotnet_gc_heap_size_bytes` rose +48.0 MB/h against working set's +51.9, so the climb IS the managed heap |
| peer construction vs that sawtooth | fleet level swings 32.5 MB, healthy gap spread **0.9 MB** — **97% of common-mode movement cancelled** |

### Historical replay through the guard's own cycle, 2026-08-08

`Tests/Anomalies/Diagnostics/AnomalyGuardReplayDiagnostics.cs` (`[LabFact]`), 12 `lab-workload` pods, 288
cycles at 5-minute cadence, against the lab's live Prometheus.

**Read every figure below as an order-of-magnitude ceiling, not as a baseline** — this is
`overfit-perf-claim-auditor`'s verdict on the run, and the caveats are the reason: **one run per sample**
(four runs, not best-of-N), taken on the dev box that **also hosts the lab it queries**, with **no canary and
no ABAB interleave**. The ~23% spread between runs matches this box's known drift and is not evidence of
anything structural. These are the first numbers of their kind in this repository; a controlled repeat
supersedes them.

| subject | measured |
|---|---|
| 288-cycle replay, wall clock | **4.8–5.9 s** per run (against the diagnostic's 30-minute failable ceiling) |
| of which in-memory guard processing + logging | 2.15–3.16 s, i.e. **~10.7–15.7 ms per *completed* cycle** |
| `PrometheusMetricWindowSource.ReadAsync`, live round trip | ~5 ms/call |
| `PrometheusTopologySource.RefreshAsync` | ~4 ms/call |
| whole cycle, distribution | p50 19 ms, p95 30–52 ms, max 137 ms |
| cycle outcomes | **201 completed, 87 blind, 0 failed** — the blind ones are a real 7.67 h scrape gap, re-verified against Prometheus by the auditor rather than taken on trust |
| replay determinism | unpinned anchor: **47 / 43 / 37** incidents opened over the same window; pinned anchor: identical twice |

**Denominator trap, paid for on this very run.** The first report of it said "≈11 ms per completed cycle",
dividing the 2.15–3.16 s processing total by all **288** cycles when only **201** completed — a blind cycle
returns before `_guard.RunCycle` and contributes nothing to the numerator. **~10.7–15.7 ms is the corrected
figure; do not re-cite the ~11 ms one.**

**The anchor is the measurement, not a detail of it.** Three back-to-back replays of "the same" window whose
start anchors differed by 7 s and 2.5 min opened 47, 43 and 37 incidents — a threshold change worth ±10%
would have been indistinguishable from the anchor moving. `OVERFIT_REPLAY_START_UTC` is now required by the
diagnostic and an unparseable value is refused rather than silently replaced.

## Semantic navigator (`Tools/SemanticNavigator`), 2026-08-06

| phase | cost |
|---|---|
| open solution (25 projects) | 3.2 s |
| build semantic model (1587 documents) | 3.9 s |
| CLI, per invocation | ~9 s |
| MCP server startup | 7.0–7.7 s once |
| warm **symbol** query (`refs`/`impls`/`callers`) | **3–23 ms** |
| warm `find_unused`, `Cli` / `Anomalies` / `Main --public` | 0.09 s / 2.75 s / **6.02 s** (694 candidates) — 9–45 ms per candidate |

**Measure twice.** Every warm query above was run in two passes: the first costs 100–1990 ms while
per-document state faults in, the second 3–23 ms. A single sample would have reported a number two orders of
magnitude too high.

## Analyzer capability: what a built-in rule does NOT catch — measured 2026-08-16 (`XC-64`)

**`CA1305` cannot see a culture-sensitive plain interpolation, so it is not a guard against this defect
class.** Measured rather than assumed, because *"just turn on CA1305"* is the obvious proposal and it does
not work: `dotnet build Sources/Anomalies -p:AnalysisMode=All -t:Rebuild` produces **exactly 2 `CA1305`
diagnostics in the entire project, and neither is one of the 42 defective sites** — both are
`StringBuilder.Append(ref AppendInterpolatedStringHandler)` (`AnomalyGuardConfigReader.cs:348`,
`MetricMap.cs:240`), both benign. The rule fires on `IFormatProvider` **overload selection**; a bare
`$"…{value:P0}…"` selects no overload at all, so there is nothing for it to flag.

**What that leaves.** The 42 sites were fixed and are pinned by
`Tests/Anomalies/DiagnosticTextIsCultureInvariantTests.cs`, which compares the same producer's output under
the invariant culture against `pl-PL` and `tr-TR`. That test protects **today's** sites; it cannot see site
43. A durable guard has to be an `OVERFIT0xx` analyzer with its own `AnalyzerReleases.Unshipped.md` entry,
`.editorconfig` severity and `Tests/Analyzers/` case — not yet written.

**The general point, which is why this sits in a measured-baselines file rather than a task comment**: a
built-in analyzer's *name* describes its intent, not its reach. Before adopting one as a guard, run it
against known-defective code and count how many of the known defects it names. Two hits and zero relevant
is a capability measurement, and it is worth as much as a timing.

## Which format specifiers actually move with the culture — measured 2026-08-17 (`XC-65`)

.NET 10 SDK 10.0.111, ICU, this Windows dev box. Invariant culture against **pl-PL, ar-SA, sv-SE, fi-FI**;
a value is "moves" if any of the four differs from invariant by an ordinal comparison. This is the predicate
`OVERFIT047` is built on, and **two of the three families are narrower than they look**:

| hole type | moves | does NOT move |
|---|---|---|
| `double` (and `float`/`decimal`/`Half`/`BigInteger`) | none, `F2`, `N0`, `G`, `R`, `E2`, `P1` — every specifier tested | — |
| `DateTime` / `DateTimeOffset` | none, `G`, `g`, `F`, `d`, `D`, `T` | **`o` `O` `s` `u` `R` `r`** — the BCL formats these five against `DateTimeFormatInfo.InvariantInfo` whatever provider is passed |
| `TimeSpan` | **only `g` and `G`** (they take the fractional-second separator from the culture) | **none at all, `c`, `t`, `T`** — all the invariant constant format |
| `int` (negative, `-1234567`) | none, `D`, `G`, `N0` | `X` |

**`R` is in both columns depending on the hole's type** — round-trip means *invariant* for a date and means
*the culture's decimal separator* for a `double`. A rule keyed on the specifier character alone gets one of
those two wrong whichever way it is written.

**What it cost to not know this.** The first build of `OVERFIT047` fired on
`Sources/Anomalies/Monitoring/SuppressionStore.cs:58` — `$"…{suppression.Until:u}…"`, already correct — and
that was the *only* site in the directory the rule was about to be armed at `error`. Arming it would have
forced a pragma onto correct code, which is how a guard teaches people to stop reading it. The negative
integral residual of the same measurement (`-1234567` differs under ar-SA/sv-SE/fi-FI, U+2212 or U+061C) is
real and is deliberately **not** flagged: reaching it means flagging every `{count}` in the tree.

## Measurement traps already paid for

- **Cross-process before/after does not work on this box.** A prefill change read +5% while the *untouched*
  decode path in the same run moved +32%. Interleave configurations **ABAB in one process** and time a canary
  path in every sample.
- **`OVERFIT_TILED_PREFILL` is dead whenever a `*.gguf.repack` sidecar sits next to the model** —
  `IsPrepacked` short-circuits it, so both arms run identical code. **Count the paths actually taken** before
  believing any kernel A/B.
- **Wrong job for the workload.** The shared `BenchmarkConfig` pins `InvocationCount=1`/`UnrollFactor=1`,
  which suits multi-millisecond model runs and leaves a microbenchmark measuring timer noise: a ~15 µs
  operation produced `RatioSD` 0.44 and a phantom 1.61× regression that was 1.01 under `[SimpleJob]`.
- **The scaffolding outweighing the subject.** A saturating `float`→`long` cast in a synthetic branch body
  made a non-inlined call measure *faster* than inlining it. **If a result is backwards, suspect the
  benchmark before the runtime.**
- **A thermally-throttled or loaded box invalidates an A/B.** Detect it with a canary — re-measure an
  unchanged path; if it moved, the box did, not your change.
- **Native-AOT publishes to the BASELINE instruction set unless told otherwise — no AVX2.** SIMD decode ran
  **~6x slower** under AOT than under the JIT for this reason alone, and nothing in the build said so. Fixed
  with `IlcInstructionSet=avx2` in `Cli.csproj`; `fma` and `x86-x64-v3` were rejected by ILCompiler. **Any
  AOT-vs-JIT performance comparison is invalid until the instruction set is pinned**, and this is invisible
  in the output.
- **A parity failure is as likely to be the test's premise as the code's.** The Q4_K_M "parity bug" was a
  **test-premise defect**: native Q4_K_M decode was correct all along, and the test was comparing against the
  wrong reference. Fixed by comparing against an F32 dequantisation **of the same file**. Before debugging a
  kernel against a parity failure, state what the reference is and why it is the right one.
- **Code-coverage instrumentation makes this codebase 10x–900x slower.** `XPlat Code Coverage` instruments
  the hot loops; on Linux CI that turned a fast suite into an unusable one. `coverlet.runsettings` excludes
  `Ops`, `Kernels`, `Maths`, `Intrinsics`, `Autograd`, `Optimizers`, `Tensors` and `LanguageModels.Runtime`
  — **always pass `--settings coverlet.runsettings`**, and remember a coverage figure quoted without it is a
  different number. Related: finite-difference gradient tests need an absolute-difference floor (5e-4) near
  zero, or they fail on values that are correctly tiny.
- **Correctness tests do not notice a performance cliff.** Theil–Sen shipped with **66 correctness tests and
  zero benchmarks** while hiding **8.1 ms per call, 99% of it inside `Sort()`**. For any algorithm that will
  be called in a loop, the order is: working version → correctness tests **plus a benchmark as the
  performance reference** → only then optimisation.

## Platform and load-time results

- **Peak RAM during load matters more than steady state on the low end.** Prefer *unpooled* buffers for
  weights and keep scratch `byte[]` out of read paths — the target includes machines where the transient
  peak, not the resident set, is what fails.
- **`ValueStopwatch` instead of `Stopwatch`.** `Stopwatch.StartNew` and the constructor allocate; the ban is
  surgical and the static `GetTimestamp`/`Frequency` remain allowed. Use `ValueStopwatch.StartNew` →
  `GetElapsedTime`.
- **ARM NEON `SDOT` on Android: correct and pointless — decode is dequant-bound, not dot-bound.** The port
  was verified correct and produced no throughput gain, because the time is spent unpacking quantised
  weights rather than in the dot product. ~3.8 tok/s for a 0.5B Q4_K model on a Motorola Edge 50 Fusion,
  pure managed .NET-for-Android. **A negative result, recorded so the same port is not attempted again** —
  and a reminder that the bottleneck's identity is a measurement, not an assumption.
