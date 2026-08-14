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

## Android exposes NO ARM hardware intrinsics — every `Arm.*` kernel is dead code there — 2026-08-14

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

**Check the probe, not the build flag.** This was found only because the app logs what the *runtime*
reports at startup. A build flag says what was requested; `IsSupported` says what will execute.

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
| **ONNX Runtime** (`Microsoft.ML.OnnxRuntime`, `Sources/Benchmark` only — it is in **no** Overfit code path) | **1.28.0** | 1.29.0 shipped 2026-08-12 and the pin is held deliberately, because moving it moves every published Overfit-vs-ORT ratio without a line of Overfit changing. **Measured, six clean A/B process pairs (`PB-ORT1`, 2026-08-12): the two versions do not separate.** Every steady-state ORT arm was faster on 1.29.0 (0.9848–0.9974) and it means nothing — the canaries moved the same way and `Overfit_Batch64` moved further, at 0.9825. **Resolving power stated before the verdict: median cross-process canary spread 4.26%, against an effect of 0.3–1.5%.** So the experiment is *silent* in that band, not negative, and the flattery any published ratio carries is bounded at **≤1.5% — under the noise floor.** No ratio needs restating; the version needs naming. |
| **llama.cpp** (the `~1.13× uniform` row below) | **not recorded — this is a gap** | The build, commit and quantisation of the llama.cpp side are not written down anywhere in this file, which is the same defect the ORT row above fixes. Anyone re-running that comparison should record them; anyone citing `1.13×` should say they do not know which build it was against. |

**The cross-process floor is intrinsic here, not background load.** Canary spread was median 3.17% on a
loaded box and median 4.26% after a reboot to a single process — slightly *worse* clean. An effect under a
few percent therefore needs a different experiment shape, not a quieter machine.

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
| `OverfitParallelFor` vs `Parallel.For`, **decode** | **455 µs / 0 B** vs **2059 µs / 925 KB** |
| `OverfitParallelFor` vs `Parallel.For`, **Conv2D** | the opposite — see the regression table |

## Throughput and memory

| subject | measured |
|---|---|
| Bielik decode after the CPU sprint | 12.55 → **17 tok/s** (bit-identical output) |
| Qwen-3B with `OVERFIT_REPACK_GEMV` | **24.4 tok/s** (+30%) |
| gap to llama.cpp | **~1.13× uniform** — this is *not* parity; always best-of-N on both sides. **Which llama.cpp build this was measured against is not recorded** — see the baseline table at the top of this file |
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
