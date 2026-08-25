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

## `XC-92`: VGG-16's 14x14 convolutions were running on seven cores of sixteen — 2026-08-25

**Provenance.** VGG-16 (`C:\onnxmodels\vgg16.onnx`) through `Scripts/ProfHarness` at `PROF_NODES=1`,
driven by `Scripts/xc92_conv_scaling.py`. 9950X3D, Windows, .NET 10, Release. Three sittings, arm order
rotated, `machine.quiet_guard` on every sitting, canary (`OverfitParallel` on balanced register-only
work) **1.211-1.228 ms across all six readings, spread 1.1%**.

**The diagnosis, and it is the part that decided the fix.** Those layers produce `N = 196` output
positions and an AVX-512 panel is 32 columns, so `GemmFusedIm2Col` dispatches `ceil(196/32) = 7` work
items — at most seven workers can ever be busy. Measured per layer: **3.93 ms at one core, 0.87 at seven
cores, 0.83 at sixteen.** Seven to sixteen buys nothing, while a many-panel control layer in the same run
keeps improving (node 1, 1568 panels: 3.85 -> 2.40 ms). **Nine of sixteen cores were idle, not
inefficient**, and that distinguishes a decomposition fix from a memory-supply one.

**The fix: the fused path splits M two ways** when a layer has fewer than half the workers' worth of
panels (`Conv2DGemmKernels.FusedMBlocksFor`, and `OVERFIT_CONV_FUSED_M_SPLIT` now defaults ON). Shipped
binary, `=0` against the default, nodes 14/15/16 median ms:

| pool | before | after | conv total | conv scaling 1 -> 16 |
|---|---|---|---|---|
| 16 workers, one per physical core | 0.84 / 0.83 / 0.82 | **0.63 / 0.65 / 0.64** | 17.46 -> 16.58 ms | 6.88x -> **7.25x** |
| 32 logical (the shipping default) | 0.85 / 0.89 / 0.90 | **0.58 / 0.67 / 0.66** | 17.08 -> 16.52 ms | — |
| 1 core (control — a single worker never splits) | 3.96 / 3.94 / 3.94 | 3.96 / 3.96 / 3.95 | 120.36 -> 120.13 ms | — |

Per-layer scaling on those three layers goes **4.71/4.77/4.82x to 6.29/6.09/6.17x**. Against ONNX
Runtime measured in the same sitting, whose equivalent nodes scale **9.50/8.84/9.34x**, that closes
about a third of the per-layer gap. Whole-convolution in the same run: ours 120.16 -> 16.79 = **7.16x**
against ORT's 95.46 -> 7.58 = **12.59x**.

**Two negative results from the same session, both worth not re-discovering.**

| tried | outcome |
|---|---|
| `OVERFIT_CONV_EXPAND_PANELS=1` — the MLAS-shaped expand-then-GEMM path already in the tree, which gathers each panel once into a shared 4.1 MB buffer and then splits M 64 ways over it | **17-25% SLOWER on exactly those layers** (0.82 -> 0.97/1.02/1.01 ms), three sittings, every other layer unmoved. It removes the duplicated gather and pays more for it elsewhere; the buffer leaves L2 |
| The obvious `floor(workers / nPanels)` rule, which gives 4 blocks at the 32-logical pool | **neutral** (0.85/0.88/0.91 -> 0.82/0.89/0.88). Cost is monotone in the block count: cap 4 neutral, cap 3 0.71/0.82/0.77, cap 2 0.59/0.67/0.65 — the duplicated gather is the term that matters, and the extra sixteen workers are SMT siblings that add no gather throughput |
| `ceil(workers / nPanels)` = 3 blocks at 16 workers, which is what the unfused `ResolveMBlocks` asks for | **no better than no split at all** (0.81/0.81/0.80): 21 items over 16 workers is two rounds |

**What this does NOT establish.** One model and one layer shape (`m = 512`, `k = 4608`, 7 panels). The
constant 2 is measured at 7 panels on a 16-core part, not derived. Whole-convolution totals on this
harness occasionally jump ~15% with the target layers unchanged — node 0 reads 1.05 ms in most runs and
1.63-1.71 in others — so read the per-layer rows, not the totals. **And the first attempt at the shipped
A/B measured nothing**: `Scripts/ProfHarness/bin` carries its own copy of `DevOnBike.Overfit.dll`, it was
two edits stale, and both arms ran the old default. The tell was arms identical to three decimals with a
1-core control that moved 1.6%. Assert the harness's copy hashes equal to the one you just built.

## `XC-92`: VGG-16's 28x28 convolutions are NOT decomposition-starved, and splitting M is a loss at every block count — 2026-08-25

**Provenance.** The instrument of the section above: VGG-16 (`C:\onnxmodels\vgg16.onnx`) through
`Scripts/ProfHarness` at `PROF_NODES=1`, driven from `Scripts/xc92_conv_scaling.py`'s runner by the
`PROF_AFFINITY` + `DOTNET_PROCESSOR_COUNT` route. 9950X3D, Windows, .NET 10, Release. Three sittings,
arm order rotated, canary **1.204-1.221 ms across six readings, spread 1.4%**. `machine.quiet_guard`
condemned sitting 1 (MsMpEng, 1.36% foreign load); sittings 2 and 3 are quiet and all three agree to
within 2% on every arm, so nothing here rests on the condemned one.

**The question.** Nodes 10/11/12 produce `ceil(784/32) = 25` panels. `OverfitParallel.For` slices that
into `ceil(25/16) = 2` items per chunk, so **three of sixteen chunks are empty** and occupancy is 78%.
The proposal was to split M until the items round-fit, as the 14x14 layers' two-way split did.

**Measured: every split is a loss, and the loss is monotone in the block count.**
`OVERFIT_CONV_FUSED_M_BLOCKS=n`, 16 workers pinned one per physical core, median ms across the three
sittings:

| blocks | node 10 | node 11 | node 12 | the three together | against the shipping rule |
|---|---|---|---|---|---|
| **1 (ships)** | **0.970** | **1.690** | **1.610** | **4.27 ms** | — |
| 2 | 1.010 | 1.770 | 1.690 | 4.47 ms | **+4.7%** |
| 3 | 1.080 | 1.860 | 1.760 | 4.70 ms | **+10.1%** |
| 4 | 1.130 | 1.980 | 1.890 | 5.00 ms | **+17.1%** |

There is no better value on the other side of the gate, so it is not a threshold to tune. **The lever is
kept** (`OverfitEnvironment.ConvFusedMBlocks`) for the same reason as `ConvExpandPanels`: the loss is the
useful part, and without it this question returns.

**The reason, and it is the transferable half.** The decomposition is not what binds. A worker sweep at
1/4/8/12/13/16 physical cores, one sitting, quiet, canary 1.215 -> 1.219 ms — measured speedup against
what the slicing arithmetic alone permits, `items / ceil(items / min(w, items))`:

| node | shape | items | w=4 | w=8 | w=12 | w=13 | w=16 | model at w=16 |
|---|---|---:|---|---|---|---|---|---|
| 1 | 224x224 | 1568 | 3.52x | 5.88x | 6.11x | 6.57x | **7.38x** | 16.00x |
| 4 | 112x112 | 392 | 3.70x | 6.70x | 7.09x | 8.34x | **8.78x** | 15.68x |
| 8 | 56x56 | 98 | 3.72x | 6.56x | 7.88x | 8.53x | **8.53x** | 14.00x |
| 10 | 28x28 | 25 | 3.29x | 5.15x | 5.49x | 7.01x | **6.79x** | 12.50x |
| 11 | 28x28 | 25 | 3.38x | 5.41x | 6.26x | 7.92x | **7.78x** | 12.50x |
| 12 | 28x28 | 25 | 3.35x | 5.33x | 6.44x | 8.32x | **8.12x** | 12.50x |
| 16 | 14x14 | 7 | 3.05x | 4.91x | 4.91x | 4.91x | **6.44x** | 7.00x |

**Two things follow, and they point in opposite directions.**

**The slicing model is real.** It predicts a step of exactly 1.5x between 12 and 13 workers for 25 items
(`ceil(25/12) = 3` -> `ceil(25/13) = 2`) and equal times from 13 to 16. Measured: **+21.7 / +21.0 /
+22.7%** at 12 -> 13, and **-1.8 to -3.2%** at 13 -> 16 — the three layers are *slower* on sixteen cores
than on thirteen. The step is at the predicted core count and nowhere else.

**And it is not what binds at sixteen cores.** The realised fraction of the model **falls as the model's
ceiling rises**: 65% at 25 items, 61% at 98, 56% at 392, 46% at 1568. **No convolution node in this model
exceeds 8.78x at 16 cores**, whatever its item count. Nodes 11 and 12 already reach 7.78x and 8.12x, so
they sit inside that envelope — they are not the outlier the 14x14 layers were, which measured 4.8x
against a decomposition ceiling of 7.0x. **The whole prize on these three layers, if they were lifted to
the best-scaling node in the model, is 4.27 -> 3.74 ms: 3.2% of a 16.44 ms convolution.**

**The duplicated gather, measured instead of modelled.** Under the override a single worker still runs
`nPanels * mBlocks` items, so the 1-core arm prices the duplication directly: three blocks add
**+14.3 / +15.0 / +14.6%** of single-core work on nodes 10/11/12 and **+19.5%** on the 14x14 layers. One
extra gather is therefore about **7%** of a panel here — far cheaper than the `m / 145` cost model in
`Conv2DGemmKernels.MaxFusedMBlocks` predicts (22%) — **and the split still loses**. Do not reach for that
cost model to predict a decomposition trade; measure the 1-core arm.

**What this does NOT establish.** One model, one box, one panel width. The envelope of ~8.8x is a
property of this machine and these layer shapes; nothing here identifies its cause, and per-core stall
attribution at 1 core against 16 is still the open question the row points at uProf for. The 12 -> 13
step was measured in **one** sitting, not three — it is the shape of the curve that is evidence, not its
last digit. And `PROF_NODES=1` costs a fixed amount per call, which depresses the 16-core arm more than
the 1-core arm, so every scaling figure here is conservative.

## What FP16 costs in accuracy, and it corrected a signed plan by 30 % — 2026-08-22

Measured on the host by `GpuProbe --fp16-bound`, so **no GPU is involved and none is needed**. Relative
L2 against an F32 reference, at the real Qwen-3B shapes. Three columns because the middle one is the
arm the probe actually builds, and it was the column the plan omitted.

| cell | FP32 accumulate, F32 out | **FP32 accumulate, FP16 out** | FP16 accumulate |
|---|---|---|---|
| `ffn_gate_up` k 2048 -> m 11008 | 2.88e-4 | **3.56e-4** | 6.53e-3 |
| `ffn_down` k 11008 -> m 2048 | 2.87e-4 | **3.52e-4** | 1.58e-2 |
| `attn_qo` k 2048 -> m 2048 | 2.88e-4 | **3.56e-4** | 6.58e-3 |
| `lm_head` k 2048 -> m 151936 | 2.88e-4 | **3.55e-4** | 6.70e-3 |
| `attn_kv` k 2048 -> m 256 | 2.97e-4 | **3.64e-4** | 6.91e-3 |

**The correction.** `XC-107` `AMENDMENT 2` section `A2.3` predicted a **3.4x** margin under the
`ParityResult.Fp16Fp32AccumulateCeiling` of 1e-3, taken from the left-hand column. But arm `X3` writes
into an **FP16 output buffer**, which pays one more rounding than that bound accounts for. The real
margin is **2.7x**. The plan's figure was not wrong arithmetic; it was the wrong column.

**What the ceiling still does, and it is the reason the arm is safe to hand to a stranger.** The
likeliest implementation mistake in a `cublasGemmEx` call is passing compute type **64**
(`CUBLAS_COMPUTE_16F`) where **68** (`CUBLAS_COMPUTE_32F`) was meant — adjacent values in the same enum,
verified against NVIDIA's `cublas_api.h:205` and `:207`. That mistake lands in the right-hand column,
**6.5x to 15.8x over the ceiling**, so it fails parity instead of printing a fast wrong number.

**The k-dependence in the right-hand column is a confirmed mechanism, not a fitted curve.** `hgemm`
rounds its running sum once per multiply-add, so error grows as `sqrt(k)`. Fitting `C*sqrt(k)` to the
k=11008 point ALONE gives `C=1.51e-4`, which then **predicts** 6.82e-3 at k=2048 against 6.53e-3,
6.58e-3 and 6.91e-3 measured. The left two columns are flat in k, as FP32 accumulation requires.

**NOT CHECKED, and it bounds every figure above.** These are host computations. **No line of any FP16
device path has ever executed** — this machine has no NVIDIA card. The FP16-accumulate column is a
subsample estimator over **4096 sampled output elements per shape**, not the full tensor, and the report
line says so.

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

### Verified headline after the 2026-08-19 session, and a regression check across the published table

Two repeats per large model with ONNX Runtime in the same process, parity asserted on every run
(cosine 1.000000, same argmax):

| model | Overfit | ONNX Runtime | ratio | at the start of this line of work |
|---|---:|---:|---:|---|
| VGG-16 | 27.34 / 27.39 ms | 18.82 / 18.88 ms | **1.45x** | 79.01 ms, 3.72x |
| CNN, 60.9 MB | 18.47 / 18.53 ms | 9.72 / 9.90 ms | **1.89x** | 67.70 ms, 5.24x |

**2.9x and 3.6x faster in absolute terms**, with the gap closing from 3.72x and 5.24x **while ONNX Runtime
itself got faster** over the same period (12.93 to 9.8 ms on the CNN).

**No regression on the small models**, which is what the check was for: the MNIST CNN runs **5.324 us**
against 5.36 previously, and the zero-allocation MLP path **6.949 us** against 6.71 — the second is +3.6%,
inside this box's run-to-run band and on a different arm label, so it is not read as a change.

### `XC-96` closed and the branch removed from the tree (2026-08-19)

Three attempts, four defects found and fixed, and it still executed overlapping chunks. **The code is gone.**
Dead code that ends the process when a switch is flipped is a liability, and nothing in the findings needs it
to survive: the four defects, the exoneration of the claim protocol and the two remaining untested causes are
all written up below and in `XC-96`.

Removed: `StealChunks` and its environment name, `_regionClaims`, `_nextRegion`, `_regionCount`,
`_regionSubChunks`, `_stealingDispatch`, `_stealGeneration`, `DrainRegions`, the dispatch branch and the
worker-loop branch. Zero references left.

Verified after removal: `Scripts/DispatchStress` clean at **22,500 dispatches** both with and without the
now-meaningless environment variable set, and the suite **2748/0 twice**.

### Load-time memory fully accounted on BOTH models: from "470 MB unknown" to ~55 MB (2026-08-21)

Supersedes the partial split below it. Every resident-weight path in `GgufLlamaLoader` was stamped and both
models measured, so this is a breakdown rather than a calculation.

| component | Qwen2.5-3B (Q4_K) | Qwen2.5-0.5B (Q5_0) |
|---|---|---|
| .NET runtime baseline at entry | 23.3 | 26.6 |
| harness probe tokenizer — **an artefact, since REMOVED** | 16.9 | 13.5 |
| chat-template `GgufReader` | 12.2 | 10.0 |
| tokenizer (sibling files) | 37.0 | 38.0 |
| FFN / embedding / LM head | **0 — all mapped** | **515.5** |
| **`attn_output` via the F32 fallback** | **153.0** | 19.5 |
| session KV cache at 2048 | ~151 | 50.5 |
| **attributed** | **393.4** | **673.6** |
| measured, with the probe | 452.8 | 727.2 |
| **measured, probe removed** | **435.9** | **713.7** |
| **unattributed** | **~59** | **~54** |

**The remainder is the same size on both models, and that is itself the evidence.** A term that does not
scale with the model is not weights; it looks like the object overhead of several hundred per-head weight
instances plus the tensor index, and it is small enough to stop here.

**Three findings, in order of size.**

**(1) `attn_output` costs 153 MB on the 3B and is filed as `XC-101`.** On a model whose every other weight is
mapped at zero cost, this one tensor is the largest managed term in the load. It is structural: `Wo` is split
per head, a head is `headDim` wide, and `headDim` is smaller than the 256-element K-quant super-block — so
no per-head K-quant representation exists and the F32 fallback is the only path.

**(2) The 0.5B is expensive for reasons the 3B is not.** 515.5 MB of weights against zero, because its
quantisations are the two this loader cannot hold verbatim: Q5_0 (no native type; +144.1 MB over its on-disk
size) and a Q8_0 LM head (interleaved on disk, split in memory; ~145 MB — `XC-100`). A model four times
larger loads in 452.8 MB against 727.2.

> **(3) The instrument was inflating every reading, and the first correction of it was ALSO wrong.**
> `Scripts/GenHarness` carried a `GgufTokenizer.Load` probe added the previous day to test a hypothesis that
> was refuted the same hour; it was never removed and held memory for the life of the process. Every
> load-time figure quoted from that harness — including the 727.2 MB above and the 759.2 that preceded it —
> was high by it.
>
> **The size was first written down here as "40 MB", from the reading at entry, and that was wrong.**
> Removing the probe and re-measuring gives **13.5 MB on the 0.5B and 16.9 MB on the 3B**; the rest of that
> 40 was the .NET runtime's own baseline, which the entry stamp had swept in with it. Corrected figures:
> **713.7 MB and 435.9 MB**. The attribution table above is otherwise unchanged, and the ~55 MB remainder
> survives on both models.
>
> Two lessons, and the second is the sharper one. **A probe that allocates is part of the subject from the
> moment it exists** — the tell was never in the totals, only in a split whose parts had to add up. And
> **a correction is a measurement too**: "40 MB" was one reading, attributed by eye, published as a fact,
> and wrong by 2.5x within the hour. This is the fifth instrument defect in two days and the first one whose
> fix needed its own fix.

### Where the 727 MB actually goes on Qwen2.5-0.5B: 515 attributed, 212 still open (2026-08-21)

`XC-97` took the load from 1,132 MB of managed heap to 727 MB and left "~470 MB unattributed". That figure
was itself too large, and the correction is worth stating: **it never counted what the layer weights legally
cost.** The loader was made to stamp every resident weight it builds, so the split below is measured, not
calculated — the same discipline `XC-60` failed, where a whole row hung on one constant taken from a document
instead of from a running process.

| component | managed | how |
|---|---|---|
| `token_embd.weight` | **146.1 MB** | Q5_0 -> Q8, streamed from the mapping |
| `blk.N.ffn_gate.weight` (24) | **112.2 MB** | Q5_0 -> Q8 |
| `blk.N.ffn_up.weight` (24) | **112.2 MB** | Q5_0 -> Q8 |
| LM head `output.weight` | **~145 MB** | Q8_0, **copied** — `LoadQ8Native` has no mmap path (`XC-100`) |
| `blk.N.ffn_down.weight` (24) | **0** | 12 Q6_K + 12 Q4_K, both sliced from the mapping |
| attention projections | **0** | per-head loaders, sliced from the mapping |
| **attributed** | **~515 MB** | |
| **still open** | **~212 MB** | tokenizer via the sibling-file path is part of it; the rest is unknown |

**The cost of having no native Q5_0 weight type is now a number rather than an estimate: +144.1 MB.** Those
49 tensors occupy 226.4 MB on disk as Q5_0 and 370.5 MB resident as Q8 — Q5_0 is 0.6875 bytes per parameter
against Q8's 1.0625, so the conversion is 1.55x, and that is the price of reusing a first-class weight type
instead of adding a fifth one.

> **The control explains itself and is the useful comparison.** Qwen2.5-3B is four times the model and loads
> in **484 MB** against the 0.5B's 727. Every one of its weights is Q4_K or Q6_K, whose on-disk layout IS the
> resident layout, so they are slices of the mapping and cost nothing; its LM head is Q6_K and also free.
> **The 0.5B is expensive precisely because its quantisations are the two this loader cannot map** — Q5_0
> (no native type) and Q8_0 (interleaved on disk, split in memory). Neither is a defect in the model file;
> both are gaps in what the loader can hold verbatim.

**Session KV cache measured separately**: `CreateSession(2048)` adds **50.5 MB** on this model, matching the
arithmetic (24 layers x 2 x 2 KV heads x 64 head_dim x 2048 x 4 B).

### `XC-97` fixed: a Q5_0 embedding no longer dequantizes to F32 — 373 MB saved, prediction 374 (2026-08-20)

`GgufLlamaLoader.LoadEmbedding` kept the embedding verbatim and mmap-backed only for Q4_K and Q6_K. Q5_0 fell
through to *"F32 fallback — full dequant into a flat [vocab x dModel] row-major buffer"*, which on
Qwen2.5-0.5B measured **519 MB for a 469 MB model file**.

| | before | after | |
|---|---|---|---|
| Qwen2.5-0.5B, managed heap after load | 1,132 MB | **759.2 MB** | **-373 MB (-33%)** |
| Qwen2.5-3B (Q4_K control) | 484.1 MB | 484.1 MB | unmoved |
| suite | 2,760 / 0 | 2,764 / 0 | +4 tests |

**The arithmetic and the measurement agree to a megabyte.** The F32 embedding was 519 MB; in Q8 the same
136M elements cost about 145 MB, predicting a 374 MB saving. Measured 373. The `FALLBACK embedding [Q5_0]`
line the loader printed under instrumentation is simply gone, and the model still generates sensibly.

**Q8 rather than a native Q5_0 weight type**: a fifth `DecodeWeight` case would touch every switch in the
runtime, while `Q8Weight` is already first-class with kernels — the loader already converts to it for a tied
LM head. **The accuracy argument is the point, not a concession**: a Q5_0 block carries 32 distinct levels
and a Q8 block carries 256, so the target has eight times the resolution the source actually uses, and the
only error is a second rounding of an already-quantized value.

**Peak, not steady state.** The conversion streams row by row out of the mapping with a one-row scratch
rented once per worker band. Converting through the full F32 table would have halved the steady state and
left the load-time spike exactly where it was — and minimising peak RAM at load is the constraint this
repository states for itself, so the cheaper implementation would have missed the point of the fix.

> **The oracle is an equality, not a tolerance, and that was a deliberate choice.** The streaming route and
> the full-F32-table route perform the same decode and the same quantization in the same order; only the
> buffering differs. So they must agree **bit for bit**, and a tolerance would have hidden precisely the
> mistake a streaming rewrite can make and a batch one cannot. Mutation-proven by corrupting the row stride:
> `quant[896] differs: expected -87, got -127 (row 1, column 0)`. A second test checks the accuracy claim
> against the Q5_0 source directly, because an equality against one's own reference cannot catch both routes
> being wrong the same way.

**Not finished: 759 MB is still 1.6x the model file.** The embedding was the largest single term, not the
only one. About 470 MB remains unattributed, and the instrumentation only covered the embedding fallback and
`AllocAndLoad` — other Q5_0 tensors may take a path nothing has looked at.

### `XC-60` measured: the saving is 3.3 MB, not 67.5 MiB — one wrong constant, twenty-fold (2026-08-20)

`XC-60` records that `CachedSingleHeadAttention._scoreScratch` is sized to the MODEL's context rather than the
`maxContextLength` the caller asked for, and derives **86.4 MiB per stack** with **67.5 MiB saved per client**.
Its own text says to confirm with a real measurement before quoting that anywhere. Measured, with managed heap
read either side of the `CachedGptStack` constructor in the same process and the same run:

| | the row's arithmetic | measured |
|---|---|---|
| `config.ContextLength` | 32,768 | **8,192** |
| whole stack | 86.4 MiB | **22.5 MB** |
| saving from capping the scratch to 2048 | 67.5 MiB | **3.3 MB** |

**The mechanism is real and the magnitude is not.** The scratch genuinely ignores the caller's
`maxContextLength` — a session asking for 2048 still pays for 8192 — but closing that is worth 3.3 MB against
a change that crosses the engine constructor, so the row is closed as **not worth doing**, not as wrong.

> **The whole error is one constant.** The row assumed 32,768 because that is what the GGUF metadata
> advertises and what `overfit doctor` prints; the engine's config carries **8,192**, capped somewhere between
> the file and the config. Every figure in the row — 128 KiB per head, 72.0 MiB of scratch, 86.4 MiB per
> stack, the 4.6-session break-even in `XC-58` — was a multiple of that one value, and none of them was
> checked against a running engine. **An arithmetic chain is only as sound as its least-verified input, and
> the input that came from a document rather than from a process is the one to check first.**

### Where our managed memory goes at load: weights are NOT copied, and `XC-60` is 20% of it (2026-08-20)

Chasing the unexplained gigabyte from the server comparison. **First, a correction that must not be blurred:
the server's 1,059 MB and the 438 MB below are NOT the same number.** The server carries ASP.NET and its own
sessions on top; this section is about `OverfitClient.LoadGguf` alone, measured with checkpoints inside a
harness. Do not add or compare them.

Managed heap after `LoadGguf` on Qwen2.5-3B Q4_K_M: **438.6 MB** (452.9 MB when the model directory has
sibling tokenizer files). Private commit at that point is 462 MB, of which only 24 MB is unmanaged — so this
is the managed heap, not native allocation.

| component | size | how it was established |
|---|---|---|
| session KV cache at 2048 tokens | **~146 MB** | measured as the delta across `CreateSession(2048)`; arithmetic agrees (2048 x 72 KiB) |
| tokenizer | **40 MB** | probed directly; the sibling-file path and the GGUF-embedded path differ by only 14 MB |
| attention scratch (`XC-60`) | **~86 MB** | that row's arithmetic, not separately measured here |
| F32 tensors | **0.6 MB** | the loader was made to name every `AllocAndLoad` — all of it is layer norms |
| copies of quantized tensors | **0 MB** | the loader was made to name every `new byte[...]` and printed **nothing** |
| **unattributed** | **~165 MB** | — |

**The main risk is closed by measurement rather than by reasoning: we do not copy the weights.** Every
quantized tensor takes an mmap slice — `LoadQ6KNative` and its siblings return a `Q6KWeight` over the mapping
and never reach their `new byte[]` fallback. Doing to ourselves what dotLLM's 1.92 GB "repacked" buffer does
was the thing worth ruling out, and it is ruled out. `OVERFIT_REPACK_GEMV` is opt-in and off.

**`XC-60` is worth more than the earlier estimate said.** Against the 1,059 MB server figure its 86.4 MiB
looked like 8% and a distraction; against the 438 MB that load actually costs it is **about 20%** — and it is
scratch the caller's own `maxContextLength` was supposed to bound and does not.

**Four candidates were killed by measurement, each of which had a persuasive arithmetic behind it**, and the
sequence is the point: a plausible size calculation is not evidence.
1. *Embedding dequantized to F32* — 151,936 x 2,048 x 4 B = 1,244 MB against a measured 1,059. Killed by
   reading: `LoadEmbedding` keeps K-quants verbatim and mmap-backed.
2. *Vocabulary-driven* — killed by the model sweep: Qwen2.5-**0.5B** has the same vocabulary as the 3B and a
   469 MB file, yet takes **1,119 MB**, the most of the three. (Its `dModel` is 896, and `896 % 256 != 0`
   fails the native-path guard, so it really does dequantize to F32 — a separate defect, and a large one.)
3. *A repacked or Q8 copy of the LM head* — killed by the byte-array probe printing nothing.
4. *The sibling-tokenizer path* — killed by running the same GGUF from a directory with no siblings: 452.9 vs
   438.6 MB, a 14 MB difference.

**Left open: ~165 MB.** Not guessed at. Candidates never measured are `ArrayPool` retention, `GgufReader`
metadata and the object overhead of 576 per-head weight objects.

**Also found, and filed as `XC-97`: Qwen2.5-0.5B takes 1,132 MB of managed heap for a 469 MB model** — 2.4x
its own file. **The cause was measured, not inferred, and the first inference was WRONG.** It looked like the
`dModel % 256 != 0` guard (896 is not a multiple of 256, and every K-quant path in the loader tests that).
Instrumenting the loader to name its fallbacks printed `FALLBACK embedding [Q5_0] 519.3 MB`: the type is
**Q5_0**, and `LoadEmbedding` handles only Q4_K and Q6_K, so it dequantizes to F32 whatever the hidden size
is. The 3B control on the same probe shows 0.6 MB of F32, all layer norms. Same class of defect as the one
this section was opened to look for, on a model this repository ships tests against.

### Peak RAM of the two SERVERS: 6.3x less committed memory, and ~1 GB of ours is unexplained (2026-08-20)

The first RAM reading measured a harness I wrote, and it measured a configuration no product ships:
`OverfitClient.LoadGguf` already creates a session at its default 2048-token context, and the harness then
called `Engine.CreateSession()` with no argument, which resolves to the MODEL's full context. This one
measures the two **products** — `overfit serve` against `dotllm serve` — one at a time, each driven to a
fully-initialised state by one real `/v1/chat/completions` request before the peak is read.

| unit | Overfit | dotLLM | ratio |
|---|---|---|---|
| peak working set | **2,906 MB** | 3,924 MB | 1.35x |
| peak private commit | **1,059 MB** | **6,675 MB** | **6.3x** |

Identical to the megabyte across three repeats on both sides, so the allocation is deterministic. Model file
is 2,007 MB. **This is the one place the two engines are not close**, and it is the axis this repository's
stated identity is about.

**Their server commits 3.3x the model file and 3.4x what their own CLI committed** (1,989 MB for `dotllm run`
on the same model). What their server does that their CLI does not was not investigated.

> **Unexplained, and it is OUR number: what is our 1,059 MB?** The weights are mapped, so a mapping should
> not appear in private commit. Three things are ruled out rather than assumed:
> * **The KV cache is not the dominant term.** The harness reading (1,052 MB) and the server reading
>   (1,059 MB) agree to 7 MB despite creating sessions at wildly different context lengths. Whatever this is,
>   it does not move with context.
> * **Repacked GEMV weights** — `OVERFIT_REPACK_GEMV` is opt-in and off by default, so we do not hold the
>   second committed copy we criticise dotLLM for.
> * **The embedding table** — read rather than guessed: `GgufLlamaLoader.LoadEmbedding` keeps Q4_K/Q6_K
>   verbatim and mmap-backed, with zero managed bytes. Its own comment records this as already having cut
>   1.2 GB to 255 MB for exactly this model. The arithmetic was tempting — vocab 151,936 x 2,048 in F32 is
>   1,244 MB against a measured 1,059 — and it is wrong.
>
> **`XC-60` is real but it is not this.** Its 86.4 MiB per stack is about 8% of the figure, and the
> context-independence above is consistent with its claim that the scratch ignores the caller's
> `maxContextLength` — but fixing it would move the number by under a tenth. **Find the gigabyte first.**

### Peak RAM against dotLLM: 1.85x lower on both units — and it points a question back at us (2026-08-20)

The throughput comparison came out level, so memory is where the two designs actually differ. Measured on the
same file, Qwen2.5-3B Q4_K_M, 2,007 MB on disk, three repeats each with the order alternated.

**Two units, because they answer different questions and only one is the mechanism's.** *Peak working set* is
physical pages resident INCLUDING memory-mapped file pages, which are reclaimable and shared — for an mmap'd
model it counts something that is not really consumed. *Peak private commit* is what must be backed by RAM or
the pagefile, and it is what decides whether a model fits on a low-end box.

| unit | Overfit | dotLLM | ratio |
|---|---|---|---|
| peak working set | **2,084 MB** | 3,838 MB | **1.84x** |
| peak private commit | **1,052 MB** | 1,989 MB | **1.89x** |

Spread was +/-3 MB across three repeats. **This is the largest measured difference between the two engines
and the only one outside noise.** Our peak working set is 2,084 MB against a 2,007 MB file — the peak is
essentially the mapping and little else, which is the "map, do not copy" identity shown as a number rather
than asserted in a README.

**The prediction was half wrong, and that is the useful part.** dotLLM's own CLI attributes 1,834 MB to
"repacked" weights held alongside the mapping, so the expected gap in private commit was about that. The
measured gap is **+936 MB**, roughly half. Either that repacked buffer is not private commit in the way their
report implies, or **we commit something large that nobody here has named** — the measurement cannot tell
which.

> **The question it turns back on us: why is our private commit 1,052 MB at all, when the weights are
> mapped?** A mapping should not appear there. What is left is the KV cache and the attention scratch — and
> `XC-60` already records that `CachedSingleHeadAttention._scoreScratch` is sized to the MODEL's context
> length rather than the `maxContextLength` the caller asked for. That row was derived arithmetically and
> **never measured**; this is the first number to check it against.

### Overfit against dotLLM: level on throughput — and an hour lost to an instrument I wrote myself (2026-08-20)

dotLLM (kkokosa) is the closest thing to a direct competitor: a pure-C# LLM inference engine, GGUF, Q4_K_M,
SIMD, .NET 10, no llama.cpp underneath. Same file on both sides — `C:\qwen3b\qwen.q4km.gguf`, Qwen2.5-3B
Q4_K_M — measured against their SOURCE at PR #414+, not the `0.1.0-preview.3` on nuget.org, because comparing
a competitor against a stale package is the dishonesty this repository criticises elsewhere.

**The result, from one instrument driving both engines through their own OpenAI-compatible servers**, one
server at a time, `/v1/models` read back and asserted before each measurement, machine quiet at 2.00%:

| | Overfit | dotLLM |
|---|---|---|
| throughput | **26.3 tok/s** | 25.4 tok/s |
| ITL p50 | 38.08 ms | 38.67 ms |
| ITL p95 | **39.24 ms** | 41.32 ms |
| TTFT p50 | 0.8 ms | 41.0 ms |
| errors | 0/16 | 0/16 |

**Level.** 3.5% on throughput and 1.5% on median inter-token latency is not a difference worth a headline; we
are marginally steadier in the tail. **The TTFT column is NOT a 50x win and must not be quoted as one**: 0.8
ms cannot contain a prefill, so ours is certainly serving a cached prefix, and the benchmark sends the SAME
prompt sixteen times, which measures the cache rather than prefill. Both engines enable prompt caching by
default, so "we cache and they do not" is ruled out; why theirs costs 41 ms was not established.

> **The method failure is the part worth keeping.** The first version of this comparison used dotLLM's own
> mature CLI on one side and a harness I had written that morning on the other. **Two defects were found in
> my harness during the run** — invented environment-variable names that set nothing, and reporting one
> thread-pool size while setting two — and every suspicious number came through it.
>
> From it came a claim that **our decode degrades 28-41% with context while dotLLM's stays flat**. That claim
> is **WITHDRAWN**. Measured properly — three repeats, order shuffled, our engine alone — the context penalty
> from 5 to 673 tokens is **2.4-3.4 ms/token, about 7%**, and it is the same at 10, 16 and 32 general workers.
> Two hypotheses were built on the withdrawn number and both were refuted: KV-band duplication across
> head-parallel workers (no jump at the branch's own 2-vs-3-worker threshold once repeated), and an
> oversubscribed general pool (pool size makes no difference).
>
> **Three rules, each earned here:**
> 1. **Do not look for the cause of a number measured once.** Establish the effect with repeats first. An
>    hour went into explaining an effect whose existence was never checked.
> 2. **A smooth curve from single readings is more dangerous than a noisy one.** The original sweep ran
>    contexts in ASCENDING order, so drift over time read as dependence on context, and the smoothness
>    suppressed suspicion instead of raising it.
> 3. **Before building an instrument for a comparison, enumerate what both sides already expose and take the
>    intersection.** Both projects ship an OpenAI-compatible `serve`, and this repository already ships
>    `overfit bench` to measure exactly that. Both were visible in greps run BEFORE the harness was written;
>    the question asked was "how do I replicate their `run` command" rather than "what do we already share".

**Cross-validation, after the fact rather than by design:** `overfit bench` puts our decode at 38.08 ms/token
and the hand-written harness put it at ~40, so our own figure was not a harness artefact. That agreement was
luck, not method — it was not set up as a control.

**Not comparable and not measured:** memory. dotLLM's CLI reports 4.05 GB for this model, of which 1.92 GB is
a second committed copy of the weights for R4 interleaving, alongside the 2.1 GB mapping. Nothing on our side
reports the equivalent, so the row this repository would most want — peak RAM — is absent.

**Outputs diverge.** Greedy on both, same file, prompt tokenised identically to 5 tokens, and they agree for
eight characters before parting. Both continuations are ordinary English and neither is obviously wrong, but
they cannot both be the argmax path. Settling it needs a logit comparison on identical input, which was not
done. It does not affect the throughput comparison: the work per token is the same shapes over the same cache
length whichever token wins.

### What the vectorised GELU is worth per TOKEN: 6.5-7.1% on GPT-2 small, and WHY the microbenchmark under-predicted it (2026-08-19, explained 2026-08-21)

The activation was measured at 9.4-12.5x. That says nothing about a token, so it was measured end to end:
GPT-2 small, 64 generated tokens, both arms **in one process** through
`CachedFeedForwardBlock.VectorGelu`, so they share a JIT, a heap and a thermal state.

**The lever was proven live first.** Making `ApplyGeLU` throw reddens all three GPT-2 KV-cache tests. Without
that check a flat 1.00 would have read as "GELU does not matter" when it would have meant "the switch is not
connected".

| | sitting 1 | sitting 2 | movement |
|---|---|---|---|
| `Decode_ScalarGelu` | 835.3 ms | 833.8 ms | -0.18% |
| `Decode_VectorGelu` | 775.8 ms | 779.2 ms | +0.44% |
| `CanaryScalarMath` | 2.668 ms | 2.688 ms | +0.74% |
| `CanaryMemory` | 0.351 ms | 0.353 ms | +0.51% |
| **saving** | **929.6 us/token = 7.1%** | **852.7 us/token = 6.5%** | |

**GPT-2 small is the upper bound**: plain GELU over the whole hidden layer, where a GeGLU model such as Gemma
applies it only to the gate branch.

**The canary was missing for the first two sittings and that cost a wrong reading.** Separated in time, the
same binary gave savings of 1200 and 514 us/token — a 2.3x swing, with the SCALAR arm moving 4.8% and the
vectorised one 0.7%. Back to back with canaries in place, everything is flat to under 1% and both sittings
agree. **The two canaries are deliberately a PAIR**: one is a fixed scalar-transcendental loop, aimed at the
specific suspect, and one is a fixed pass over memory. A canary made only of memory traffic could not have
distinguished a scalar-throughput excursion from anything else.

> **The microbenchmark under-predicted the end-to-end saving 4.3x. Measured 2026-08-21, and two thirds of
> it is now attributed rather than hypothesised.** Both hypotheses recorded here on 2026-08-19 were wrong.
>
> **First, a denominator correction, because the original number was built on it.** The element count was
> recorded as *"36,864 per token over exactly 12.00 calls, matching the arithmetic to the digit"*. It is
> **40,320 over 13.125 calls**. The decode also runs the **6 prompt positions**, so the work is 12 layers x
> **70** positions / 64 tokens, not 12 x 64 / 64. The earlier probe agreed with the arithmetic because both
> used the same wrong denominator — *agreement between a probe and a calculation is not a check when the
> calculation is the thing being assumed*. Per element the decode saves **22.1 ns**, not 24.1, and the
> factor is **4.3x**, not 4.7x.
>
> **The instrument: `ApplyGeLU` timed from INSIDE a real decode**, nine repeats, ABAB, medians, with a
> fixed scalar-transcendental canary per repeat. The in-GELU numbers hold to 5-9% across repeats and
> **reproduced across two separate processes to 0.2%**, which matters because in one of those processes the
> canary moved 21%: the box moved and the measured quantity did not follow it.
>
> **The ladder. Same loop, same array, same process; one condition changes per rung.**
>
> | condition | ns/element | vs the microbenchmark |
> |---|---|---|
> | `GeluActivationBenchmark`, quiet, its own array | 5.77 | 1.00x |
> | the same loop quiet inside the probe process | 6.25 / 6.29 | **1.08x** |
> | during a decode, private array copied into just before timing (L1-warm) | 8.92 | **1.55x** |
> | during a decode, the real activation buffer | 11.29 | **1.96x** |
>
> **The 1.08x rung is the instrument validating itself.** A probe that disagreed with BenchmarkDotNet on a
> quiet machine would have made every rung above it unreadable. Two quiet readings were taken, one before
> the repeats and one after, agreeing to 0.6%, so a warmup artefact cannot pass as a quiet-machine number.
>
> So the 1.96x splits into **1.43x environment** (the decode's own concurrency and clock, which a
> single-threaded microbenchmark does not reproduce) and **1.27x buffer** (the real activation buffer
> against a private array pulled into L1 immediately before the loop reads it).
>
> **(a) INPUT DISTRIBUTION — REFUTED.** The penalty is the SAME on both arms: scalar 1.94-1.96x, vector
> 2.27-2.34x. `TensorPrimitives.Sigmoid` is branchless and never calls `MathF.Tanh`, so no property of the
> argument can slow it. *An effect that is equal on a branchy transcendental and on a branchless SIMD path
> is not a property of the function.* The real distribution was captured anyway and is **not** the uniform
> [-4, 4] the benchmark feeds — median **-1.685**, p25 -2.130, p75 -1.131, max 11.491, only 1.66% of values
> past |x|>4. It is a genuinely different distribution and it turned out not to be the cause.
>
> **(b) CACHE RESIDENCY of the transcendental's constants — REFUTED as stated, and partly right by
> accident.** It predicted the tables start cold between GELU calls. But the reference loop runs
> immediately after the real call, with those tables at their warmest, and still reads 1.55x. What the
> measurement does support is a **buffer** effect of 1.27x, which is a different claim about a different
> object.
>
> **THE SPIN POOL — REFUTED AGAIN, this time in the loop's own unit.** It was ruled out on 2026-08-19 by
> comparing end-to-end savings (514 vs 547 us/token), which is the wrong instrument for a claim about one
> loop. Re-tested in two processes, because `OverfitParallel._decodePool` is `static readonly` and setting
> the variable inside a running process changes nothing. **The lever was proven to have moved before the
> result was read** — the probe writes `DecodePoolEnabled` and `DecodePoolSize` into every row, and the arms
> read 1/10 and 0/10. In-GELU cost: scalar 11.21 -> 11.29 ns (**+0.7%**), vector 1.40 -> 1.35 ns
> (**-3.2%**), both inside the repeat spread. The pool is not part of this.
>
> **STILL OPEN, and it is the smaller half.** The in-situ GELU saving is **395 us/token**. The wall saving
> is **629 us/token** in the probe harness and **853-930** in BenchmarkDotNet. So **37-57% of the wall
> saving is not inside `ApplyGeLU`**, and that part is not explained. The candidate consistent with the
> 1.43x environment rung is that removing 395 us/token of serial scalar work lowers package power and
> raises the clock for everything else — untested, and the probe harness cannot settle it: its wall spread
> is 13-15% against an effect of 1.7%.
>
> **The transferable rule, which is still worth more than the cause.** A kernel microbenchmark here
> under-reported a real end-to-end effect by 4.3x, and **1.96x of that was the kernel genuinely costing
> more in situ than in a benchmark of itself**. Extrapolating from one to the other is not conservative in
> either direction. Where a kernel number must be trusted in context, put the reference loop INSIDE the
> harness on a private array — the ladder above cost one afternoon and turned two hypotheses into four
> measurements.
>
> Method: `.claude/do-insitu.py`, `do-pool.py`, `do-ref.py`, `do-quiet.py` (scratch, not durable).

### GELU in the decode FFN was scalar and cost 6.0 ns per element: vectorised, 9.4-12.5x (2026-08-19)

`CachedFeedForwardBlock.ApplySiLU` carried the note that *"the scalar path's per-element `MathF.Exp` was the
bottleneck"* and had been vectorised through `TensorPrimitives.Sigmoid`. `ApplyGeLU`, forty lines below it,
still ran one `MathF.Tanh` per element and said so: *"Vectorization can be done later if this becomes a
bottleneck."* The twin path had been measured; this one never had.

**The rewrite is an algebraic identity, not a second approximation.** `tanh(z) = (1 - e^-2z)/(1 + e^-2z)`, so
`0.5 * (1 + tanh(z))` is exactly `sigmoid(2z)`, and the published form `0.5 * x * (1 + tanh(z))` is exactly
`x * sigmoid(2z)`. Nothing about the accuracy of the GELU approximation changes; only how it is evaluated.

**The prediction that the fast arm might LOSE is recorded with the result, because it was reasonable.** The
vectorised form makes three passes over the buffer where the scalar made one, and at these widths (12-56 KB)
the buffer sits in L1 or L2. It lost anyway, and by a wide margin — scalar `MathF.Tanh` costs **6.0 ns per
element**, about 28 cycles.

| width | scalar | vectorised | shipped method | ratio (shipped) |
|---|---|---|---|---|
| 3072 (GPT-2 small) | 17,719 ns | 1,836 ns | 1,890 ns | **9.4x** |
| 11008 (Qwen2.5-3B) | 74,515 ns | 6,721 ns | 7,787 ns | **9.6x** |
| 14336 (Gemma-2 9B) | 100,210-113,216 ns | 8,785 ns | 9,051 ns | **11.5-12.5x** |

**The 14336 baseline is quoted as a range on purpose.** It read 100,210 ns ±0.3% in the first sitting and
113,216 ns ±12.5% in the second. The canary was flat in both (463.85 ns ±0.13%), so the box did not move —
that one arm was disturbed. The vectorised arm read 8,751 and 8,785 across the same two sittings, stable to
0.4%, so **the whole uncertainty in that row is in the baseline**, and the honest ratio is a range.

**The identity earns its keep**: routing through `TensorPrimitives.Tanh` instead costs 1.12-1.21x more, so
the algebra is doing work rather than tidying.

**The shipped method is measured, not a copy of its shape.** `ShippedApplyGeLU` calls the real
`CachedFeedForwardBlock.ApplyGeLU` through `InternalsVisibleTo`. It runs 3-16% behind the standalone arm
because it includes a source copy and a `PooledBuffer` rental. This arm exists because a stale harness binary
once made an entire sweep describe the previous library.

**The parity test found a real regression in the change, before it shipped.** `TensorPrimitives` **rejects an
empty span** — `ArgumentException: Input span arguments must not be empty` — where the scalar loop simply did
not execute. The general lesson is worth more than the fix: **a vectorised rewrite inherits its primitive's
argument contract, and that contract is usually not the one the loop had.** Guarded, with the reason recorded
at the site.

**Mutation-proven**, green/red/green, by dropping the factor of two from the identity — the one way to get it
wrong that still looks plausible. Suite 2760/0 after the work. Benchmark:
`Sources/Benchmark/GeluActivationBenchmark.cs`; parity: `Tests/LanguageModels/Runtime/GeluVectorisationParityTests.cs`.

**NOT measured: what this is worth per token.** The activation was measured, not a decode step. For Gemma it
is the gate branch of GeGLU and for GPT-2 the whole hidden layer, but its share of a token is a separate
measurement and is not guessed here.

### `TG-T14` RESOLVED: one test wrote a process-wide setting and eight tests failed somewhere else (2026-08-19)

Eight `Anomalies` acceptance tests had been failing since at least 2026-08-13 with nobody knowing. The
leading hypothesis on the row was a shifted metric-channel registry in static state. **That was wrong**, and
the code says so directly: `LabWindowFixture.Load` maps channels **by name** through
`Enum.TryParse<MetricIndex>(parts[0])`, and `MetricWindow` carries only `readonly` fields. There is no
mutable static anywhere on that path.

**The cause.** `AffineTrendOnLabFixtureDiagnostics` set `OVERFIT_LAB_FIXTURE_NAME` and never restored it —
one `SetEnvironmentVariable`, no `try`, no `finally`. That variable selects the recording
`LabWindowFixture` hands to **every** test in the process. From that point on, five plain `[Fact]` tests in
three unrelated classes loaded the twelve-replica *healthy* window where they expected the four-replica one
with an injected throttle.

**Proven from the artefacts rather than argued.** `lab-window-healthy-12pod.csv` contains **zero**
`CpuThrottleRatio` rows against one in `lab-window.csv`, and the validator's own message reads *"not reported
by any of the 12 replicas"* — twelve, not four. One cause accounts for all three failure shapes: an empty
collection where a degraded replica was expected, an out-of-range index where a faulted pod was, and a
calibration refusing a window that failed its own gate.

**Three layers of silence stacked, which is why it lasted.** The culprit is a `[LongFact]`, so the ordinary
suite never ran it. The area gate that would have shown it hit its own timeout on 2026-08-13 (`exit 124`,
63.31 min) and no complete earlier run existed at all. And the victim count **varied** — 7, 8, 9 — with the
order xunit happened to pick, which reads as flakiness rather than as a leak.

**The measurement that settled it cost 18 seconds.** With `OVERFIT_RUN_LONG` unset the area runs 592 tests
with **zero** failures. The attributes were then checked test by test rather than assumed: **six of the eight
victims are plain `[Fact]`**, so they genuinely ran and genuinely passed. Had they been `[LongFact]`, that
zero would have meant *did not execute* and reading it as a pass would have repeated the exact defect this
subsystem exists to catch.

| arm | passed | failed | skipped |
|---|---|---|---|
| area gate, before | 606 | **7** | 10 |
| area gate, after (same command, 64.2 min) | 615 | **0** | 8 (all lab) |
| full fast suite, after | 2748 | 0 | 281 |

**Fix**: the diagnostic now resolves the path locally and calls `LabWindowFixture.Load(path)`, which already
existed. Nothing process-wide is written. The `OVERFIT_AFFINE_FIXTURE` knob is unchanged.

**Guard**: `Tests/Diagnostics/LabFixtureSelectionLeakTests.cs`, one plain `[Fact]`, fails if any test source
writes that variable. **Mutation-proven green/red/green** by re-introducing the exact line. It also asserts
its own walk scanned more than 100 files, because "no offenders" and "no files read" are the same green.

**Deliberately NOT written: the general rule.** *"Any test that writes a process-wide setting must restore
it"* is the real rule, but a restore in source text can be a `finally`, a `Dispose`, or a captured value
written back elsewhere. A first pass of that heuristic flagged `ModelFactTests`, which is correct and
restores in `Dispose`. This repository already carries two analyzers that failed in the two opposite
directions — one reporting code that did not exist, one silently reporting nothing — so the shipped check is
the narrow one that cannot produce a false positive.

### `XC-88`: the 29% ONNX Runtime drift does NOT reproduce, and the guard that judged the window was broken (2026-08-19)

`XC-88` was filed because ONNX Runtime read **12.93, then 9.79, then 9.19 ms** on the same untouched binary
across three sittings of the 60.9 MB CNN, and the task asked whether that tracked thread count, power state or
ORT's own arena warm-up. The discriminating shape was stated before measuring: *stable inside a process but
different between processes is per-process state; drifting inside one process is warm-up or the box; both
engines drifting together is the box.*

**Nine process launches across two independent instruments put ONNX Runtime between 9.69 and 10.04 ms.**

| instrument | arm | readings | spread |
|---|---|---|---|
| plain loop, 5 launches | ORT, its own thread choice | 9.83..9.93 ms | **1.0%** |
| plain loop, 5 launches | ORT, 16 threads forced | 9.69..10.04 ms | **3.6%** |
| BenchmarkDotNet, 4 launches | ORT | 9.72..9.96 ms | 2.5% |
| both instruments | **Overfit, the control** | 18.45..18.74 ms | **1.0-1.2%** |

**Two of the three filed candidates are ruled out as live mechanisms.** Timing each process in halves put the
first-to-second-half movement at **-3.2%..+1.5%**, so nothing is still warming after the 40 warm-up calls —
that closes the arena. Thread count is closed from the other direction, and with a result worth keeping:
**forcing 16 threads is measurably *worse* than letting ONNX Runtime choose**, 3.6% spread against 1.0%. The
published comparisons use ORT's own choice and should keep doing so. The pin has not moved either —
`Microsoft.ML.OnnxRuntime` has been 1.29.0 since 2026-04-06, so no package bump explains it.

> **What is NOT established: what actually happened in the original three sittings.** Power state was the
> third candidate and I did not measure it; it cannot be reconstructed now, because the sittings are gone and
> the effect does not reproduce on demand. **"It does not reproduce" is not "it was the box"** — it only
> means the cause is not a standing property of the engine, the harness or the pin. A reading taken today is
> reproducible to 1-2.5% across launches; the 12.93 remains unexplained and stays quoted with its sitting.

**The measurement also condemned itself, and that defect was real.** Both windows came back
`*** CONTAMINATED ***` while the share probe beside the verdict printed **2.31% against an 8% ceiling** — a
guard disagreeing with its own printed number. The cause: `DESKTOP_LOUD_SECONDS` condemned a window when one
desktop application spent **8.0 core-seconds**, an absolute, and every sample it was calibrated on ran 30-32
seconds. The denominator was never written down. `XC-88` measured for 296 s, where Parsec's ordinary
unchanged background reaches 21.84 core-s of 4745 — **0.46%, a third of the calibrated share** — and was
condemned for it.

The failure direction is the bad one: **the rule got stricter the longer the window**, so the runs most
expensive to repeat were the ones most likely to be thrown away. It is now `DESKTOP_LOUD_SHARE`, expressed as
a share of the window, and the calibration is preserved rather than re-guessed: `8.0 / (30.0 x 16)` is still
exactly 8.0 core-seconds at 30 seconds and becomes 80 at 300. Both `XC-88` windows clear at 0.51% and 0.46%.
The **scanner** probe stays absolute on purpose — Defender and Windows Update thrash L3 and the disk far
beyond what their CPU seconds suggest, so "any activity contaminates" is the intended rule there.

### `XC-96` third attempt: the packed CAS claim does NOT close it, which exonerates the claim protocol (2026-08-19)

The design named at the end of the second attempt was implemented: the per-region counters were replaced with
**packed claim words taken by compare-and-swap**, reusing `DecodeChunkClaim` rather than inventing a second
packing convention — the same `Publish` / `TryClaim` pair the decode pool already ships and already has a
concurrency test for. The generation now travels **inside** each word, so a claim from a stale generation
loses its CAS instead of being validated separately.

**It still fails.** `Scripts/DispatchStress` with stealing on: length 2 at two chunks per worker fails at
dispatch 10, length 100 at four fails at dispatch 4, length 4096 at two fails at dispatch 5. Always the same
signature — an index executed twice, none missing.

> **That is a finding, and it is the useful part.** With atomic, generation-validated claims **no index can be
> claimed twice within a generation**, and the stealing branch was checked to `return` rather than fall
> through into the ordinary descriptor loop. So the duplicate does not come from the claim protocol at all —
> **the claim protocol is exonerated.** What remains are the two paths a chunk can be executed on without
> being claimed: the token accounting, and the interaction between `_stealingDispatch` and the worker loop.

**Stopped after the fourth attempt**, and this time not for lack of an idea but for lack of a *premise worth
testing*. Four patches have each moved the failure later without closing it, and the fifth would be a guess.
Recorded so the next attempt starts from what is ruled out rather than from the beginning:

| # | defect | status |
|---|---|---|
| 1 | mode flag cleared at end of dispatch, late worker takes stale path | fixed |
| 2 | caller drains greedily, dispatch runs single-threaded | fixed |
| 3 | drainer captures shape while counters are reset under it | fixed |
| 4 | generation published before the shape rather than after | fixed |
| - | **claim protocol** | **exonerated by the CAS version still failing** |
| ? | **token accounting, or `_stealingDispatch` versus the worker loop** | **untested** |

**The branch remains off by default and the shipping path is unaffected**: `DispatchStress` clean at 22,500
dispatches, suite 2748/0 after the work.

### `XC-96` second attempt: two more defects found, still not closed, and the design that would close it (2026-08-19)

The work-stealing branch was re-opened with `Scripts/DispatchStress` as the instrument — a per-index ledger
over thousands of dispatches a second, instead of a thirty-second suite that dies without printing a failure.
It reproduced the defect **in under four dispatches**, which is the whole point of building it.

**Defect three: the drainer captures the shape once, and the counters are reset under it.** A drainer reads
`_regionCount` and `_regionSubChunks` on entry, but `_regionClaims` is shared and zeroed by the next
dispatch. A worker slow enough to straddle two dispatches mixes the **old shape with the new counters** and
computes `region * oldSubChunks + sub` where the dispatcher wrote `region * newSubChunks + sub` — a chunk
nobody assigned it. Guarded with a generation counter, re-checked before every claim.

**Defect four: the generation was published FIRST, which is backwards.** Bumping it before writing the shape
lets a drainer read the new generation and the old shape, pass its own consistency check, and index with the
stale formula — the same defect moved one line earlier. The generation must be published **last**, after the
shape and the counters.

**Both fixed, and it still fails.** The failure moved from dispatch 3 to 5 to 12, which is the signature of a
race that needs a different design rather than another guard.

> **The remaining window cannot be closed by checking.** A drainer passes the generation check and can then be
> preempted for arbitrarily long before its `Interlocked.Increment`, by which time the counters have been
> reset. Any check-then-act pair has this window.
>
> **The design that closes it is already in this file.** The decode pool packs the generation and the next
> index into ONE 64-bit word and claims with compare-and-swap (`DecodeChunkClaim`), so a claim from a stale
> generation fails atomically rather than being validated separately. Its own comments explain why a counted
> semaphore was not enough for the same problem. **A per-region packed CAS claim is what the stealing
> protocol needs**, and it is a proven shape here rather than a new invention.

**Stopped and left off by default.** Shipping default verified after the work: `DispatchStress` clean at
22,500 dispatches, suite 2748/0. Four design defects are now written down against this idea — mode-flag
lifetime, caller greed, shape-versus-counter staleness, and generation publication order — which is worth
more than a fourth failed patch.

### `XC-98` refuted: the pool size buys nothing, and cutting it without pinning costs (2026-08-19)

The task was filed on one reading — **16 workers 31.22 ms against 32 workers 31.55 ms** — concluding that
SMT is worth nothing here and the pool should be sized to physical cores. **That reading had affinity
applied**, and separating the two arms reverses it.

| configuration | CNN, 60.9 MB | VGG-16 |
|---|---:|---:|
| 32 logical (shipping) | 18.63 ms | 27.56 ms |
| **16 workers, NOT pinned** | 19.23 **(+3.2%)** | 28.08 **(+1.9%)** |
| 16 workers, pinned one per core | 18.68 (+0.2%) | 27.37 (-0.7%) |
| 16 not pinned + region-major x4 | 18.46 (-0.9%) | 26.93 (-2.3%) |
| **16 pinned + region-major x4** | **18.23 (-2.2%)** | **26.67 (-3.2%)** |

> **Cutting the pool to physical-core count without pinning is worse than leaving it alone** — the scheduler
> puts two workers on one core and leaves others idle. **The size alone buys nothing** (+0.2% / -0.7%). What
> pays is the **region-major layout**, and the smaller pool only helps it show.

**So the premise was wrong and the task closes as a negative.** The measurement that produced it was not
wrong — it was read as being about the worker count when it was about worker *placement*, and those need
different code: the count is one line in `ResolveWorkerCount`, the placement is per-thread affinity, which is
`SetThreadAffinityMask` on Windows and `sched_setaffinity` on Linux, in a library whose whole identity is
portable pure C#.

**What is still on the table, and it is small.** The best configuration reachable without any affinity code
is **pool at physical-core count plus region-major x4**: **-0.9% on the CNN and -2.3% on VGG-16**. The CNN
figure sits inside this box's run-to-run band, so it is one solid result and one that is not. It also needs
portable physical-core detection, which `Environment.ProcessorCount` does not give and which cannot be
assumed to be half the logical count — this repository already has a big.LITTLE note saying exactly that.

**Recommendation: do not build it for 1-2%.** The remaining gap to ONNX Runtime is 1.43x on VGG-16 and 1.94x
on the CNN, of which convolution is 92%, and inside that scaling is the larger factor. A percent of pool
placement is not where that is.

### `XC-97` closed, and the region-major gain turns out to need a pool sized to physical cores (2026-08-19)

**The defect was mine and it was committed.** `OverfitParallel` shipped with `StealChunks` **on**, so any
`chunksPerWorker` above 1 entered the abandoned work-stealing branch, executed **overlapping chunks**, and
signalled the countdown past zero — a background-thread throw that ends the process without printing a
failure. Whole suite runs finished at 176, 188 and 222 of 2748 tests, each reporting "Passed!".

**It cost hours because the instrument could not see it.** Reverting `OverfitParallel.cs` from `HEAD` restored
the *broken* file, because the branch was in `HEAD` — so every "revert and retest" reproduced the defect and
pointed the blame back at the innocent parts of the same commit.

**`Scripts/DispatchStress` is what found it, in seconds.** It hammers `For` with a per-index ledger and stops
at the first dispatch that runs an index twice or not at all:

| shape | before | after switching the branch off |
|---|---|---|
| length 2, 2 chunks per worker | fails at dispatch 16 | clean |
| length 100, 3 chunks per worker | fails at dispatch 6 | clean |
| length 4096, 2 chunks per worker | fails at dispatch 37 — 396 duplicated, 40 missing | clean |
| **all 15 lengths x 5 factors** | - | **22,500 dispatches, 0.5 s, clean** |

Suite 2748/0 twice, and the coverage test is back to its full 1/2/3/4/8 range.

#### The re-measurement, and the condition nobody had checked

With the dispatcher correct, region-major x4 was measured again — three ABAB passes per model, **all six
negative**: the 60.9 MB CNN **18.72 to 18.20 ms (-2.8%)**, VGG-16 **27.63 to 26.70 ms (-3.4%)**, pool use
65.0% to 69.5% and 76.3% to 79.4%. The original direction held and the magnitude grew.

**Then the benchmark disagreed with the harness**, and the difference was the worker count:

| pool | CNN | VGG-16 |
|---|---:|---:|
| **16 physical cores, pinned** | **-2.8%** | **-3.0%** |
| **default, 32 logical** | **+0.0%** | -1.0% |

> **The gain exists only when the pool is sized to physical cores.** The product uses
> `Environment.ProcessorCount`, which is 32 here, so it gets none of it. Both switches therefore ship **off**:
> a switch goes on when it pays in the configuration that ships, not in the one that measured best.

**And that points somewhere better than the 3%.** SMT was already measured to be worth nothing on this
machine — **16 workers 31.22 ms against 32 workers 31.55 ms** — so the pool is simply the wrong size. Sizing
it to physical cores takes that ~1% directly *and* unlocks this ~3%. The two belong in one change with one
measurement, which is `XC-98`.

### CORRECTION: `chunksPerWorker > 1` overlaps chunks, and that voids the region-major measurement (2026-08-19)

**A conclusion recorded earlier today was wrong and is withdrawn.** Three cold test runs ended at 176, 188
and 222 of 2748 tests, each reporting "Passed!", and that was attributed to the committed dispatcher change
with a recommendation to revert the commit. **The commit was not the cause.** Moving one test file aside
gives **2738 passed, 0 failed, 32 seconds** — green and fast.

**The cause is `OverfitParallelChunkGridTests` itself, and it is the messenger rather than the defect.** It
calls `For(..., chunksPerWorker)` **explicitly** with 2, 3, 4 and 8, so lowering the *default* to 1 did not
touch it. At those values it reports overlapping work — one run recorded **"0 index(es) never ran and 629 ran
more than once"** at length 4096 with three chunks per worker — and it takes the test host down often enough
to end whole suite runs part-way.

> **So `chunksPerWorker > 1` is broken in the dispatcher**, and the coverage test written to guard the grid
> layout is what found it. Nothing in the product passes a value above 1: the default is 1 and the
> convolution fan-out passes `ChunkFactor`, which is also 1. **Exposure is nil, and the defect is real.**

#### What this voids

**The region-major measurement (-2.9% on the 60.9 MB CNN, -2.1% on VGG-16) was taken at four chunks per
worker — i.e. with a dispatcher that overlaps chunks.** Six ABAB passes all favoured it, and overlapping work
should be *slower* rather than faster, so the direction is not obviously an artefact. But the arm was running
a different computation from the one it was compared against, and **a measurement of code that does the wrong
work is not a measurement.** It has to be repeated after the overlap is fixed.

#### The three instrument failures of this session, in order

They are recorded together because each one made the next harder to see.

1. **A mutation harness restored source with `shutil.copy2`'s preserved timestamp.** MSBuild saw the object
   as newer than the source and skipped the rebuild, so runs after the restore executed the *mutated*
   library. 100 tests failed against byte-identical source; a file-by-file bisection against `HEAD` was what
   finally showed it. Fixed in `Scripts/mutate.py`.
2. **`subprocess.run(timeout=...)` kills the direct child only.** A timed-out `dotnet test` leaves
   `testhost.exe` and `DevOnBike.Overfit.Tests.exe` alive holding
   `Global\DevOnBike.Overfit.MachineMeasurement`, and the build guard then refuses every later build. Each
   timeout poisoned the next run, which read as "the code hangs". Fixed with `kill_tree` / `kill_test_hosts`.
3. **Overlapping test runs.** Two `dotnet test` processes do not produce two results; they take one
   machine-exclusive mutex and produce none. Several arms reported HUNG or NO SUMMARY purely because a
   previous background run was still alive. This one was self-inflicted, diagnosed mid-session, and then
   repeated three more times.

**The common shape: none of the three announces itself.** A skipped build, an orphaned process and a
colliding run all present as "the code is broken", and all three pointed at the same innocent commit.

### The full work-stealing protocol: attempted, abandoned, and what it cost (2026-08-19)

`XC-95`'s region-major layout is an approximation — it holds locality only while workers happen to finish in
step. The full design gives each worker a **home region** with its own claim counter, drained before it
steals from any other. It was implemented behind `OVERFIT_PARALLEL_STEAL` and **abandoned without a usable
measurement**.

**Two distinct concurrency defects were found and fixed, and it still did not run.**

**1. The mode flag's lifetime.** The dispatch set `_stealingDispatch = true` and cleared it on the way out. A
dispatch can leave tokens unconsumed — the caller may drain every region before a slow worker has woken — so
a late worker wakes with the flag already cleared and takes the **single-counter path with the stealing
dispatch's state still in place**: `_nextChunk` is zero and `_chunkCount` is not, so it re-executes chunk
zero and signals a countdown that has already reached zero. The fix is to write the flag for **every**
dispatch under the lock and never clear it, so a late worker always describes the dispatch it is joining.

**2. The caller was the greedy drainer.** In the old protocol the calling thread runs exactly one chunk and
waits, leaving the rest to the pool. The new one had it drain every region — and it starts before any worker
has been woken, with the semaphore's wake latency as a head start. On dispatches whose chunks are
microseconds long, which is most of the test suite, **the caller finishes the whole range alone and the
fan-out runs single-threaded**. Correct, roughly sixteen times too slow, and indistinguishable from a hang
from outside: the suite went past a four-minute-per-arm timeout. The fix is for the caller to drain its own
region, then wait with a timeout and steal only if the pool has not finished — which also keeps a wake that
never happens from blocking forever.

**Both were fixed and the coverage test still did not complete.** A third failure remains unfound.

> **Stopped rather than iterated.** This is a tuned primitive with recorded incidents of its own — a leaked
> semaphore token that stopped the pool sleeping at all, and a claim protocol whose comments explain why the
> decode pool needed a different one. Debugging a concurrency defect in it through a ten-minute feedback loop
> by patching and re-running is how the next incident gets written, not avoided. **Reverted to the committed
> state.**

**What survives, and it is not nothing.** The two defects above are real properties of the design, not of the
implementation, and any future attempt inherits them: the mode flag must be per dispatch and never cleared,
and the caller must not be the primary drainer. The region-major layout that ships is the measured part of
this idea (**-2.9% on the CNN, -2.1% on VGG-16**), and pool use is 69.4% — so **most of the idle time this
protocol was meant to recover is still there.**

### Region-major chunks: -2.9% on the CNN, -2.1% on VGG-16, and the first version of this measurement was wrong (2026-08-19)

`XC-95` wanted a worker pinned to a contiguous region. The cheap approximation keeps the claim protocol and
changes only the index-to-range mapping: chunk `i` becomes region `i % regions`, sub-chunk `i / regions`, so
the worker that took index `k` takes `k + regions` next — the continuation of its own region rather than the
start of somebody else's. Behind `OVERFIT_PARALLEL_REGION_MAJOR`, on by default, with four chunks per worker
(`OVERFIT_PARALLEL_CHUNK_FACTOR`, also measured).

| model | current default | region-major x4 | change | pool use |
|---|---:|---:|---:|---|
| CNN, 60.9 MB | 18.89 ms | **18.34 ms** | **-2.9%** | 64.8% -> **69.4%** |
| VGG-16 | 27.31 ms | **26.74 ms** | **-2.1%** | 76.9% -> **79.5%** |

Three ABAB passes per model, **all six negative** (-2.0, -2.6, -4.2 and -2.1, -1.8, -2.2). Suite 2748/0 in
every arm of both switches. Headline on the shipping default: **VGG-16 27.12 ms against ONNX Runtime's 19.03
(1.42x)** and the **60.9 MB CNN 18.63 against 9.74 (1.91x)**, parity unchanged in all four runs.

**The layout and the split only work together.** With slice-major, four chunks per worker is **3.4% slower**;
with region-major it is 2.9% faster. Either alone loses, which is why the two defaults were flipped together.

#### The first version of this measurement said the opposite, and the cause was my own harness

It reported region-major as neutral at two and four chunks and **worse at eight**, and that was recorded as a
negative. It was an artefact.

**`shutil.copy2` preserves the modification time.** A mutation harness copied the source, edited it, built and
tested — then restored the copy, which put the file's timestamp *back* to before the build. MSBuild compares
those timestamps, decides the output is current, and **does not rebuild**. Every run after that restore
executed the **mutated** library against correct source.

Measured while diagnosing it: the restore moved the timestamp **backwards by 113.5 seconds**, and the next
full suite reported **100 failures** — with source byte-identical to a green commit. It took a file-by-file
substitution of `HEAD` versions to establish that the source was never the problem. The failing assertions
read `Expected: 0.698840261, Actual: 0`, which is exactly what "each chunk is one element short" produces.

**Two things came out of that.** `Scripts/mutate.py` now owns the pattern and re-stamps the file on restore,
with the incident in its docstring; and this measurement was redone as a **runtime switch rather than a code
swap**, with the live layout echoed into `OccupancyReport` and asserted by the harness before any reading is
used. Both arms in one binary cannot be built stale.

> **The general lesson is not about `copy2`.** A build that silently does not happen produces a measurement
> of the wrong code that looks exactly like a measurement of the right code — the same shape as the stale
> `ProfHarness` binary that voided a sweep earlier the same day. **Echo the thing being switched, and refuse
> the reading when the echo disagrees.**

### The finer-split cost IS locality, and it is the first confirmed mechanism in this thread (2026-08-19)

The premise was stated before the run, and the arithmetic did **not** support it: the packed kernel matrix is
read once per panel whatever the split, and a worker's input working set fits L2 in both arms (917 KB at one
chunk per worker, 229 KB at eight). The measurement was run against that reasoning rather than to confirm it.

AMD uProf `data_access`, 30 s per arm, our module only, **per thousand instructions**:

| where the data came from | factor 1 | factor 8 | change |
|---|---:|---:|---:|
| **this core's L2** (nearest) | 6.853 | 6.461 | **-5.7%** |
| same-CCX L3 | 1.724 | 2.418 | **+40.3%** |
| **another CCD's cache** | 0.244 | 0.396 | **+62.1%** |
| **DRAM** (farthest) | 0.283 | 0.481 | **+70.0%** |
| L1 data-cache accesses | 403.8 | 411.6 | +1.9% |
| retired instructions | 553601 | 561751 | +1.5% |
| **CPI** | **0.3928** | **0.4351** | **+10.8%** |
| L2 DTLB misses | 0.132 | 0.190 | +43.6% |

> **Supply moves outward through the hierarchy monotonically with distance**, and the near level is the only
> one that falls. Instruction count and L1 access count are flat, so this is the same work executing more
> slowly: **CPI rises 10.8%**, which is the +6-7% of worker time `XC-94` measured and could not attribute.

**Two limits on this reading, both material.** Wall times under the profiler are **inverted** — 26.27 ms at
factor 1 against 23.64 at factor 8, the opposite of the un-profiled 18.9 against 19.7 — because the ~35%
profiling overhead is not uniform across the arms. **Only the per-instruction ratios from this run are
usable; its timings are not.** `Scripts/machine.py` also flagged `TiWorker` (Windows Update) at 1.05 of 1313
core-seconds; the ratios are robust to a load that small, but it was there.

#### What it means for the 35% idle pool

`XC-92` measured a straggler ratio of 1.28x at one chunk per worker: work is unevenly spread and a fifth of
the pool waits at the barrier. **Rebalancing it costs cache locality**, and the two are in direct tension —
contiguous chunks give locality and imbalance, fine chunks give balance and lose locality. That is why every
finer split measured worse, and it is a property of the work rather than of any one implementation.

**The resolution is the one work-stealing schedulers use, and our claim counter is the wrong shape for it.**
`_nextChunk` hands out chunks in global order, so an early-finishing worker takes the chunk adjacent to
someone else's region rather than a continuation of its own. A locality-preserving variant gives each worker
a contiguous region, has it claim sub-chunks **inside its own region first**, and steals from another region
only when its own is exhausted. That keeps the contiguity the cache wants and still drains the tail.

**This is a design, not a measurement.** It has not been built or measured, and the two attempts before it
both lost.

### The per-chunk buffer was not the cost, and the code that assumed it was is reverted (2026-08-19)

`XC-93` left one attribution unverified: the convolution worker rents a panel buffer of up to 590 KB per
invocation, so a finer split pays it again per chunk. **The arithmetic never supported it** — an
`ArrayPool` rent measures ~9 ns here, and the observed cost was **~16 us per extra chunk**, three orders of
magnitude apart. `XC-94` built the per-thread replacement as the test.

| panel buffer | factor 1 -> 8: worker time | wall clock |
|---|---:|---:|
| rented per work item | +7.4% | +4.4% |
| **held per thread** | **+6.1%** | **+3.8%** |

**The buffer accounts for 1.3 points of a 7.4-point cost.** At one chunk per worker — where the product
actually runs — the two arms are **18.91 against 18.81 ms**, inside the noise. So the change delivers
nothing at the default and does not explain the finer-split cost either.

**It also broke a real contract.** `ResNetBlock_DAG_InferenceAllocatesZeroBytes` failed in the full suite
while passing in isolation: a thread-held array grows when a larger layer arrives, and in a mixed workload
that growth lands inside a measured window. The zero-allocation guarantee is a product property, not a
detail, and the test was right.

**Reverted.** The measurement stays here; the code does not, because keeping an arm that buys nothing and
can break a contract when switched on is a liability rather than an option.

**Still not identified: what the remaining ~6% per-chunk cost is.** It is inside the worker (worker time
rises, dispatch wall falls), it is not the buffer, and the worker's other per-invocation work — the pin, one
`stackalloc` of 128 bytes, reading the context — is far too small. The next candidate is locality rather
than overhead: with eight times more chunks, a worker's chunk spans fewer consecutive panels and consecutive
chunks land on different cores, so the packed kernel matrix is re-read from L3 instead of staying in one
core's L2. **That is a hypothesis and it has not been measured.**

### Finer chunking makes workers busier and the model slower (2026-08-19)

`OverfitParallel.For` gained an opt-in `chunksPerWorker`, and the convolution fan-out asks for it. Both arms
green at 1, 2, 4 and 8 (suite 2738/0 each).

| CNN, 16 cores | wall | mean chunks | pool use |
|---|---:|---:|---:|
| 1 chunk per worker | **18.95 / 18.82 ms** | 14.4 | 64.9 / 64.5% |
| 2 | 19.31 ms (+1.9%) | 22.6 | 65.3% |
| 4 | 19.41 ms (+2.4%) | 35.8 | **68.5%** |
| 8 | 19.73 ms (+4.1%) | 56.8 | **70.9%** |

VGG-16 shows the same shape more weakly: pool use 76.4% to 80.8%, wall 27.33 to 27.49 ms.

> **Workers get busier and the model gets slower**, monotonically in both. The extra busy time is not
> productive: `GemmFusedPanelWorker512` rents a `PooledBuffer` of up to 590 KB **per invocation**, so a 4x
> finer split pays that rent four times and the occupancy metric counts it as work. **This was written into
> `XC-93` as the risk before the measurement ran**, which is the only reason it was recognised rather than
> re-derived.

**So the straggler is real and rebalancing is not the fix.** 35% of the pool's time is idle at one chunk per
worker; finer chunks convert that idle time into overhead at slightly worse than parity. The per-chunk cost
has to fall before a finer split can pay, which means **per-worker scratch instead of per-chunk** — a
`[ThreadStatic]` buffer, which needs no worker id and so avoids the coupling described below.

#### The switch had to become per-call-site, and memory corruption is why

Applying the factor to every fan-out turned **100 tests red**. The first cause was a
`SemaphoreFullException`: the start semaphore's maximum count was the worker count, and a wider release
exceeds it. Fixing that left **two** failures, and they were the important ones.

**`TensorMath.LayerNorm`'s backward sizes its partial buffers `WorkerCount x C`, dispatches over `numRows`,
and picks its slot with `chunkIdx = chunkStart / perChunk`.** More chunks than workers makes that index run
past the buffer — and it is a pinned write, so the result is a corrupted heap, not a wrong number. One arm
of the sweep took the test host down mid-run, and **that arm first reported GREEN with 1575 of 2738 tests**,
because the absence of a `[FAIL]` line read as a pass. The count check that turns a short run back into a
failure is now in the harness.

**The dispatcher cannot detect this.** It holds a function pointer; nothing in the signature says whether a
body treats its chunk as a worker slot. So the caller declares it, and the default stays where every
existing body was written against.

#### Two of the three occupancy metrics only hold at one chunk per worker

Measured at four chunks per worker, `occupancy` read **26.6%** and `overhead` **44.7%** for a dispatch whose
`pool use` had **improved** from 64.6% to 68.3%. `occupancy` divides by the chunk count; `overhead`
subtracts the single longest chunk, which stops being the critical path once chunks are small. **`pool use`
is the one to read**, and the report now says so.

#### An instrument failure that cost a whole sweep

The first sweep showed `mean chunks 14.4` at every factor — the switch appeared dead. It was not: **only
`Main.csproj` had been rebuilt, and `ProfHarness` keeps its own copy of the library in its output
directory.** Rebuilding the library is not enough when the consumer copies it. The factor is now echoed into
the report and the harness refuses a reading whose echoed factor does not match what was asked.

### Worker occupancy: 65% of the pool's time at 16 cores, and it explains almost the whole scaling gap (2026-08-19)

**Written because a hardware profiler cannot answer this.** AMD uProf's counters key on
`CYCLES_NOT_IN_HALT`, and **a worker parked waiting for work produces no samples at all**. uProf did settle
what it could — our code's CPI is **0.4413 at one core and 0.4205 at sixteen**, identical, so nothing about
the executed instructions degrades with thread count, and memory is not the limiter. It could not see the
part that was missing.

`OverfitParallel.MeasureOccupancy` (`OVERFIT_PARALLEL_OCCUPANCY=1`) times every chunk against its dispatch.
It is off by default and checked before any timestamp; **measured cost with it on: 18.97 / 18.78 ms against
18.87 / 18.87 off**, i.e. inside the noise.

| cores | ms/call | **pool use** | occupancy | straggler | overhead |
|---:|---:|---:|---:|---:|---:|
| 2 | 66.83 | 96.8% | 96.8% | 1.03x | 0.4% |
| 4 | 36.70 | 93.9% | 93.9% | 1.06x | 0.6% |
| 8 | 22.35 | 86.7% | 88.1% | 1.11x | 1.7% |
| **16** | **18.88** | **65.0%** | 71.5% | **1.28x** | **7.9%** |

**At 16 cores only 65% of the pool's time is executing.** The three metrics are deliberately separate
because they have opposite fixes, and a single "efficiency" number would merge them:

| cause | share | mechanism |
|---|---|---|
| **straggler 1.28x** | ~21% | `chunkCount = min(workers, totalWork)` — **exactly one chunk per worker**, so the claim counter has nothing left to hand out and everyone waits for the slowest |
| **overhead 7.9%** | ~8% | 69 us per dispatch over 9010 dispatches, against 5.51 us measured for the dispatch alone: the rest is waking 15 workers through one semaphore |
| **pool underfill 6.5%** | ~6% | mean chunks 14.4 of 16 — small-N layers produce 7 chunks and leave nine workers idle |

> **`pool use` had to be added after the first reading, and the reason is worth keeping.** `occupancy`
> divides by the chunks a dispatch created, so a layer that produces 7 chunks on a 16-worker pool scores as
> a full house. It reads 71.5% where the honest figure is 65.0%. **A metric that cannot see idle workers is
> the wrong metric for a question about idle workers.**

**At 100% pool use, 18.88 ms becomes 12.27 ms — scaling 10.84x against ONNX Runtime's 12.12x.** Worker
occupancy therefore accounts for almost the entire scaling gap; only 1.12x is left unexplained.

**The indicated fix is to create more chunks than workers.** The claim counter (`_nextChunk`) already exists
and workers already claim dynamically — with one chunk each there is simply nothing to balance. This is a
change to the central dispatch primitive, which decode and attention also use, so it belongs behind a switch
with both arms measured rather than in a convolution-shaped patch.

### The convolution gap is 1.71x scaling and 1.31x per-core work, and the pool is not at fault (2026-08-19)

**This correction matters more than the numbers, because it reverses a conclusion recorded the same
morning.** That conclusion — "scaling is not the problem, our scaling matches theirs" — came from
**whole-model** figures. VGG-16's whole model is 32% dense layers that neither engine can scale, because
they are memory-bound; averaging them in hides what convolution is doing. ONNX Runtime's per-node profiler
at 1 and 16 cores separates it.

| | Overfit | ONNX Runtime | ratio |
|---|---:|---:|---:|
| convolution, 1 core | 125.34 ms | 95.68 ms | **1.31x** |
| convolution, 16 cores | 17.70 ms | 7.89 ms | **2.24x** |
| **convolution scaling** | **7.08x** | **12.12x** | |

`1.31 x 1.71 = 2.24` — the decomposition closes. **Scaling is the larger of the two factors**, and it is the
one that was written off this morning.

> **It also retires the caveat that protected the old conclusion.** The 14.30x figure is a pure-FMA ratio
> with no memory traffic, so "convolution touches memory, therefore 14.30x does not apply to it" was a
> reasonable objection. **ONNX Runtime reaches 12.12x — 85% of it — on the same box and the same layers.**
> The ceiling is real for convolution.

#### Per layer, one sitting, both arms

| layer | K | 1 core | 16 cores | scaling |
|---|---:|---:|---:|---:|
| conv1_1 | 27 | 2.32 ms | 1.06 ms | **2.19x** |
| conv1_2 | 576 | 18.29 | 2.50 | 7.32x |
| conv2_1 | 576 | 7.87 | 1.21 | 6.50x |
| conv2_2 | 1152 | 15.61 | 1.77 | **8.82x** (best) |
| conv3_x | 2304 | 13.87 | 1.65 | 8.41x |
| conv4_x | 4608 | 13.70 | 1.74 | 7.87x |
| conv5_x | 4608 | 4.32 | 0.84 | **5.14x** |
| MaxPool | - | 2.47 | 0.37 | 6.68x down to 2.56x |
| fc1 | - | 8.90 | 7.06 | 1.26x (memory-bound, expected) |

**No layer exceeds 8.82x**, so this is systematic rather than a few bad shapes. The first layer (2.19x) and
the three 14x14 layers (~5.0x) are worse still, but fixing only those would leave most of the loss.

#### The pool is exonerated, by measurement

`OverfitParallel` on perfectly balanced, register-resident, memory-free work:

| cores | ms | scaling |
|---:|---:|---:|
| 1 | 18.02 | 1.00x |
| 4 | 4.54 | 3.97x |
| 8 | 2.30 | 7.84x |
| **16** | **1.22** | **14.82x** |

**14.82x, above the 14.30x reference** — that workload is scalar rather than 512-bit, so it costs less
all-core clock. The fan-out mechanism is not the problem; the convolution's **work decomposition** is.

**Not identified: what caps convolution at ~7-8.8x.** Item count is not it — conv1_2 has 1568 balanced work
items for 16 workers and still reaches only 7.32x. One observation to start from: the same conv3_2 layer
scales **10.5x measured in isolation** (`NchwcConvProbeBenchmark`) against **8.41x inside the model**, which
points at state carried between layers rather than at the kernel. **This is where AMD uProf finally earns
its place** — per-core stalls and cache-miss attribution at 1 core against 16.

#### What it is worth

At ONNX Runtime's 12.12x, our convolution would be `125.34 / 12.12 = 10.34 ms` instead of 17.70 — **7.36 ms**.

| model | now | with their scaling | ratio to ONNX Runtime |
|---|---:|---:|---:|
| VGG-16 | 27.42 ms | **~20.1 ms** | 1.66x -> **~1.22x** |
| CNN, 60.9 MB | 18.73 ms | **~11.4 ms** | 1.92x -> **~1.19x** |

#### And the two models are the same convolution stack

The first per-layer profile of the 60.9 MB CNN shows it is **VGG-16's convolutional stack** — identical
layer shapes and K values — with a global average pool and a 1000-way classifier instead of the dense stack.

| | convolution | pooling | dense | total |
|---|---:|---:|---:|---:|
| CNN, 60.9 MB | **17.70 ms (94.5%)** | 0.98 | 0.05 | 18.73 |
| VGG-16 | 17.70 | 0.92 | 8.69 | 27.42 |

That is why the CNN's ratio (1.92x) is **worse** than VGG-16's (1.66x) despite being the smaller model: VGG's
dense stack is at parity with ONNX Runtime and dilutes the convolution gap. **The CNN is the cleaner
instrument for convolution work**, and the two models do not need separate treatment.

### `XC-91`: the NCHWc probe wins where the shape says it should and loses everywhere else (2026-08-19)

`NchwcConvProbeBenchmark` implements the direct convolution over 16-float channel blocks and races it
against the shipped im2col path on two real VGG-16 layers, both **3.70 GFLOP**. Blocked input and filter are
prepared in setup, so **no reorder cost is charged and the cache is warm** — every advantage is given to the
candidate. Correctness is asserted against `ConvLayer` before either number is read; both kernels agree to a
relative 1.5e-6, which is fp32 reassociation.

#### The kernel shape mattered more than the layout did

| kernel | loads per FMA | single core | GFLOP/s | % of 359 |
|---|---:|---:|---:|---:|
| im2col + GEMM (shipped) | - | **12.44 ms** | 296 | **83%** |
| NCHWc, 1 block x 8 positions | 1.125 | 22.4 | 165 | 46% |
| NCHWc, 4 blocks x 4 positions | **0.5** | **14.55** | 252 | 70% |

The first shape loads one weight vector and issues eight broadcasts per `(input channel, kernel position)`,
which saturates the load ports. Inverting the ratio — four output-channel blocks driven by one broadcast —
was worth **1.54x** on the candidate and the arithmetic predicted it. Sixteen accumulators is the budget the
im2col micro-kernel already holds in registers on this JIT, so it was chosen as a shape known to fit.

#### At 16 threads, on two shapes, the verdict splits

| layer | N/K | im2col | NCHWc parallel | result |
|---|---:|---:|---:|---|
| **conv1_2** — K=576, N=50176 | **87** | 1.529-1.550 ms | **1.148-1.168 ms** | **NCHWc 1.34x faster** |
| **conv3_2** — K=2304, N=3136 | **1.36** | 1.170-1.193 ms | 1.343-1.363 ms | im2col 1.15x faster |

Four repetitions per shape, spread under 1.5%, and the direction reproduced in a second sitting. `conv3_2`
was re-measured after the shape became a parameter, as the control on whether parameterising cost anything;
it reproduced.

> **The mechanism is clean and it is the ratio `N/K`.** NCHWc's saving is the im2col matrix it never builds,
> and that matrix costs `K x N`. At `N/K = 87` it is 115 MB and avoiding it wins; at `N/K = 1.36` there is
> little to avoid, and the managed kernel's lower arithmetic density loses. **A single-shape probe would have
> produced a confident verdict that did not cover the case** — the first run measured only `conv3_2`,
> concluded NCHWc loses, and was wrong about the layers where it matters.

#### What a hybrid would be worth, and why it is still not recommended

VGG-16's high-`N/K` convolutions are conv1_1, conv1_2, conv2_1 and conv2_2 — **43.80 ms of the 124.21 ms
single-core convolution budget, 35%**. At the measured 1.34x that is ~1.6 ms of the 27.30 ms model at 16
cores, taking VGG-16 to ~25.7 ms and the ratio to ONNX Runtime from 1.66x to **~1.56x**.

**That is roughly 6%, and it needs nearly the whole port**: the blocked kernel, blocked **pooling** (a MaxPool
sits between conv1_2 and conv2_1, so an unblocked pool would force a reorder round trip mid-stack), the
reorder itself, and layout propagation through the importer. ONNX Runtime gets 1.51x from the same layout
because their assembly wins on the compute-dense layers too; ours does not, so the hybrid only ever covers
the early stack. **Recommendation: do not build it.** The probe cost one benchmark file and settled it.

#### Two instrument failures worth keeping

**The first run reported 60.02 ms for an arm that measures 12.48 ms warm** — a warmup artefact that would
have made the candidate look 2.29x faster than the baseline. It was caught by reconciling against a
standalone run of the same layer, which is why the benchmark now times both arms inside its own setup as
well: three independent instruments agreeing is what makes the reading usable.

**`Scripts/machine.py` flagged Windows Defender (`MsMpEng`) during two of these runs**, at 1.05 and 6.50
core-seconds, while the total foreign share was only 2.18-2.32%. The named-scanner probe fired where the
share probe would have passed the window — which is the reason there are two probes.

### Sizing the NCHWc port: the layout is worth 1.51x, the rest of their advantage is 1.53x (2026-08-19)

**The split was measured, not argued, and it cost one afternoon rather than a port.** ONNX Runtime ships both
structures. At `ORT_ENABLE_ALL` its NCHWc graph transform runs and `Conv` goes through `MlasNchwcConv` over
a blocked channel layout. At `ORT_ENABLE_EXTENDED` the transform does not run and `Conv` falls back to
`MlasConv` - **im2col plus SGEMM, the same structure Overfit uses**. Same assembly, same threads, same box,
same process. The difference between their two arms is the layout's worth with everything else held still.

| path | convolution |
|---|---:|
| ONNX Runtime, NCHWc | **7.65 ms** |
| ONNX Runtime, im2col + GEMM | **11.57 ms** |
| Overfit, im2col + GEMM | **17.74 ms** |

- **Layout: 1.51x** (their im2col to their NCHWc)
- **Everything else: 1.53x** (our im2col to their im2col — hand-written AVX-512 against managed C#)
- **Product 2.32x, against the 2.29x measured directly.** The decomposition closes.

**Capability checked before the result was read**: the NCHWc arm carries `ReorderInput`/`ReorderOutput`
nodes and the fallback arm carries none. An arm that had silently stayed on NCHWc would have reported the
layout as worthless, which is the answer that stops the work.

Whole-model wall moved 16.52 to 20.49 ms (+24.1%) between the two arms; dense, pooling and flatten were
unchanged, so the lever was isolated to convolution. Their fallback op is `FusedConv`, i.e. bias and
activation fused into the convolution even on the im2col path — the same change `XC-86` made here today.

#### What the ceiling would be worth

At the full 1.51x, convolution goes 17.74 to 11.75 ms and VGG-16 goes **27.30 to 21.31 ms**, taking the
ratio to ONNX Runtime from 1.66x to **1.29x**. That is the ceiling and it assumes a managed NCHWc kernel
captures as much of the layout benefit as their assembly does.

#### What the port contains, counted from their source

| piece | MLAS | needed for a VGG-class model |
|---|---|---|
| orchestration | `snchwc.cpp`, **2040 lines** | most of it |
| activation reorder in/out | `reorder.cpp` | yes |
| **filter** reorder `OIHW -> OIHWBiBo` | `reorder.cpp`, load-time | yes |
| general blocked kernel | `MlasConvNchwcFloatKernel` | **yes** |
| first-layer kernel (unblocked input) | `MlasConvNchwFloatKernel` | **yes** — VGG's 3-channel input |
| depthwise kernel | `MlasConvDepthwiseFloatKernel` | no |
| pointwise 1x1 kernel | `MlasConvPointwiseFloatKernel` | no for VGG, yes for ResNet |
| pooling over the blocked layout | `SpoolKernelAvx512F.asm` | **yes** — otherwise reorders per layer |
| assembly | `SconvKernelAvx512F.asm` 25 KB + `SconvKernelCommon.inc` 29 KB | to be re-derived in `Vector512` |

Block size is **16 floats** for AVX-512 (`platform.cpp`). VGG-16 is entirely 3x3 stride 1, so it needs
**two** of their four kernels, not four.

#### The dominant risk, and this repository has the evidence for it

A direct convolution over blocked channels holds an output tile live across the whole kernel window, so it
wants **more** simultaneously-live vector registers than the im2col micro-kernel does. **The largest single
win on this branch — 3.46x — came from stopping the existing micro-kernel spilling its 16 accumulators**,
and it spilled because the method also carried nine parameters, a `stackalloc` and eight store calls. The
NCHWc kernel is a harder register-allocation problem than the one the JIT has already failed once.

**So the 1.51x is a ceiling whose realisation in managed code is genuinely uncertain**, and the uncertainty
is concentrated in the one mechanism that has been most expensive here.

#### Recommendation: a bounded probe, not the port

Prototype **only the NCHWc micro-kernel**, for one VGG layer shape, as a benchmark arm against the current
im2col path on the same layer — pre-blocked data prepared in setup, no reorders, no graph changes, no
importer work. If it does not beat the im2col path in isolation, the port is dead and the probe cost a
fraction of it. This is the same shape as the cost ladder that settled the register question, and the same
discipline that killed the N-block and the shared expansion before they reached the product.

### ONNX Runtime's own per-node budget: the whole remaining gap is convolution (2026-08-19)

ONNX Runtime has a per-node profiler (`SessionOptions.EnableProfiling`), wired into `Scripts/ProfHarness`
behind `PROF_ORT_PROFILE=1`. It is the only source that can say whether their advantage is spread across the
model or concentrated. Three runs at 16 cores, agreeing within 1%: totals 16.62 / 16.54 / 16.45 ms.

| operator | Overfit | ONNX Runtime | ratio |
|---|---:|---:|---:|
| **Conv (13 nodes)** | **17.74 ms** | **7.76 ms** | **2.29x** |
| Dense (3 nodes) | 8.58 | 8.18 | **1.05x** |
| MaxPool (5) | 0.93 | 0.43 | 2.16x |
| AveragePool | 0.05 | 0.03 | - |
| Relu (15) | **0.00** (fused) | **0.00** (fused) | - |
| ReorderOutput | - | 0.03 | - |
| **TOTAL** | **27.30** | **16.45** | **1.66x** |

**`fc1`: ours 6.97 ms, theirs 6.815 ms — 1.02x.** Two independent engines hit the same wall on the same
411 MB of fp32 weights, which corroborates the `XC-87` negative from outside our own code. **The dense stack
as a whole is at parity.**

> **Of the 10.85 ms difference, 9.98 ms is convolution — 92%.** Everything else in VGG-16 is level or too
> small to matter. This retires the whole-model ratio as a planning number: there is one problem left, not
> five.

**And their convolution is structurally different, confirmed in their source rather than inferred.**
`snchwc.cpp` implements `MlasNchwcConv` over an **NCHWc blocked channel layout** with a direct convolution
kernel — `MlasConvNchwcFloatKernel`, backed by `SconvKernelAvx512F.asm`. **They never build an im2col matrix
at all.** The `ReorderOutput` node in their profile is the NCHWc-to-NCHW conversion at the end, and it costs
0.03 ms. Their kernel flags include `MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION` and
`MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION`, so bias and activation are fused into the convolution — the same
change made here today, arrived at independently.

**Their per-node spread is also much flatter than ours**: their heaviest convolution is 0.875 ms and their
lightest 0.33 ms, where ours run 2.47 ms down to 0.83 ms.

**Measurement conditions, stated because the guard objected.** All three runs were flagged by
`Scripts/machine.py` at 5.99-6.44% foreign CPU against a then-5% ceiling. The load was the desktop's constant
background — Parsec, the compositor, Task Manager and the agent — not an event, and the three readings agree
within 1%. The ceiling was **re-derived from five samples rather than moved to let these three pass**; see
the module docstring.

### fc1 is already at its memory ceiling, and the 2.4 ms estimate came from the wrong ceiling (2026-08-19)

VGG-16's `fc1` is `25088 -> 4096` at batch 1: **411 MB of fp32 weights read per inference**, 6.97 ms in the
model and **25.5% of the whole 27.30 ms budget** - larger than any convolution. It also scales worst of
anything in the model. `LinearGemvAccumulatorBenchmark` runs it at its real shape.

**The premise, stated before the run**: `AccumulateRowsAvx512` loads four accumulator vectors and stores four
back **for every input row**, so four weight loads carry eight extra memory operations. They hit L1 and are
individually cheap, but they occupy load and store ports. **What would refute it**: a stream arm with the
same workers over the same 411 MB and no accumulator at all.

| arm | ms | GB/s | vs current |
|---|---:|---:|---:|
| **current kernel** | **5.587** | **73.6** | - |
| stream ceiling (read only, no accumulator, no FMA) | 6.106 | **67.3** | **+9.3%** |
| accumulator held across 2 input rows | 5.703 | 72.1 | +2.1% |
| across 4 rows | 6.029 | 68.2 | +7.9% |
| across 8 rows | 6.376 | 64.5 | +14.1% |

**Refuted, and by the arm that was written to refute it.** The shipped kernel is **faster than a loop that
reads the same bytes and does no arithmetic at all**. The accumulator traffic is hidden behind the memory
wait, and every blocked variant is worse - narrower inner loop, more register pressure, no benefit to trade
it against.

> **The 2.4 ms estimate was wrong, and the mistake has a name.** It divided 411 MB by **90.8 GB/s measured
> with a different loop**. The ceiling for *this* access pattern is 67-74 GB/s and the kernel sits at 73.6.
> That is this repository's own rule broken again: **check the number is in the mechanism's unit.**

**What is left on this layer is not tuning.** 411 MB at fp32 is the traffic, and the only lever that changes
it is reading fewer bytes - quantised weights. That changes numerics and is a product decision, not a kernel
one. Nothing in the current shape is available.

**Isolated it runs 5.587 ms against 6.97 ms in the model**, a 25% difference not accounted for; the in-model
figure follows the whole convolution stack, so cache state differs. Not investigated.

### The vectorised im2col gather: -23.8% on a single core, and 2.00x against ONNX Runtime becomes 1.41x (2026-08-19)

At `stride == 1` and a fixed `(ky, kx)`, consecutive output positions read consecutive input addresses. The
gather is therefore a contiguous copy per run of output positions sharing an output row, not a scatter of
indices. The run table depends on the panel and not on `kk`, so it is built once and reused for all K rows —
which also removes the integer division the old code ran on every gathered element.
`OVERFIT_CONV_VECTOR_GATHER=0` restores the element-at-a-time form.

#### Single core, per layer — the arm the cost model made its prediction for

| layer | scalar | vector | change |
|---|---:|---:|---:|
| K=576, out=3.2M (28.9 M elements) | 34.28 ms | **19.27 ms** | **-43.8%** |
| K=1152, out=1.6M | 22.70 | 14.75 | -35.0% |
| K=576, out=1.6M | 11.54 | 7.53 | -34.8% |
| K=2304, out=0.8M | 18.42 | 13.41 | -27.2% |
| K=4608, out=0.4M | 15.64 | 13.36 | -14.6% |
| K=4608, out=100k | 4.49 | 4.02 | -10.5% |
| **MaxPool x4** | 2.39 / 1.21 / 0.61 / 0.34 | **unchanged** | **+0.0%** |
| **Linear x3** | 8.91 / 2.95 / 0.35 | **unchanged** | -0.8% / -0.3% |
| **WALL** | **185.23** | **141.05** | **-23.8%** |

**The gain rises with the element count, layer by layer, and the layers that do not gather do not move.**
Four MaxPool nodes at +0.0% and three dense layers inside 1% are the canary *inside* the measurement: one
thing changed, and it is the thing that was meant to.

**Predicted 59 ms at a 4x gather; measured 44.2 ms, which implies 2.28x.** The model had the shape right and
the magnitude high by a third. Recorded because the estimate was published before the fix existed.

#### Both engines, one sitting, same loop and same core mask

| cores | Overfit | ONNX Runtime | ratio | was |
|---:|---:|---:|---:|---:|
| 1 | **148.12 ms** | 105.28 ms | **1.41x** | 2.00x |
| 4 | 47.98 | 33.92 | **1.41x** | 1.92x |
| 16 | 27.26 | 16.09 | 1.69x | 1.98x |

#### All cores, ABAB, ONNX Runtime in the same process

| model | scalar gather | vector gather | change | canary |
|---|---:|---:|---:|---:|
| VGG-16 | 30.20 / 29.90 ms | **29.87 / 27.86 ms** | **-3.9%** | 18.84 / 19.23 / 18.94 / 19.05 |
| CNN, 60.9 MB | 21.10 / 21.02 ms | **18.41 / 18.84 ms** | **-11.5%** | 9.69 / 9.69 / 9.59 / 9.75 |

Parity unchanged in all eight runs. **The whole-model gain at 16 cores is far smaller than the single-core
gain**, and that is not a contradiction: the gather was compute-bound work that parallelised well, so
sixteen cores were already hiding most of it.

> **It moved the bottleneck, and the next task is visible in the same table.** We now scale 5.43x across 16
> cores where ONNX Runtime scales 6.54x — **the first time on this branch that their scaling is the better
> of the two.** Removing per-core compute leaves a higher proportion of memory-bound work, so the ratio is
> 1.41x at one and four cores but 1.69x at sixteen. The per-core problem is now smaller than the scaling
> problem, which is the reverse of this morning.

**Four of five mutations caught** — left-padding columns unzeroed, right-edge columns unzeroed, the input
row offset sign flipped, and the last element of every run left stale. **The fifth escaped and should
have**: the fill of lanes past `nrEff` is a performance guard against NaN in uninitialised pool memory, and
the partial micro-kernel computes those lanes without storing them, so by construction it cannot change a
result.

### The 2.00x is the im2col gather, not the micro-kernel: a fitted cost model checked on seven held-out layers (2026-08-19)

Single-core, per layer, VGG-16 (`Scripts/ProfHarness`, `PROF_AFFINITY=1`, `DOTNET_PROCESSOR_COUNT=1`). Two
layers were used to fit two coefficients; the other seven were never used in the fit.

```text
layer time  =  0.964 ns x (K * N gathered elements)  +  3.32 ns x GFLOP
```

| layer | predicted | measured | error |
|---|---:|---:|---:|
| K=1152, out=1.6M | 26.2 ms | 27.16 ms | 3.5% |
| K=576, out=1.6M | 13.11 | 13.81 | 5% |
| K=576, out=3.2M | 40.15 | 43.03 | 7% |
| K=1152, out=0.8M | 9.63 | 9.68 | **0.5%** |
| K=4608, out=100k | 3.94 | 4.45 | 13% |
| K=27, out=3.2M | 1.88 | 3.39 | **80% — outlier** |

**Eight of nine layers inside 13%.** The K=27 first layer is the one it does not describe; its inner sweep is
27 deep and something else dominates there, which is a separate question.

**What the two coefficients say.**

- **The GEMM term is 3.32 ns per GFLOP = 301 GFLOP/s, which is 84% of this box's 359 GFLOP/s single-core FMA
  ceiling.** The micro-kernel is not the problem. It is within a few percent of what ONNX Runtime achieves
  across its whole model.
- **The gather term is 0.964 ns per element, about 4.8 cycles at 5 GHz** — the price of a scalar loop that
  reads one element at a time behind two bounds comparisons.

**Across VGG-16 the gather is 81.7 M elements = 78.8 ms of the 204.4 ms single-core total, or 39%.**

**And it does not have to be a gather.** At `stride = 1`, for a fixed `(ky, kx)` consecutive output positions
read **consecutive input addresses** — a contiguous run, not a scatter of indices. That is one `Vector512`
load and store where the current code runs 32 scalar iterations with a branch each. Edge handling stays
scalar; the interior, which is nearly all of it on a 224x224 input, does not.

**Sizing it honestly**: a 4x faster gather removes 59 ms of 204, giving 145 ms against ONNX Runtime's 105 —
**2.00x becomes 1.38x**. That is an estimate from the fitted coefficient, not a measurement of a fix that
exists, and the achievable speedup of the vectorised form has not been measured.

### The gap to ONNX Runtime is 2.00x per core, and our scaling is slightly BETTER than theirs (2026-08-19)

**This measurement overturns the conclusion recorded earlier the same day, and the reason it does is worth
more than the number.** The earlier reading compared our single-core figure against the *machine's*
theoretical all-core FMA ratio and concluded the remaining gap was parallel scaling. ONNX Runtime's own
single-core figure was never measured. It is the only comparison that could have settled it.

`Scripts/ProfHarness` runs both engines through the same loop, the same warmup and the same affinity mask,
with ONNX Runtime's `IntraOpNumThreads` set explicitly to the same core budget.

| cores | Overfit | speedup | ONNX Runtime | speedup | ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 209.71 ms | 1.00x | **104.96 ms** | 1.00x | **2.00x** |
| 2 | 111.00 | 1.89x | 56.80 | 1.85x | 1.95x |
| 4 | 64.22 | 3.27x | 33.43 | 3.14x | 1.92x |
| 8 | 40.46 | 5.18x | 21.43 | 4.90x | 1.89x |
| 16 | 31.14 | **6.74x** | 15.72 | **6.68x** | 1.98x |

**Our scaling matches theirs and is marginally better at every rung.** ONNX Runtime scales exactly as badly
as we do — 6.68x across 16 cores against the machine's 14.30x. **The whole difference is per-core work, it
is 2.00x, and it is flat.**

> **ONNX Runtime runs at 82% of this box's single-core FMA ceiling (294 of 359 GFLOP/s). We run at 41%
> (147 GFLOP/s).** That is the entire problem stated in one line, and it is not a threading problem.

**It also explains the day's failures as one pattern.** The 128-column N-block, the M-split and the shared
panel expansion all targeted parallelism, and all three lost. The three changes that worked — the register
spill fix (3.46x on the kernel), the MR-major pack and the Relu fusion — all reduced per-core work. The
selection was measured each time; the *reason* only became visible here.

**What this retires.** `XC-85`'s premise for the small-N layers is refuted: they are not short of work
items. 448 balanced items in place of 7 changed their rate from 1028 to 955 GFLOP/s, which is what "not
core-starved" looks like. **And a second hypothesis died with it** — those layers are not weight-bandwidth
bound either: this is a Ryzen 9 9950X3D with **128 MB of L3**, so a 9.44 MB weight matrix never leaves
cache. Both were arithmetic that sounded right and neither survived a measurement.

**Machine, for the record**: AMD Ryzen 9 9950X3D, 16 physical cores / 32 threads, L2 1 MB per core, L3
128 MB across two CCDs (96 MB V-Cache + 32 MB). **The V-Cache asymmetry does not matter here**: eight cores
on the V-Cache CCD run VGG-16 in 40.88 ms against 40.33 ms on the other one — 1.3%, and in favour of the CCD
*without* it. **SMT does not help either**: 16 workers on 16 cores is 31.22 ms against 31.55 ms for 32
workers, so `OverfitParallel`'s `Environment.ProcessorCount` pool is twice the useful size.

### Folding Relu into the convolution epilogue: -5.7% on VGG-16, -8.4% on the 60.9 MB CNN (2026-08-19)

`OVERFIT_FUSE_CONV_RELU=0` restores the separate node. ABAB, both models, ONNX Runtime in the same process.

| model | Relu as its own node | fused | change |
|---|---:|---:|---:|
| VGG-16 | 34.66 / 34.68 ms | **32.54 / 32.82 ms** | **-1.99 ms (-5.7%)** |
| CNN, 60.9 MB | 25.69 / 25.50 ms | **23.62 / 23.27 ms** | **-2.15 ms (-8.4%)** |

**The canary moved the way that cannot fake this result.** ONNX Runtime was **1.9% slower** in the fused
arms (18.84 -> 19.18 ms). A box drifting in our favour would have sped the canary up alongside us; it did
the opposite, so the reading is real and if anything understated. Parity unchanged in all eight runs.

**Predicted 1.7 ms, measured 1.99.** The prediction counted only the removal of the Relu pass. Bias and
clamp now share one traversal as well, and that second saving was not in the estimate.

**Why fusion and not threads** is in the per-layer section below: the Relu nodes were at **65.9 GB/s against
a 90.8 GB/s DRAM ceiling**, so they were never short of parallelism. Only checking the unit showed that; the
1.08x speedup on its own says the opposite.

**Three mutations, all caught** - the clamp removed from the epilogue (caught by
`AFusedConvolution_ClampsItsOwnOutput` and `Fusing_DoesNotChangeTheOutput`), the second-reader guard
weakened to `readers >= 1` (`AConvolutionWhoseOutputHasASecondReader_IsNotFused`), and the `Relu` node left
in place beside a convolution that now clamps (`AConvolutionWhoseOutputHasOneReader_IsFused` and
`Importing_TheMnistCnn_FoldsReluIntoConvolution`). **The third mutation is the one worth having a test for**:
it produces the *right answer* - `max(0, max(0, x))` is `max(0, x)` - so no parity check anywhere could
see it. Suite 2738/0 in both arms of the switch.

### Where the remaining gap to ONNX Runtime actually is: parallel scaling, not the kernel (2026-08-19)

Measured with `artifacts/prof` — one process, one workload, no BenchmarkDotNet host/child split, 40 warmup
calls then ten seconds of steady state. It reproduces the benchmark to within 1% (34.83 against 34.51 ms),
so it is the same subject seen with a cleaner instrument. Thread count set with `DOTNET_PROCESSOR_COUNT`
and **read back out of the process** before each reading is used.

#### VGG-16 whole model

| threads | ms/call | speedup | efficiency |
|---:|---:|---:|---:|
| 1 | 198.55 | 1.00x | 100% |
| 2 | 107.73 | 1.84x | 92% |
| 4 | 63.91 | 3.11x | 78% |
| 8 | 42.64 | 4.66x | 58% |
| 16 | 34.80 | 5.70x | 36% |
| 32 | 34.56 | **5.74x** | 18% |

**16 to 32 threads buys 0.7%.** The machine's measured single-core-to-all-core FMA ratio is **14.30x**
(359 to 5136 GFLOP/s); this reaches **5.74x**, which is **40% of the available scaling**.

> **The arithmetic that reframes the whole effort.** At 198.55 ms on one thread, scaling at the machine's
> 14.30x would put VGG-16 at **13.9 ms — ahead of ONNX Runtime's 18.9 ms**. It sits at 34.56.
> **The single-thread kernel is competitive; every remaining millisecond of the 1.8x gap is scaling.**
> The register-spill fix, the packing and the tiling were all real wins - and the work that is left is not
> in the micro-kernel at all.

#### Per operator

| operator | 1 thread | 4 | 32 | speedup | share at 32 |
|---|---:|---:|---:|---:|---:|
| ConvLayer (13) | 181.66 | 54.83 | 22.50 | 8.07x | 65.2% |
| LinearLayer (3) | 11.11 | 8.92 | **9.12** | **1.22x** | 26.4% |
| ReluActivation (15) | 1.85 | 1.78 | 1.72 | **1.08x** | 5.0% |
| MaxPool2DLayer (5) | 5.07 | 1.71 | 1.13 | 4.49x | 3.3% |

`LinearLayer` is **slower at 32 threads than at 4**. That is what a bandwidth-bound operator looks like once
the extra threads only add contention.

#### Per layer, and the four separate defects it separates

| node | shape | 1 thread | 32 | speedup | GFLOP/s at 32 |
|---|---|---:|---:|---:|---:|
| 33 | Linear fc1 | 8.42 | **7.42** | **1.13x** | - |
| 2 | Conv 224^2 K=576 | 41.22 | **3.99** | 10.33x | **927** |
| 24/26/28 | Conv 14^2 K=4608 | 13.00 | **3.32** | **3.9x** | **830** |
| 19/21 | Conv 28^2 K=4608 | 30.51 | 3.67 | 8.3x | **2077** (best) |
| 0 | Conv 224^2 K=27 | 3.06 | **1.16** | 2.64x | **149** |
| 1..29 | ReLU x15 | 1.85 | 1.72 | **1.00x** | - |

**Checking the unit reversed the ReLU conclusion, which is the reason the rule exists.** A 1.00x speedup
reads as "not parallelised". Node 1 is 64x224x224 = 3.21 M elements, so it reads 12.8 MB and writes 12.8 MB
in 0.39 ms = **65.9 GB/s against a 90.8 GB/s DRAM ceiling**. ReLU is **not** an unparallelised compute
operator - it is at 73% of the memory ceiling and parallelising it can return almost nothing. The available
fix is the opposite one: **fuse it into the convolution's epilogue** so the tensor is written once instead of
written, read and written again.

`fc1` is the same shape of problem: 102.8 M parameters = **411 MB read per inference**, and
411 MB / 7.42 ms = **55.4 GB/s** against the same 90.8 GB/s ceiling.

Nodes 24/26/28 have only **196 output columns** - 6.1 panels of 32 - and reach 830 GFLOP/s where node 21,
the same K on a larger output, reaches 2077. Node 0 has **K=27**, so the micro-kernel's inner sweep is 27
deep and setup dominates: 149 GFLOP/s.

### The 128-column N-block: refuted, and it is the same half-structure mistake twice (2026-08-19)

MLAS packs a **128-column** B panel and sweeps all of M through it; this kernel packs **32**. The worker was
restructured to gather `ConvNBlock * 32` columns per work item and loop M outer, sub-panel inner, so the same
eight rows of A serve every sub-panel back to back.

| N-block | VGG-16 | against 32 | ONNX Runtime canary |
|---|---:|---:|---:|
| 32 (1 sub-panel) | **34.51 / 34.55 ms** | - | 19.00 / 18.80 ms |
| 64 (2) | 38.68 ms | **+12%** | 18.92 ms |
| 128 (4) | 50.88 ms | **+47%** | 18.97 ms |

**Monotonically worse, and the arithmetic should have been done before the code.** The packed B panel grows
from `K x 32` to `K x 128` — **589 KB to 2.36 MB** — against a 1 MB L2. At 32 columns the panel fits; at 128
it does not, and every micro-kernel call streams it from L3.

**MLAS's panel is 64 KB because it blocks K at 128 at the same time.** Widening N without blocking K blows
the cache budget, and the curve's slope is that budget being exceeded further.

> **This is the second time half of this structure has been built and lost.** K-blocking without A-packing
> lost; the N-block without K-blocking lost. In BLIS and MLAS these are **one construction with four coupled
> parameters** - MR/NR, KC, MC, NC - chosen so that three products land in three cache levels: `KC x NR` in
> L1, `MC x KC` in L2, `KC x NC` in L3. **Any subset breaks the balance it exists to hold.** Either all four
> move together or none should.

The switch stays, defaulting to 1, so the measurement is reproducible rather than re-derived.

### `XC-82` re-measured: memory-only, and ONNX Runtime moved again (2026-08-19)

Removing the dense layer's unread output-major weight copy touches VGG-16's fully-connected layers, so the
figures published before it were unconfirmed on the current tree. Re-measured, two runs per model, ONNX
Runtime in the same process:

| model | Overfit before | after | ORT before | after |
|---|---:|---:|---:|---:|
| VGG-16 | 33.41 ms | **32.98 ms** (-1.3%) | 19.09 ms | **18.28 ms** (-4.2%) |
| CNN, 60.9 MB | 24.13 ms | **24.12 ms** (-0.1%) | 9.78 ms | **9.19 ms** (-6.0%) |

**`XC-82` is time-neutral.** Overfit did not move: -1.3% and -0.1% are inside the run-to-run spread. It was a
memory change - 592 MiB of managed heap - and it cost nothing in time, which is what the change claimed.

**The ratios got worse anyway, and only because ONNX Runtime got faster.** VGG-16 reads 1.75x -> 1.80x and the
60.9 MB CNN 2.47x -> 2.63x, from a denominator that dropped 4-6% on a binary nobody touched.

**That is the third sitting in which ONNX Runtime's large-CNN figure has moved on its own**: 12.93, then 9.79,
then 9.19 ms - a **29% spread**. Ours moved 0.1% between the last two. **What moves it is still not
identified**, and until it is, a large-CNN ratio quoted from one sitting cannot be compared with one quoted
from another.

> **A ratio has two operands and the other one is not under your control.** When a published number worsens,
> check whose half moved before attributing it to your own change.

### The pooled panel rent, measured — and the batch anomaly resolves by elimination (2026-08-18)

The last untested candidate for the batch anomaly was the convolution worker's packed-panel rent: `K * 32`
floats, **147,456 floats or 589 KB**, once per worker per dispatch, about **18.9 MB across 32 workers per
convolution call**. 589 KB is well over the 85,000-byte large-object threshold, so a pool miss would be a
large-object allocation on every convolution from every worker.

| arm | mean | per unit | allocated |
|---|---:|---:|---:|
| single thread, 64 rents | 2.004 us | **31.3 ns** per rent+return | **none** |
| parallel, dispatch only | 380.4 us | 5.94 us per dispatch | none |
| parallel, every worker rents | 399.7 us | 6.24 us per dispatch | none |

**The pool hits.** Nothing is allocated at this size, so there is no large-object path. The parallel
difference is 19.2 us across 2,048 rents - about 9 ns each - and it sits **inside error bars of 25-37 us**.
Against 0.79 ms of real work per image, 18.9 MB of rents costs at most 0.3 us. **Not the mechanism.**

**One small correction to a documented figure.** `PooledBuffer`'s remarks record "Rent+Return on
ArrayPool.Shared ~4 ns regardless of size". At 589 KB single-threaded it is **31 ns** - eight times that.
Still negligible, but "regardless of size" is not exact.

### So what the batch anomaly was

By elimination, with each step measured rather than argued:

| candidate | verdict |
|---|---|
| arithmetic, cache, algorithm | **eliminated** - the effect vanishes entirely at one worker |
| parallel dispatch, hot | 5.51 us |
| parallel dispatch, cold wake | 12.4 us penalty - 130x too small |
| pooled panel rent | ~0.3 us per call, no allocation |
| BenchmarkDotNet overhead on sub-millisecond calls | **the only candidate left** |

That last one fits every observation. At one worker a call takes 6.7 ms and the harness overhead is
invisible - and there batching does nothing at all (6.729 against 6.774 ms per image across an eightfold
batch). At thirty-two workers a single image is 0.79 ms of work, and the anomaly appears exactly where
per-call time falls below a millisecond. The non-monotonic points - batch 2 slower than batch 4 - are what a
fixed per-iteration cost looks like when it is comparable to the work.

> **A "gain" that disappears when you remove the parallelism, and that no component in the parallel path can
> account for, is a property of the instrument.** Four candidates were measured and eliminated to reach that;
> none of them needed to be guessed.

**Nothing to build.** Batched graph inference is not worth writing, and neither is a cheaper dispatch: the
effect that motivated both was the harness.

### The dispatch cost, measured — and the direction it was supposed to open is closed (2026-08-18)

The section below concluded that the batch anomaly was per-dispatch overhead and named reducing it as the
lever. **That was wrong, and measuring the dispatch is what showed it.**

`ParallelDispatchCostBenchmark` times one `OverfitParallel.For` with a trivial body against sixty-four of
them chained inside a single measured call, so only the first of the chain can find the pool parked:

| arm | mean | per dispatch |
|---|---:|---:|
| `Single` (the pool has an iteration gap to park in) | 17.86 us | **17.86 us** |
| `Chained`, 64 back to back | 352.61 us | **5.51 us** |

**Two results, and the second kills the plan.**

**The hot dispatch costs 5.51 us, which validates a published claim.** `OverfitParallel`'s own remarks say
the bulk-semaphore wake brought dispatch "from 32-47 us to ~5 us". Measured independently here: 5.51 us. The
documentation is correct.

**The cold-wake penalty is 12.4 us, and that is far too small to be what the batch sweep saw.** The batch
measurement implied roughly **1.6 ms** of per-call overhead at Hw=14. The parking penalty is **12 us** -
smaller by a factor of 130. Against 0.79 ms of real work per image, dispatch is negligible hot or cold.

> **The benchmark's own refutation clause fired.** It was written to say that if hot and cold dispatch cost
> the same, parking was not what the batch sweep measured. They differ - and both are so far below the
> effect that the conclusion is the same: **parking is not the mechanism.**

**So "reduce the dispatch cost" is closed as a direction.** There is nothing to win: 5.5 us for a 32-way
fan-out is already good, and the entire parking penalty is 12 us once after idleness. Porting the decode
pool's spin-park protocol - measured there at +23-25% - would burn cores at idle to win microseconds here.

**The batch anomaly stays unexplained, but the elimination is now substantial**: not arithmetic, not cache,
not the algorithm (it vanishes at one worker), and not dispatch. What remains untested is the per-worker
`PooledBuffer` rent of `k * 32` floats - 589 KB per worker, about 18.9 MB per convolution call at 32 workers
- and BenchmarkDotNet's own overhead on sub-millisecond calls. **Neither has been measured, and neither
should be assumed.**

### The batch "gain" was per-dispatch overhead, and one worker proved it (2026-08-18, `XC-78`)

The section below measured about 3x less time per image at batch 4 on an occupancy-starved shape, and
recorded that **the mechanism was not established** because the implementation processes images
independently. It is established now, and it removes the result.

**The question was posed backwards.** At Hw=14 a single image takes 2.429 ms and four take 3.103 ms in
total. That is not "batching is fast" — it is **"one image is slow"**. In absolute terms batch 1 runs at
381 GFLOP/s and batch 4 at 1192.

**A worker-count sweep answers it outright**, per image:

| workers | batch 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| **1** | 6.729 ms | 6.735 | 6.669 | 6.774 |
| 8 | 3.980 | 1.139 | 1.150 | 1.157 |
| 32 | 2.409 | 1.786 | 0.791 | 0.807 |

**At one worker batching does nothing at all** — 6.729 against 6.774 across an eightfold batch, inside a 0.3%
error. The entire effect lives in the parallel layer: not in arithmetic, not in cache, not in the algorithm.

> **Batching amortises nothing here; it dilutes a per-dispatch cost.** At one worker there are 6.7 ms of work
> to hide it in and nothing to gain. At thirty-two there are 0.8 ms of work per image and the overhead
> dominates. **The lever is the dispatch cost, not the batch.**

**What was NOT separated, and it matters for anyone acting on this.** At 32 workers batch 2 measures 3.572 ms
and batch 4 measures **3.162 ms** — less time for twice the work, which is internally inconsistent, so those
two points are not usable. Three candidates remain unseparated: the fixed `OverfitParallel.For` cost
(~0.22 ms measured elsewhere), scheduling quantisation of 35 work items across 32 workers, and
BenchmarkDotNet's own per-iteration overhead on sub-millisecond calls. **The one-worker row is what carries
the conclusion**, because it is long enough per call for all three to be negligible.

**Consequence for the roadmap: batched graph inference is not worth building.** The measured gain is an
artefact of an overhead a correct implementation would still pay per image. What the numbers do point at is
reducing the per-dispatch cost, or dispatching once for several panels rather than once per convolution —
neither of which needs a batch API.

### Batched convolution: measured before building, and the answer is "not yet" (2026-08-18, `XC-78`)

**It is a capability, not a tuning knob.** `OnnxGraphModel.RunInference` throws unless
`input.Length == _inputSize`, and every intermediate buffer is sized for one image. Supporting a batch means
resizing those buffers, propagating a batch size through every node, and roughly 12.8 MB per extra image for
VGG-16's largest activation.

**And the usual justification does not survive arithmetic here.** "Batching raises arithmetic intensity
because the weights are reused" is false for this loop order: A is read once per N-panel and there are
`N / 32` panels, so a batch of B gives B times the panels and **B times the A traffic, exactly
proportional**. Nothing is amortised. That claim was made in an earlier proposal in this repository and it
was wrong.

**Measured on the path that already accepts a batch** (`Conv2DKernels.ForwardNchw`), per image:

| shape | batch 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| Hw=28, N=784, 25 panels | 1.482 ms | 1.480 | 1.562 | 1.556 |
| Hw=14, N=196, 7 panels | 2.429 ms | 1.763 | **0.776** | 0.797 |

**Flat where the shape is well conditioned; about 3x per image where panels are scarce, saturating at
batch 4.** On VGG-16 the second case is conv11-13, about 16% of the model, so the ceiling on this whole
direction is around 10% — at batch 4 or more, for throughput workloads only.

**The mechanism for the Hw=14 gain is NOT established, and that is a reason to wait.** The implementation
processes batch items independently — one `GemmFusedIm2Col` call each, its own dispatch, its own gathered
panels — so on the face of it a batch should change nothing. It changes a lot. Until that is explained, the
gain cannot be assumed to survive a real batched implementation.

**The first version of this measurement was worthless, and the reason generalises.** At batch 1 and Hw=28 it
reported **5.929 ms**; with more warmup the same call measures **1.482 ms**. The distribution was bimodal
with modes at roughly 6.5 ms and 21 ms — a 3.2x split, the signature of tier-0 code still being sampled.
**The project's shared benchmark config uses five warmup iterations with one invocation each**, which is not
enough for calls of this length. Pinning affinity did not help, because this was never a placement problem.

> **Check the warmup before believing a long-running benchmark.** Five iterations at one invocation each is
> five calls; a method that needs more than that to tier up will be measured cold, and the tell is a bimodal
> distribution whose modes differ by roughly the tier-0 penalty.

**Recommendation: do not build batched graph inference on this evidence.** The payoff is bounded at about
10% of one model, confined to one layer shape, and its mechanism is unexplained.

### Fixing the fallback kernels, and what a broken off-arm was hiding (2026-08-18, `XC-78`)

The full-tile split had been applied only to the packed AVX-512 kernel. Two other bodies remained.

**One of them needed nothing, and reading it saved half the work.** The AVX2 path already has the split:
`GemmNPanelWorker` chooses between `MicroKernel8x8` and `MicroKernelTail` on `mrEff == Mr`. **The AVX2
kernel was written correctly and the AVX-512 kernel, added later, lost the pattern.**

**The other one matters for a reason that is not performance.** `MicroKernel8x32Avx512` is reached only with
`OVERFIT_CONV_PACK_A=0` or `OVERFIT_CONV_FUSED_IM2COL=0` - the A/B switches themselves. Production never
takes it. But left unfixed, `PACK_A=0` measures *"no packing AND a spilling kernel"*.

> **A switch whose off-arm is broken makes its own measurement dishonest.** It is the same trap the prefetch
> experiment fell into hours earlier, in the opposite direction.

**Measured on VGG-16, all cores:**

| configuration | before the fix | after |
|---|---:|---:|
| default | 33.20 ms | 33.42 ms |
| `OVERFIT_CONV_PACK_A=0` | **54.95 ms** | **39.13 ms** (-28.8%) |
| `OVERFIT_CONV_FUSED_IM2COL=0` | - | 49.72 ms |
| AVX2 (`OVERFIT_CONV_AVX512=0`) | 74.06 ms | unchanged, already split |

**And that changes a published number.** With a broken off-arm the packing switch would have claimed -39%.
With both arms sound it is **39.13 -> 33.42 ms = -14.6%** — while the original measurement of the same
change, taken when *neither* arm had the full-tile body, said **-8.4%**. **The packing is worth nearly twice
what its own measurement reported**, because the spill was masking it on both sides.

> **A ratio between two arms is only as good as the worse arm.** Fixing an unrelated defect can move a
> number you already published, in either direction.

**Coverage.** Three mutations, all three caught under `OVERFIT_CONV_PACK_A=0` — the only configuration that
reaches this kernel. Two by wrong values, one by killing the test host. `ConvGemmMSplitTests`, which calls
`Gemm` directly, is what catches them.

### The partial-tile body, and an estimate that was wrong by twenty times (2026-08-18, `XC-78`)

The full-tile fast path left partial tiles - fewer than eight rows, or fewer than thirty-two columns - on the
general kernel, which still carries a `stackalloc` and a store helper and therefore still spills.

**The estimate said not to bother. It was wrong, and the error is the point.** Weighing partial tiles by
their OUTPUT - the last panel of VGG's conv11-13 produces 4 columns of 32 - gave 1.8% of those layers' work
and **0.55% of the model**. But **a partial tile runs the full K sweep**: 4,608 k-steps, exactly as many as a
full one. Only the store is shorter. Weighed by COST it is one tile in seven, **14.3%**.

> **Weigh an edge case by what it costs, not by what it produces.** A tile that emits an eighth of the
> columns still does all of the arithmetic.

**Measured**, both engines all cores, two runs each:

| model | before | after | change | ONNX Runtime canary |
|---|---:|---:|---:|---:|
| VGG-16 | 37.88 ms | **33.60 / 33.22 ms** | **-11.8%** | 18.90-19.27 ms |
| CNN, 60.9 MB | 26.75 ms | **24.11 / 24.16 ms** | **-9.8%** | 9.74-9.81 ms |

Against ONNX Runtime: VGG-16 **2.00x -> 1.75x**, the 60.9 MB CNN **2.72x -> 2.47x**. Parity
unchanged. Even 11.8% is more than the corrected 2.2% arithmetic predicts, so part of the gain is dispatch
overhead the estimate did not model at all.

**Three bodies, not one method with branches**, because the full-tile body sits at the register limit -
three prefetch instructions added to it measured 40% slower. Even the extra live parameters an edge case
needs are not free, so the selection happens at the call site. The partial body takes its scratch from the
caller, allocated once per worker: a `stackalloc` inside it would reintroduce exactly the frame this is
avoiding.

**Coverage.** Four mutations on the body, three caught (two by killing the test host on an out-of-bounds
store, one by wrong values). The fourth - routing full tiles to the partial body - escapes, **correctly**:
that body computes the same values, only more slowly, so it is an alternative rather than a defect. Routing
partial tiles to the full body is caught. **The first version of this change routed everything to the partial
body and the escaped mutation is what exposed it.**

### The conv micro-kernel was spilling its accumulators, and it cost 3.46x (2026-08-18, `XC-78`)

**The largest single finding of the investigation, and it is not in the memory hierarchy at all.**

**How it was isolated.** A rung was added to the cost ladder that inlines the micro-kernel's FMA sequence
into the benchmark method: same buffers, same addresses, same 1,600 iterations, same k = 4608, same MR-major
A layout, same contiguous B. **The only difference is whether the accumulator loop lives in its own method.**
Pinned, two runs each:

| arm | run 1 | run 2 | GFLOP/s | of the 359 GFLOP/s single-core peak |
|---|---:|---:|---:|---:|
| inlined | 11.31 ms | 11.25 ms | **335** | **93%** |
| calling the production kernel | 39.19 ms | 38.87 ms | 97 | 27% |

**3.46x, from removing the call alone**, both arms stable to 0.8%.

**Why.** Sixteen `Vector512` accumulators need sixteen zmm registers. In a method that also carries nine
parameters, a `stackalloc` scratch tile and eight `StoreTile` calls, the register allocator does not keep
them there. The fix is a full-tile body holding nothing but the accumulators, with the general kernel
routing `mrEff == 8 && nrEff == 32` to it and keeping the edge-case machinery for everything else.

**This is why every memory hypothesis measured null.** Cache blocking, software prefetch, tile shape and
working-set capacity were each tested here and each moved nothing — **because none of them touches register
allocation**. Most of a day of memory-hierarchy hypotheses was spent on a problem in the generated code.

> **When several independent hypotheses about one subsystem all measure null, the shared premise is the
> suspect.** Here the shared premise was "the gap is in operand delivery", and it was wrong.

**Measured end to end**, both engines all cores:

| model | before | after | change | ONNX Runtime canary |
|---|---:|---:|---:|---:|
| VGG-16 | 50.96 ms | **37.83 / 37.92 ms** | **-25.7%** | 18.94-18.98 ms |
| CNN, 60.9 MB | 41.95 ms | **27.11 / 26.40 ms** | **-36.2%** | 9.79-9.88 ms |

Against ONNX Runtime: VGG-16 **2.77x -> 2.00x**, the 60.9 MB CNN **4.49x -> 2.72x**. Parity unchanged
in every arm.

**Coverage, and it exposed a gap plus a broken harness.** Three mutations were applied. Two produce wrong
values and are caught. The third — routing partial tiles through the full-tile body, which then stores eight
rows of thirty-two columns outside the valid region — **was reported as ESCAPED twice, and that was the
harness lying**: the out-of-bounds store kills the test host, so no `[FAIL]` line is ever printed, and a
harness that looks only for `[FAIL]` reads a dead process as a clean pass. **The canary cannot catch this,
because the unmutated baseline "escapes" in exactly the same way.** The harness now requires proof that the
suite ran — a summary line and a plausible test count — and reports `NO RUN` or `PARTIAL RUN` otherwise. With
that in place the third mutation reports "only 141 tests passed".

**And no test produced a partial tile at all**, so both the fast path's guard and the general path's edge
handling were unexercised. `Tests/Core/Kernels/ConvPartialTileTests` now covers five shapes chosen against
the micro-tile — output channels not a multiple of 8, output positions not a multiple of 32, and both — each
against a direct convolution written in the test, with a guard band that catches a store running past the
output.

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
