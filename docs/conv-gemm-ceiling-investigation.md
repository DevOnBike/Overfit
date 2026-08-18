# Why the convolution GEMM is far from the machine ceiling — an investigation, including its failures

*2026-08-18. Companion to `XC-78` in [`docs/TASKS.md`](TASKS.md); the raw numbers live in
[`docs/measured-baselines.md`](measured-baselines.md) and `artifacts/`.*

This file records an investigation that produced **one shipped win, three refuted hypotheses, one
self-inflicted arithmetic error, and one instrument that turned out not to work.** It is written out in full
because the failures are the part that transfers: the shipped win took an afternoon and the reasoning that
led to it is ordinary, while the three refutations and the broken instrument cost more and are the kind of
thing that is otherwise re-derived by the next person.

---

## The question

After five optimisations, VGG-16 inference runs at **50.7 ms against ONNX Runtime's 18.3 ms** — 2.77x
behind. Convolution is about 70% of that time. The question is why the convolution kernel is so far from
what the silicon can do, and what would close it.

## The first thing to get right: what "the ceiling" actually is

**It was wrong, for most of a day, and every percentage computed against it was about 2.35x too generous.**

The figure in use was **2190 GFLOP/s** all-core, carried forward from an earlier note rather than
recomputed. It is not this machine's AVX-512 ceiling.

Derived from the benchmark's own work unit instead: `MachineRooflineBenchmark.FmaChains512` issues **12 FMAs
per iteration over 2,000,000 iterations**, each 512-bit and therefore 32 FLOP — **768 MFLOP per worker**.

| arm | measured time | throughput |
|---|---:|---:|
| 1 worker | 2.139 ms | **359 GFLOP/s** |
| 32 workers, 32x the work | 4.785 ms | **5136 GFLOP/s** |

**How it was caught, and why nothing else could have caught it.** A percentage of a wrong ceiling still
looks like a percentage — no downstream result contradicts it, because everything downstream is smaller.
It surfaced only when a separate benchmark reported the 8x32 micro-kernel at **340 GFLOP/s single-core**,
which would have been **222% of the 153 GFLOP/s single-core figure then in use.**

> **A rate that exceeds its own ceiling deserves more attention than a rate that merely disappoints.** An
> impossible percentage is the only observation that can falsify a denominator.

The 14.30x ratio between the two arms was never affected, because it is a ratio of two measurements of the
same thing. Every conclusion drawn from *that* still stands. Only the absolute denominator was wrong.

## What was tried, in order

### 1. The patch gather folded into the GEMM's pack — SHIPPED, −13.7%

The unfused path built a `K x N` column matrix — **115.6 MB at VGG's conv2** — written once and read once,
to hold values readable straight from the 3.2 MB input. The gather alone measured 19.8% of convolution time.

MLAS does the same thing at block granularity (`MlasConvExpandThenGemmSegmented` expands `CountK x CountN`
into a column buffer). Ours goes one level finer: the destination is the micro-kernel's own packed panel, so
there is no intermediate buffer at all.

### 2. MR-major kernel packing — SHIPPED, −8.4%

BLIS states the contract outright (`docs/KernelsHowTo.md`): A's micropanel is *"stored by columns with
leading dimension PACKMR"*, so the MR values one k-step consumes are adjacent. Ours read `rows[r][kk]` —
eight floats `k * 4` bytes apart, **18 KB at K = 4608**.

**The transferable part is not the fix but why it was available to us and not to BLIS.** BLIS packs A on
every call because it is a general GEMM and A is caller data. **In inference A is the convolution's own
weights, which never change** — so the pack is a load-time cost. Measured both ways: packing per call gave
−4.1%, packing at load gave **−8.4%**, almost exactly double, which is what had to happen if the pack cost
roughly equalled the gain.

> **Reading a competitor's source is worth most where it exposes a premise you do not share.** The same
> premise difference retires an earlier negative recorded in this repository for "BLIS-style blocking and
> packing": that experiment paid the packing cost every call.

### 3. K-blocking — REFUTED

MLAS blocks both dimensions at 128, keeping its packed B panel a constant 64 KB. Ours contracts the whole of
K, so the panel is `K x 32` = 589 KB at K = 4608. A specific, cheap hypothesis.

Swept on VGG-16 with an unblocked run at each end of the sweep: off 56.18 / 56.12 ms (agreeing to 0.1%),
Kc=64 **+5.2%**, Kc=128 +0.8%, Kc=256 −1.4%, Kc=512 −0.7%, Kc=1024 **−2.1%**.

**The ONNX Runtime canary moved 2.1% across the same runs, so the best result is the size of the
instrument's own error. Verdict: not resolved, not "a small win".** The shape of the curve is the finding —
the gain rises monotonically toward larger Kc and the best value is the one nearest to no blocking. That is
the C re-accumulation cost dominating: contracting the whole of K keeps the C tile in registers so C is
written once, while blocking turns conv6's 3.2 MB of C traffic into about 115 MB.

### 4. Panel grouping and BLIS blocking-with-packing — REFUTED earlier, and one of the two reasons was false

Both were already measured negative in this repository before today. Worth recording that **one of the two
recorded reasons did not survive checking**: the BLIS negative was justified by *"most im2col K values are
at most a few hundred, so a single K-block means no blocking benefit"*, and **VGG-16's K runs 576 to 4608**.
The result stood; its stated reason did not. That is why hypothesis 3 was worth testing again despite
"already refuted" being on the record.

> **A refuted result and a refuted explanation are different objects.** Re-read the explanation before
> letting an old negative close a question.

## The instrument that did not work

To attribute the remaining gap, a cost ladder was built: the same micro-kernel, the same FLOP count, one
surrounding cost added per rung — real A, then real B, then real C, then production.

**It produced a backwards result on its first run** (a 9.4 MB working set measuring *faster* than a 736 KB
one), which this project's rules say to blame on the benchmark before the silicon. That was correct: the
first rung was mis-designed, issuing 36x more C stores per FLOP than the others.

**After the fix it produced a different problem, and this one is fatal to the instrument.** Two runs of the
same code, minutes apart:

| rung | run 2 | run 3 |
|---|---:|---:|
| L1 — minimal working set | 97.2 | 97.0 |
| L1b — streaming, 590 KB | 96.4 | **134.9** |
| L2 — real A, 9.4 MB | 95.8 | 96.6 |
| L3 — real B, 14.7 MB | 95.0 | **117.8** |
| L4 — real C, 1.6 MB | 94.9 | **117.2** |
| L5 — production | 62.8 | **70.6** |

Some rungs moved 24-40%, others not at all. **The inter-rung comparisons are therefore not measurements of
what they claim to measure.**

**The first explanation offered for this was wrong, and it is left here because the correction is the
point.** The claim was that the rungs share buffers, so each warms the caches for the next. They do share
buffers - but `BenchmarkConfig` uses `Job.Default`, and **BenchmarkDotNet's default toolchain runs every
benchmark case in its own process**. Separate address spaces cannot inherit each other's cache state. The
explanation was plausible, fitted the data, and was refuted by reading twenty lines of the config.

**A controlled pair then made the failure sharper rather than explaining it away.** Two runs of the ladder
back to back, nothing else on the box:

| rung | run A | run B | spread |
|---|---:|---:|---:|
| L0 — L1-resident | 131.4 | 129.1 | **1.8%** |
| L2 — real A | 96.5 | 96.6 | **0.1%** |
| L3 — real B | 95.1 | 106.9 | 12.4% |
| L1 — full K | 97.1 | 134.7 | 38.6% |
| L1b — streaming | 97.0 | 134.3 | 38.4% |
| L4 — real C | 117.6 | 94.5 | 24.5% |
| L5 — production | 100.1 | 60.9 | **64.2%** |

**This rules out box drift**: L0 and L2 are stable to 1.8% and 0.1% inside the same pair, so the machine was
not warming up or clocking down under the others. **The variance is per arm, not per run.** And the values
are not scattered - they sit in two discrete states (97 or 134; 61 or 100), which is the signature of a
binary condition rather than of noise.

**The obvious suspect on this particular part.** A Ryzen 9 9950X3D has two CCDs with different L3 capacity
and different maximum clocks, and a single-threaded process lands on one or the other. That is testable by
pinning affinity, and it matters far beyond this ladder: **every single-threaded measurement ever taken on
this box is subject to it.**

**What actually causes it is not established.** The remaining candidates are thermal or frequency drift
across a long multi-process run, and something inside the arms themselves. Between the two runs above,
other benchmarks were executed on the same box, so the runs are not a controlled pair either.

> **An instrument that disagrees with itself by 24-40% cannot attribute a 1.5x effect**, whatever the
> cause. Establishing its reproducibility is a prerequisite, not a refinement - and a plausible cause is
> not the same as a measured one.

**What survives from it**, because it repeated across both runs:

- **L1 and L2 are stable at ~97 GFLOP/s.** Growing the A working set from 736 KB to 9.4 MB costs nothing.
- **Production (L5) is the slowest rung in both runs**, at 62.8 and 70.6 against the hand-issued kernel's
  ~97-117. The wrapper — patch gather, panel setup, dispatch — costs somewhere between 1.4x and 1.7x.
- **The L1-resident rung reached 130.7 GFLOP/s** against ~97 for the same call count at full K, in the one
  run where it exists.

What does **not** survive: any claim that the memory hierarchy beyond L1 is free, and any specific
attribution between the rungs.

### The CCD hypothesis, tested and refuted — but the test found the fix anyway

The two discrete states looked like CCD placement, so the ladder was pinned to each CCD in turn, twice each.

| rung | CCD0 run 1 | CCD0 run 2 | CCD1 run 1 | CCD1 run 2 |
|---|---:|---:|---:|---:|
| L0 — L1-resident | 92.7 | 92.0 | 93.3 | 91.1 |
| L1 — full K | **135.2** | **96.9** | **96.5** | **134.3** |
| L1b — streaming | 134.7 | 133.6 | **96.6** | 132.5 |
| L2 — real A | 97.0 | 96.7 | 96.9 | 96.2 |
| L3 — real B | 117.1 | 95.0 | 116.6 | 95.0 |
| L4 — real C | 95.0 | 106.3 | 95.4 | 106.6 |
| L5 — production | 70.4 | 70.3 | 70.4 | 70.5 |

**Refuted: the bimodality happens within a single CCD.** L1 measured both 135.2 and 96.9 on CCD0 alone, and
both 96.5 and 134.3 on CCD1 alone. Placement is not the variable.

**But pinning stabilised three of the seven arms, including the only one that matters.** L5, the production
rung, moved 60.9 to 100.1 unpinned - a 64% spread - and reads **70.3 to 70.5 across all four pinned runs**.
L0 and L2 are likewise stable. So affinity was never the *cause*, and it is still the *fix* for the arms
that measure something real.

> **A refuted hypothesis can still leave a working procedure behind.** The CCD explanation was wrong and the
> experiment was worth running: **pin affinity before taking any single-threaded measurement on this box.**

**What remains bimodal, and the leading candidate.** L1, L1b, L3 and L4 still flip between two states within
one CCD. What separates them from the stable arms is that each hammers a small, fixed set of buffers - L1
reuses exactly one A block and one B panel 1,600 times - while L0's working set fits L1d entirely and L2 and
L5 spread their accesses over many addresses. **That points at allocation address and cache-set aliasing**:
two buffers that happen to alias in the L2 sets on one run and not on the next. It is a candidate, not a
finding; the way to settle it is to offset the buffers deliberately and see whether the two states can be
selected on demand.

**The operational consequence is already usable, whatever the cause.** The production rung is reproducible
to 0.3% once pinned, which is enough to attribute the wrapper cost - and the arms that are not reproducible
are the synthetic ones, not the one that measures the real code.

### Attribution of the production wrapper, and a third broken arm

With affinity pinned, two ablation rungs were added to the ladder: production with the gather removed, and
production with the micro-kernel removed.

**One of the two is trustworthy and one is not.**

*Trustworthy:* the gather-only rung is stable to **0.1%** across runs and its time is **about 6% of
production**. It runs no FMAs, so nothing about it depends on operand values. **The patch gather is not
where the wrapper's cost is.**

*Not trustworthy:* the no-gather rung reads 65.2 and 66.3 GFLOP/s, which looks stable and is not usable.
Ablating the gather leaves the packed panel holding **uninitialised pool memory**, and denormals or NaNs in
an FMA chain are slow on this class of part. That arm may be measuring a denormal penalty rather than a
wrapper cost. **An ablation that removes the code producing a value must also define what that value is**,
or it measures whatever the allocator left behind.

So the split is: gather about 6%, micro-kernel sweep the rest, and **the wrapper beyond the gather is not
separable with the ablation as it stands.**

### The micro-kernel tile shape: refuted for issue rate, untested for traffic

MLAS and BLIS both hold **24 accumulators** on AVX-512 — MLAS's `FgemmKernelAvx512FCommon.inc` declares
`zmm4-zmm27`, BLIS's SKX configuration uses `MR=32, NR=12`. Ours holds 16, leaving 13 of the 32 zmm
registers unused. Per k-step that is 2 B loads plus 12 A broadcasts for 24 FMAs — **4.36 FLOP per byte
fetched against 3.20** for our 8x32.

A 12x32 arm was added to `GemmMicroKernelShapeBenchmark` and measured, pinned, twice:

| shape | accumulators | TFLOP/s |
|---|---:|---:|
| Avx512_8x32 (current) | 16 | 0.34 |
| Avx512_6x48 | 18 | 0.34 |
| **Avx512_12x32** (the MLAS/BLIS shape) | **24** | **0.34** |

**Identical, to 0.2% across runs.** Sixteen accumulators already saturate the FMA units; more buy nothing.

**But this benchmark cannot answer the question the wider tile was proposed for, and says so itself**: its
docstring records that the panels are "sized to sit in L1 so the result reflects instruction issue, not
memory bandwidth". A wider tile pays by fetching fewer bytes per FLOP, and this benchmark removes byte
fetching as a constraint by construction.

> **A shape that is neutral at the issue-rate limit may still pay at the traffic limit.** The refutation is
> real and it is narrow: tile shape is closed as an *issue-rate* question and untouched as a *traffic* one.
> Answering the second means testing 12x32 in the production kernel, not in L1.

**Pinning fixed this instrument too**: two runs agree to 0.2% on every arm, where the ladder unpinned
disagreed with itself by up to 64%.

### The measured 8.4% never reached a default build, and a null result is what caught it

`UsePackedA` shipped reading `Environment.GetEnvironmentVariable(...) == "1"` while the four other
convolution switches read `!= "0"`. **So the MR-major packing was opt-in**, the 8.4% it measures was
unreachable without setting a variable, and `README.md` was publishing a VGG-16 figure the library did not
produce as built.

**How it surfaced.** A prefetch sweep came back perfectly flat - 55.31, 55.22, 55.53, 55.58 ms across
distances of off, 4, 8 and 16. The prefetching micro-kernel is selected by
`if (UsePackedA && ConvPrefetchDistance > 0)`, so with packing off **the code under test never executed
once**. The sweep measured nothing and looked like a clean null.

**The tell was the baseline, not the flatness.** 55.3 ms is the unpacked number; the packed path measures
50.7. A flat sweep whose baseline does not match the configuration you believe you are in is a dead lever,
not a null result.

> **A flat result is the signature of a disconnected lever at least as often as of a real null.** This
> project already records the same failure from 2026-08-17, where `AblatePackB` existed only in the AVX2
> worker while the box ran AVX-512 and the diagnostic printed three numbers of pure noise. **Twice now, and
> both times the fix is the same: prove the lever moves something before believing that it does not.**

The default is corrected to `!= "0"`, matching the other four, and the re-run carries a deliberately absurd
prefetch distance as a liveness canary - prefetching far past the panel has to cost something if the path
executes at all.

### What is left from the sources

**Software prefetch, and it is the one technique read from MLAS that is still untested here.**
`FgemmKernelAvx512FCommon.inc` issues `prefetcht0` against the B panel at a tuned offset, two per block, on
every inner iteration. .NET exposes `Sse.Prefetch0`, so this is portable. Given that the kernel reaches 95%
of peak with L1-resident operands and a fraction of that in production, hiding operand latency is exactly
the lever the measurements point at.

### THE ANSWER: the micro-kernel was spilling its accumulators

Everything above searched the memory hierarchy. The gap was in the generated code.

A rung was added to the ladder that inlines the micro-kernel's FMA sequence into the benchmark method -
**same buffers, same addresses, same 1,600 iterations, same k, same layouts, the call the only difference**:

| arm | run 1 | run 2 | GFLOP/s | of the 359 single-core peak |
|---|---:|---:|---:|---:|
| inlined | 11.31 ms | 11.25 ms | **335** | **93%** |
| calling the production kernel | 39.19 ms | 38.87 ms | 97 | 27% |

**3.46x from removing the call**, both arms stable to 0.8%. Sixteen `Vector512` accumulators need sixteen
zmm registers; in a method carrying nine parameters, a `stackalloc` and eight store calls, the register
allocator does not keep them there.

**Shipped as a full-tile body** holding nothing but the accumulators, with the general kernel routing
`mrEff == 8 && nrEff == 32` to it. Measured end to end: **VGG-16 50.96 -> 37.88 ms (-25.7%)** and the
**60.9 MB CNN 41.95 -> 26.75 ms (-36.2%)**, parity unchanged, ORT canary steady. Against ONNX Runtime:
**2.77x -> 2.00x** and **4.49x -> 2.72x**.

> **When several independent hypotheses about one subsystem all measure null, suspect their shared premise.**
> Cache blocking, prefetch, tile shape and working-set capacity were each measured and each moved nothing -
> not because each was individually wrong, but because all four shared the assumption that the gap was in
> operand delivery. **Four nulls in a row is itself a measurement, and it was pointing at the premise.**

**A fourth broken instrument surfaced on the way, and it is the worst of the four.** The mutation that routes
partial tiles through the full-tile body stores outside the valid region and **kills the test host**. No
`[FAIL]` line is printed, so a harness that greps for `[FAIL]` reads a dead process as a clean pass and
reports ESCAPED. It did, twice. **The canary cannot catch it, because the unmutated baseline "escapes"
identically.** A mutation harness must require proof that the suite *ran* - a summary line and a plausible
test count - not merely the absence of failures.

## Where things actually stand

Restated on the corrected ceiling, single core:

| | GFLOP/s | of the 359 GFLOP/s single-core peak |
|---|---:|---:|
| 8x32 micro-kernel, L1-resident panels | 340 | **95%** |
| the same kernel at full K, hand-issued | ~97 | 27% |
| production convolution | ~63-71 | 18-20% |

And all-core, whole conv stack: **786 GFLOP/s = 15.3% of the 5136 GFLOP/s roofline**. ONNX Runtime's
whole-model rate is 30.94 GFLOP in 18.3 ms = **1690 GFLOP/s = 33%**.

**The one conclusion that is safe: the arithmetic is not the problem.** The micro-kernel reaches 95% of
single-core peak when its operands are in L1. Everything lost is lost in operand delivery and in the code
around the kernel — not in the FMA sequence.

## What to do next, and what not to

**Do not** propose a fifth structural rewrite. Four have now been tried; two paid, two did not, and the two
that did not were both plausible.

**Establish the instrument's reproducibility first.** The ladder is the right idea and the right question,
but it disagrees with itself. The next step is not a redesign: it is running it twice back to back, with
nothing else on the box, and seeing whether the disagreement survives a controlled pair. Only then is it
worth guessing at a cause. Without that, no attribution from it is worth acting on.

**Then the ordered questions, all of which the fixed ladder answers directly:**

1. Is the L1-to-L2 transition really worth 3.5x (340 to 97)? If so, the operand working set must fit L1, not
   L2 — which is what BLIS's `KC` exists for. **But K-blocking was measured null**, and that contradiction
   has to be resolved before anything is built on either number.
2. What exactly is the production wrapper's 1.4-1.7x? The `AblatePackB` / `AblateMicroKernel` switches
   already exist to split it.
3. Only after those: tile shape. `GemmMicroKernelShapeBenchmark` prices shapes without writing them, and it
   already says the 8x32 and 6x48 shapes both reach 340 GFLOP/s while AVX2 shapes reach 130-180.

## The same defect elsewhere: the pattern priced, the target named

The register-spill finding is a pattern, not an incident, so the obvious question is where else it lives.

**This repository already documented the rule and then broke it.** `GemmMicroKernelShapeBenchmark`'s remarks
say accumulators must be named locals and never a `stackalloc` span, "because a span forces an L1 round-trip
per accumulator per iteration", and record that breaking it once cost a 2.8x error in an earlier roofline.
**`Q4KGemvKernel.GemmTiled512` declares five `stackalloc` spans of `Vector512` as its accumulators.**

**The pattern was priced before proposing any work on the kernel**, because pricing a layout against an
already-measured shape is one benchmark arm while reimplementing Q4_K's accumulation is a day. Same 8x32
tile, same loads, broadcasts, FMAs and order; only where the sixteen accumulators live changes. Pinned, two
runs:

| accumulators | run 1 | run 2 | TFLOP/s |
|---|---:|---:|---:|
| named locals | 6.953 ms | 6.949 ms | **0.34** |
| `stackalloc` span | 21.562 ms | 17.263 ms | 0.11 / 0.14 |

**2.5x to 3.1x.** The named-local arm is stable to 0.06%; the span arm is itself bimodal, but even its
faster reading is 2.48x.

**What that does and does not license.** It establishes the pattern's price on this box. It does **not**
establish what share of `GemmTiled512`'s time is in that accumulation loop rather than in dequantisation,
scale handling and shuffles - a kernel can use a slow pattern somewhere that does not matter. Filed as
`XC-84` with that measurement required first.

**And decode is explicitly not a candidate.** The repository measures decode GEMV at 82% of this box's DRAM
read ceiling, so it is bandwidth-bound and register residency cannot help it. The candidate is prefill,
which is compute-bound.

## Re-running the nulls against the fixed kernel — and what that cost

Four hypotheses measured null earlier, all against a kernel running at 27% of single-core peak. With the
full-tile body at 93%, they deserved a re-run. **The re-run was not the cheap "just run it again" it looked
like, and it produced two findings before any benchmark started.**

**The prefetch and K-blocked paths were separate methods that had NOT been fixed.** Only the full-tile body
lost its `stackalloc` and its store helper; `MicroKernel8x32Avx512PackedAPrefetch` and
`MicroKernel8x32Avx512Accumulating` still carried both. Re-running as planned would have compared a fixed
kernel against unfixed ones and refuted them for the wrong reason.

**K-blocking turned out to be silently BROKEN, by the packing default being corrected.** Turning
`OVERFIT_CONV_PACK_A` on pointed `ctx.A` at the MR-major packed matrix while the accumulating kernel still
read A row-major. It had been green earlier that day only because packing was accidentally opt-in. The
correctness sweep caught it — 14 tests red at Kc=256 — **only because the sweep was run with the switch set**.
Production was never affected, the path being default-off, but **incorrect code behind a switch is worse
than no code**: whoever sets it gets silent wrong answers. The path, its kernel, `LoadTile` and the
environment name were deleted; the measurement stays here.

> **A default-off path is not covered by "the suite is green".** Every switch has to be exercised, or it
> rots into a trap.

**Prefetch, re-measured against the fixed kernel, is not null — it is 40% WORSE**, and the number says why:

| distance | VGG-16 | ONNX Runtime canary |
|---|---:|---:|
| off | **35.79 / 37.53 ms** | 18.07 / 18.29 ms |
| 4 | 51.32 ms | 18.26 ms |
| 8 | 51.09 ms | 18.41 ms |
| 16 | 51.20 ms | 18.36 ms |
| 4096 (liveness canary) | 52.97 ms | 18.51 ms |

**51 ms is exactly the pre-fix figure** (50.7-51.1). The prefetching variant was generated from the full-tile
body, so it has the same sixteen accumulators and no `stackalloc` — and it lands back at the spilling
kernel's speed. **Three prefetch instructions with their address arithmetic push it back over the register
limit.**

> **The full-tile body sits exactly at the register cliff.** Anything added to it costs about 40%. That is a
> constraint on every future change to this kernel, and it is worth more than the prefetch result itself.

Prefetch is now refuted twice — null against the spilling kernel, harmful against the fixed one — and both
times for the same underlying reason. Its two methods and the environment name are removed.

**A sixth intermittent test also surfaced**: `NeuralNetwork_TrainsOnXORProblem` failed once in a pair of
full-suite runs minutes apart. Same signature as `XC-83` — a convergence threshold over unseeded
initialisation. Two instances make it a family, and the fix is to seed, not to move thresholds.

## The general lessons, separated from the subject

1. **Recompute a denominator; never carry one forward.** A percentage of a wrong ceiling is
   indistinguishable from a correct one until something exceeds 100%.
2. **A refuted result and a refuted explanation are different objects.** Check the explanation before an old
   negative is allowed to close a question.
3. **Reading a competitor's source pays most where it exposes a premise you do not share** — here, that a
   general GEMM must pack A per call and an inference engine need not.
4. **A benchmark whose arms share state measures their order.** Run-to-run variance that hits some arms and
   not others is the signature.
5. **When a result is backwards, the benchmark is the suspect** — twice in this investigation, and both
   times the benchmark was at fault.
