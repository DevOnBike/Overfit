# GpuProbe

Measures **one operation** — the frozen quantized matmul a QLoRA fine-tune spends its time in — on this
machine's CPU and on this machine's GPU, in one sitting, and prints a single block of text. The one
question it answers is *"how much faster is this card than this CPU at that operation"*, and it is
built so that a number it prints can be trusted or is not printed at all.

Design and the reasoning behind every choice: [`docs/specs/gpu-probe-route-and-design-plan.md`](../../docs/specs/gpu-probe-route-and-design-plan.md).

---

## If somebody sent you this and asked you to run it

### 1. What you need

| | what | how to check it is there |
|---|---|---|
| required | an NVIDIA graphics **driver** (the ordinary one, from GeForce Experience or nvidia.com) | open a terminal and run `nvidia-smi`. It prints your driver version and your card's name. If the command is not found, the driver is not installed |
| required | the **CUDA 12 redistributable**, for the `cublas64_12.dll` library. **The version matters — see below** | see step 2 — the probe tells you itself, and that is a better check than looking for the file |

You do **not** need Python, Visual Studio, a model file, or the full CUDA Toolkit SDK. You do **not**
need the .NET SDK if you were sent a built folder.

The driver alone is enough for most of the probe. The `cublas64_*.dll` library is what the **most
important** arm needs — the one that measures what NVIDIA's own code does on your card. Without it the
probe still runs and still prints numbers, but those numbers are a floor rather than an answer.

**Install CUDA 12, and CUDA 13 will not do instead.** The probe reaches cuBLAS through ILGPU 1.5.3, and
the only cuBLAS library names that version of ILGPU contains are `cublas64_10`, `cublas64_11` and
`cublas64_12` — read out of the assembly's own bytes, where a major 13 appears nowhere. So a machine
carrying a CUDA 13 redistributable and nothing older is expected to have no cuBLAS that ILGPU can name.
That inference has not been run against a CUDA 13 install, and this project has not confirmed what
NVIDIA calls the CUDA 13 library — but the direction is not in doubt, and the failure is the quiet kind:
**X1 and X2 both skip**, and the report that comes back looks entirely
normal apart from one `NOT MEASURED` line. X3 resolves its own cuBLAS and may still run — see the arm
table below — so the run is not necessarily empty, but the FP16 floor and its FP32 control are gone and
the numbers that remain cannot be compared with anybody else's. If you already have CUDA 13, install 12 as well. The probe
prints which `cublas64_*.dll` it actually loaded, and, when none loaded, which of the three names it
tried, so the report names the problem instead of leaving you to guess.

### 2. Check it in under a minute, before the real run

```
GpuProbe.exe --quick --parity-only --x1
```

This runs the correctness checks on tiny shapes and prints no timings. Look at two lines:

- `accelerator type : Cuda` — the probe found your card. If it says `OpenCL` or `CPU`, see the warnings
  section below.
- `ARM X1 (cuBLAS): NOT MEASURED - ...` — if this line appears, cuBLAS did not load, and the reason is
  on the same line. If the line is absent, cuBLAS is working.

### 3. The one command

```
GpuProbe.exe --x1
```

If you were sent the source folder and have the .NET 10 SDK, use `dotnet run -c Release --project . -- --x1`.

### 4. How long it takes

**Budget an hour, start it and leave the machine alone.** It prints its progress to the terminal as it
goes, one line per shape, so you can see it is alive.

Being honest about that number, because it is the one thing here nobody has measured on the right
hardware: **the probe has never been timed end to end on an NVIDIA card.** On the developer's machine —
a Ryzen 9 9950X3D with an *integrated* Radeon reached through OpenCL, which is the slowest
configuration the probe supports — the first three of fifteen shape/batch combinations took about
eight minutes. Almost all of that is arm G1, the deliberately naive kernel, which took 1706 ms per
call at the smallest real shape. A discrete card should be far faster, but *how much faster is exactly
what the probe is measuring*, so any estimate here would be a guess dressed as a fact.

The full sweep was **not run to completion** on that machine — it was stopped after the third
combination, because that integrated GPU is not the hardware the answer is wanted about and the
box was needed for other work. So the hour is a budget, not a measurement.

If it is taking too long, two levers, in this order: `GpuProbe.exe --x1 --cells=attn` runs only the two
small shapes, and `--reps=5` is already the minimum but `--warmup-max=20` will cut the warm-up phase.
Say which you used when you send the result, because `--cells=attn` leaves out the shapes that carry
most of the work.

The probe does not use a fixed number of warm-up runs. It repeats each measurement until the timings
stop moving, so a slower or busier machine simply takes longer. It prints how many rounds each shape
needed. There is a hard limit so it always finishes.

**About 5 GB of free RAM.** The largest shape holds a 1.2 GiB weight in memory and uploads it to the
card. If it runs out of memory, `GpuProbe.exe --x1 --cells=ffn` skips that shape.

**Close your browser and anything heavy first.** The probe times a fixed reference workload before and
after everything else. If the machine drifted between the two, it says so and withholds its conclusion.

### 5. What to send back

**Copy everything between the `=== BEGIN GPU PROBE REPORT ===` and `=== END GPU PROBE REPORT ===`
lines, and send that.** The same text is also saved as `gpu-probe-report.txt` beside the executable, and
as `gpu-probe-report.json` — sending either file instead is fine.

**Two lines in the report say `PLEASE FILL IN MANUALLY`.** They ask for your RAM type and speed, and for
what else was running on the machine. The probe cannot read either. Please fill them in; a timing from a
machine that was also compiling something is not comparable to one from an idle machine.

**What it sends back:** your GPU model, your CPU model, your OS version, and the timings. It does **not**
read your user name, your machine name, or any file path, and it writes nothing outside this folder.

---

## What a good run looks like

Four things, and if all four hold the report has a conclusion in it:

1. The first line about the canary ends **`- within the 5 % threshold`**, not `CANARY MOVED`.
2. `accelerator type` says **`Cuda`**.
3. Every shape's `parity` column says **`PASS`**, and the `warm` column says **`ok`**.
4. Each shape ends with a line beginning **`HEADLINE forward  C3 cpu f32 -> G2 gpu tiled:`** followed by
   a number such as `14.30x`.

If the last line instead says `NOT PRINTED, and that is deliberate`, the probe found a reason not to
trust the comparison and printed the reason where the number would have gone. That is the probe working,
not the probe failing. Send the report anyway — the reason is the useful part.

## What each warning line means

**`CANARY MOVED - SITTING SUSPECT`** — the probe times an identical, fixed piece of work at the start and
at the end of the run, and the two disagreed by more than 5 %. The machine changed underneath the
measurement: something else started, or the machine got hot and slowed down. Every timing in the report
was taken against a moving baseline, so no ratio is printed. *Close everything else, let the machine sit
idle for a minute, and run again.*

**`the device arms ran on OpenCL, not CUDA`** (or `CPU`) — the probe did not find an NVIDIA card and fell
back. OpenCL is a much weaker path than the NVIDIA one, and `CPU` means ILGPU's software emulator, which
is not a GPU at all. Numbers from either say nothing about what an NVIDIA card would do, so no ratio is
printed. *Check `nvidia-smi` works, and that you are not running on an integrated graphics chip.*

**`ARM X1 (cuBLAS): NOT MEASURED`** — the `cublas64_*.dll` library did not load, and the line says why.
The probe still measures its own hand-written GPU kernels, but those are a **floor**: they are what a
first attempt at a port would get, not what the card can do. The gap between the two is usually large.
*Install the **CUDA 12** redistributable and run again with `--x1`. If the line names `cublas64_10`, `cublas64_11` and `cublas64_12` as the names it tried, none of them is on this machine — CUDA 13 does not satisfy it.*

**`TENSOR CORES: ...`** — without `--x1` this says `NOT engaged`, because no FP16 arm ran and nothing
could have used them; every number is then a lower bound for your card. With `--x1` it says
`NOT OBSERVABLE` and lists the preconditions instead. That is not evasion: **no CUDA or ILGPU API reports
whether tensor cores were actually used.** The FP32 arm X2 sits beside the FP16 arm X1 precisely so the
size of the gap between them can be read as the answer. *Nothing to do.*

**`WARM-UP NOT SETTLED`** — an arm's timings were still changing when the clock started, so its number
is not a measurement of that arm. Usually the machine was busy. *Run again on an idle machine, or pass
`--warmup-max=300`.*

**`PARITY FAILED - TIMING WITHHELD`** — a GPU kernel computed the wrong answer at that shape, so its
timing is not printed at all. A fast wrong kernel is the failure this probe is most exposed to. *This is
a bug in the probe, not on your machine. Please send the report — that is exactly what we need to see.*

---

## Options

```
--parity-only             run the correctness checks only, print no timing
--quick                   small shapes, seconds not minutes; NOT the real shapes
--fp16-bound              measure what FP16 costs in ACCURACY at each shape, then stop. No GPU needed
--x1                      also measure cuBLAS. Needs the CUDA 12 redistributable, not 13
--allow-cpu-accelerator   measure even if there is no GPU at all
--device=cuda|opencl|cpu  force a backend instead of taking the best available
--reps=N                  timed repetitions per arm (minimum 5)
--warmups=N               MINIMUM warm-up rounds (minimum 10). The real count is decided by measurement
--warmup-max=N            hard cap on warm-up rounds (default 100)
--warmup-budget-ms=N      wall-clock budget for one shape's warm-up (default 30000)
--warmup-tolerance=F      settled when two window medians are within F of each other, e.g. 0.05
--cells=SUBSTRING         only shapes whose name contains SUBSTRING
--batches=16,128,256      token counts to sweep
--seed=N                  seed of the synthetic weights and activations
```

---

## Handing it to somebody else

Preferred — they then need nothing installed but the graphics driver and the CUDA 12 redistributable:

```
dotnet publish Demo/GpuProbe/GpuProbe.csproj -c Release -r win-x64 --self-contained -o <folder>
```

Zip `<folder>` and send it. **Do not publish with `PublishAot=true`.** ILGPU generates IL and PTX at
runtime; an ahead-of-time compiled build cannot work and the failure is not obvious.

---

## What it does, so a number from it is not misread

Ten arms per shape, run **interleaved** so drift over the sitting falls on all of them equally:

| arm | what it is |
|---|---|
| C1 | the real training op — dequantize each Q4_K row, then F32 dots across the batch |
| C2 | the real backward, input gradient only |
| C3 | the same forward with the dequantize removed. **This is the honest CPU baseline for the GPU** |
| C4 | the same backward with the dequantize removed |
| G1 | a naive GPU kernel, one thread per output element — the floor a first port gets |
| G2 | a shared-memory tiled GPU kernel |
| G3 | the GPU backward, which reads the weight in the transposed direction |
| X1 | **cuBLAS FP16 (`cublasHgemm`) — the FP16 FLOOR.** It accumulates in FP16, which is neither the fastest nor the most accurate FP16 path a card offers. Quote it as a lower bound, never as what the hardware can do. Off by default; needs the CUDA 12 redistributable |
| X2 | cuBLAS FP32 (`cublasSgemm`), the control that makes the FP16 number readable |
| X3 | **cuBLAS `cublasGemmEx` with `CUBLAS_COMPUTE_32F` — FP16 storage, FP32 accumulate, the path a tensor core actually takes.** Reached through this project's own P/Invoke, because `ILGPU.Algorithms` exports no `GemmEx` at all. Same `--x1` flag |

**X1 and X3 are not two names for the same thing, and the difference cuts both ways.** X1 is
`cublasHgemm`, which accumulates in FP16; X3 accumulates in FP32. X1 is what a naive port reaches for and
is a FLOOR; X3 is the ceiling. **X3 also resolves its own cuBLAS**, and its candidate list starts at
`cublas64_13.dll`, which ILGPU 1.5.3 contains no name for — so on a machine carrying only a CUDA 13
redistributable **X3 can be the only cuBLAS arm that runs.** The report prints each arm's library, and
they must be checked against each other before any X1-to-X3 ratio is read.

**The headline is C3 against G2, never C1 against G2.** C1 includes a dequantize the GPU arms do not
perform, so comparing against it would flatter the GPU. C1 minus C3 is reported separately as the cost of
the dequantize.

**Two correctness gates run before anything is timed, and a shape whose check fails prints no timing at
all.** First a deliberately non-square 3x5x7 case against a hand-written triple loop; then, at every real
shape, the GPU output against the CPU F32 result — **cosine at least 0.9999 and relative L2 below a
ceiling that depends on the arm's arithmetic.** A fast wrong kernel is the failure this probe is most
exposed to, and a number printed beside a failed check gets quoted without the check.

Maximum relative error is printed but does **not** gate, and that is deliberate. It is an extreme-value
statistic over millions of elements, decided by the single worst one, so on an FP16 arm one
near-cancellation fails a kernel that is right everywhere else. Relative L2 is the whole-tensor error and
it is what a training run would feel. The ceilings are measured rather than assumed, by
`GpuProbe.exe --fp16-bound`, which needs no GPU: **1e-4** for an F32 arm (the arms measure 1.5e-7 to
5.9e-7, so a hundredfold margin), and for FP16 a ceiling that grows as `sqrt(k)`, because `cublasHgemm`
rounds its running sum once per multiply-add. Measured relative L2 for FP16: **6.5e-3 at k = 2048** and
**1.6e-2 at k = 11008** — a sqrt(k) fit to the second predicts 6.8e-3 for the first, so the mechanism is
confirmed rather than curve-fitted.

**Arm X3 is judged against a flat 1e-3, and the bound behind it was measured on 2026-08-22** by the same
`--fp16-bound` run, which now prints a third column. FP32 accumulate with the result left in F32 costs
**2.87e-4 to 2.97e-4**; X3 writes into an FP16 output buffer, so it pays one more rounding and its real
bound is **3.52e-4 to 3.64e-4**, flat across every k as the mechanism predicts. That leaves a **2.7x**
margin under the ceiling, not the 3.4x a reading of the first column alone would give. The same ceiling is
what catches the one mistake that would matter: passing `CUBLAS_COMPUTE_16F` (64) instead of
`CUBLAS_COMPUTE_32F` (68) would give FP16 accumulate at 6.5e-3, **6.5x over the ceiling**, so a wrong
compute type fails parity instead of printing a fast wrong number.

**A suspect sitting suppresses the headline ratio; it does not merely annotate it.** The same principle
as the parity gate, and it exists because of a measured failure on 2026-08-21: the probe printed
`HEADLINE forward ... 1036.58x` while its own `CANARY MOVED - SITTING SUSPECT` line sat higher up the
same page. A number printed next to a warning gets quoted without the warning. The reasons are now
printed in the place the ratio would have gone, so a copy-paste cannot separate them.

**Warm-up is measured, not chosen.** Each arm repeats until the median of its last five readings is
within 5 % of the median of the five before it — or within its own scatter, because a trend smaller than
a measurement's noise is not resolvable and demanding it would never terminate. There is a floor of ten
rounds, a cap, and a wall-clock limit, and the report prints how many rounds each arm actually took.
Measured on 2026-08-21 on a Ryzen 9 9950X3D with the `--quick` shapes: at the old fixed default of two
warm-ups the C3 host arm reported **0.560 ms** against a settled median of **0.145 ms** — a factor of
about four, from warm-up alone. The stopping rule finds the number of rounds instead of assuming it, and
prints how many it took.

> **A correction, kept because it is the more useful lesson.** An earlier draft of this paragraph quoted
> **6.841 ms** and **0.007 ms** here, and said the two inflated the ratio "by about a million". Those
> figures are real, but they came from a run of a **deliberately mutated build** with the synchronise
> below removed — an experiment whose report artefact was left in the working tree, looked exactly like a
> normal result, and was committed. Two separate people then read it as evidence of a warm-up problem.
> Warm-up **is** a real effect, at about 4x; the barrier is a different defect worth about 8000x. A
> mutated build's OUTPUT is mutated too, and it does not restore itself when the source does.

**Every GPU timing ends at `accelerator.Synchronize()`, and the synchronise lives inside the timing
helper rather than at each call site**, so no arm can be written that forgets it. A GPU launch is
asynchronous, and a timer stopped before the work finishes measures the enqueue and reports a
spectacular, meaningless speedup.

## What it does NOT measure

The probe prints this list itself, at the end of its own report. The one that matters most: **nobody has
ever measured what fraction of a real QLoRA training step is spent in this operation**, so a speedup here
does not convert into a fine-tune speedup in either direction. This project has already measured one
kernel benchmark under-reporting a real end-to-end effect by 4.3x.

Second, and it limits the headline: **arm X1 is `cublasHgemm`, which accumulates in FP16.** A tensor
core accumulates in **FP32**, through `cublasGemmEx` with `CUBLAS_COMPUTE_32F` — and ILGPU's cuBLAS
wrapper does not expose `GemmEx` at all (verified against the package metadata: `Gemm` has overloads for
`Half`, `float`, `double`, `Float2` and `Double2`, and there is no `GemmEx`). So X1 is neither the
fastest nor the most accurate FP16 path the card offers, and the card's real FP16 ceiling is above what
this probe can reach. Closing that gap needs P/Invoke, not ILGPU.

Third: **nothing here says FP16 is numerically usable for QLoRA training.** Speed and numerical viability
are two questions and this probe answers only the first — the measured FP16 error above is where a
decision on the second would have to start.

## Why this project is not in `Overfit.sln`

It references the published `DevOnBike.Overfit` package rather than the source, and it carries the two
empty `Directory.Build.props` / `Directory.Build.targets` sentinels so the repository root build files do
not reach it. That keeps ILGPU out of the solution's central package list — where one security advisory
against a GPU package would fail the build of everything — and keeps the repository's Roslyn analyzers off
a machine that only wants to run a benchmark.

The cost of that choice, stated because it is a real one: **the CPU arms run the published package, not
this repository's `HEAD`.** The probe prints the resolved assembly version so a reader can tell.
