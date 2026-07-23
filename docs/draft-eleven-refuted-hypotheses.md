# DRAFT — Our C# matmul turned out to be faster than llama.cpp's. It took eleven refuted hypotheses to find out.

> Status: draft. Every number here is measured on one machine (AMD Ryzen 9 9950X3D, .NET 10, Windows 11)
> and reproducible from the repository. Nothing is estimated; where something is estimated it says so.

---

## The claim, and why you should not believe it yet

Overfit is a pure-C# CPU inference engine — no native binaries, no Python, no ONNX Runtime. Its Q4_K
matrix-multiply kernel, run on llama.cpp's own benchmark shape, at the same instruction set and the same
thread count:

| | time | TFLOP/s |
|---|---:|---:|
| llama.cpp (AVX2 build, `test-backend-ops`) | 38 559 µs | 1.56 |
| **Overfit `GemmTiled`** | **35 308 µs** | **1.70** |

That is a 1.09× win in managed C# over hand-written C++ SIMD.

Now the honest part: **the whole model is still slower than llama.cpp**, and I spent most of a day being
wrong about why. This article is about the being-wrong, because that is the part nobody publishes and the
only part that generalises.

---

## Act I: the comfortable, wrong story

Prefill — processing the prompt before the first token — was **3.76× behind** llama.cpp. Decode was only
1.13× behind. Two very different numbers for the same engine, which should have been the first clue.

I read llama.cpp's `ggml_gemm_q4_K_8x8_q8_K`. Beautiful code: constant-index unrolled accumulators, a
`block_q8_Kx4` activation interleaving, four hand-tuned ISA variants. I formed a comfortable narrative:
*their kernel craft is better than ours; the gap decomposes as 2.34× kernel quality × 1.60× AVX-512.*

I then spent several hours building things that followed from that narrative. Every single one failed:

- **Register pressure.** Their kernel keeps a tile in registers; ours spills to `stackalloc` scratch.
  Obvious cause. I split the pass in half to reduce live state — **exact tie**.
- **Activation interleaving.** Copy their `block_q8_Kx4` layout — never got built, because of Act II.
- **Fixed-tile specialisation.** Unrolled the tile loop to constant indices — **inconclusive, reverted**.

Three failures, one after another, each of which sounded correct while I was writing it.

**The mistake was not the hypotheses. It was that I never measured their kernel.** I read it and inferred.

---

## Act II: measure the thing you are comparing against

llama.cpp ships `test-backend-ops`, which times individual operators. Thirty seconds of work:

```
q4_K m=4096 n=512 k=14336 → 38558.65 us/run, 60.13 GFLOP/run → 1.56 TFLOPS
```

I put that exact shape into our benchmark. We came out at **1.70 TFLOP/s**.

**The premise of everything I had built that day was false.** Their kernel was not better. The gap lived
somewhere else entirely, and three days of planned work evaporated in one measurement.

> **Lesson 1.** Reading someone else's code produces *plausible explanations*, which are worse than no
> explanation because they feel like knowledge. If you are comparing against a project, measure it.

---

## Act III: the ceiling is not where you think

With the kernel exonerated, I needed a real ceiling. So I wrote one — a standalone probe that measures what
the machine can actually do, since a bare "1.7 TFLOP/s" is meaningless until you know whether the box tops
out at 2 or at 20.

The first version reported a peak of **0.79 TFLOP/s** — *below* what our real matmul achieved. Impossible
for a loop that touches no memory.

The bug: I had put the accumulator chains in a `stackalloc` span.

```csharp
Span<Vector256<float>> acc = stackalloc Vector256<float>[Chains];   // an L1 round-trip per accumulator
```

A span indexed by a loop variable does not live in registers. It forced a load and a store per accumulator
per iteration, so the benchmark measured L1 latency instead of FMA issue rate. Named locals with constant
indices fixed it: **2.19 TFLOP/s**.

> **Lesson 2.** `stackalloc` is not "registers". Constant-index named locals are. And if a synthetic peak
> comes out below your real code, the benchmark is broken — real code cannot exceed a true ceiling.

And then I made a *different* error with the same shape. A working-set sweep, meant to expose the L1→L2→L3
steps, came out perfectly flat at ~75 GB/s from 8 KB to 128 MB. I concluded the machine had no cache cliff
and that cache blocking therefore could not pay — a conclusion I acted on.

It had one accumulator. It was measuring the dependency chain's *latency*, which was below every cache
level's bandwidth, so no level could show. With eight independent streams:

| working set | 1 core | all cores | scaling |
|---|---:|---:|---:|
| 16 KB – 2 MB | ~76 GB/s | 700–900 GB/s | **9–12×** |
| 8 MB | 67 | 113 | **1.7×** |
| 128 MB | 60 | 64 | 1.1× |

There is a cliff, and it lands exactly where **16 cores × 8 MB = 128 MB = this chip's L3 including
V-cache** — a number the benchmark was never given. That self-consistency is the only reason I now trust it.

> **Lesson 3.** One accumulator measures latency. Several measure throughput. This is the same mistake as
> Lesson 2 wearing a different hat, and I made both in the same afternoon.

---

## Act IV: what actually paid

Once the ceilings were honest, the wins were unglamorous.

**The biggest one was not making anything faster — it was doing it less often.** The Q4_K kernel widens F16
scales to float once per weight block. That reads as amortised. But the kernel is invoked once per *column
tile* — 84 times for a 672-token prompt — so every scale was widened **84 times over**. Ablation priced it
at 12% of the kernel.

x86 has `vcvtph2ps`, which would widen eight halves in one instruction. .NET exposes neither an `F16C`
intrinsic class nor a `Half` overload of `Vector128.Widen`, so that instruction was unavailable.

I hoisted the decode to once per projection instead. **12% → 0.14%.**

The missing instruction would have made the work ~4× faster. Restructuring deleted 83/84 of it. And had the
instruction been available, I would very likely have used it, banked the 4×, and never asked the better
question.

> **Lesson 4.** Before looking for a faster way to do the work, count how many times you do it.

**Second: the same technique inverted between two kernels.** Porting the Q4_K kernel to AVX-512 by pairing
two activation columns per instruction gave **+13.8%**. The identical port applied to the Q6_K kernel gave
**−20%**, and was reverted.

Why: pairing pays for the `vinserti64x4` that broadcasts shared weights with the arithmetic subsequently done
on them. Q4_K broadcasts eight vectors per sub-block and then issues sixteen paired statements against them.
Q6_K broadcasts six per `k`, sixteen times per block, for far less arithmetic each. The lane-crossing traffic
outran the savings.

A second attempt on Q6_K — pairing what was *already adjacent in memory* rather than broadcasting — gave
**+12%**. Same kernel, same instruction set, same shape. Only the choice of what shares a register changed.

> **Lesson 5.** A wider vector is not a property of the ISA. It is a ratio between broadcast cost and work
> done per broadcast, and that ratio is per-kernel. Do not extrapolate a port from one kernel to another.

---

## Act V: the measurement discipline, in one page

Everything above reduces to a handful of rules that survived the day:

1. **Two identical arms are the cheapest canary there is.** One sweep reported the same configuration 21%
   apart under two different names. Without that accidental duplicate I would have believed the whole table.
2. **Interleave arms; never run all-A-then-all-B.** A 20% thermal drift between two sequential runs once
   inverted a result completely.
3. **A single-arm measurement after a hot-path change is worthless.** One change "gave" +4% while an
   untouched component moved with it — that was the machine, not the change.
4. **An impossible ordering means a broken benchmark, not a discovery.** When 512-bit measured *slower* than
   256-bit, the cause was a helper method the JIT declined to inline: I was timing the calling convention.
5. **Ablate inside the real kernel; do not micro-benchmark the part.** Toggling a piece off in production
   measures its real share. A synthetic harness measures the harness.
6. **Match the ceiling to the instruction mix your code actually issues.** I claimed our kernel ran at 78%
   of the float ceiling. It performs one `vpmaddubsw` per 32 MACs; against the ceiling that applies it was
   at 15%. The number was arithmetic, not measurement.

---

## Where it ended

Prefill went from **143 to ~299 tok/s** — the gap to llama.cpp's AVX-512 build from 3.76× to **1.81×**, and
to their AVX2 build to about **1.14×**. On machines without AVX-512, a managed C# engine is roughly at
parity with llama.cpp on prompt processing.

Decode is memory-bound and stays 1.13× behind; the probe shows our GEMV already runs at 82% of the DRAM read
ceiling with compute headroom to spare, so a wider instruction set cannot help there. That is a closed
question rather than an open one, which is worth as much as a speedup.

Final tally for the campaign: **six wins, about fifteen refuted hypotheses, three corrected arithmetic errors
of my own.** The corrections included counting FLOPs at half their true value for an entire afternoon,
because the commonly quoted "15.5 GFLOPs" for VGG-16 is a *MAC* count.

I do not think the wins are the interesting part.

---

## Try it on your machine

The probe is a single xUnit test with no external dependencies. It reports peak FMA at every vector width
(one core and all cores), sustained memory bandwidth, and the working-set sweep — and asserts that its own
measurement loops allocate zero bytes, so it cannot silently degrade into a GC benchmark.

```
dotnet test -c Release --filter FullyQualifiedName~MachineProbe
```

Read your numbers before you believe anyone's — including mine.

---

### Notes for revision (remove before publishing)

- Decide the audience: kernel authors (narrow) vs anyone who optimises anything (wide). Lessons 1–6 are
  discipline and generalise to SQL, pipelines, allocation work. The SIMD is the *illustration*. Leaning wide
  is probably the difference between a few hundred readers and a few thousand.
- Charts worth making from data already in the repo: the worker sweep, the working-set cliff, the
  before/after prefill bar, the tile-shape ceilings.
- Consider splitting Act V into its own follow-up piece — it is the most reusable and the most quotable.
- Verify every figure against ROADMAP before publishing; several numbers in this draft were themselves
  corrected mid-session.
