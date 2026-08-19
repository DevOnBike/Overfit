# Overfit against ONNX Runtime: the whole curve, and what is behind it

The headline table lives in [`README.md`](../README.md). This document is the long form: how the ratio moves
with model size, what the remaining gap is made of, what has been measured **not** to fix it, and the
uncertainties that a single number would hide.

Everything here is measured on one machine — AMD Ryzen 9 9950X3D, 16 physical cores, ONNX Runtime 1.29.0 —
with ONNX Runtime running **in the same process** as the canary, so a number that moved because the box moved
shows up as both engines moving together. Raw provenance, including the hypotheses that were refuted and the
instrument failures that produced two wrong conclusions before they were caught, is in
[`measured-baselines.md`](measured-baselines.md).

**What the two ends actually measure.** On a 7,840-parameter `Linear` the arithmetic takes a few hundred
nanoseconds, so the result is dominated by ONNX Runtime's ~1.6 µs of per-call dispatch — that 8.2× is a real
property you feel on small models called at high rates, and it is mostly a measure of **call overhead**. On
VGG-16 the arithmetic dominates and the result measures **kernel quality**, where Microsoft's MLAS is
about 1.45× ahead of us.

**Where those two started.** The first figures recorded on this line of work were **79.0 ms for VGG-16 and
67.7 ms for the 60.9 MB CNN** — 3.72× and 5.24× behind ONNX Runtime. They are now **2.9× and 3.6× faster**
in absolute terms, and the gap has closed to 1.45× and 1.89× over the same period. ONNX Runtime's own CNN
figure also fell over it, 12.93 to 9.8 ms, but **that movement is not attributed** — see the uncertainty
section below. Every figure here is measured with ONNX Runtime running in the
same process as the canary, and parity is checked on every run: cosine 1.000000, same argmax.

**Where the remaining gap is, decomposed rather than guessed.** ONNX Runtime's own per-node profiler puts
**92% of it in convolution**: the dense layers are at **1.05×** — `fc1` is 6.97 ms against their 6.815, two
engines hitting the same wall on 411 MB of fp32 weights — and activations cost nothing on either side because
both fuse them into the convolution epilogue.

Convolution then splits cleanly into **1.31× per-core work × 1.71× scaling**, and the product closes to the
2.24× measured directly. The per-core half is nearly spent: a cost model fitted on two layers and checked on
seven held-out ones puts our **GEMM micro-kernel at 301 GFLOP/s, 84% of this machine's single-core FMA
ceiling**. The scaling half is where the room is — **their convolution scales 12.12× across 16 cores against
our 7.08×**, and worker-occupancy instrumentation shows **35% of the pool's time idle**, split between an
uneven work split and dispatch overhead.

**What is known not to fix it**, each measured and each written up in
[`docs/measured-baselines.md`](docs/measured-baselines.md): wider column blocks, splitting the M dimension,
expanding panels into a shared buffer, blocking the dense accumulator, sizing the pool to physical cores, and
a managed port of MLAS's NCHWc layout — which wins 1.34× on gather-heavy early layers and loses 1.15× on
compute-dense ones, so a hybrid would cost most of the port for about 6%. Rebalancing the work split is
measured to **cost cache locality**: at eight chunks per worker, supply from L2 falls 5.7% while L3 rises
40.3%, another CCD's cache 62.1% and DRAM 70.0%, with the instruction count flat and CPI up 10.8%.

Everything above is a position, not a finished story. The provenance — every measurement, every refuted
hypothesis, and the instrument failures that produced two wrong conclusions before they were caught — is in
[`docs/measured-baselines.md`](docs/measured-baselines.md).

**Every row is a like-for-like thread count, and that was checked rather than assumed.** The three small
rows run ONNX Runtime pinned to one thread, so they are only honest if Overfit is single-threaded there too.
Re-running them with `OVERFIT_PARALLEL_WORKERS=1` moved Overfit by 0.9% on the MLP and 1.2% on the MNIST CNN
— i.e. those two models never reach a parallel path, and the comparison is one thread against one thread.
The `Linear` row is the exception: forcing one worker made Overfit **faster**, 226 → 190 ns, which would
raise the ratio to 9.9×. **The table publishes the slower, default-configuration number**, because that is
what a consumer gets without setting anything. The two large CNN rows give both engines the whole machine.

**The CNN gap is kernels, not threading**, and that is measured rather than assumed: with thread counts
matched on both sides the gap is roughly constant across thread counts, and the two engines scale about
equally with cores (Overfit 4.2×, ONNX Runtime 4.9× going from one thread to the whole machine; measured
2026-08-17). Allocation stays 0 B on the Overfit side throughout, against 224–904 B per call for ONNX
Runtime.

**The output is identical, only the speed differs.** Both large-CNN rows carry a parity check in the
benchmark's own setup: cosine 1.000000 against ONNX Runtime, max absolute difference 6.7e-8 on the 60.9 MB
CNN and 3.2e-7 on VGG-16, same argmax. This is a performance gap, not an accuracy trade.

**The two large-CNN rows carried a session-level uncertainty the small rows did not, and it was ONNX Runtime's figure that moved.** Across three sittings its 60.9 MB CNN result read 12.93, then 9.79, then 9.19 ms — a **29% spread on an untouched binary** — while ours moved 0.1% between the last two and all three small benchmarks stayed inside 1.3%.

**That was investigated on 2026-08-19 and it does not reproduce.** Nine process launches across two independent instruments put ONNX Runtime between 9.69 and 10.04 ms: **1.0% spread** letting it choose its own thread count, 3.6% with 16 forced, 2.5% under BenchmarkDotNet, and Overfit's control 1.0–1.2%. Timing each process in halves put the within-process movement at −3.2%..+1.5%, which rules out arena warm-up, and the ONNX Runtime pin has not moved since 2026-04-06, which rules out a package bump. Forcing the thread count turned out to be *worse* than ONNX Runtime's own choice, so these comparisons keep the default.

**What the original 12.93 was is still not identified, and "does not reproduce" is not an explanation.** Power state was the third candidate and it was not measured; the sittings are gone and the effect cannot be summoned. So: **a large-CNN figure measured today is good to 1–2.5% across launches, but do not compare one against a figure quoted in a document older than 2026-08-19.** The three sittings were each internally consistent — both engines were re-measured together — so the *ratios* survive even where the milliseconds do not.

**One caveat on the near-parity rows.** Across process repeats Overfit's MLP figure moved 0.4% while ONNX
Runtime's moved 7% and ML.NET's moved 29%, so the 1.3× is quoted from the repeat where the *opponent* was
fastest. Treat the three rows between 1.2× and 1.3× as "about even", not as a ranking.

**Pick on the shape of your workload, not on one ratio**: many small in-process inferences favour Overfit;
one big convolutional model favours ONNX Runtime, and if you can ship native dependencies you should use it
for that. Method and full caveats: [`docs/measured-baselines.md`](docs/measured-baselines.md).

Full benchmark tables and caveats live in [`docs/TECHNICAL.md`](docs/TECHNICAL.md).

---
