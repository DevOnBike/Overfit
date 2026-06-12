# MNIST / CNN training performance audit (2026-06-11)

A full, measured audit of the CNN training path (the `Mnist_FullTrain60k_CnnBeastMode_Benchmark` workload:
Conv(1→8,3×3) → ReLU → MaxPool2×2 → Linear(1352→64) → ReLU → Linear(64→10), 60k samples, batch 64, Adam).
Box: AMD Ryzen 9 9950X3D (16C/32T). Every claim below was A/B-measured; negative results are recorded so the
same trees don't get barked up again.

## Headline results

| Lever | Effect | Status |
|---|---|---|
| **DataParallelTrainer ×8 + lr×8 (linear rule)** | 5 epochs **6135 → 2666 ms = 2.3×**, final loss identical (0.0357 vs 0.0350) | ✅ shipped (`Mnist_FullTrain60k_CnnDataParallel8_Benchmark`) |
| **+ batch 128 per replica** | → **2247 ms = 2.7× total** | ✅ shipped |
| **MaxPool2D pool=2 AVX2 tail** | op **145.8 → ~107 ms (−26%)**, steady epoch **1103 → ~1035 ms (−6%)**, bit-identical values **and argmax indices** | ✅ shipped (`PoolingKernels.MaxPool2DForwardWithIndicesPool2Avx2` + `MaxPoolPool2Avx2ParityTests`) |
| Net | full 5-epoch train **~6.1 s → ~2.1 s** at equal quality | |

## Per-op audit (steady-state epoch, single replica)

| Op | ms/epoch | Implementation | Verdict |
|---|---:|---|---|
| Conv2D fwd | ~256 | im2col + GEMM, `OverfitParallelFor` | ⛔ investigated to death earlier (≈22% of FP peak; 3 micro-opt hypotheses refuted; OPF-migration attempt was +13% WALL — keep as is) |
| ReLU fwd | ~82 | `TensorPrimitives.Max` | ✅ **at memory floor** (~48 GB/s counting read+write+RFO) |
| ReLU bwd | ~86/ep | manual `Vector<float>` | ✅ at floor — **Vector512 variant tried & reverted** (444→431 ms aggregate = noise; 3-stream memory-bound, SIMD width >256 buys nothing) |
| MaxPool2D fwd | 146 → **107** | was: SIMD vertical max + **scalar branchy horizontal/argmax tail** | ✅ **fixed** — AVX2 deinterleave (shuffle+permute), GE-masks → blends, exact INT32 index math, overlapping-last-window (no scalar tail) |
| MaxPool2D bwd | (in backward) | scatter by recorded indices | ⛔ random scatter — not vectorizable profitably |
| Linear fwd/bwd | ~91 + ~189 | `LinearKernels` on `OverfitParallelFor` | ✅ near floor; no raw `Parallel.For` (mentions in file are comments) |
| SoftmaxCE, Reshape, copies | ~35 | — | negligible |
| Adam.Step | ~51 | **3× raw `Parallel.For`** (master-side, outside replicas — legal) | 🟡 only remaining cosmetic: →OPF would cut ~2.7 MB/epoch of TPL allocation (the dominant allocator). Zero expected speed. |

Allocations: ~4.35 MB/epoch total (≈4.6 KB/batch), 2× gen0/epoch — training is effectively allocation-quiet
already; "zero-alloc training" would be a banner, not a speedup.

## Why there is no more parallelism to extract (the "51% CPU" myth)

- Measured "effective cores": single-replica **16.3**, DP×8 **16.6** — both ≡ **all 16 physical cores
  saturated**. The "51%" figure divides by 32 *logical* cores; SMT siblings share the FP/FMA units, so they
  add ~nothing for vectorized FP. **There is no idle hardware.**
- Replica suppression (`OverfitParallelFor.SuppressParallelismOnCurrentThread`) **works and is load-bearing**:
  disabling it (`runWorkerOpsInline:false`) measured **1.8× slower** (oversubscription).
- "Give each replica 2–4 OPF threads" is **architecturally impossible**: `OverfitParallelFor.For` holds the
  global `_gate` lock across dispatch **and** `_completion.Wait()` — the pool runs ONE job at a time by
  design, so concurrent intra-replica dispatches would serialize.
- 16 replicas: worse (2837 ms, SMT collision). Sync-frequency halving (batch 128): only +9% → the serial
  sections (averaging + Adam + broadcast) are not the dominant limiter either.

## Dead ends (measured — do not retry)

| Attempt | Result |
|---|---|
| Training forward via int8 `ProjectBatched` (instead of dequant-row + F32 `TensorPrimitives.Dot`) | **0.61× — slower** + 2% activation-quant noise. `TensorPrimitives.Dot` gets AVX-512 on Zen5; rows live in L1. |
| Scale-unpack hoist in `ProjectBatchedChunk` (loop order o→sb→n) | **−25%** despite being bit-identical — destroys activation locality (512 KB L2 walk per weight super-block). |
| AVX-VNNI (`vpdpbusd`) in the batched int8 dot | ≈0 (40.2 vs 38.9 ms, noise). Third VNNI burial in this repo (decode 2026-05-31 ×2, batched 2026-06-11). |
| ReLU backward on Vector512 | ≈0 (memory-bound). |
| Conv2D → OverfitParallelFor | +13% wall (earlier session) — Conv keeps its current dispatch. |
| All-32 spin-barrier, attention thread-widening, etc. | see `docs/llamacpp-cpu-analysis.md` / perf-sprint notes. |

## Incidental findings

- **`MaxPool2DForwardWithIndicesGeneric` and the pool=2 path have always had different tie-breaking rules**
  (first-in-scan-order vs left-column-then-row0). Ties only, pre-existing; documented in
  `MaxPoolPool2Avx2ParityTests`. The AVX2 path is bit-identical to the pool=2 scalar rule (the one in
  production for pool=2).
- The beast benchmark with `enableTelemetry=false` is honest (~1.1 s/epoch steady) — its per-op
  `OperationStats` cost is negligible. (An earlier claim that instrumentation dominated it was wrong and
  retracted; the ~21 s/epoch note in its docstring refers to the old BatchNorm/Residual architecture.)
- Large-batch LR scaling on Adam: the **linear rule (lr×R) beat the √R rule on final loss** in this setup
  (0.117 vs 0.133 after one epoch; over 5 epochs lr×8 fully recovers single-replica quality).

## Addendum 2026-06-12 — one-cycle LR (time-to-quality) + the PyTorch reference

**One-cycle LR closes epochs, not milliseconds** (`MnistOneCycleBenchTests`, 3 runs, init noise ±0.006
— unseeded layers, compare bands not digits). Schedule = `LearningRateSchedule.WarmupCosine` (warmup
10% of steps → peak → cosine to 0.0005) on the DP×8 / batch-128 rig:

| Arm | Final-epoch loss | Wall | vs baseline |
|---|---:|---:|---|
| 5 ep, const lr 0.008 (baseline) | 0.036–0.049 | ~2.8 s | — |
| 3 ep, peak **0.048** | **0.0397** | **1.47 s** | **−47% time, matches the baseline band** |
| 4 ep, peak **0.032** | **0.0304** | **2.0 s** | **−28% time, BEATS the baseline band** |
| 3 ep, peak 0.008 (low) | 0.088–0.101 | 1.5 s | low peaks don't work — peak 4–6× base lr is the regime |

**PyTorch 2.11 CPU reference** (`Scripts/bench_mnist_torch.py`) — identical arch / data / batch 128 /
AdamW on the same box, thread-count SWEPT to find torch's true optimum (4→710 ms, 6→565, **8→524–570
(optimum)**, 10–16→600–615, 24→685, 32→13–14 s/epoch — the same SMT-oversubscription lesson as our
worker-count findings). Overfit side measured with **BenchmarkDotNet** (`MnistTrainingEpochBenchmark`,
1 op = 1 full epoch, 10 iterations): **503.0 ± 4.2 ms/epoch, 1.31 MB allocated/epoch**.

**Verdict: Overfit ≈ PyTorch CPU on this workload, consistently ~5–10% faster** (503 ± 4 vs ~524–570
at torch's optimal 8 threads — the bands don't overlap, but the margin is small). ⚠️ RETRACTION of the
first claim in this addendum's draft: an earlier "1.2× faster" compared against torch at 16 threads,
which a later sweep showed is NOT its optimum (8 is) — always sweep BOTH sides' knobs before claiming
a ratio (same lesson as the llama.cpp "parity" retraction). Honest scope: a tiny 1→8-channel conv is
dispatch-bound, where oneDNN's conv kernels can't shine and Python's per-step overhead hurts; on large
convs/transformers PyTorch would likely win. `torch.compile` untested (Inductor needs MSVC `cl`).

**Allocation breakdown of the 1.31 MB/epoch** (`MnistAllocBreakdownTests`, GC.GetTotalAllocatedBytes,
all threads): tape path **754 KB** (1 624 B/batch/replica — `AutogradNode`/`TapeOp` objects, ~8 ops/batch)
+ trainer/Adam/TPL dispatch **656 KB** (~11.3 KB/step). At 503 ms/epoch that is ~2.8 MB/s → ~2 gen0/epoch
→ GC cost ≤1 ms (~0.2%), **below the benchmark's own ±4 ms stddev — full zero-alloc training would buy
nothing measurable here** (it remains a possible "0 B/step" banner: node/tape pooling + closure-free
dispatch, ~1–2 sessions, 0 perf).

## Reproduce

```powershell
# flip [LongFact] → [Fact] on the benchmark(s) first, restore after
dotnet test ./Tests/Tests.csproj -c Release --filter "FullyQualifiedName~CnnBeastMode"        # single replica + per-op stats
dotnet test ./Tests/Tests.csproj -c Release --filter "FullyQualifiedName~CnnDataParallel8"    # DP×8 twin (same Epoch-line format)
dotnet test ./Tests/Tests.csproj -c Release --filter "FullyQualifiedName~MnistDataParallelBench"  # 5-arm sweep (single / 4 / 8 / lr rules)
dotnet test ./Tests/Tests.csproj -c Release --filter "FullyQualifiedName~MaxPoolPool2Avx2Parity"  # bit-identity gate (fast, always on)
dotnet test ./Tests/Tests.csproj -c Release --filter "FullyQualifiedName~OneCycle"                # one-cycle LR arms (flip [LongFact] first)
python Scripts/bench_mnist_torch.py 8                                                             # PyTorch reference at ITS optimum (sweep showed 8 thr)
dotnet run -c Release --project Sources/Benchmark -- --filter "*MnistTrainingEpoch*"              # BDN-grade ms/epoch for the Overfit side
```
