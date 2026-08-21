---
name: project-gpu-probe-route
description: 2026-08-21 GPU probe route decision (ILGPU) and QLoRA GEMM shapes — what was verified, what was rejected and why, what stayed unmeasured.
metadata:
  type: project
---

Plan: `docs/specs/gpu-probe-route-and-design-plan.md`, signed `STATUS: APPROVED` 2026-08-21. Scope was a
measurement probe only, not a GPU backend.

**Route: ILGPU 1.5.3.** Verified from the nupkg on nuget.org that day: **5 managed DLLs, `runtimes/` empty —
zero native assets**, no transitive deps on the `net7.0` asset. Ships `CPUAccelerator`, so a kernel is
developable and provable with no GPU. **Not verified: that it runs without the CUDA toolkit** (this box has
no NVIDIA card) — that is the route's load-bearing unknown.
Rejected: ComputeSharp 3.2.0 (Win/DX12 only, 43.7 MB source generator, no tensor-core or vendor-BLAS route,
so a poor number would be a fact about DirectX); CUDA P/Invoke (needs the toolkit, not the driver);
TorchSharp (measures NVIDIA's kernels, not ours — which is the exact unknown being priced).

**The AOT objection to ILGPU is weaker than it looks.** The banned symbols and AOT rules bind `Sources/Main`
and anything reachable from `Tests/AotSmokeTest`. GPU is already on the private/commercial moat side
(`ROADMAP.md:1168-1169`), so a GPU assembly never stands under those rules. Do not reject a GPU route on AOT
grounds; reject it, if at all, on the moat decision.

**The op that dominates QLoRA is `ComputationGraph.FrozenQuantizedLinear`** — shared by the Llama path
(`TrainableLlamaBlock.cs:198`, `TrainableLlamaModel.cs:193`) and the GPT-1 path
(`Gpt1LoRAFineTuner.cs:555`). Two facts a port inherits: the dequant is INSIDE the op and its share falls as
the batch grows (`O(k*m)` against `O(n*k*m)`); and the backward reads the weight transposed, which is a
different memory pattern on a GPU.

**Qwen2.5-3B QLoRA shapes, 253 calls/step (36 x 7 + head).** ffn 2048->11008 (x72) and 11008->2048 (x36) are
**78.9% of forward MACs** (derived arithmetic, not measured); attn_qo 2048->2048 9.8%; lm_head 2048->151936
10.1%; attn_kv 2048->256 1.2%. The LM head is **Q6_K, not Q4_K** (`GgufLlamaLoader.cs:647`) — a device-side
dequant needs two block formats.

**Never measured, and it blocks any end-to-end claim:** the fraction of a QLoRA step that is
`FrozenQuantizedLinear`. `QwenGgufTrainStepTimeTests` times the whole step and decomposes nothing, and its
own comment says it has never been run. Recorded as Risk R3.

**Do not cite the 1.96 TFLOP/s prefill figure as the QLoRA CPU baseline.** That is `GemmTiled`, the
*inference* prefill kernel (`ROADMAP-COMPLETED.md`, 672 rows, 2048->11008, 15.44 ms, 9950X3D, 2026-07-22).
The training path's closest analogue in the same table is `WeightStationary`, 53.62 ms, about 0.57 TFLOP/s —
roughly 3.5x slower. See [[reference-measured-baselines]].

**Isolation trick worth reusing:** a probe project outside `Overfit.sln`, with its own empty
`Directory.Build.props` sentinel and `ManagePackageVersionsCentrally=false`, referencing the **published**
`DevOnBike.Overfit` package. Keeps a risky dependency out of `Directory.Packages.props`, where
`NuGetAudit`/`NuGetAuditMode=all` + `NU1901-1904`-as-error would turn one advisory into a solution-wide build
failure. Cost: the CPU arm then runs the PUBLISHED method, not `HEAD`.

## Amendment 1 — the dotLLM precedent (same day)

`D:\dotLLM` (kkokosa/dotLLM, GPLv3) is a C#/.NET-10 inference engine with a WORKING CUDA backend, and
`docs/CUDA.md` is a written evaluation of this exact route question. **Read it before re-deriving any of
this.** They evaluated ILGPU 1.5.3 and rejected it for a SHIPPING engine: no tensor cores from custom
kernels (no `wmma`/`mma.sync`), no BF16, `~60-80% of native CUDA` (their word: **estimated**, not measured)
against `~98-100%` for their route — own P/Invoke + `.cu`->PTX via nvcc + cuBLAS, isolated assembly.

**Route unchanged (ILGPU) for four reasons.** Their negative evidence is `quantized_gemv.cu` = **GEMV,
n=1, decode** — the worst GPU case; QLoRA training is a GEMM at n=16..256, their *prefill* class, which
they route through `cublasHgemm`. **dotLLM has no backward pass, no autograd, no optimizer at all**
(`LoraAdapter.cs` is a serving-side record). Their bar was 98-100% for a product; a go/no-go probe can
spend a bounded 20-40%. Their route costs 6,323 lines + nvcc — the outcome the probe exists to justify.
And only ILGPU runs on this box (`CPUAccelerator`).

**I WAS WRONG that ILGPU has no tensor cores.** `ILGPU.Algorithms` ships a cuBLAS FP16 GEMM wrapper WITH
automatic tensor cores; the true gap is tensor cores from *custom* kernels. So the primary GPU arm is
cuBLAS HGEMM — and because cuBLAS is the SAME library the production route would call, the number
transfers across routes and ILGPU is only the harness. That dissolves "a probe in a tech we will not use
measures the wrong thing".

**The claimed dotLLM negative result is NOT in the tree.** No "underperforms CPU on small models" wording
anywhere in `*.md`/`*.cs`; `docs/GPU.md:289` 10-50x prefill / 3-10x decode are **targets**, and
`docs/BENCHMARKS.md:84`'s small-model sentence is dotLLM **CPU** vs llama.cpp. The real evidence is
uncited data in `docs/AOT.md:113-119`: GPU decode **1.6 tok/s** on Llama-3.2-1B Q4_K_M, 4.3 Q8_0, 3.7
Bielik-1.5B, but 43.6 on SmolLM-135M. **Card never named; no CPU arm in that table; 27x spread on a 7.4x
param ratio; both 1B rows use `--gpu-layers 16` = partial offload.** Do not put 1.6 next to our 24.4 CPU
tok/s — different model, quant, box, engine.

**Two traps to carry forward.** Pre-Volta (Pascal) cuBLAS FP16 falls back to CUDA cores at ~FP32 speed
(`docs/GPU.md:144`) — an ~8x silent verdict error, so always print whether tensor cores engaged. And FP16
is NOT numerically free: they measured Qwen2.5-0.5B diverging at layer 1, maxDiff 4.7, growing to 14.9 by
layer 24, enough to flip argmax (`docs/ROADMAP.md:180`); llama.cpp keeps FP32 residuals. Speed and
numerical viability are separate questions.

`nvcuda.dll` is in the driver; **`cublas64_*.dll` is NOT** — it needs the toolkit or a redistributable
(`docs/CUDA.md:74,683`).
