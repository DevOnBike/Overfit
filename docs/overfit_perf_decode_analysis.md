# Overfit `perf` — decode performance analysis vs llama.cpp

Goal: increase decode throughput, i.e. generated tokens per second during model response.

Scope:

- Overfit branch: `https://github.com/DevOnBike/Overfit/tree/perf`
- Reference implementation: `https://github.com/ggml-org/llama.cpp`
- Focus: token-by-token decode path, not training, not general architecture cleanup.

This is an architecture and hot-path review based on code structure and public repository inspection. It is not a runtime benchmark result.

---

## Executive summary

The `perf` branch already has many important pieces in place:

- GGUF / mmap loading
- K-quant support: Q4_K / Q6_K / Q8_0-style paths
- KV cache
- batched prefill
- GQA-related optimization
- prompt-lookup speculative decode
- quant projection kernels
- ASP.NET / local-agent demo paths

The biggest remaining opportunity is not “add SIMD somewhere”. The main bottleneck is likely the decode hot path structure:

- too many separate projections per token,
- repeated activation quantization,
- per-head output projection,
- fragmented dispatch,
- intermediate scratch traffic,
- sampler/logits work that can be avoided in fast paths,
- speculative decode buffers that allocate in the current path.

The direction to copy from `llama.cpp` is not the exact C code, but the design shape:

- build a decode plan up front,
- use larger fused operations,
- pack weights for runtime layout, not loader convenience,
- separate prefill and decode threading policies,
- avoid intermediate score/logit buffers where possible,
- reduce per-token dynamic branching and dispatch.

---

## Current good signs in Overfit

The `perf` branch appears to contain the right runtime areas:

```text
Sources/Main/LanguageModels/Runtime/
  CachedLlamaInferenceEngine
  CachedLlamaSession
  CachedMultiHeadAttention
  CachedSingleHeadAttention
  CachedFeedForwardBlock
  Q4KDotKernel
  Q6KDotKernel
  BatchedQuantProjection
  BatchedAttentionKernel
  SpeculativeSampler
  TokenSampler
```

This is the right place to optimize. The repo already seems to be beyond the “toy runtime” phase. The remaining work is mostly hot-path consolidation and runtime layout.

---

## What llama.cpp does that matters here

`llama.cpp` / ggml gets much of its CPU performance from:

1. **Graph-level execution**
   - Work is scheduled as large graph operations.
   - The runtime avoids many small object-level dispatches.

2. **Typed tensor operations**
   - Quantized tensor types have specialized `vec_dot` implementations.
   - Runtime layout is optimized for matmul/dot traversal.

3. **Backend scheduler**
   - Prefill and decode can use different threading strategies.
   - Work is chunked based on size/type/backend.

4. **Flash / streaming attention paths**
   - Attention can avoid materializing unnecessary intermediate score buffers.

5. **Packed layout and blockwise kernels**
   - Data layout is chosen to feed hot kernels efficiently.

For Overfit, the best strategy is a managed equivalent of this style:

- precompute a `LayerDecodePlan`,
- pack layer weights for decode,
- fuse common projection groups,
- avoid repeated quantization of the same activation,
- reduce intermediate arrays and full-vocab passes.

---

# Recommended implementation plan

## Priority 0 — Add decode benchmark harness first

Before changing kernels, add one benchmark that isolates decode.

### Benchmark dimensions

```text
Model: Qwen2.5-3B Q4_K_M
Prompt: fixed, e.g. 128 tokens
Generate: 256 tokens

Modes:
- greedy, no constraints
- temperature/top-p
- JSON constraint
- tool-call constraint, if applicable
```

### Metrics

```text
- tokens/sec after prefill
- per-token latency p50/p95/p99
- allocated bytes/token
- attention time
- FFN time
- LM head time
- sampler time
- total layer time
```

### Minimal instrumentation

```csharp
DecodeProfiler.BeginToken();
DecodeProfiler.BeginLayer(i);
DecodeProfiler.Mark("qkv");
DecodeProfiler.Mark("attention");
DecodeProfiler.Mark("wo");
DecodeProfiler.Mark("ffn");
DecodeProfiler.Mark("lm_head");
DecodeProfiler.Mark("sampler");
DecodeProfiler.EndLayer(i);
DecodeProfiler.EndToken();
```

Without layer breakdown, you will not know whether the next bottleneck is FFN, attention, LM head, sampler, memory layout or thread overhead.

---

# PR 1 — LayerDecodePlan: remove hot-path dispatch

## Problem

Decode likely still contains dynamic decisions per token/layer:

- projection kind checks,
- shape guards,
- model-family checks,
- quant path dispatch,
- head/GQA conditional branches.

Each branch is small, but repeated across every layer and every token.

## Proposal

Build a `LayerDecodePlan` once at model load time.

```csharp
internal sealed class LayerDecodePlan
{
    public ProjectionKind QKind;
    public ProjectionKind KKind;
    public ProjectionKind VKind;
    public ProjectionKind OKind;

    public ProjectionKind GateKind;
    public ProjectionKind UpKind;
    public ProjectionKind DownKind;

    public bool IsSwiGlu;
    public bool IsGqa;

    public int DModel;
    public int HeadDim;
    public int HeadCount;
    public int KvHeadCount;
    public int Dff;
}
```

Compile a layer decode delegate once:

```csharp
internal delegate void DecodeLayerFn(
    LayerRuntimeState state,
    ReadOnlySpan<float> input,
    Span<float> output,
    int position);
```

Runtime loop:

```csharp
for (int i = 0; i < _layers.Length; i++)
{
    _decodeFns[i](_states[i], current, next, position);
    Swap(ref current, ref next);
}
```

## Acceptance criteria

```text
- no extra allocations
- same output as old path within existing tolerance
- decode benchmark improves or stays neutral
- dynamic dispatch moved out of token loop
```

## Expected impact

Small to medium alone, but important as foundation for fused kernels.

---

# PR 2 — Fused QKV projection

## Problem

Attention likely performs separate projections for Q, K and V, sometimes at per-head or per-group granularity.

Even if the dot kernels are fast, repeated projection setup and repeated activation preparation hurt decode throughput.

## Proposal

For one token, compute all Q/K/V projections in one fused operation:

```text
hidden[dModel]
  -> Q all heads:   [nHeads * headDim]
  -> K all kvHeads: [nKvHeads * headDim]
  -> V all kvHeads: [nKvHeads * headDim]
```

API sketch:

```csharp
public static class QkvProjectionKernel
{
    public static void ProjectQkv(
        ReadOnlySpan<float> hidden,
        QkvWeights weights,
        Span<float> q,
        Span<float> k,
        Span<float> v,
        QuantizedActivationScratch scratch);
}
```

For Q4_K / Q6_K:

```text
- quantize hidden once to Q8_K-style activation scratch
- reuse that same activation for Q, K and V
- run quant dot projections over packed Q/K/V weights
```

## Better runtime layout

Create a packed structure at load/finalization time:

```csharp
internal sealed class PackedQkvWeights
{
    public QuantizedMatrix Wq;
    public QuantizedMatrix Wk;
    public QuantizedMatrix Wv;
    public ProjectionKind Kind;
}
```

Later, evolve this into physically interleaved packed QKV if it improves cache locality.

## Acceptance criteria

```text
- one activation quantization for Q/K/V
- all Q/K/V outputs match old path
- 0 B allocation
- layer attention projection benchmark improves
- full decode benchmark improves
```

## Expected impact

High. This is one of the top ROI changes.

---

# PR 3 — Full output projection (`Wo`) instead of per-head O projection

## Problem

If the current attention output projection is done per head, the runtime performs many small `Wo` projections.

That is likely slower than:

```text
attention output per head -> concatenate to [dModel] -> one full Wo projection
```

## Proposal

Change attention output path:

```text
1. Compute each head attention output into attnConcat[dModel].
2. Run one full Wo projection: dModel -> dModel.
3. Add residual / bias as needed.
```

New scratch:

```csharp
internal sealed class AttentionScratch
{
    public float[] Q;
    public float[] K;
    public float[] V;
    public float[] AttnConcat;
    public float[] AttnProjected;
    public QuantizedActivationScratch AttnQ8;
}
```

New API:

```csharp
public static class OutputProjectionKernel
{
    public static void ProjectFull(
        ReadOnlySpan<float> attnConcat,
        PackedWoWeights weights,
        Span<float> output,
        QuantizedActivationScratch scratch);
}
```

## Acceptance criteria

```text
- output matches old per-head projection path
- one Wo call per layer per token
- no per-head Wo dispatch
- 0 B allocation
- decode benchmark improves
```

## Expected impact

High, especially for many-head models.

---

# PR 4 — Fused SwiGLU gate + up projection

## Problem

SwiGLU FFN typically does:

```text
gate = hidden @ W_gate
up   = hidden @ W_up
ffn  = silu(gate) * up
out  = ffn @ W_down
```

If gate and up each quantize the same hidden activation separately, that is wasted work.

FFN often dominates FLOPs per layer, so this is a major target.

## Proposal

Add fused gate/up projection:

```csharp
public static class SwiGluProjectionKernel
{
    public static void ProjectGateUpAndActivate(
        ReadOnlySpan<float> hidden,
        PackedFfnWeights weights,
        Span<float> ffnHidden,
        QuantizedActivationScratch hiddenQ8,
        Span<float> gateScratch,
        Span<float> upScratch);
}
```

Minimum version:

```text
- quantize hidden once
- project gate
- project up
- apply silu(gate) * up into ffnHidden
- run down projection
```

Better version:

```text
- compute gate/up by blocks
- immediately apply silu * up
- avoid storing full gate/up if possible
```

## Acceptance criteria

```text
- parity vs old FFN path
- one hidden activation quantization for gate/up
- 0 B allocation
- FeedForwardDecodeBenchmark improves
- full decode benchmark improves
```

## Expected impact

High. This is probably the best first kernel PR because it is less risky than attention but affects a large part of decode.

---

# PR 5 — Streaming / flash-style attention for single-token decode

## Problem

If attention materializes score arrays per head/token/context, memory traffic grows with context length.

For single-token decode, attention can be computed using online softmax without storing all scores.

## Proposal

Implement stable streaming attention:

```csharp
public static void ComputeSingleHeadStreaming(
    ReadOnlySpan<float> q,
    KeyValueCacheView kv,
    Span<float> output,
    int visibleLength,
    float scale)
{
    // online softmax:
    // m = running max
    // l = running denominator
    // output = running weighted sum of V
}
```

Online algorithm:

```text
m_new = max(m, score)
alpha = exp(m - m_new)
beta  = exp(score - m_new)

out = out * alpha + v * beta
l   = l * alpha + beta
m   = m_new

final: out /= l
```

## Benefits

```text
- no scoreScratch array
- less memory traffic
- better long-context decode behavior
- closer in spirit to llama.cpp flash attention path
```

## Acceptance criteria

```text
- numerical tolerance vs old attention
- no per-token score array allocation
- lower memory traffic
- long-context decode benchmark improves
```

## Expected impact

Medium for short context, high for long context.

---

# PR 6 — Zero-allocation speculative decode buffers

## Problem

The speculative decode path appears to allocate temporary arrays for:

- candidate tokens,
- embeddings,
- hidden states,
- logits,
- residual logits / last logits.

That cancels part of the benefit of speculative decode in streaming scenarios.

## Proposal

Create one scratch object per session:

```csharp
internal sealed class SpeculativeDecodeScratch
{
    public int[] CandidateTokens = Array.Empty<int>();
    public float[] CandidateEmbeddings = Array.Empty<float>();
    public float[] CandidateHidden = Array.Empty<float>();
    public float[] CandidateLogits = Array.Empty<float>();
    public float[] LastLogits = Array.Empty<float>();
    public float[] ResidualLogits = Array.Empty<float>();

    public void EnsureCapacity(int maxBatch, int dModel, int vocabSize)
    {
        // grow only when needed
    }
}
```

Avoid:

```csharp
new int[history.Length + 1]
new float[batch * dModel]
new float[batch * vocab]
new float[vocab]
```

Use ring buffers / spans over existing history rather than copying tokens into new arrays.

## Acceptance criteria

```text
- speculative decode path has stable allocations after session creation
- 0 B/token or near-zero allocation under benchmark
- same generated output under deterministic settings
```

## Expected impact

Medium if speculative decode is used; low otherwise. Important for credibility of low-allocation claims.

---

# PR 7 — Greedy `ProjectArgMax` fast path

## Problem

For greedy decode, if no constraints / repetition penalties / logprobs are needed, the runtime may not need to materialize the entire logits vector.

Current generic path likely:

```text
hidden -> logits[vocab]
argmax(logits)
```

For pure greedy:

```text
hidden -> best token directly
```

## Proposal

Add LM head fast path:

```csharp
public static int ProjectArgMax(
    ReadOnlySpan<float> hidden,
    PackedLmHeadWeights weights,
    QuantizedActivationScratch scratch,
    out float bestLogit);
```

Only use when:

```text
- greedy sampling
- no grammar / JSON constraint
- no repetition penalty
- no top-k / top-p
- no logprobs requested
- caller does not need full logits
```

## Acceptance criteria

```text
- same token as full logits + argmax
- no full logits buffer write
- no separate argmax pass
- no allocation
```

## Expected impact

Small to medium. More useful when vocab is large.

---

# PR 8 — Top-k/top-p sampler optimization

## Problem

Sampling paths often waste time over the entire vocabulary:

- full sorting,
- full softmax,
- repeated masking,
- full-vocab scan even when only a small allowed token set exists.

## Proposal

Add fast paths:

```text
1. Greedy: ProjectArgMax
2. Top-k: partial selection / min-heap, no full sort
3. Top-p: cap to top-k first, then top-p on subset
4. Grammar constraint: enumerate allowed tokens when possible
```

Extend constraint API:

```csharp
public interface ITokenConstraint
{
    bool CanEnumerateAllowedTokens { get; }

    ReadOnlySpan<int> GetAllowedTokens();

    void MaskInPlace(Span<float> logits);
}
```

For JSON/tool constraints:

```text
if allowed-token set is compact:
    iterate only allowed tokens
else:
    fall back to full-vocab masking
```

## Acceptance criteria

```text
- same distribution behavior within expected stochastic variance
- greedy unchanged
- top-k/top-p benchmark improves
- JSON/tool-call constrained decode improves when allowed set is small
```

## Expected impact

Medium for sampling and constrained JSON/tool calling.

---

# PR 9 — Separate decode vs prefill threading policy

## Problem

The optimal threading policy for prefill is not the same as for single-token decode.

Prefill has larger matrix work and can use more parallelism. Decode has smaller per-token tasks and can lose performance to thread scheduling overhead or cache contention.

## Proposal

Add dedicated threading options:

```csharp
public sealed class DecodeThreadingOptions
{
    public int DecodeThreads { get; init; }
    public int PrefillThreads { get; init; }
    public int MinRowsPerWorker { get; init; }
    public bool UsePersistentThreadPool { get; init; }
    public bool PinWorkers { get; init; }
}
```

Heuristic:

```text
if rows < MinRowsPerWorker * threads:
    use fewer workers or single-thread
else:
    parallelize
```

Separate:

```text
prefill: more threads, batch-friendly
decode: lower overhead, persistent workers
```

## Acceptance criteria

```text
- benchmark with 1, 2, 4, 8, 16 decode threads
- benchmark prefill separately
- default chosen by model size / CPU core count
- no regression on small models
```

## Expected impact

Medium to high depending on CPU and current thread overhead.

---

# PR 10 — Packed decode weight layout

## Problem

Loader-friendly layout is not always runtime-friendly layout.

For decode, weights should be packed to match the traversal order of the dot kernels and cache lines.

## Proposal

Add finalization after loading:

```csharp
public sealed class DecodePackedLayerWeights
{
    public PackedQkvWeights Qkv { get; init; }
    public PackedWoWeights Wo { get; init; }
    public PackedFfnWeights Ffn { get; init; }
    public PackedLmHeadWeights? LmHead { get; init; }
}
```

Goals:

```text
- Q/K/V packed for fused projection
- Wo as one full projection matrix
- gate/up packed for fused SwiGLU
- 32/64-byte alignment
- block order matches Q4_K / Q6_K dot kernel traversal
```

## Acceptance criteria

```text
- one-time load/finalization cost acceptable
- memory overhead measured and documented
- decode speed improves
- no output regression
```

## Expected impact

High after fused kernels exist. Lower if done before PR 2–4.

---

# Recommended order

## Phase 0 — Measurement

```text
1. Decode benchmark harness
2. Layer breakdown profiler
3. Allocation-per-token guard for decode modes
```

## Phase 1 — Biggest ROI kernels

```text
1. Fused SwiGLU gate+up projection
2. Full-Wo attention output projection
3. Fused QKV projection
```

## Phase 2 — Context and sampler

```text
4. Streaming attention without scoreScratch
5. Greedy ProjectArgMax
6. Top-k/top-p partial sampler
```

## Phase 3 — Production polish

```text
7. Zero-allocation speculative decode buffers
8. Decode/prefill thread policy
9. Packed decode weights
10. Specialized LayerDecodePlan
```

---

# Suggested first PR

Start with:

```text
Fused SwiGLU Q4_K / Q6_K gate+up projection
```

Why:

- FFN is a large part of decode cost.
- It is less risky than attention.
- It can be benchmarked in isolation.
- It removes repeated activation quantization.
- It should preserve semantics cleanly.

## Files likely involved

```text
Sources/Main/LanguageModels/Runtime/CachedFeedForwardBlock.cs
Sources/Main/LanguageModels/Runtime/Q4KDotKernel.cs
Sources/Main/LanguageModels/Runtime/Q6KDotKernel.cs
new: Sources/Main/LanguageModels/Runtime/SwiGluProjectionKernel.cs
new benchmark: FeedForwardDecodeBenchmark
```

## Acceptance criteria

```text
- parity vs old FFN path
- 0 B allocation
- one activation quantization for gate/up
- isolated FFN benchmark improves
- full decode benchmark improves
```

---

# Realistic throughput milestones

Do not promise immediate llama.cpp parity.

A realistic path might look like:

```text
Milestone 1:
19 -> 22–24 tok/s
via fused SwiGLU, sampler fast path, speculative buffer cleanup

Milestone 2:
24 -> 26–28 tok/s
via full-Wo, fused QKV, thread policy

Milestone 3:
approach llama.cpp-class throughput on selected CPU/model/quant
via packed decode layout, streaming attention, stronger batching/speculative path
```

Overfit’s strongest public positioning should remain:

```text
Not the fastest raw decoder.
But pure C#/.NET, in-process, Native-AOT-friendly,
low allocation, no Python, no Ollama, no model server, no data egress.
```

That is the real product angle.
