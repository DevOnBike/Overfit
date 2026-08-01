# `LanguageModels/Runtime` — the decode engine

Where a transformer actually generates tokens. This is the most performance-sensitive directory in the
repository and the one with the strictest allocation contract: **per-token decode allocates 0 bytes.**

## The layering, and why weights are never copied

```text
CachedSlmInferenceEngine       public entry; FromGpt1(model) wires the adapter
  CachedSlmSession             per-session state: KV buffers + position counter
    StackWeights               BlockWeights[] + final norm + LM head
      BlockWeights             layer norms + per-head attention + FFN
        SingleHeadWeights      ReadOnlySpan refs into TensorStorage for Q/K/V/O + biases
    KeyValueCache              pre-allocated K/V, O(N) decode
```

Every weight handle is a `ReadOnlySpan<float>` obtained from `TensorStorage` at decode time. Creating a
session therefore allocates the KV buffers (~80 MB for GPT-2 Small) and **nothing else** — no weight is
duplicated per session. `CachedGpt1ModelAdapter.RefreshWeightsFromModel()` is a deliberate no-op for
in-place weight updates, which is what the LoRA path relies on.

## Quantised paths

`Q4KDotKernel`, `Q4KGemvKernel` and `Q4KRepack` are the Q4_K decode kernels. `OVERFIT_REPACK_GEMV`
enables a repacked weight layout worth about **+30%** on Qwen-3B. Two traps, both paid for here:

- **`OVERFIT_TILED_PREFILL` is a dead flag** whenever a `*.gguf.repack` sidecar sits beside the model,
  because `IsPrepacked` short-circuits it. An A/B across that flag ran an identical mix in both arms
  and measured noise. Count the paths taken before believing any kernel A/B in this directory.
- Adding bias support to the tiled prefill GEMM measured an **exact tie (0.999×)**, because
  `ProjectBatchedWeightStationary` already amortises weight decode across the row tile — the same thing
  the tiling does. The "~3×" in the kernel docs is against re-decode-per-row, not against
  weight-stationary.

## Where the engine stands

Decode is about **1.13× behind llama.cpp, uniformly across models** — not parity, and the figure is
best-of-N on both sides. FFN and LM head are at the DRAM bandwidth floor. Whole-matrix Q4_K attention
was validated as parallel-bandwidth-bound before any refactor, per the two-pass rule.

`MoeRouter` and `MoeFeedForwardBlock` cover mixture-of-experts; `norm_topk_prob` is architecture-aware
because qwen2moe uses raw softmax and Mixtral renormalises, and getting it wrong yields plausible
gibberish. `DraftModelSpeculativeDrafter` and `PromptLookupDrafter` implement speculative decoding
behind `ISpeculativeDrafter`.
