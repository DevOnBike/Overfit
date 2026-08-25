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
- **The flag stays opt-in, and that was measured rather than assumed.** The kernel behind it is worth
  **2.98×** at `pp512` on a model with no sidecar (325.26 ± 3.30 against 109.25 ± 1.68 t/s, `-t 32`,
  Qwen2.5-3B Q4_K_M, three interleaved fits). It was made default-on on 2026-08-25 and **reverted the same
  day**: the in-process repack costs **+1194 MiB** and **+186 ms once** at the start of decode, which leaves
  a short CLI invocation **9.5% slower for 2.4× the memory**. A `*.gguf.repack` sidecar reaches the same
  kernel with no managed heap and is the route being pursued. Numbers and conditions: `XC-119` in
  `docs/measured-baselines.md`.
- Adding bias support to the tiled prefill GEMM measured an **exact tie (0.999×)** — but that is the
  marginal effect of lifting the `bias.IsEmpty` gate, **not** a measurement of the flag, and it was read as
  the second for eighteen days. Its mechanism claim is half right: `ProjectBatchedWeightStationary` hoists
  the *scale* decode out of its row loop and re-does the *nibble* unpack per row, which is what the
  `block_q4_Kx8` tiling amortises. The "~3×" in the kernel docs is against re-decode-per-row.

## Where the engine stands

Decode is about **1.13× behind llama.cpp, uniformly across models** — not parity, and the figure is
best-of-N on both sides. FFN and LM head are at the DRAM bandwidth floor. Whole-matrix Q4_K attention
was validated as parallel-bandwidth-bound before any refactor, per the two-pass rule.

`MoeRouter` and `MoeFeedForwardBlock` cover mixture-of-experts; `norm_topk_prob` is architecture-aware
because qwen2moe uses raw softmax and Mixtral renormalises, and getting it wrong yields plausible
gibberish. `DraftModelSpeculativeDrafter` and `PromptLookupDrafter` implement speculative decoding
behind `ISpeculativeDrafter`.
