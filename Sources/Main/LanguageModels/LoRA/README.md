# `LanguageModels/LoRA` — low-rank fine-tuning, including on quantised bases

Adapters that train a small pair of matrices beside a frozen base, so a model can be specialised
without touching or duplicating its weights. `ILoRAAdapter` / `ILoRAInjectable` are the seams;
`Gpt1LoRAFineTuner` and `QLoRAFineTuner` are the facades; `Gpt1LoRAFile` is the on-disk format.

## QLoRA is the reason this matters

`QLoRAFineTuner` trains adapters against a **frozen Q4_K base**: the base is never dequantised in
full, only row-by-row as the forward pass needs it (see `IDequantRowSource` in `../../Autograd`). A
real Qwen-3B fine-tune runs in about **3 GB of RAM on a CPU**, which is a thing llama.cpp cannot do —
it infers, it does not train.

One measured gotcha: `AdamEpsilon` must be raised to **1e-4** on this path. The default epsilon
interacts badly with the quantised base's gradient scale, and the symptom is a run that trains and
learns nothing rather than one that fails.

## Merging back

`Gpt1LoRAMergeAdapter` folds adapters into the base so inference takes the fast path. Measured, and the
trade is explicit:

| Path | Speed vs trainable | Fidelity |
|---|---|---|
| Merged into Q8 base | 3.49× | cosine 0.99994 |
| Merged, Q4_K-partial (opt-in) | 1.02× vs preset | cosine 0.965 |

The Q4_K-partial merge is opt-in precisely because 0.965 is a visible quality change for a 2% speed
gain — a bad default, a reasonable choice for someone who has measured their own case.
