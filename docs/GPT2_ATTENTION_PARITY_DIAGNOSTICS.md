# GPT-2 attention parity diagnostics

Prior result:

```text
embedding  OK
block0_ln1 OK
block0_attn mismatch
```

This means the first real mismatch is inside block 0 attention.

## Main suspects

```text
1. GPT-2 c_attn.bias is ignored by Overfit.
2. Q/K/V split or head order differs.
3. c_proj output mapping does not match Overfit's per-head Wo layout.
```

## Run

Generate reference JSON with attention internals:

```bash
python3 Scripts/debug_gpt2_reference.py \
  --size small \
  --fixtures Tests/test_fixtures \
  --prompt "The future of software development is" \
  --out Tests/test_fixtures/gpt2_reference_small.json
```

Remove local Skip from:

```text
Gpt2Small_CompareAttentionInternalsAgainstPyTorchReference
```

Run:

```bash
dotnet test -c Release --filter "Gpt2Small_CompareAttentionInternalsAgainstPyTorchReference"
```

## Interpretation

If these are near-perfect:

```text
q raw vs PyTorch q WITHOUT bias
k raw vs PyTorch k WITHOUT bias
v raw vs PyTorch v WITHOUT bias
```

then the QKV weight orientation and split are correct.

If the same comparisons WITH bias are much worse, then the next code change is
clear: Overfit needs GPT-2-compatible Q/K/V bias support or a separate GPT-2
attention import path.

If Q/K/V without bias already differs badly, fix converter QKV weight orientation
or Q/K/V/head split before adding bias.

If Q/K/V and context are close but output projection differs, fix `c_proj` /
per-head `Wo` mapping.
