# GPT-2 stage parity diagnostics

Current symptom:

```text
Reference next token: 287 ' in'
Overfit next token:   318 ' is'
```

Tokenizer parity is already OK. The next step is finding the first stage where
Overfit diverges from PyTorch GPT-2.

## Run

Generate/update the reference JSON:

```bash
python3 Scripts/debug_gpt2_reference.py \
  --size small \
  --fixtures Tests/test_fixtures \
  --prompt "The future of software development is" \
  --out Tests/test_fixtures/gpt2_reference_small.json
```

Remove `Skip` locally from:

```text
Gpt2Small_CompareStagesAgainstPyTorchReference
```

Run:

```bash
dotnet test -c Release --filter "Gpt2Small_CompareStagesAgainstPyTorchReference"
```

## How to read the output

The diagnostic prints these stages:

```text
embedding
block0_ln1
block0_attn
block0_after_attn_residual
block0_ln2
block0_mlp
block0_output
final_norm
```

Interpretation:

```text
embedding mismatch
  token/position embeddings or checkpoint write order is wrong

block0_ln1 mismatch
  layernorm gamma/beta/epsilon/layout issue

block0_attn mismatch with embedding/LN OK
  attention mapping issue, likely Q/K/V bias, QKV split, WO split, or causal mask

block0_mlp mismatch with attention/residual/LN OK
  MLP c_fc/c_proj weight orientation or GELU variant issue

final_norm mismatch after good block0
  later block mapping issue
```

## Current prime suspects

```text
1. GPT-2 Q/K/V bias is ignored by Overfit MultiHeadAttentionLayer.
2. Attention output projection is split per head and may not match GPT-2 c_proj exactly.
3. GELU variant may differ from HuggingFace GPT-2 gelu_new.
4. Conv1D orientation may still be wrong for some projection.
```
