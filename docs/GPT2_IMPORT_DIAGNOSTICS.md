# GPT-2 import diagnostics

Current smoke tests prove:

```text
vocab.json loads as 50257 tokens
gpt2_small.bin loads into Overfit
cached runtime executes
token ids stay inside vocab range
```

They do not prove semantic GPT-2 correctness. Current generated text is repetitive:

```text
is is is ...
time time time ...
```

This usually means the imported weight layout or model architecture mapping is not equivalent to HuggingFace GPT-2.

## Diagnostic flow

Generate/download Overfit GPT-2 fixtures:

```bash
python3 Scripts/convert_gpt2.py --size small --out Tests/test_fixtures/
```

Install Python dependencies:

```bash
pip install torch transformers numpy
```

Generate PyTorch reference logits:

```bash
python3 Scripts/debug_gpt2_reference.py \
  --size small \
  --fixtures Tests/test_fixtures \
  --prompt "The future of software development is" \
  --out Tests/test_fixtures/gpt2_reference_small.json
```

Remove `Skip` locally from:

```text
Tests/LanguageModels/Diagnostics/Gpt2ImportParityDiagnostics.cs
```

Run:

```bash
dotnet test -c Release --filter "Gpt2Small_CompareFinalLogitsAgainstPyTorchReference"
```

## What to inspect

The diagnostic prints:

```text
Reference top logits
Overfit top logits
Top-k overlap
Max abs diff on top-token union
Reference next token
Overfit next token
```

Expected current result: tokenizer ids should match, logits likely do not.

## Likely mismatch sources

```text
Q/K/V bias ignored
attention output projection per-head mapping
Conv1D weight orientation
GELU exact variant
LayerNorm epsilon/order
LM head/tied weight layout
causal attention/KV cache semantics
```

Start with the first mismatch. Do not optimize sampler or decoding until final logits match the PyTorch reference.
