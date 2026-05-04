# GPT runtime milestone

## Summary

Overfit now has an end-to-end GPT-style language-model pipeline for a small TinyShakespeare character model.

The milestone is not "ChatGPT in C#". The milestone is a working GPT-style stack:

```text
training
checkpoint save
checkpoint load
cached KV generation
legacy/cached parity validation
0 B managed allocation continuation path
```

---

## Demo flow

```text
TinyShakespeare corpus
  -> CharacterTokenizer
  -> GPT1Model training
  -> checkpoint.bin
  -> checkpoint load
  -> cached KV runtime generation
  -> validation
```

---

## Current demo model

```text
vocab = 68
context = 128
dModel = 128
heads = 4
layers = 4
dFF = 512
parameters ~= 825k
```

---

## Display generation

Current display sampling:

```text
TopK = 16
Temperature = 0.85
RepetitionPenalty = 1.15
RepetitionWindow = 64
```

This improves demo readability without changing checkpoint weights.

---

## Validations

The checkpoint show demo validates:

```text
checkpoint loads
cached generation produces text
cached greedy matches legacy greedy
cached continuation allocation is 0 B
generation latency is measured
generated text has no null characters
```

Recent local result:

```text
Time per token: around 1.2 ms/token
Legacy parity: OK
Continuation allocation check: 0 B
```

---

## Experimental performance

Data-parallel TinyShakespeare training is isolated under experimental tests.

Observed local throughput on Ryzen 9 9950X3D:

```text
8 workers:  ~360 seq/s
12 workers: ~437 seq/s
16 workers: ~457 seq/s
```

This is useful for performance exploration, but changes global batch dynamics and is not the default quality path.

---

## Limitations

Current checkpoint is not self-contained. It assumes:

```text
same GPT config
same TinyShakespeare corpus/tokenizer
same checkpoint format
```

Future improvement:

```text
checkpoint.bin
config.json
tokenizer/vocab metadata
format version
```

The model is a small character-level language model. It is not a general-purpose assistant and does not perform instruction following.
