# TinyShakespeare GPT demo

This folder contains the end-to-end GPT-style TinyShakespeare demo.

It validates:

```text
training a small GPT-style character-level model
writing checkpoint.bin
loading checkpoint.bin
cached KV runtime generation
legacy/cached greedy parity
0 B managed allocation continuation hot path
```

---

## What this demo is

A small character-level GPT-style language model trained on TinyShakespeare.

It is useful for validating Overfit's GPT training and runtime pipeline.

---

## What this demo is not

It is not a general-purpose assistant.

It does not answer arbitrary questions.

It is not a production LLM.

---

## Checkpoint

The demo writes and reads:

```text
test_fixtures/checkpoint.bin
```

The current checkpoint is tied to:

```text
GPT config
TinyShakespeare character tokenizer
test_fixtures/tiny_shakespeare.txt
```

A future model-package format should include model config and tokenizer metadata.

---

## Recommended quality training preset

```text
SeqLen = 128
BatchSize = 4
Steps = 20,000
DModel = 128
Heads = 4
Layers = 4
DFF = 512
LR = 3e-4 -> 3e-5
WeightDecay = 0.05
GradClip = 1.0
```

Training tests should remain skipped by default.

---

## Show checkpoint

After `checkpoint.bin` exists:

```bash
dotnet test -c Release --filter "Demo_LoadCheckpoint_AndShowCachedRuntimeGeneration_RepetitionAware"
```

The display-oriented show test uses:

```text
TopK = 16
Temperature = 0.85
RepetitionPenalty = 1.15
RepetitionWindow = 64
```

The repetition penalty is display-only and does not modify checkpoint weights.
