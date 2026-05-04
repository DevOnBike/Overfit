# Overfit

Pure C# deep-learning and optimization engine.

Predictable CPU performance. Explicit memory ownership. Zero-allocation inference hot paths.

No native binaries. No Python runtime. No ONNX Runtime dependency.

---

## What it does

Overfit is a managed .NET engine for small and medium CPU workloads where predictable memory behaviour matters.

Current focus areas:

- **Zero-allocation CPU inference** — caller-owned buffers, preallocated runtime state, no per-call GC pressure on hot paths.
- **ONNX import** — load PyTorch-exported models into native Overfit models.
- **Autograd training** — train small neural networks and language models in pure C#.
- **GPT-style language models** — small GPT-style character models, checkpointing, cached KV generation.
- **Evolutionary optimization** — allocation-free `Ask` / `AskThenTell` loops for black-box search.

---

## Quick start

### Inference — native model

```csharp
using DevOnBike.Overfit.Inference;

var model = new Sequential(
    new LinearLayer(784, 128),
    new ReluActivation(),
    new LinearLayer(128, 10));

model.Load("model.bin");
model.Eval();

using var engine = InferenceEngine.FromSequential(
    model,
    inputSize: 784,
    outputSize: 10);

Span<float> input = stackalloc float[784];
Span<float> output = stackalloc float[10];

engine.Run(input, output); // zero-allocation
```

### Inference — ONNX import

```csharp
using DevOnBike.Overfit.Onnx;
using DevOnBike.Overfit.Inference;

var model = OnnxImporter.Load("classifier.onnx");
model.Eval();

using var engine = InferenceEngine.FromSequential(
    model,
    inputSize: 784,
    outputSize: 10);

var prediction = engine.Predict(input); // ReadOnlySpan<float>, 0 B
```

### Inference — ONNX DAG topology

Use the graph importer for skip connections, residual blocks, or branching topology.

```csharp
using DevOnBike.Overfit.Onnx;
using DevOnBike.Overfit.Inference;

var dagModel = OnnxGraphImporter.Load(
    "resnet.onnx",
    inputSize: 784,
    outputSize: 10);

dagModel.Eval();

var backend = new OnnxGraphInferenceBackend(dagModel);
using var engine = InferenceEngine.FromBackend(backend);

var prediction = engine.Predict(input); // ReadOnlySpan<float>, 0 B
```

---

## GPT-style language model demo

Overfit includes an end-to-end TinyShakespeare GPT-style demo.

In plain terms, the demo proves that Overfit can:

1. train a small GPT-style character-level language model,
2. save the trained weights into `checkpoint.bin`,
3. load `checkpoint.bin`,
4. generate text from a prompt,
5. use a cached KV runtime,
6. verify cached generation against the legacy/reference path,
7. keep the cached continuation hot path at **0 B managed allocations**.

This is not a general-purpose assistant. It is a small Shakespeare-style character model used to validate the GPT training and runtime pipeline.

### What the checkpoint is

`checkpoint.bin` is the trained model weights for the TinyShakespeare demo.

The current demo checkpoint is tied to:

```text
GPT config
TinyShakespeare character tokenizer
test_fixtures/tiny_shakespeare.txt
```

A future model-package format should include config and tokenizer metadata next to the weights.

### Load and show checkpoint

After `test_fixtures/checkpoint.bin` exists:

```bash
dotnet test -c Release --filter "Demo_LoadCheckpoint_AndShowCachedRuntimeGeneration_RepetitionAware"
```

This uses cached KV generation with display sampling:

```text
TopK = 16
Temperature = 0.85
RepetitionPenalty = 1.15
RepetitionWindow = 64
```

Expected validations:

```text
Legacy parity: OK
Continuation allocation check: 0 B
Time per token: around 1.2 ms/token on the demo model
```

### Manual checkpoint training

Checkpoint regeneration is a manual long-running demo test.

```text
Tests/LanguageModels/Demo/TinyShakespeare/
```

Long-running training tests are intentionally skipped by default:

```csharp
[Fact(Skip = "Manual long-running GPT demo. Remove Skip locally, run once, then restore Skip.")]
```

Recommended sequential quality preset for a demo checkpoint:

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

### Experimental data-parallel training

Data-parallel TinyShakespeare training is kept separate and manual:

```text
Tests/LanguageModels/Experimental/
```

Observed local throughput on Ryzen 9 9950X3D:

```text
8 workers:  ~360 seq/s
12 workers: ~437 seq/s
16 workers: ~457 seq/s
```

This is an experimental performance path, not the default correctness path.

---

## Test policy

Default:

```bash
dotnet test -c Release
```

should run fast correctness tests only.

Manual tests must use a clear `Skip` reason:

```csharp
[Fact(Skip = "Manual long-running GPT demo. Remove Skip locally, run once, then restore Skip.")]
```

Fast GPT runtime acceptance tests should remain unskipped.

Important GPT validation areas:

```text
cached greedy == legacy greedy
GenerateNextToken loop == Generate
context overflow behavior
session reset behavior
0 B cached continuation hot path
checkpoint load/show
```

---

## Benchmark snapshot

Local machine used for recent measurements:

```text
AMD Ryzen 9 9950X3D
Windows 11 25H2
.NET 10
BenchmarkDotNet 0.15.8
```

### Linear inference — Overfit vs ONNX Runtime

```text
Overfit InferenceEngine: 0 B managed allocations on hot path
ONNX Runtime preallocated path: small managed allocation remains
```

The exact numbers depend on benchmark shape and branch state. Use BenchmarkDotNet results for final claims.

### GPT TinyShakespeare cached runtime

Recent demo result:

```text
cached generation: around 1.2 ms/token
legacy/cached greedy parity: OK
cached continuation allocation: 0 B for measured continuation tokens
```

### GPT training

Single-model sequential training is the quality baseline.

Experimental data-parallel training increases throughput but changes global batch dynamics. It is useful for performance exploration, not a replacement for the default quality checkpoint path yet.

---

## ONNX import

Supported ONNX operators include:

```text
Conv
Gemm
Relu
Tanh
Sigmoid
Softmax
MaxPool
GlobalAveragePool
BatchNormalization
Add
Reshape
Flatten
Identity / Dropout no-op in eval mode
```

Two importers:

- `OnnxImporter` — linear topology, faster for simple CNN/MLP models.
- `OnnxGraphImporter` — DAG topology, required for skip connections and residual blocks.

External `.data` files from PyTorch exports are resolved automatically.

---

## Architecture

```text
InferenceEngine
  zero-allocation inference facade

Sequential
  module composition

ComputationGraph
  autograd tape and backward execution

Parameter
  long-lived trainable state with Data + Grad storage

TensorStorage
  unmanaged memory ownership

Kernels
  pure Span-based math, no AutogradNode ownership

Optimizers
  Adam, SGD over Parameter collections

ONNX
  OnnxImporter and OnnxGraphImporter

GPT1Model
  Embedding + N x TransformerBlock + final norm + LM head

Cached SLM runtime
  cached KV token-by-token generation

Tokenizers
  CharacterTokenizer and BytePairEncoder
```

---

## Current GPT status

Complete and working:

```text
GPT1Model
TransformerBlock
MultiHeadAttention
ScaledDotProductAttention
LayerNorm
Embedding
GELU FFN
CharacterTokenizer
Save/Load checkpoint
Cached KV runtime
Legacy/cached parity checks
0 B continuation hot path
TinyShakespeare checkpoint demo
```

Experimental:

```text
data-parallel TinyShakespeare training
parallel attention backward behind experimental flag
```

Not yet complete:

```text
self-contained model package with config + tokenizer metadata
general-purpose instruction-following model
production trainer API for GPT
```

---

## Requirements

- .NET 10+
- No native dependencies
- No Python runtime required for normal use
- Native AOT friendly design goals

---

## What Overfit is not

Overfit is not a PyTorch or TensorFlow replacement.

Overfit is not a general-purpose LLM assistant.

The differentiator is pure C#, explicit memory ownership, predictable CPU behavior, and zero-allocation inference paths for small and medium models where managed runtime control matters.
