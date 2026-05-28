# Overfit for PyTorch Users

**For ML engineers coming from Python, curious about what .NET ML actually looks like in 2026.**

---

## The Honest Pitch

If you're a full-time ML researcher pushing the state of the art on transformers and diffusion, **Overfit is not for you today**. Your tools are PyTorch and JAX, your hardware is GPU, your life is fine.

Overfit is for a different problem:

- You've trained a model in PyTorch.
- The production system where it needs to run is C#.
- You've been handed the "deployment" task and you're tired of the Python sidecar story.

Or:

- You work somewhere with a strong .NET shop.
- You want to build ML features without fighting the infrastructure.
- You're curious whether C# ML tooling has matured enough to be usable.

This scenario is about **bridging the two worlds**.

---

## Mental Model Translation

If you know PyTorch, Overfit will feel broadly familiar. The core abstractions map closely:

| PyTorch | Overfit |
|---------|---------|
| `torch.Tensor` | `FastTensor<float>` |
| `nn.Module` | (interface `IModule`) |
| `nn.Sequential` | `Sequential` |
| `nn.Linear(in, out)` | `new LinearLayer(in, out)` |
| `nn.Conv2d(...)` | `new ConvLayer(...)` |
| `nn.BatchNorm1d(...)` | `new BatchNorm1D(...)` |
| `nn.ReLU()` | `new ReluActivation()` |
| `torch.optim.Adam(...)` | `new Adam(model.Parameters(), ...)` |
| `loss.backward()` | `graph.Backward(loss)` |
| `optimizer.step()` | `optimizer.Step()` |
| `optimizer.zero_grad()` | `optimizer.ZeroGrad()` |
| `model.eval()` | `model.Eval()` |
| `model.train()` | `model.Train()` |
| `torch.no_grad()` | Pass `null` for `ComputationGraph` |

The differences you'll notice:

### Autograd is explicit

PyTorch tracks gradients automatically whenever `requires_grad=True`. Overfit uses an explicit `ComputationGraph` handle that you pass in (or pass `null` for inference).

```python
# PyTorch
output = model(input)
loss = criterion(output, target)
loss.backward()
```

```csharp
// Overfit
var graph = new ComputationGraph();
var output = model.Forward(graph, input);
var loss = TensorMath.MSELoss(graph, output, target);
graph.Backward(loss);
```

This is more verbose but gives you control: `graph = null` means "just inference, don't record anything." No context manager, no implicit state.

### Memory is explicit

Tensors implement `IDisposable`. In training loops, you wrap them in `using var` to release memory deterministically:

```csharp
using var input = new FastTensor<float>(batchSize, features);
using var output = model.Forward(graph, input);
```

This is why Overfit doesn't trigger GC in the hot path — you control lifetime explicitly.

### No broadcasting (yet)

PyTorch broadcasts shapes implicitly. Overfit currently requires explicit shapes — implicit broadcasting is on the roadmap, but for now you might write a few more lines to reshape.

### CPU only (today)

Overfit is CPU-first. No `.cuda()`, no `.to(device)`. If you need GPU, you're in the wrong library.

---

## The MNIST "Hello World"

Here's a full training loop for MNIST in Overfit. Compare to a PyTorch equivalent — the structure is familiar.

```csharp
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Ops;

// 1. Build the model
var model = new Sequential(
    new LinearLayer(784, 128),
    new ReluActivation(),
    new LinearLayer(128, 64),
    new ReluActivation(),
    new LinearLayer(64, 10)
);

// 2. Set up training
using var optimizer = new Adam(model.Parameters(), learningRate: 0.001f);
model.Train();

// 3. Training loop
for (var epoch = 0; epoch < 10; epoch++)
{
    foreach (var batch in trainLoader)
    {
        var graph = new ComputationGraph();

        using var input = batch.Features; // [batchSize, 784]
        using var labels = batch.Labels;  // [batchSize, 10] one-hot

        using var logits = model.Forward(graph, input);
        using var loss = TensorMath.SoftmaxCrossEntropy(graph, logits, labels);

        graph.Backward(loss);

        optimizer.Step();
        optimizer.ZeroGrad();
        graph.Reset();
    }

    Console.WriteLine($"Epoch {epoch} done");
}

// 4. Evaluation
model.Eval();
using var testInput = testBatch.Features;
var predictions = model.Forward(null, testInput); // null graph = no autograd
```

Roughly the same lines of code as the PyTorch equivalent. Same concepts.

---

## Where PyTorch Still Wins

Being honest:

### Research agility

PyTorch lets you prototype wild architectures with minimal friction. Dynamic graphs, custom autograd functions, Python's ecosystem of papers-as-code. Overfit is not trying to compete here.

### HuggingFace Hub breadth

HuggingFace Transformers gives you thousands of pre-trained model architectures with two lines of Python. Overfit doesn't try to match that breadth, but for the model families that matter for production .NET use — Qwen2.5, Llama-2/3.x, Mistral, Mixtral, Qwen-MoE, BERT/MiniLM/BGE/E5, GPT-2 — the Transformer + MultiHeadAttention + MoE layers are shipped, and you load the same checkpoints (GGUF or safetensors) directly via `OverfitClient.LoadGguf(...)` or the native safetensors loader.

### GPU training

If your workflow involves training on a DGX or multi-GPU rig, stay in PyTorch.

### Community and tutorials

Stack Overflow has a million PyTorch questions answered. Overfit is new — expect to read source code sometimes.

---

## Where Overfit Actually Wins

Stuff you might not realize matters until you hit it:

### Deployment is trivial

A trained Overfit model is just a binary file and some C# code. No Python environment, no TorchScript compilation issues, no "it works in Jupyter but not in production" surprises.

```bash
dotnet publish -c Release -r linux-x64 /p:PublishAot=true
# ^ produces a single self-contained executable
```

### Inference is genuinely fast

On small models, Overfit beats ONNX Runtime by 10-40×. For embedded inference, real-time control loops, or latency-sensitive services, this matters.

### Debugging is the same as debugging C#

Break points in your IDE, step through the forward pass, inspect tensor contents. No Python-to-C++ boundary, no "the kernel died, restart the notebook."

### No Python versioning hell

If you've ever debugged "why does this PyTorch model break under Python 3.11 but work in 3.9" — you know the pain. `dotnet` versions it, you move on.

---

## Getting Weights From PyTorch to Overfit

For LLMs (Qwen / Llama / Mistral / Mixtral / Qwen-MoE / GPT-2), just point `OverfitClient.LoadGguf("model.gguf")` at the GGUF — done, zero conversion. For HF-style models with safetensors (BERT-family embeddings, base transformer checkpoints), the native `SafetensorsReader` + family loaders read them directly. For arbitrary PyTorch architectures, ONNX export → `OnnxImporter.Load` (linear) or `OnnxGraphImporter.Load` (DAG / ResNet skip connections) covers most cases. The hand-rolled binary path below is still the simplest option for small custom MLPs:

### Step 1: Export weights from PyTorch

```python
import torch
import struct

model = torch.load("my_model.pth")
model.eval()

with open("weights.bin", "wb") as f:
    for layer in [model.fc1, model.fc2, model.fc3]:
        # Weight matrix (output_dim × input_dim), transposed to row-major
        weights = layer.weight.detach().cpu().numpy().T.flatten()
        for w in weights:
            f.write(struct.pack('f', w))

        # Bias
        biases = layer.bias.detach().cpu().numpy().flatten()
        for b in biases:
            f.write(struct.pack('f', b))
```

### Step 2: Load weights in Overfit

```csharp
var model = new Sequential(
    new LinearLayer(784, 128),
    new ReluActivation(),
    new LinearLayer(128, 64),
    new ReluActivation(),
    new LinearLayer(64, 10)
);

model.Load("weights.bin");
model.Eval();
```

The `Sequential.Save` / `Load` format is a simple concatenation of layer parameters. Check the docstrings for the exact byte layout.

### Step 3: Validate parity

Run inference on the same input in both PyTorch and Overfit. Compare outputs element-wise. Expect differences on the order of 1e-6 (floating-point determinism across libraries isn't perfect but should be close).

```csharp
// Sanity check
var pytorchOutput = LoadReferenceOutput("pytorch_reference.bin");
var overfitOutput = model.Forward(null, testInput).DataView.AsReadOnlySpan();

for (var i = 0; i < pytorchOutput.Length; i++)
{
    Debug.Assert(Math.Abs(pytorchOutput[i] - overfitOutput[i]) < 1e-4f);
}
```

---

## What's Already Shipped That You Care About

If you last looked at Overfit a year ago, this list is probably new:

- **ONNX import** — linear topology (`OnnxImporter`) + DAG with skip connections (`OnnxGraphImporter`). Hand-rolled protobuf, no `Google.Protobuf` dependency.
- **Transformer building blocks** — `MultiHeadAttentionLayer`, `LayerNorm`, `RmsNorm`, RoPE, transformer block, full transformer training in the graph.
- **LLM inference end-to-end** — Qwen2.5 (0.5B-32B), Llama-2/3.x, Mistral, Mixtral, Qwen-MoE, GPT-2. GGUF mmap (Q4_K_M / Q6_K verbatim, Q8_0 de-interleaved) and native safetensors loader. One-line `OverfitClient.LoadGguf(...)`.
- **Sentence embeddings** — MiniLM / BGE / E5 with bit-parity vs HF/PyTorch (cosine 1.000000 on real weights).
- **Agentic primitives** — ReAct loop, Critic, CircuitBreaker, SummarizingChatSession, tool-calling, sliding-window eviction.
- **Learning rate schedules** — cosine, warmup, step (`Training/LearningRateSchedule`).
- **Quantization for inference** — Q4_K_M, Q6_K, Q8_0 via GGUF; FP32 training stays for now.

Still on the roadmap (see [ROADMAP.md](../../ROADMAP.md)):

- **Tensor broadcasting** — implicit shape broadcasting in elementwise ops.
- **Mixed-precision training** — FP16/BF16 in the training loop (inference is already quantized).
- **Decode tok/s catch-up** — closing the ~1.5× gap vs llama.cpp on the same Q4_K_M file via AVX2 dequant-fused GEMV + L2 blocking.

---

## Should You Try It?

Good reasons to try Overfit:

- You're deploying ML into a .NET application and the Python sidecar is painful.
- You need predictable inference latency (edge, real-time, game engines).
- You're curious about how ML tooling looks outside the Python monoculture.
- You want a smaller deployment footprint than ONNX Runtime provides.

Bad reasons:

- You're doing research. Stay in PyTorch.
- You need LLMs / diffusion / GPU-heavy work. Wrong library.
- You're allergic to C#. Fair enough.

---

## Further Reading

- [Main README](../../README.md) — benchmarks and overview
- [ROADMAP](../../ROADMAP.md) — current status and planned features
- [ASP.NET scenario](aspnet-microservice.md) — most common deployment target
- [Finance / Latency scenario](finance-latency.md) — where Overfit's characteristics shine