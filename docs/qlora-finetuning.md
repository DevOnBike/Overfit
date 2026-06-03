# Fine-tune an LLM on your CPU, in pure .NET

Overfit can **fine-tune a real, already-quantized LLM (Qwen/Llama GGUF) on a CPU** — no GPU, no Python,
no CUDA. The 4-bit base never leaves its quantized form (it is never expanded to F32 or rewritten); only a
small **LoRA adapter** trains. This is the QLoRA recipe (frozen 4-bit base + trainable low-rank adapters),
implemented end-to-end in C#.

This is the thing `llama.cpp` structurally can't do (it has no training) and PyTorch-QLoRA needs CUDA + Python
for.

## Quick start (API)

```csharp
using DevOnBike.Overfit.LanguageModels.LoRA;

using var tuner = new QLoRAFineTuner(@"C:\qwen3b\qwen.q4km.gguf");   // tokenizer.json must be in the same dir

tuner.FineTuneOnFile("my-notes.txt");          // teach it your text
tuner.SaveAdapter("my-notes.lora");            // ~tens of MB; the 2 GB base is untouched

Console.WriteLine(tuner.Ask("What does my-notes say about X?"));
```

Reuse a saved adapter later without retraining:

```csharp
using var tuner = new QLoRAFineTuner(@"C:\qwen3b\qwen.q4km.gguf");
tuner.LoadAdapter("my-notes.lora");
Console.WriteLine(tuner.Ask("..."));
```

## Quick start (demo console)

```powershell
# fine-tune on a text file, then drop into an interactive chat
dotnet run -c Release --project Demo/QLoRAFineTuneDemo -- C:\qwen3b\qwen.q4km.gguf my-notes.txt 3

# or just chat with a previously-saved adapter
dotnet run -c Release --project Demo/QLoRAFineTuneDemo -- C:\qwen3b\qwen.q4km.gguf --adapter my-notes.lora
```

## What it does (the demo that proves it)

Teach the model a fact it cannot possibly know — an invented metal **"Zorvex"** mined only in **"Tarnholm"** —
then ask:

```
BEFORE: "The only known mine of Zorvex is in the city of" -> "gow…"          (base is clueless)
fine-tune on 3 sentences …                                                    loss 14.67 -> 0.0000
AFTER:  "The only known mine of Zorvex is in the city of" -> "Tarnholm. Zorvex"   (recites it)
```

The frozen 4-bit base is **bit-identical** before and after — all the new knowledge lives in the adapter.

## Settings (`QLoRAOptions`)

Defaults are the empirically validated known-good values. Most callers only change `Epochs`.

| Option | Default | Notes |
|---|---|---|
| `Rank` | 8 | adapter capacity; higher = more capacity + larger adapter |
| `Epochs` | 3 | passes over the text. 1 = light; 3–5 = solid learning; many = memorization |
| `ChunkLength` | 256 | tokens per training sequence (text is chunked into these) |
| `LearningRate` | 0.002 | validated; 0.002–0.005 is stable |
| `GradientClipNorm` | 0.5 | keeps the LM-head LoRA stable |
| `LoRAOnLmHead` | true | output capacity — needed to recite specific new tokens |
| `AdamEpsilon` | **1e-4** | **do not lower.** The default 1e-8 makes Adam blow up at low loss on small sets |

> The single most important non-obvious setting is **`AdamEpsilon = 1e-4`**. With the usual 1e-8, the loss
> spikes catastrophically once it gets low on a small overfit set.

## How long / how much RAM (measured, real Qwen2.5-3B Q4_K_M on CPU)

- **Training:** ~0.084 s/token (≈ 21 s per 256-token step). Time scales with `tokens × epochs`.
  - **30 A4 pages** (~20k tokens EN / ~30k PL): **≈ 1–3.5 h** for 3 epochs (one pass ≈ 28 min EN).
- **RAM:** ~3 GB peak with gradient checkpointing (the 4-bit base is ~2 GB, never expanded). Fits a 16 GB box.
  - For reference, expanding the base to F32 would be ~11.5 GB — which is exactly what this path avoids.
- **Generation (`Ask`):** ~0.4 s/token.

## Honest scope / limits

- **CPU correctness-first, not speed-optimized.** The training kernels dequantize on the fly with parallel
  but un-fused matmuls; a repacked-GEMV kernel (roadmap "fast fine-tuned decode") would be several× faster.
  This is about *capability* (fine-tune on a laptop), not matching a GPU.
- **Batch = 1** (one sequence per step), **Qwen tokenizer** (the validated path).
- **It overfits readily** — for injecting a few facts that's the point; for broad style adaptation use more
  text and fewer epochs.
- **One-directional:** Overfit loads external models (GGUF/safetensors) and saves its own small adapter file;
  it does **not** export back to `.gguf`.

## Why this matters

A frozen 4-bit base + many small swappable adapters = a private, offline "knowledge module" system you can
train and run entirely on commodity CPUs, inside a .NET process, with your data never leaving the machine.
