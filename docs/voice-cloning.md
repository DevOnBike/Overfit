# Voice cloning (Orpheus + SNAC, pure .NET)

Teach Orpheus a target voice by QLoRA-fine-tuning it on a small dataset of that voice — **entirely in managed
.NET, on the CPU, no Python**. The frozen 4-bit base is never modified; a small LoRA adapter learns to emit the
target speaker's audio tokens. This is the training side of the codec: the SNAC **encoder** (`Snac.Encode`) turns
your recordings into audio tokens, so dataset prep needs no external tools.

> **Consent / legal.** Clone only a voice you own or have explicit permission to use. Every file Overfit synthesizes
> carries a synthetic-speech provenance marker. You are responsible for the voice rights and your jurisdiction's rules.

## Pipeline

```
recordings + transcripts
  → Snac.Encode(clip) ───────────────► SNAC codes          (SnacEncoder, bit-exact vs PyTorch)
  → OrpheusSnacBridge.Interleave ────► 7-per-frame stream
  → OrpheusTrainingSequence.Build ───► (prompt → audio-token) training example
  → VoiceCloneTrainer.Train ─────────► LoRA adapter         (QLoRA, frozen 4-bit base)
  → load adapter into Orpheus ───────► speaks in the new voice
```

The training target **mirrors the generation path** (`<|audio|>{voice}: {text}<|eot_id|>` + audio tokens + end),
so what the model learns is exactly what it later generates. The loss is **completion-only** (prompt masked).

## 1. Collect a dataset

A folder of short clips with transcripts:

```
myvoice/
  clip001.wav   clip001.txt   ("Hello, this is a short sentence.")
  clip002.wav   clip002.txt
  ...
```

- **Keep clips short — ~1–3 s each.** Audio-token sequences are long and Orpheus's vocab is large (~156 k), so the
  training logits arena (~`7·T·vocab`) dominates RAM. Short clips keep `maxSeqLen` (and memory) manageable.
- Clean mono audio, one speaker, consistent mic. Any format with a decoder (WAV/MP3) — resampled to 24 kHz.
- Transcripts: a sibling `.txt` per clip. No `.txt`? Pass a `WhisperTranscriber` and the builder auto-transcribes
  (review the output — STT is imperfect).
- More data = better; even a few minutes adapts the voice, more makes it solid.

## 2. Build examples + train (C#)

```csharp
using var trainer = new VoiceCloneTrainer(orpheusGgufPath, maxSeqLen: 768);

var builder = new VoiceCloneDatasetBuilder(
    Snac.Load(snacDir), trainer.Tokenizer, trainer.EndOfTextTokenId);

var examples = builder.BuildFromFolder("myvoice", voice: "myvoice", whisper: null);

trainer.Train(examples, onStep: (epoch, step, loss) =>
    Console.WriteLine($"epoch {epoch} step {step}  loss {loss:F4}"));

trainer.SaveAdapter("myvoice.adapter");
```

## 3. Speak in the cloned voice

Load the adapter into an Orpheus engine and synthesize as usual (the adapter is small; the base GGUF is untouched).

## Single-take recording

You can record **one file** instead of many: read the lines in order with a clear ~1 s pause between each, then
`VoiceCloneDatasetBuilder.BuildFromRecording(...)` / `BuildFromRecordingFile(wav, transcript.txt, voice)` peak-
normalizes, splits on silence (`AudioSegmenter`) and pairs each segment with its line (errors if the counts differ).
The `Demo/VoiceClone` console wraps the whole flow (`--dry-run` to check the split, `--adapter`+`--test` to synthesize).

## Status

- **End-to-end validated 2026-06-06** on a real 89 s recording (20 Harvard sentences): split → 20 examples
  (~280 tokens each) → QLoRA trained on Orpheus 3B (loss **5.87 → 2.70**, 3 epochs) → adapter saved → synthesized a
  test sentence through the adapter + SNAC. All pure managed .NET on the CPU.
- **Memory is modest** — the flat audio stream is ~`7·T/4` tokens (not `7·T`), so clips run ~200–400 tokens and
  `maxSeq 512` fits in a ~2 GB logits arena. Sequences are short; training a 3 B base on CPU is the slow part.
- **Quality scales with data + epochs.** A few minutes / 3 epochs adapts the voice; more lowers the loss further.
  Gated **P2**: validate it actually sounds like the target (and your consent/legal posture) before making it a claim.
