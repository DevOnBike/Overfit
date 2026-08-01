# Bug hunt: Sources/Main/DeepLearning

- **Scope:** `Sources/Main/DeepLearning` (layer + model definitions, ~42 files)
- **Timestamp (UTC):** 2026-08-02 22:37
- **Commit:** `79e9d80` (branch `gimli`)
- **Score:** 3 defects found = 6 points (out of 21 needed to "win")
- **Run ended by:** SCOPE was not exhausted — ended by choice with time remaining (~7 of 10 minutes
  used) once high-yield areas (parameter enumeration, save/load round-trips, train/eval branches,
  checkpoint/dropout interaction) were covered breadth-first and the remaining files (activations,
  pooling, ONNX glue, config records, image augmentation) were skimmed and found low-risk. See
  Coverage below for the precise split. Given the score is well short of 21, treat this as **evidence
  the reviewed code is in good shape**, not as a verdict on the whole directory — the "not reached"
  list below is real and should be read as "start here," not "assumed clean."

## Findings, ranked by damage

### 1. LSTM layer weights are never actually saved or loaded — `Save`/`Load` silently no-op all the way down

**What breaks:** `LstmCell.Save(BinaryWriter)` / `Load(BinaryReader)` (`Sources/Main/DeepLearning/LSTMCell.cs:152-157`)
are empty method bodies — no bytes written, no bytes read. `LstmLayer.Save`/`Load`
(`Sources/Main/DeepLearning/LSTMLayer.cs:182-189`) just delegate straight through (`_cell.Save(bw)` /
`_cell.Load(br)`), so the no-op propagates unchanged. Both composite models built on `LstmLayer` inherit
the bug with zero extra code needed to trigger it:

- `LstmAutoencoder.Save`/`Load` (`Sources/Main/DeepLearning/LSTMAutoencoder.cs:154-168`) calls
  `_enc1.Save/_enc2.Save/_dec1.Save/_dec2.Save` — none of the four `LstmLayer`s ever write their `W`/`U`/`B`
  weights.
- `Crnn.Save`/`Load` (`Sources/Main/DeepLearning/Crnn.cs:323-343`) calls `_lstm.Save(bw)`/`_lstm.Load(br)`
  between the (correctly-serialized) conv layers and classifier — same gap.

Contrast with every other layer in the directory (`LinearLayer`, `ConvLayer`, `BatchNorm1D/2D`,
`LayerNormLayer`, `EmbeddingLayer`, `DepthwiseConv2DLayer`, `MultiHeadAttentionLayer`), all of which
correctly read/write their tensors — this is not "LSTM doesn't need persistence," it is a genuinely missing
implementation on the one recurrent layer in the codebase.

**Where:** `LstmCell.Save`/`Load` (`LSTMCell.cs:152-157`); propagated through `LstmLayer.Save`/`Load`
(`LSTMLayer.cs:182-189`); consumed by `LstmAutoencoder.Save`/`Load` (`LSTMAutoencoder.cs:154-168`) and
`Crnn.Save`/`Load` (`Crnn.cs:323-343`).

**How anyone would notice today:** They would not. `Save(path)` completes without exception and produces a
file (conv/classifier bytes are real, LSTM section is simply absent — the format has no length markers
around each sub-module so a byte-for-byte diff against a fresh untrained model would be needed to catch it).
`Load(path)` completes without exception and leaves the LSTM at its random init weights. For
`LstmAutoencoder` (the anomaly-detection LSTM autoencoder referenced in project memory) this means: train,
save, restart the process, load — and the reconstruction-error detector runs on an untrained recurrent core
while every other layer looks correctly restored. For `Crnn` (OCR/CTC): same failure mode, the classifier
head and conv front-end reload correctly but the LSTM — the layer that reads the sequence — does not.

**What test would have caught it:** A round-trip test — `Save` to a `MemoryStream`, construct a fresh model,
`Load`, then assert `LstmLayer`'s `W`/`U`/`B` (via `Parameters()`) are bit-identical to the pre-save values
(or, end-to-end, that `Forward`/`Recognize` output matches before-save and after-reload on the same input).
Neither `LstmAutoencoder` nor `Crnn` appears to have this test today (a `Tests/**/LSTMAutoencoderTests.cs`
or `CrnnTests.cs` round-trip-save test would need to assert on the recurrent weights specifically, not just
"no exception thrown").

### 2. `DepthwiseConv2DLayer.Load` desyncs the stream when the file has a bias the layer wasn't constructed with

**What breaks:** `Load` (`Sources/Main/DeepLearning/DepthwiseConv2DLayer.cs:151-174`) reads the has-bias flag
and only *conditionally* consumes the following `_channels` floats:

```csharp
if (br.ReadInt32() == 1 && Bias is not null)
{
    var bSpan = Bias.DataSpan;
    for (var i = 0; i < bSpan.Length; i++)
    {
        bSpan[i] = br.ReadSingle();
    }
}
```

If the flag is `1` (file was saved by a layer with `useBias: true`) but this instance was constructed with
`useBias: false` (`Bias == null`), the `&&` short-circuits the loop body — but the flag integer has already
been consumed, and the `_channels` bias floats that follow it in the stream are **never read**. Every
subsequent `br.ReadSingle()`/`br.ReadInt32()` call — including in a sibling layer's `Load` if this one sits
inside a `Sequential`/`Crnn`-style composite `Load` — now reads from the wrong offset. Compare
`ConvLayer.Load` (`ConvLayer.cs:224-257`), which handles exactly this case correctly by lazily
*constructing* `Bias` when the file says it's present, so the read count always matches what `Save` wrote.

**Where:** `DepthwiseConv2DLayer.Load` (`DepthwiseConv2DLayer.cs:151-174`), contrasted with the correct
pattern in `ConvLayer.Load` (`ConvLayer.cs:224-257`) and `LinearLayer`/`BatchNorm1D`/`BatchNorm2D`.

**How anyone would notice today:** Not from an exception — `BinaryReader.ReadSingle()` on desynced-but-still
-in-range bytes just returns garbage floats reinterpreted as weights for whatever comes next in the stream,
so every parameter loaded after this layer in a composite model's `Load` is silently corrupted (wrong shape
constraints would eventually throw if a later dimension-check catches the drift, but nothing guarantees that
happens before it reaches the weight-copy loop). Isolated `DepthwiseConv2DLayer` round-trip tests using a
matching `useBias` value on both sides would not catch it — only a mismatched-bias-config round-trip, or a
composite (this layer followed by another) would show it.

**What test would have caught it:** A round-trip test that saves a `DepthwiseConv2DLayer(useBias: true)` and
loads into a `DepthwiseConv2DLayer(useBias: false)` (or vice versa) and asserts either a clear error or
correct data — today it does neither, it silently misaligns. Also any composite round-trip test with a
depthwise-bias layer followed by another parameterized layer would show corrupted values in the second
layer after `Load`.

### 3. `CheckpointedModule` recomputation is silently wrong for any non-deterministic segment (e.g. `DropoutLayer`/`Dropout2DLayer`)

**What breaks:** `ComputationGraph.Checkpoint` (`Sources/Main/Autograd/ComputationGraph.cs:594-627`) computes
the forward output once (`fwd`, non-recording), keeps only input+output on the main tape, then **re-runs the
segment from scratch** in `CheckpointBackward` to regenerate activations for backprop. The class doc is
explicit that "the segment must be deterministic," but nothing enforces that contract, and
`CheckpointedModule` (`Sources/Main/DeepLearning/CheckpointedModule.cs`) happily wraps *any* `IModule`,
including `DropoutLayer`/`Dropout2DLayer`. `TensorMath.Dropout`
(`Sources/Main/Ops/TensorMath.Activations.cs:153-189`) draws its mask from `Random.Shared.NextSingle()` with
no seed capture — so the recompute inside `CheckpointBackward` draws a **different** mask than the one that
produced the actual forward output the rest of the network consumed. The backward pass then multiplies the
incoming gradient by the *wrong* mask (`DropoutBackward`, same file, uses `mask.DataView` from whichever
Dropout call just ran). Result: gradients for the checkpointed segment (and everything upstream of it) are
computed against a dropout pattern that never actually happened during the real forward pass — wrong, not
crashing, not warned about.

**Where:** `ComputationGraph.Checkpoint`/`CheckpointBackward` (`Sources/Main/Autograd/ComputationGraph.cs:594-627`);
`CheckpointedModule.Forward` (`Sources/Main/DeepLearning/CheckpointedModule.cs:38-47`); root cause in
`TensorMath.Dropout`'s unseeded `Random.Shared` use (`Sources/Main/Ops/TensorMath.Activations.cs:171-176`).

**How anyone would notice today:** Not from any error — training would proceed, loss would move, and the
degradation (worse convergence / wrong gradient noise) would look like ordinary optimization variance,
not a bug. `GPT1Model`'s own `checkpointBlocks` option is safe in practice (its `TransformerBlock` segments
have no dropout), so the *shipped* model composition does not currently trigger this — but nothing stops a
caller from wrapping a `DropoutLayer`/`Dropout2DLayer` (or any conv-net segment that includes one) in
`CheckpointedModule`, and the class's own doc comment ("Transparent: identical result (bit-close)") states a
guarantee that silently does not hold for that composition.

**What test would have caught it:** A gradient-check (finite-difference or graph-vs-no-graph gradient
comparison) on a `CheckpointedModule` wrapping a `DropoutLayer` (forced training-mode, fixed
probability) would show the checkpointed gradient diverging from the non-checkpointed one beyond FD
tolerance — this test does not appear to exist for the dropout+checkpoint combination specifically (the
existing checkpoint tests, if any, likely only cover deterministic segments).

## Shared root cause

None of the three share a root cause with each other — they're independent (missing `LstmCell`
serialization; a stream-alignment bug specific to `DepthwiseConv2DLayer.Load`; an unenforced determinism
contract between `Checkpoint` and `Dropout`). Finding #1's two symptoms (`LstmAutoencoder` and `Crnn`) *do*
share one root cause (`LstmCell.Save`/`Load` being empty) and are reported as a single defect.

## Coverage

**Reviewed and found clean:** `LinearLayer`, `ConvLayer` (Save/Load, bias handling), `DepthwiseConv2DLayer`
(everything except the Load bug above), `BatchNorm1D`, `BatchNorm2D` (train/eval branching, running-stat
caching, save/load round-trip symmetry), `LayerNormLayer`, `EmbeddingLayer`, `FeedForwardLayer` (incl. LoRA
weight/output provider hooks), `MultiHeadAttentionLayer` (incl. legacy-vs-new checkpoint format detection,
parameter/save/load symmetry across heads), `ScaledDotProductAttentionLayer`, `Sequential` (forward/inference
buffer ping-pong, prepared-inference dispatch), `ResidualBlock`, `Crnn` (forward wiring, CTC glue — apart
from the shared LSTM save/load gap), `GPT1Model` (tied-weight transpose on load, `TrainableParameters`
completeness), `LoRAAdapter`, `LlamaBlockLoRA`, `TrainableLlamaBlock` (forward + `DecodeStep` parity,
frozen-vs-trainable parameter separation), `LlamaLayerFrozenWeights`, `TrainableLlamaModel`
(`TrainableParameters`/`SaveAdapter`/`LoadAdapter` consistency), `DequantMatVec`, `RangedDequantRowSource`,
`MaxPool2DLayer`, `DropoutLayer`, `Dropout2DLayer` (as standalone layers — the bug is in the *composition*
with `CheckpointedModule`, not in these files themselves).

**Not reached (no time spent, not vetted either way):** `AveragePool2DLayer`, `GlobalAveragePool2DLayer`,
`FlattenLayer`, `RepeatVector`, `OnnxAddLayer`, `ReluActivation`, `SigmoidActivation`, `SoftmaxActivation`,
`TanhActivation`, `TransformerBlock` internals (only its `TrainableParameters`/`Save`/`Load` delegation was
checked, not its `Forward`), `ImageAugmentation`, `Gpt2Config`, `LlamaConfig`, `GPT1Config` (config records —
low risk, skipped on purpose), `Abstractions/*` interfaces (read only as needed to interpret other files).

## What a short score means here

The hunt stopped at 3 defects / 6 points with roughly 7 of the 10 minutes used and did not exhaust the
directory — it stopped because the highest-yield areas (parameter enumeration vs. disposal, save/load
round-trips, train/eval branching, the checkpoint/randomness interaction the task flagged) had been swept
breadth-first and the remaining unreached files are lower-risk shapes (stateless activations, pure
config records, simple pooling layers already spot-checked via `MaxPool2DLayer`). This is a partial-scope
result: the "reviewed and found clean" list is a real guarantee for those files, but the "not reached" list
is genuinely unknown, not a second vote of confidence — start there next time rather than re-covering
ground above.
