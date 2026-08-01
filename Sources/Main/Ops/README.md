# `Ops` — tensor operations, the layer above the kernels

`TensorMath` is a single static facade split across partials by area — attention, convolution,
normalisation, RoPE, activations, losses, shape manipulation. It sits between the layer types in
`../DeepLearning` and the inner loops in `../Kernels`: it knows shapes, broadcasting and which kernel
to dispatch to, and it does not contain the hot loop itself.

## Two calling conventions

Most operations exist twice:

- **Plain**, taking spans and writing into a caller-owned output. Used by the inference path, which
  allocates nothing.
- **Graph-recording**, taking a `ComputationGraph` first, which appends to the tape so a backward pass
  can walk it.

The graph-recording form is being migrated onto the `ComputationGraph` facade itself
(`graph.Linear(...)` rather than `TensorMath.Linear(graph, ...)`) — prefer the graph method where both
exist. See `docs/OverfitArchitectureRefactorPlan.md`.

## Layout conventions that bite

`TensorMath.Rope` implements rotary embeddings, and **the pair layout differs between formats**:
HuggingFace rotates halves, GGUF rotates adjacent pairs. Loading a GGUF model with the HF convention
produces fluent, confident, wrong output — the loaders permute rows so this file only ever sees one
convention. `ExpandKvHeads` covers grouped-query attention, where K/V heads are fewer than Q heads and
must be repeated rather than reshaped.

`CtcLoss` and `CtcDecoder` are the sequence-labelling path used by the CRNN OCR model; the loss is
verified by finite differences, and `NGramCtcLanguageModel` is the optional decode-time prior.
