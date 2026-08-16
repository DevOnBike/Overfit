# `Inference` — the zero-allocation execution path

The public entry point for running a trained model. `InferenceEngine.Run(input, output)` takes
caller-owned buffers and allocates **nothing** per call; that property is the product identity, not an
optimisation, and it is why this path exists separately from `../Autograd`.

## The contract

- Buffers belong to the caller. The engine writes into `output` and never hands back an array it owns.
- No tape, no `ComputationGraph`, no `AutogradNode`. Nothing on this path knows gradients exist.
- **Do not call `model.Forward(...)` in a hot path.** It is the training-shaped entry and it allocates.
  Go through `InferenceEngine.Run`.
- No LINQ, no `.ToArray()`, no hidden allocation. `Sources/Main/BannedSymbols.txt` enforces the parts a
  compiler can check.

## Backends

`IInferenceBackend` is the seam. `SequentialInferenceBackend` runs a linear stack;
`OnnxGraphInferenceBackend` runs a DAG, which is what skip connections need — see `../Onnx` for which
importer produces which. `InferenceEngine.FromBackend(backend)` wraps either.

`MnistPredictor` is the small worked example: load, run, read a class out, with the buffer ownership
visible in a few lines.
