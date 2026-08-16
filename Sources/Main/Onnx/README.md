# `Onnx` — importing models from the interchange format

Reads `.onnx` files into Overfit models. **There is no protobuf dependency** — the parser in
`OnnxProtoParser` is hand-rolled, because a code-generated protobuf runtime drags reflection and
reflection is banned in this assembly for Native-AOT reasons.

## Two importers, and picking the wrong one fails late

| Entry point | Produces | Use when |
|---|---|---|
| `OnnxImporter.Load(path)` | `Sequential` | The graph is a linear chain. Faster and simpler. |
| `OnnxGraphImporter.Load(path, inputSize, outputSize)` | `OnnxGraphModel` (DAG) | Skip connections — ResNet, DenseNet, anything with a residual add. |

The graph model is then wrapped in `OnnxGraphInferenceBackend` and handed to
`InferenceEngine.FromBackend(...)`. Feeding a residual network to the sequential importer does not
fail at load; it fails when the numbers come out wrong, which is much later and much more expensive.

## Behaviour worth knowing

- **External `.data` sidecars are resolved automatically.** PyTorch ≥ 2.x writes weights beside the
  graph by default, and a model that loads with zero weights and no error is the failure this avoids.
- **Unsupported operators throw**, naming the operator. Silently skipping an op would produce a model
  that runs and is wrong.
- `OnnxShapeContext` carries inferred shapes through the import, which is why the graph importer needs
  input and output sizes when the file does not pin them.

## Direction

Import only. Overfit reads external formats — ONNX, GGUF, safetensors, `.bin` — and **does not export
to them**. There is no exporter here and none is planned; proposals to add one are out of scope rather
than unimplemented.
