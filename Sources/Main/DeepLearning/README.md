# `DeepLearning` — layers and model definitions

The building blocks a model is assembled from — `LinearLayer`, `ConvLayer`, `DepthwiseConv2DLayer`,
the pooling and normalisation layers, activations, `LSTMLayer`/`LSTMCell`, `MultiHeadAttentionLayer`,
`EmbeddingLayer`, `Dropout` — plus the assembled models: `Sequential`, `GPT1Model`, the Llama block
types, `Crnn`, `ResidualBlock`, `LSTMAutoencoder`.

Layers own their parameters. `layer.TrainableParameters()` is the canonical way to enumerate them and
what optimisers consume: `new Adam(model.TrainableParameters(), lr)`. A `Parameter` is a first-class
type (`../Parameters`) with `Parameter` ownership in the autograd sense, meaning the layer disposes it
and `graph.Reset()` does not.

## Where the arithmetic lives

Not here. A layer knows its shapes, its parameters and its lifetime; the maths is
`TensorMath.*` in `../Ops`, which dispatches to `../Kernels`. Keeping the loop out of the layer is what
lets a kernel be replaced and A/B-measured without touching model definitions.

`CheckpointedModule` wraps a segment for gradient checkpointing. `LlamaLayerFrozenWeights`,
`TrainableLlamaBlock`, `LlamaBlockLoRA` and `DequantMatVec` are the QLoRA training path — a frozen
quantised base with trainable adapters — and are the reason a 3B fine-tune fits in ~3 GB.

`OnnxAddLayer` exists so an imported residual connection has a layer to be; see `../Onnx` for why a
DAG import is required for those models.
