# `Autograd` — the training-side execution path

Reverse-mode automatic differentiation: `ComputationGraph` records a tape of `AutogradNode`s as the
forward pass runs, `Backward(loss)` walks it in reverse, and `Reset()` reclaims the temporaries.

**This is one of the two execution paths in the engine and mixing them is the most common
architectural mistake here.** Inference (`../Inference`) has no tape, no graph and no per-call
allocation; training has all three. A `ComputationGraph` reached from an inference hot path is a bug
regardless of whether it produces the right numbers.

## Ownership is explicit, not inferred

Every node carries an `AutogradNodeOwnership` set at creation, and that tag alone decides who disposes
it:

| Ownership | Disposed by |
|---|---|
| `GraphTemporary` | `graph.Reset()` |
| `GraphAuxiliary` | `graph.Reset()` — MaxPool index maps, Softmax probabilities |
| `Parameter` | the owning layer's `Dispose()` |
| `ExternalBorrowed` | the caller |
| `View` | never — no backing storage |

Without this, a graph that reset itself would free a layer's weights, and a graph that freed nothing
would leak every intermediate of every step. `IDisposableAnalyzers` is wired into this project; heed
its diagnostics rather than suppressing them.

## Operations live on the graph

Anything that records tape is a method on `ComputationGraph` — `Linear`, `Conv2D`, activations,
`SoftmaxCrossEntropy` — split across `ComputationGraph.*.cs` partials by area. The older
`TensorMath.*(graph, …)` style is being migrated onto this facade; prefer the graph method when both
exist.

`CheckpointSegment` implements gradient checkpointing, which is the memory lever for training: on a
12-layer GPT-1 it cut peak RAM **24×**, recomputing forward activations during the backward pass
instead of holding them. `FrozenQuantizedLinear` and the `IDequantRowSource` implementations are the
QLoRA path — a frozen Q4_K base with trainable adapters, so a 3B model fine-tunes in about 3 GB.
