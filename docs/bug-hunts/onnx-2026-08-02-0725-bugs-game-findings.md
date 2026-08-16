# Bug hunt: Sources/Main/Onnx

- **Scope**: `Sources/Main/Onnx` (ONNX importer — `OnnxProtoParser`, `OnnxImporter`, `OnnxGraphImporter`,
  `OnnxGraphModel`, `Operators/`, `Schema/`, `Protobuf/`)
- **Timestamp (UTC)**: 2026-08-02 07:25
- **Commit**: `44c0433`
- **Score**: 12 points / **6 defects**
- **Ended**: by scope narrowed to time — stopped voluntarily after covering the parser, both importers, the
  DAG execution model, and 10 of 14 operator files, at roughly 4 of the 10 minutes available. Not a
  time-cap stop; treat the unopened operators as genuinely unreached, not as "checked and clean".
- **README present**: yes, `Sources/Main/Onnx/README.md`, read first. `Sources/Main/README.md` also read
  for the shared hot-path/AOT rules.

---

## Findings, ranked by damage

### 1. `OnnxGraphImporter` never checks the declared graph output — trusts "last node processed" instead

**What breaks**: `OnnxGraphModel.RunInference` copies out `_nodes[^1]`'s output slot as the model's result
(`OnnxGraphModel.cs`, `RunInference`, `// Last node's output slot → caller's output span.`). Nothing in
`OnnxGraphImporter.LoadFromBytes` ever reads `model.Graph.Outputs` — I grepped the whole directory for it
and got zero hits outside the parser that populates the field. The importer never verifies that the tensor
produced by the last node in the file's node list is the tensor the file actually names as its output.

**Where**: `Sources/Main/Onnx/OnnxGraphImporter.cs::LoadFromBytes`, `Sources/Main/Onnx/OnnxGraphModel.cs::RunInference`.

**How anyone would notice today**: they would not. Most exporters happen to emit the output-producing node
last, so common ResNet/DenseNet exports work by coincidence. Any exporter that appends trailing nodes after
the real output (an unused auxiliary head left over from training-mode export, a debug/logging branch, a
second declared output in a multi-output graph) makes `RunInference` silently return the wrong tensor's
values — same shape typically (both come from 4-D conv/FC output paths), no exception, no message. This is
exactly the "loads with zero weights and no error" class of failure the README calls out for the sidecar
case, except here it isn't caught anywhere.

**What test would have caught it**: a graph-importer fixture with a trailing node computing an unrelated
tensor after the true output (e.g. an extra `Conv` that isn't wired to `graph.Outputs[0]`), asserting the
returned values match a reference computed from the tensor actually named in `graph.Outputs`, not from
whatever node happened to be listed last.

---

### 2. `OnnxGraphImporter.ResolveExternalData` is a duplicated, unguarded reimplementation — path traversal and no offset/length bounds check

**What breaks**: `OnnxImporter.ResolveExternalDataPath` explicitly rejects absolute `location` values and
calls `IsPathInsideDirectory` to reject `..`-escapes out of the model directory, and its
`ResolveExternalData` bounds-checks `offset`/`length` against the sidecar file size before slicing,
throwing `OverfitFormatException` on failure. `OnnxGraphImporter`'s own copy of this logic (comment: *"To
avoid duplication we just re-implement the minimal version here"*) has neither. It does
`Path.Combine(externalDataDir, init.ExternalData.Location)` with no rooted-path check and no
directory-escape check, and then does `fileBytes.AsSpan(offset, length).CopyTo(raw)` with no validation
that `offset`/`length` (both read straight from the untrusted `.onnx` file) fit inside `fileBytes`.

**Where**: `Sources/Main/Onnx/OnnxGraphImporter.cs::ResolveExternalData` (private, ~line 299).

**How anyone would notice today**: not under normal use — only a hostile or corrupt `.onnx` triggers it,
and the DAG importer is exactly the path the README recommends for third-party model-zoo files (ResNet,
DenseNet, EfficientNet). A crafted `external_data` `location` of `..\..\secret.bin` or an absolute path
reads and returns an arbitrary file's bytes as tensor weights with no error at all — silent local file
disclosure, not even a crash. A corrupt offset/length instead throws a bare `ArgumentOutOfRangeException`
from `Span` internals rather than the clear, reportable `OverfitFormatException` the sibling importer
gives for the identical situation.

**What test would have caught it**: feed `OnnxGraphImporter.LoadFromBytes` a model whose external-data
`location` is `..\..\x` or rooted, assert it throws the same class of exception `OnnxImporter` throws;
feed one with an out-of-range offset/length, assert `OverfitFormatException` rather than an unhandled
`ArgumentOutOfRangeException`.

---

### 3. `OnnxGraphImporter`'s shape/size casts skip the overflow guard `OnnxImporter` enforces

**What breaks**: `OnnxImporter` routes every `long` dimension through `CheckedToInt32`, which throws a
named `OverfitRuntimeException` on overflow. `OnnxGraphImporter.SeedInitializerShapes`,
`SeedInputShapes`, and `ComputeSize` use bare `(int)` casts and an unchecked `int` multiply instead — a
dimension outside `int` range silently wraps, and `ComputeSize`'s `size *= dim` loop has no overflow check
either. The wrapped value then sizes a real `TensorStorage<float>` buffer
(`new TensorStorage<float>(bufferSizes[i])`), so a wrap that lands on a small positive number produces an
undersized buffer with nothing flagging the mismatch.

**Where**: `Sources/Main/Onnx/OnnxGraphImporter.cs::SeedInitializerShapes`, `::SeedInputShapes`, `::ComputeSize`.

**How anyone would notice today**: only with a corrupt/adversarial file or an unusually large declared
dimension — no message distinguishes this from a normal-looking allocation failure, and a wrap that stays
positive wouldn't even fail loudly.

**What test would have caught it**: a graph-importer fixture with a dimension (or an output-shape product)
that overflows `int`, asserting a clear thrown error rather than a wrapped buffer size.

---

### 4. `Flatten`'s `axis` attribute is silently ignored

**What breaks**: ONNX `Flatten` takes an `axis` attribute (default 1) — `nn.Flatten(start_dim=k)` exports
as `axis=k`. `OnnxOperatorMapper` routes both `"Reshape"` and `"Flatten"` through the same
`ReshapeOperator.Build`, but that method never reads `"axis"` at all. `Flatten` nodes never carry a second
shape-input tensor (that's a `Reshape`-only feature), so every `Flatten` node falls into the hardcoded
`"Fallback: assume flatten [batch, rest]"` branch — i.e. it always behaves as `axis=1`, whatever the file
actually says. Every other constraint in this directory that can't be honored throws a named error (Conv's
group/dilation/padding, MaxPool's ceil_mode/padding, ReduceMean's axes, AveragePool's auto_pad/dilations) —
`Flatten`'s axis is the one constraint that's assumed rather than checked.

**Where**: `Sources/Main/Onnx/Operators/ReshapeOperator.cs::Build`.

**How anyone would notice today**: not at all for `axis=1` exports (the common case). For any exported
`axis != 1`, the shape recorded in `OnnxShapeContext` is simply wrong; inference either throws deep inside
the next layer's shape-mismatched matmul, or — if the miscomputed size happens to coincide with the next
layer's expected width — silently produces wrong numbers.

**What test would have caught it**: a `Flatten` node with `axis=2` on a rank-4 input, asserting either the
axis-2 shape or a clear "unsupported axis" throw; today's fallback silently produces the axis=1 shape
either way.

---

### 5. `AddOperator`'s documented "both inputs must have identical shapes" is never checked at import time

**What breaks**: the XML doc on `OnnxAddLayer` and the comment on `AddOperator` both assert equal-shape
inputs, but `AddOperator.Build` never compares `shape0`/`shape1` — it just takes whichever one is non-null
and moves on. The actual equality check only happens implicitly, deep in
`System.Numerics.Tensors.TensorPrimitives.Add`, on the first inference call. So a graph with a genuine
shape-mismatched residual add (malformed export, or a broadcasting pattern this importer doesn't support)
passes `OnnxGraphImporter.Load` cleanly and only fails much later, on first `RunInference`, with a
`TensorPrimitives`-internal exception that names neither the node nor the tensors involved.

**Where**: `Sources/Main/Onnx/Operators/AddOperator.cs::Build`; comment contract on
`Sources/Main/DeepLearning/OnnxAddLayer.cs`.

**How anyone would notice today**: at first inference, via an exception with no reference back to which
ONNX node or tensor caused it — a worse diagnostic than every neighboring operator's load-time check gives
for its own constraints.

**What test would have caught it**: a graph with an `Add` node whose two inputs have different shapes,
asserting `OnnxGraphImporter.Load` throws a clear, node-named error rather than deferring to a generic
`TensorPrimitives` exception at first `RunInference`.

---

### 6. README's residual-network claim is now stale

**What breaks**: `Sources/Main/Onnx/README.md` states "Feeding a residual network to the sequential
importer does not fail at load; it fails when the numbers come out wrong, which is much later and much
more expensive." `OnnxImporter.ValidateLinearTopology` counts consumers per tensor and throws
`"Branching topology detected: tensor '...' has N consumers..."` at load for exactly the case a residual
connection produces (the skip-connection tensor is consumed both by the shortcut `Add` and by the main
branch). For the common skip-connection shape, this now fails loudly at load, not silently later — the
documented failure mode no longer matches the code. This is the safe direction (a real guard replaced an
absent one), but it's still a false claim in the file that's supposed to be the accurate contract for this
directory, and a reader relying on it to justify *not* adding load-time validation elsewhere would be
relying on stale information.

**Where**: `Sources/Main/Onnx/README.md` (the "Two importers" section) vs.
`Sources/Main/Onnx/OnnxImporter.cs::ValidateLinearTopology`.

**How anyone would notice today**: only by reading both and comparing — nothing exercises this
inconsistency, since the code's actual behavior is *safer* than documented, not worse.

**What test would have caught it**: none needed for correctness (behavior is fine); a doc-accuracy check
isn't something a unit test covers.

---

## Shared root causes

- **Findings 1–3 share one root cause**: `OnnxGraphImporter` re-derives its own copy of logic that
  `OnnxImporter` already gets right (external-data resolution, checked-cast dimension handling) rather than
  reusing it, and the copy dropped every safety check along the way. The `ResolveExternalData` comment even
  says this out loud ("re-implement the minimal version here" to avoid a "reflection workaround") — the
  duplication is deliberate, the regression in what it checks is not. Fixing this by sharing
  `OnnxImporter`'s internal helpers (same assembly, `internal` is already usable across these two files)
  would close findings 2 and 3 in one change and remove the divergence finding 1 sits next to.
- **Findings 4 and 5 share a pattern, not a cause**: both are operators whose documented constraint
  (`Flatten`'s `axis`, `Add`'s equal-shape requirement) is asserted in a comment but never actually
  enforced by code, unlike every neighboring operator (Conv, MaxPool, AveragePool, ReduceMean, Softmax),
  which all throw a named `OverfitRuntimeException` for the constraint they can't satisfy.

---

## Coverage

**Reviewed and found clean**:
- `Sources/Main/Onnx/Protobuf/ProtoReader.cs` — varint/length/fixed32/fixed64 reads are all bounds-checked
  before the cast; the documented negative-length-turns-into-a-hang attack is closed correctly
  (`ReadLength` compares the raw `ulong` before truncating).
- `Sources/Main/Onnx/OnnxProtoParser.cs` — all message parsers default-skip unknown fields via
  `SkipField`, no unbounded recursion, packed/unpacked repeated-field handling for floats/ints looks right.
- `Sources/Main/Onnx/OnnxImporter.cs` — `ResolveExternalData`/`ResolveExternalDataPath`/
  `IsPathInsideDirectory` correctly reject absolute paths and directory escapes, and bounds-check
  offset/length against the sidecar file before slicing; `CheckedToInt32` is applied consistently to every
  dimension; `ValidateLinearTopology` correctly flags any tensor with >1 non-initializer consumer.
- `Sources/Main/Onnx/Operators/ConvOperator.cs`, `MaxPoolOperator.cs`, `AveragePoolOperator.cs`,
  `SoftmaxOperator.cs`, `BatchNormOperator.cs`, `ReduceMeanOperator.cs`, `GlobalAveragePoolOperator.cs`,
  `GemmOperator.cs` — each throws a named, clear error for every constraint it can't satisfy (group,
  dilation, asymmetric padding, ceil_mode, auto_pad, non-`axis=-1` softmax, non-`[2,3]` ReduceMean axes,
  `transA`/alpha/beta on Gemm).
- `Sources/Main/Onnx/Operators/OnnxOperatorMapper.cs` — unsupported op types genuinely throw naming the
  operator; no silent-skip path exists in the dispatcher itself.
- `Sources/Main/Onnx/OnnxGraphNode.cs`, `OnnxShapeContext.cs` — simple, no defects found.

**Not reached** (ran out of self-imposed budget before the time cap, not because the clock ran out):
- `Sources/Main/Onnx/Operators/ReluOperator.cs`, `TanhOperator.cs`, `SigmoidOperator.cs` (small,
  likely-low-risk activation wrappers, but not opened).
- `Sources/Main/Onnx/Schema/*.cs` (record/DTO definitions — glanced at the file list only, not the bodies).
- Cross-check of `OnnxImporter`'s own node-order-vs-topological-order assumption for `Sequential` (the
  same "last/order in file is trusted" pattern as finding 1, but on the linear-importer side — worth a
  follow-up look since `ValidateLinearTopology` checks *branching* but not that node order matches actual
  dataflow order).
- `DeepLearning/GlobalAveragePool2DLayer`, `FlattenLayer`, `MaxPool2DLayer`, etc. — the layer
  implementations the operators build were not opened; only the operator-side shape/attribute handling was
  reviewed.

## If more time were available

The run stopped by choice at 6 defects / 12 points, short of 21, with roughly 6 of the 10 minutes still
available. That is a genuine shortfall against the target, not a "clean scope" result — the unopened
Relu/Tanh/Sigmoid operators and the Schema DTOs are real gaps, and the `Sequential` node-order question
flagged above looks like it could be a seventh finding on a closer look.
