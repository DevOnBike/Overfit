# ONNX — manual review, 2026-08-03

Read rather than hunted: the review agent was unavailable, so this is a hand review of
`Sources/Main/Onnx` (32 files). Coverage: `Protobuf/ProtoReader.cs` in full, both importers' external-data
paths in full, the schema records, and a targeted scan of dimension arithmetic. **Not read**: the sixteen
operator implementations, `OnnxProtoParser`'s attribute and node parsing beyond the tensor path, and
`OnnxGraphModel`'s execution. No claim is made about those.

**Five findings, and four of them are in one method.**

## What is already right, so nobody re-does it

`ProtoReader` has clearly had a hostile-input pass and it is a good one. `ReadVarint` is bounded twice
(buffer end and a 64-bit shift limit), and `ReadLength` compares the declared length against the remaining
bytes **as `ulong`, before the cast** — with a comment explaining the attack it closes: a length whose low 32
bits are negative would otherwise move the read position backwards and turn the parse loop into a permanent
one, with no exception and no crash. That is a subtle case handled deliberately, and it raises rather than
lowers the expectation for the rest of the module.

## 1. The DAG importer resolves external data with no path validation at all

`OnnxGraphImporter.ResolveExternalData` (line 319):

```csharp
var fullPath = Path.GetFullPath(Path.Combine(externalDataDir, init.ExternalData.Location));
fileBytes = File.ReadAllBytes(fullPath);
```

`OnnxImporter.ResolveExternalDataPath` — the sibling, same assembly — rejects an empty location, rejects an
absolute one, and proves the result stays inside the model directory (`IsPathInsideDirectory`). The graph
importer does none of the three.

**`Path.Combine` is the sharp edge.** When its second argument is rooted it discards the first entirely and
returns the second — documented .NET behaviour. So a model naming `C:\Users\…\id_rsa` as its external data
location is not combined with anything; the process simply reads that file. A relative `../../../..` escapes
just as easily, since nothing checks containment.

**Why this importer matters more than the other one.** `OnnxGraphImporter` is the one required for
ResNet/DenseNet-style skip connections — the models anybody would actually import — while the linear
`OnnxImporter` handles the simple topologies. The guarded path is the less used one.

**The root cause is written in the file.** The method opens with:

```csharp
// Delegate to OnnxImporter's implementation (same assembly, internal access).
// Reflection workaround: call Load which resolves internally, or duplicate.
// To avoid duplication we just re-implement the minimal version here.
```

"To avoid duplication we just re-implement" contradicts itself in one sentence, and "the minimal version" is
where the three checks went. This is the same shape as the loaders reviewed on 2026-08-02: one implementation
of a pattern carries the guard and its sibling does not.

**Fix**: make `ResolveExternalDataPath` internal-shared and call it from both. The duplication the comment
worried about is the correct thing to remove — by sharing the code, not by rewriting it shorter.

## 2. Unchecked narrowing of a file-supplied offset and length

Same method, lines 328–331:

```csharp
var offset = (int)init.ExternalData.Offset;                    // Offset is long
var length = init.ExternalData.Length > 0
    ? (int)init.ExternalData.Length                            // Length is long
    : fileBytes.Length - offset;
```

The sibling uses `CheckedToInt32(...)` with the initializer's name in the message. Here a value above
`int.MaxValue` wraps to a negative, and `AsSpan(offset, length)` then throws `ArgumentOutOfRangeException`
rather than the `OverfitFormatException` every other malformed-model path produces — so a caller that handles
bad models does not handle this one.

## 3. A negative allocation is reachable

Line 331, when the model declares `Length == 0` ("read to end of file"):

```csharp
var length = fileBytes.Length - offset;
var raw = new byte[length];
```

Nothing establishes `offset <= fileBytes.Length`. A large offset makes `length` negative and `new byte[]`
throws `OverflowException`. Again the wrong exception type, and again the sibling computes this through
`GetExternalDataLength`.

## 4. No bound on the external file itself

`File.ReadAllBytes(fullPath)` reads the whole file into memory with no size check, and with finding 1 the
path is attacker-chosen. Pointing it at a large file is an out-of-memory kill — uncatchable in the sense
that matters, because the process dies with it. The linear importer has the same read but a path it has
proved is inside the model directory, which bounds the damage to files somebody deliberately put there.

## 5. Dimension products are computed unchecked

`OnnxGraphImporter.cs:276` (`size *= dim`) and `OnnxTensor.cs:52` (`count *= d`).

Identical to `GgufTensorInfo.ElementCount`, found in the GGUF loader the day before. A crafted set of
dimensions wraps the product to a small or negative value that can pass a later shape check while the real
layout disagrees — silently wrong weights rather than an exception.

## Fix order

1 first, and 2–4 with it, because they are four defects in one twenty-line method and reading it once is
cheaper than reading it four times. 5 belongs with the GGUF `ElementCount` fix already queued — same defect,
same reasoning, two files.

## What would have caught these

Findings 2, 3 and 5 are exactly the shape `OVERFIT024` is proposed for: a value from an untrusted file
sizing an allocation without validation. Finding 1 is not — no analyser will notice that one method validates
a path and its neighbour does not. That one needed a reader, which is the argument for continuing to do this
by hand alongside the rules.
