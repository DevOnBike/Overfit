# `Exceptions` — the library's own error types

Three types, and the rule for when to use them rather than a BCL exception.

```text
OverfitException                 base — catch this to catch "the engine failed"
  OverfitFormatException         a file, model or payload is not what it claims to be
  OverfitRuntimeException        the engine cannot do what was asked in its current state
```

## Which to throw

| Situation | Type |
|---|---|
| Malformed GGUF/ONNX/safetensors, bad header, truncated tensor | `OverfitFormatException` |
| Unsupported operator, invalid state, an operation that cannot proceed | `OverfitRuntimeException` |
| The **caller** passed a bad argument | `ArgumentException` family — stays BCL |
| Use after `Dispose()` | `ObjectDisposedException` — stays BCL |

The split is by *whose* mistake it is. Argument validation and disposal are the caller's contract with
any .NET API and are expected to look like every other .NET API; a corrupt model file is the engine's
own domain and is worth being able to catch by base type.

`InvalidDataException` maps to `OverfitFormatException`, and `InvalidOperationException` /
`NotSupportedException` map to `OverfitRuntimeException` — that mapping is what the migration applied,
so prefer the Overfit type in new code inside the library.

An unsupported ONNX operator throws with the operator **named**. An exception that says "unsupported"
without saying what costs the reader a debugging session.
