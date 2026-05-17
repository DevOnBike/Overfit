# MiniInstruction demo

## Summary

MiniInstruction is the next step after TinyShakespeare.

TinyShakespeare proves:

```text
train style model -> checkpoint -> cached generation
```

MiniInstruction proves:

```text
train User/Assistant format -> checkpoint_instruction.bin -> cached answer generation
```

## Scope

This is a synthetic overfit demo. It is intentionally small and controlled.

It should not be described as a general-purpose assistant.

## Dataset

The dataset is embedded in the xUnit file as simple User/Assistant pairs:

```text
User: What is 2 plus 2?
Assistant: 4.

User: What is Overfit?
Assistant: Overfit is a C# deep-learning engine.
```

This avoids external files and keeps the demo deterministic.

## Runtime validation

The load/show test validates:

```text
checkpoint loads
generated text is printable
cached greedy == legacy greedy
cached continuation allocation = 0 B
```

## Future improvements

```text
- load OpenAI Cookbook toy chat JSONL as an optional fixture
- add checkpoint package with config + tokenizer metadata
- add BPE tokenizer
- add a larger public instruction subset
```
