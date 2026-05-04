# MiniInstruction demo

This demo trains a small GPT-style character-level model on a synthetic
User/Assistant corpus.

It is an overfit instruction-format demo. It is not a general-purpose assistant.

## What it proves

```text
- GPT training can learn a User/Assistant response format
- checkpoint_instruction.bin can be written
- checkpoint_instruction.bin can be loaded
- cached KV runtime can generate instruction-style text
- cached greedy matches legacy greedy
- cached continuation hot path stays at 0 B managed allocations
```

## Files

```text
MiniInstructionCheckpointTests.cs
```

## Manual training

The training test is skipped by default:

```csharp
[Fact(Skip = "Manual long-running mini instruction demo. Remove Skip locally, run once, then restore Skip.")]
```

Remove Skip locally, run once, then restore it.

Optional step override:

```powershell
$env:OVERFIT_MINI_INSTRUCTION_STEPS="10000"
```

Default:

```text
ContextLength = 128
BatchSize = 8
Steps = 5000
DModel = 128
Heads = 4
Layers = 4
DFF = 512
LR = 3e-4 -> 3e-5
WeightDecay = 0.02
GradClip = 1.0
```

## Show checkpoint

After training:

```bash
dotnet test -c Release --filter "Demo_LoadCheckpoint_AndShowMiniInstructionGeneration"
```

The checkpoint path is:

```text
test_fixtures/checkpoint_instruction.bin
```
