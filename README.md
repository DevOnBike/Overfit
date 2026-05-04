# MiniInstruction answer trim fix

This ZIP replaces:

```text
Tests/LanguageModels/Demo/MiniInstruction/MiniInstructionCheckpointTests.cs
```

## Change

The load/show test now prints only the first assistant answer instead of the full
continuation transcript.

Before:

```text
Assistant: 4.

User: What is two plus two?
Assistant: 4.
```

After:

```text
Answer:
4.
```

## How

It adds:

```text
ExtractFirstAssistantAnswer(...)
FindFirstStopMarker(...)
AssertInstructionAnswerLooksValid(...)
AssertExpectedDemoAnswer(...)
```

The generation still uses cached runtime and the same sampling. The change is
only display trimming and validation.

## Run

```bash
dotnet test -c Release --filter "Demo_LoadCheckpoint_AndShowMiniInstructionGeneration"
dotnet test -c Release
```

## Expected validation

```text
Legacy parity: OK
Continuation allocation check: 0 B for 8 tokens
```
