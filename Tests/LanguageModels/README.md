# Language model tests

## Stable correctness

```text
LanguageModels/GPT1/
LanguageModels/Runtime/
LanguageModels/Tokenization/
```

These should be fast and run under normal `dotnet test`.

## Manual demos

```text
LanguageModels/Demo/TinyShakespeare/
```

Training/checkpoint regeneration tests are manual and should use:

```csharp
[Fact(Skip = "Manual long-running GPT demo. Remove Skip locally, run once, then restore Skip.")]
```

## Experimental

```text
LanguageModels/Experimental/
```

This is for data-parallel training and other non-default performance experiments.

## Diagnostics

```text
LanguageModels/Diagnostics/
```

Profilers and bottleneck investigations. These are manual by default.
