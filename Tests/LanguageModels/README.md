# Language model tests

Language model tests are split by stability and purpose.

---

## Stable correctness

```text
LanguageModels/GPT1/
LanguageModels/Runtime/
LanguageModels/Tokenization/
```

These should be fast and should run under normal:

```bash
dotnet test -c Release
```

Examples:

```text
GPT1 model shape/save/load tests
cached runtime acceptance tests
cached vs legacy greedy parity
GenerateNextToken loop behavior
tokenizer tests
```

---

## Runtime

```text
LanguageModels/Runtime/
```

Runtime tests cover cached token-by-token generation and SLM runtime behavior.

Important coverage:

```text
cached greedy == legacy greedy
GenerateNextToken loop == Generate
session reset
context overflow
0 B cached continuation hot path
factory/runtime mode behavior
sampling behavior
```

---

## Manual demos

```text
LanguageModels/Demo/TinyShakespeare/
```

This folder contains user-facing GPT demos:

```text
train TinyShakespeare checkpoint
load checkpoint
show cached generation
display sampling with repetition penalty
```

Long-running training tests must be skipped by default:

```csharp
[Fact(Skip = "Manual long-running GPT demo. Remove Skip locally, run once, then restore Skip.")]
```

---

## Experimental

```text
LanguageModels/Experimental/
```

This is for non-default GPT performance experiments:

```text
data-parallel TinyShakespeare training
experimental parallel attention backward
throughput exploration
```

Experimental tests should not be treated as stable public API.

---

## Diagnostics

```text
LanguageModels/Diagnostics/
```

Profilers and bottleneck investigations live here.

These are manual by default.
