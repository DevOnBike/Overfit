# GPT experimental tests

This folder is for non-default GPT performance and training experiments.

Examples:

```text
data-parallel TinyShakespeare training
experimental parallel attention backward
throughput comparison
training bottleneck exploration
```

These tests should be manual and skipped by default:

```csharp
[Fact(Skip = "Manual experimental long-running GPT data-parallel training demo. Remove Skip locally, run once, then restore Skip.")]
```

Experimental tests are not stable public API.

Keep the stable GPT correctness path under:

```text
Tests/LanguageModels/GPT1/
Tests/LanguageModels/Runtime/
Tests/LanguageModels/Tokenization/
```
