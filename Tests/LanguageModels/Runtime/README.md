# GPT runtime tests

This folder contains fast runtime tests for cached SLM/GPT inference.

These tests should remain part of normal:

```bash
dotnet test -c Release
```

---

## What belongs here

```text
cached KV runtime
runtime factory
session behavior
cache behavior
cached kernels
sampling
cached vs legacy parity
runtime acceptance tests
```

---

## What does not belong here

Long-running training demos do not belong here.

Use:

```text
Tests/LanguageModels/Demo/
Tests/LanguageModels/Experimental/
```

for those.

---

## Required fast coverage

```text
cached greedy == legacy greedy
GenerateNextToken loop == Generate
session reset clears runtime state
context overflow throws controlled exception
0 B managed allocation continuation hot path
runtime dispose behavior
```
