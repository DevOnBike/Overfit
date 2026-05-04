# Overfit test project layout

The test project is organized by product area first, then by purpose.

Default rule:

```bash
dotnet test -c Release
```

should run fast correctness tests only.

Manual / long-running / experimental tests must be skipped by default with a clear comment:

```csharp
[Fact(Skip = "Manual long-running GPT demo. Remove Skip locally, run once, then restore Skip.")]
```

---

## Top-level structure

```text
Tests/
  Core/
    Algorithms/
    Autograd/
    Distributions/
    Kernels/
    Memory/
    Parameters/
    Randomization/
    TensorMath/

  DeepLearning/
    Attention/
    Cnn/
    Inference/
    Modules/
    Training/
    Transformer/

  LanguageModels/
    GPT1/
    Runtime/
      Acceptance/
      Adapters/
      Blocks/
      Cache/
      Engine/
      Factory/
      Kernels/
      Parity/
      Sampling/
      Session/
    Tokenization/
    Demo/
      TinyShakespeare/
    Experimental/
    Diagnostics/

  Data/
    Mnist/

  Evolutionary/
    Algorithms/
    Noise/
    QualityDiversity/

  Forecasting/

  Preprocessing/
    Normalizers/

  Integrations/
    K8s/
    Onnx/
    Shap/

  Diagnostics/
    Probes/
    Tracing/

  Examples/
    TicTacToe/

  TestSupport/
    GradientChecking/
    Helpers/
    Prepare/
```

---

## Naming rules

- One public class per file.
- Test class name should mirror the subject under test.
- Manual demos use `Demo_...`.
- Diagnostics/profilers use `...ProfilerTests` and are skipped by default.
- Long-running tests must use `[Fact(Skip = "...")]`.
- Runtime acceptance tests stay fast and unskipped.

---

## GPT policy

Stable GPT tests:

```text
Tests/LanguageModels/GPT1/**
Tests/LanguageModels/Runtime/**
Tests/LanguageModels/Tokenization/**
```

Manual GPT demos:

```text
Tests/LanguageModels/Demo/TinyShakespeare/**
```

Experimental GPT training/perf:

```text
Tests/LanguageModels/Experimental/**
```

GPT bottleneck profilers:

```text
Tests/LanguageModels/Diagnostics/**
```
