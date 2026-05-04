# Test project cleanup plan

## Why

The root `Tests/` folder currently mixes unrelated domains:

```text
Autograd
CNN
GPT
MNIST
Forecasting
Evolutionary algorithms
K8s
SHAP
Diagnostics
Tokenizers
Inference
Training
```

This makes it harder to see what belongs to GPT, what is core math, and what is
a long-running/manual demo.

## Target

Move tests by domain and purpose. Do not change test code during this cleanup.

## Move strategy

Use:

```powershell
.\Tests\MoveTestsToProposedLayout.ps1
```

Dry-run first:

```powershell
.\Tests\MoveTestsToProposedLayout.ps1
```

Apply:

```powershell
.\Tests\MoveTestsToProposedLayout.ps1 -Apply
```

Then run:

```bash
dotnet test -c Release
```

## Files that should be deleted after move

If these still exist after the move, delete them:

```text
Tests/LanguageModels/TinyShakespeareDataParallelTrainingTests.cs
```

Keep the experimental one:

```text
Tests/LanguageModels/Experimental/TinyShakespeareDataParallelTrainingTests.cs
```

## Suggested commit

```bash
git add Tests
git commit -m "test: organize test project layout"
```

## Do not combine with logic changes

This should be a pure file move / folder cleanup commit.
Do not change production code in the same commit.
