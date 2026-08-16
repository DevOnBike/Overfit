# `Trees` — gradient-boosted tree inference

Read-only GBDT: `XgboostModelLoader` parses an XGBoost model file, `BoostedTreeModel` evaluates it, and
`TreeObjective` applies the objective's link function so the output matches what XGBoost itself would
print.

**Inference only.** Training is deliberately not implemented — the value here is deploying a model
someone already trained into a .NET process with no Python and no native library, and that is a much
smaller and much more verifiable surface than a trainer.

## Measured

Parity with XGBoost 3.3.0 is better than **1e-4** on the validated models. Speed, against the reference:

| Shape | Result |
|---|---|
| Single row | **18.7×** faster |
| Batch, branchless traversal | **1.55×** faster, all cores |

The single-row gap is the interesting one and it is not about the arithmetic: it is the per-call
overhead a cross-runtime boundary imposes, which disappears when the model lives in the same process.
That is the case this directory exists for — scoring one row inside a request handler.
