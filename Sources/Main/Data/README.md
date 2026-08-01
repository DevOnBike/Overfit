# `Data` — tabular data, preparation and interpretation

Everything between a CSV and a tensor, plus the tools for deciding whether the features are worth
training on. This is the classical-ML side of the library; nothing here is on an inference hot path.

| Directory | Role |
|---|---|
| `Abstractions` | `IDataLayer`, `IFeatureNormalizer`, `IFeatureNameProvider`. |
| `Contracts` | Schemas, reports, verdicts — `TableSchema`, `FeatureImportanceReport`, `PipelineContext`. |
| `Prepare` | The pipeline layers: filters, scalers, selectors. |
| `Normalizers` | Min-max, z-score, log1p, date-time expansion. |
| `Features` | Feature-importance analysis. |
| `Interpretation` | `ModelInterpreter` — explaining a trained model's outputs. |
| `Tabular` | `TabularToTensorConverter`, the final step into the engine. |
| `Serialization` | `ModelSerializer` and the source-generated JSON context. |

## `Prepare` is stateful, and that is the whole point

`DataPipeline` composes layers — `ConstantColumnFilterLayer`, `CorrelationFilterLayer`,
`DuplicateRowFilterLayer`, `OutlierClipLayer`, `RobustScalingLayer`, `LogTransformLayer`,
`BorutaSelectionLayer`, `ShapSelectionLayer`, `TechnicalSanityLayer` — each of which **learns
parameters during `Fit` and applies them during `Transform`**.

The failure this causes is silent and was found across the whole directory during the nullable
migration: **`Transform` called before `Fit`**. A scaler with no learned bounds does not throw, it
scales by defaults, and the model trains on quietly wrong numbers. `ScalerParams` is the learned state;
if you add a layer, make the unfitted case fail loudly.

`GlobalShapAnalyzer` and `ShapKernel` live in `../Statistical` and are what `ShapSelectionLayer` and
`ModelInterpreter` call.
