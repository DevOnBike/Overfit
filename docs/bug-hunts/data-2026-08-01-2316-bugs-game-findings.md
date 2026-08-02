# Bug hunt — Sources/Main/Data

- **Scope:** `Sources/Main/Data` (tabular pipeline, normalizers, tabular→tensor conversion, interpretation, serialization)
- **Timestamp (UTC):** 2026-08-01 23:16 (2026-08-02 local irrelevant — UTC is authoritative)
- **Commit:** `44c0433`
- **Score: 8 points / 4 defects**
- **Ended by:** scope narrowed to a natural stopping point well inside the 10-minute cap (~5 minutes used). Not every file was opened (see Not Reached), so this is a partial-scope result, not an exhaustive one — treat the "clean" list as spot-checked, not proven.
- **README:** `Sources/Main/Data/README.md` exists and was read first. It names the exact failure this hunt was pointed at ("Transform called before Fit is silent"), and finding #2 below is a structural variant of exactly that failure that the README's own inventory of fitted layers missed. `Sources/Main/README.md` was also read for the hot-path/AOT rules (not directly applicable — nothing here is on an inference hot path per the Data README itself).

---

## Findings, ranked by damage

### 1. `FastRandomForest.BuildRecursive` never partitions data at the chosen split — the forest silently degenerates to predicting the global mean, and "importance" is noise

**Where:** `Sources/Main/Data/FastRandomForest.cs`, `BuildRecursive` (lines ~116–150).

**What breaks:** A split node picks a random feature and a random threshold within its range, records `importance[featureIdx] += 1/(depth+1)`, and then recurses **twice with the identical, unfiltered `x`/`y`**:

```csharp
var leftIdx = BuildRecursive(x, y, depth + 1, nodes, importance);
var rightIdx = BuildRecursive(x, y, depth + 1, nodes, importance);
```

Neither call slices rows by `features[featureIdx] <= threshold` before recursing. Since `rows` therefore never shrinks with depth, the only stopping condition is `depth >= _maxDepth`, and every leaf computes `CalculateMean(y)` over the **entire original dataset** — the same value at every leaf, in every tree, regardless of which branch was taken to reach it. `Predict()` therefore returns (approximately) the constant `mean(y)` for any input, independent of the features passed in, and the accumulated "importance" reflects nothing but how often a feature was randomly chosen as a split feature, weighted by `1/(depth+1)` — not any measured reduction in error.

**Consumers, both silently broken by this:**
- `BorutaSelectionLayer.Process` calls `forest.TrainAndGetImportance` and compares real-feature vs. shadow-feature hit counts — with importance being noise, "confirmed" vs. "rejected" features are effectively a coin flip dressed up as a principled statistical test.
- `ShapSelectionLayer.Fit` trains a `FastRandomForest`, then wraps `forest.Predict` in a `ShapKernel` for SHAP-based importance — SHAP explanations built on a near-constant model function are themselves near-constant/meaningless, so the resulting "kept" feature set is not driven by actual feature-target relationships.

**How anyone would notice today:** They would not. Both layers run to completion, return a plausible-looking (non-empty, non-full) subset of columns, and never throw. The only symptom is a model trained on the "selected" features performing no better than one trained on a random subset — which looks like "feature selection didn't help much," not "feature selection is broken."

**What test would have caught it:** A unit test training `FastRandomForest` on a synthetic dataset with one informative feature (e.g. `y = x[3] > 0 ? 1 : 0`, rest random noise) and asserting `TrainAndGetImportance` ranks column 3 highest, or asserting `Predict` tracks `x[3]` at all. No such test exists under `Tests/` for this class (searched for "Boruta" — zero matches).

---

### 2. `BorutaSelectionLayer` has no fit/transform gate — it re-selects features on every `Process()` call, unlike every sibling layer in `Prepare/`

**Where:** `Sources/Main/Data/Prepare/BorutaSelectionLayer.cs`, `Process`.

**What breaks:** Every other stateful layer in `Prepare/` (`ConstantColumnFilterLayer`, `CorrelationFilterLayer`, `OutlierClipLayer`, `RobustScalingLayer`, `ShapSelectionLayer`) has a `_fitted` flag and a `RequireFitted` guard so that a second call to `Process()` (the intended "transform on new/inference data" call, per the README's own description of the layer contract) reuses the parameters learned on the first call. `BorutaSelectionLayer` has neither field — it runs the full shadow-feature Boruta procedure, including training `_numIterations` fresh `FastRandomForest` instances, from scratch on **every** call. Called a second time (e.g. on a held-out/inference batch through the same `DataPipeline`), it will pick a different-shaped, potentially different-membership column subset than the one it picked during training, silently desynchronizing every downstream layer's column indices from what they were fitted against. It also unconditionally reads `context.Targets`, so it requires labels to be present at whatever time it's called — a contract violation if the pipeline is ever reused for genuine label-less inference the way the README's fit/transform framing implies the other layers are meant to be.

**How anyone would notice today:** They would not, until predictions on new data look wrong for reasons that trace back to a column-index mismatch several layers downstream — a very hard bug to localize, because `DataPipeline.Execute` only checks row-count desync (`Features` vs `Targets`), not column-schema desync between two calls to the same layer.

**What test would have caught it:** A test that calls `BorutaSelectionLayer.Process` twice with two different (but same-shaped) datasets and asserts the second call's kept-column set equals the first call's — this is exactly the pattern `Tests/TestSupport/Prepare/DataPreparationTests.cs` should exercise for every stateful layer, and (per grep) does not for Boruta.

*Shared root cause with #1*: both bugs live in the "learned feature selection" family and compound — Boruta's selection is already noise (#1) and also non-reproducible run-to-run (#2).

---

### 3. `ConstantColumnFilterLayer.IdentifyByUniqueRatio` off-by-one + degenerate fallback: `minUniqueRatio = 1.0` (a validated, legal value) silently turns the filter into a no-op

**Where:** `Sources/Main/Data/Prepare/ConstantColumnFilterLayer.cs`, `IdentifyByUniqueRatio` (lines 142–167) and `Process` (line 64).

**What breaks:** The constructor accepts `minUniqueRatio` in the closed range `[0, 1]`. Internally, `minUnique = (int)(rows * _minUniqueRatio)`, and a column is kept only if `uniqueValues.Count > minUnique` (strict `>`). With `minUniqueRatio = 1.0` and `rows = N`, `minUnique = N`, and the maximum possible unique count for any column is also `N` — so `count > minUnique` (`N > N`) can **never** be true, for any column, including ones with zero duplicates. Every column fails the check, so `keptList.Count == 0`. That result then hits the fallback in `Process`:

```csharp
_keptIndices = keptList.Count == cols || keptList.Count == 0 ? null : keptList.ToArray();
```

`null` is the sentinel this layer uses to mean "no filtering needed" (used deliberately for the `keptList.Count == cols` case). Reusing it for the `keptList.Count == 0` case means "we determined literally nothing is worth keeping" is silently treated identically to "everything already passed" — the layer disables itself and returns the data completely untouched. So the single most aggressive, fully-valid setting of this parameter produces the *opposite* of aggressive filtering: it does nothing, including leaving genuinely constant columns in place.

**How anyone would notice today:** They would not — no exception, no warning, and the returned column count is unchanged, which looks identical to "nothing was constant." Only comparing against a manual check of the input for constant columns would reveal the layer never ran.

**What test would have caught it:** A test constructing the layer with `minUniqueRatio: 1.0` against a dataset containing at least one genuinely constant column and asserting that column is removed.

---

### 4. `TabularToTensorConverter` has no persistence for learned category mappings — categorical schema must be re-`Fit` at inference, and nothing checks it matches training

**Where:** `Sources/Main/Data/Tabular/TabularToTensorConverter.cs` (whole class — `_categoryMaps`), contrasted with `Sources/Main/Data/Serialization/ModelSerializer.cs`.

**What breaks:** `Fit(data)` learns `_categoryMaps[col.Name] = sortedUniqueValuesSeenInData` for every categorical column, entirely in `Dictionary<string,string[]>` process memory — there is no `Save`/`Load` for it anywhere in the class or in `ModelSerializer` (which only persists raw `ConvLayer`/`LinearLayer` weight tensors, validated by rank+dimension equality, not by any notion of schema). Two independent hazards follow directly from this:
- `Convert(data)` reads `_categoryMaps[col.Name]` (line 93) with no check that `Fit` was ever called; on a fresh instance this throws a generic `KeyNotFoundException` deep inside the row loop, with nothing pointing at "you forgot to call Fit," unlike every layer in `Prepare/` which now has a named `RequireFitted` exception for exactly this class of mistake.
- If a process restart forces a second `Fit()` call at inference time (the only way to reconstruct `_categoryMaps` without persisting it), and the inference dataset's categorical columns happen to contain the same *count* of unique values as training but not the same *set* (e.g. a rare training-time category absent from the inference sample, offset by an inference-only category), `_featureWidth` still matches, `ModelSerializer.LoadModel` still validates cleanly (it only checks tensor shape), and every one-hot column from that point on is silently shifted — the model consumes structurally valid but semantically wrong input with no error anywhere in the chain.

**How anyone would notice today:** Only by manually diffing the category ordering between the training-time and inference-time `_categoryMaps`, or by predictions being inexplicably wrong for rows containing categorical features near a boundary where the vocab differs.

**What test would have caught it:** A round-trip test that `Fit`s on a training set, `Fit`s a second converter instance on an inference set missing one rare category present in training, and asserts either an explicit schema-mismatch error or identical column ordering to the training instance — today there is no such check to write a test against.

---

## Coverage

**Reviewed and found clean** (read in full, no defect found):
- `Sources/Main/Data/Prepare/RobustScalingLayer.cs` — `_fitted` guard, `RequireFitted`, zero-IQR fallback all correct.
- `Sources/Main/Data/Prepare/OutlierClipLayer.cs` — same pattern, correct, `lowVal >= highVal` degenerate-range guard present.
- `Sources/Main/Data/Prepare/CorrelationFilterLayer.cs` — `_fitted` guard correct; `ChooseColumnToDrop` only reads `targetCorrelations` when it was actually computed.
- `Sources/Main/Data/Prepare/ShapSelectionLayer.cs` — `_fitted` guard correct (the layer itself; its *dependency* `FastRandomForest` is broken — see finding #1).
- `Sources/Main/Data/Prepare/DuplicateRowFilterLayer.cs`, `TechnicalSanityLayer.cs` — logic is internally consistent (row-hash/bucket dedup, corruption-ratio filtering); flagged only as a lower-confidence architectural note, not a scored defect: neither this layer nor `DataPipeline`/`IDataLayer` distinguishes a "training/fit" call from a "serving/inference" call, so a row-dropping layer run against a live single-row or small-batch inference request would silently return fewer predictions than inputs with no count or index reported back to the caller. Not scored because it's plausibly intentional (dataset-curation-only usage) and this repo's docs don't state the pipeline is used on live single-row inference — but it is worth the caller's own judgement call.
- `Sources/Main/Data/Prepare/LogTransformLayer.cs` — stateless, throws loudly on out-of-range column index.
- `Sources/Main/Data/Prepare/DataPipeline.cs` — row-desync check between `Features`/`Targets` after every layer is correct.
- `Sources/Main/Data/Normalizers/MinMaxNormalizer.cs`, `ZScoreNormalizer.cs`, `Log1pNormalizer.cs` — all correctly gate `Transform` on frozen state and guard divide-by-zero (range/stdDev floors).
- `Sources/Main/Data/Normalizers/DateTimeNormalizer.cs` — stateless, no issues.
- `Sources/Main/Data/Contracts/*.cs` — plain data contracts, config records; no logic to break.
- `Sources/Main/Data/Serialization/ModelSerializer.cs`, `OverfitJsonContext.cs` — internally consistent for what they cover (see finding #4 for the gap in what they *don't* cover).
- `Sources/Main/Data/Features/IndexedFeatureNameProvider.cs`, `CustomFeatureNameProvider.cs` — trivial, correct.
- `Sources/Main/Data/Features/FeatureImportanceAnalyzer.cs` — entire file body is commented out (dead code, not compiled); not scored.
- `Sources/Main/Data/Interpretation/ModelInterpreter.cs` — permutation-importance and Pearson-correlation helpers look correct on inspection.

**Not reached** (no time spent — open these first on a follow-up pass):
- `Sources/Main/Data/Contracts/FeatureImportanceReport.cs`, `FeatureImportanceResult.cs`, `FeatureImportanceVerdict.cs`, `FeatureReport.cs`, `CorrelationPair.cs`, `TrainingProgress.cs` — skimmed only enough to confirm they're plain contracts; not scrutinized field-by-field.
- `../Statistical/GlobalShapAnalyzer.cs`, `../Statistical/ShapKernel.cs` — referenced by `ShapSelectionLayer` and named in the Data README as living outside this directory; out of the stated scope, but worth a follow-up hunt given finding #1 means they're currently being fed a broken model function.
- `Tests/TestSupport/Prepare/DataPreparationTests.cs` — only grepped for "Boruta" (zero hits), not read in full; would likely show which of the four findings above already have partial coverage.

## What a short score means here

The hunt stopped at 4 confirmed defects (8 points) well inside the 10-minute budget, not because the scope ran out but because it reached a natural checkpoint with two clearly independent, high-confidence findings (the shared-root-cause pair #1/#2, plus #3 and #4) and a residual "not reached" list that's honestly still open. This is **not** a clean bill of health for the unreached files — treat the Not Reached list as the next hunt's starting point, not as evidence of anything.
