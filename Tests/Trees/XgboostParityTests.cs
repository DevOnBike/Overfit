// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text.Json;
using DevOnBike.Overfit.Trees;

namespace DevOnBike.Overfit.Tests.Trees
{
    /// <summary>
    /// Parity of the pure-managed <see cref="BoostedTreeModel"/> against reference XGBoost predictions.
    /// Fixtures are produced by <c>Scripts/export_xgboost_fixture.py</c> (real XGBoost 3.3.0): the model
    /// JSON plus, for 200 held-out rows (some with missing/NaN features), XGBoost's raw margins
    /// (<c>output_margin=True</c>) and transformed predictions. Three objectives cover the three
    /// output transforms: logistic / softmax / identity.
    /// </summary>
    public sealed class XgboostParityTests
    {
        [Theory]
        [InlineData("clf", TreeObjective.Logistic, 1)]
        [InlineData("multi", TreeObjective.Softmax, 3)]
        [InlineData("reg", TreeObjective.Identity, 1)]
        public void ManagedPredictor_MatchesXgboost(string task, TreeObjective objective, int groups)
        {
            var dir = Path.Combine(AppContext.BaseDirectory, "test_fixtures", "xgboost");
            var model = XgboostModelLoader.Load(Path.Combine(dir, $"{task}_model.json"));

            Assert.Equal(objective, model.Objective);
            Assert.Equal(groups, model.NumGroups);

            var io = ReadFixture(Path.Combine(dir, $"{task}_io.json"));
            Assert.Equal(model.NumFeatures, io.Features[0].Length);

            var margins = new float[model.NumGroups];
            var preds = new float[model.NumGroups];
            var maxMarginDiff = 0f;
            var maxPredDiff = 0f;

            for (var r = 0; r < io.Features.Count; r++)
            {
                model.PredictRawMargins(io.Features[r], margins);
                model.Predict(io.Features[r], preds);

                for (var g = 0; g < model.NumGroups; g++)
                {
                    maxMarginDiff = MathF.Max(maxMarginDiff, MathF.Abs(margins[g] - io.Margin[r][g]));
                    maxPredDiff = MathF.Max(maxPredDiff, MathF.Abs(preds[g] - io.Pred[r][g]));
                }
            }

            // float32 traversal vs XGBoost's float32 predictor — agreement is tight.
            Assert.True(maxMarginDiff < 1e-4f, $"{task}: max raw-margin diff {maxMarginDiff:E3}");
            Assert.True(maxPredDiff < 1e-4f, $"{task}: max prediction diff {maxPredDiff:E3}");
        }

        [Fact]
        public void ScalarPredict_BinaryAndRegression_MatchSpanOverload()
        {
            var dir = Path.Combine(AppContext.BaseDirectory, "test_fixtures", "xgboost");

            foreach (var task in new[] { "clf", "reg" })
            {
                var model = XgboostModelLoader.Load(Path.Combine(dir, $"{task}_model.json"));
                var io = ReadFixture(Path.Combine(dir, $"{task}_io.json"));

                var scalar = model.Predict(io.Features[0]);
                Assert.Equal(io.Pred[0][0], scalar, 4);
            }
        }

        [Theory]
        [InlineData("clf")]
        [InlineData("multi")]
        [InlineData("reg")]
        public void PredictBatchParallel_IsBitIdentical_ToSequential(string task)
        {
            var dir = Path.Combine(AppContext.BaseDirectory, "test_fixtures", "xgboost");
            var model = XgboostModelLoader.Load(Path.Combine(dir, $"{task}_model.json"));
            var io = ReadFixture(Path.Combine(dir, $"{task}_io.json"));

            // Tile the 200 fixture rows to 2000 so the work crosses the parallel-dispatch grain
            // (below ~512 rows PredictBatchParallel runs inline) — this exercises real fan-out.
            const int tiles = 10;
            var features = model.NumFeatures;
            var rows = io.Features.Count * tiles;
            var flat = new float[rows * features];

            for (var r = 0; r < rows; r++)
            {
                io.Features[r % io.Features.Count].CopyTo(flat.AsSpan(r * features, features));
            }

            var sequential = new float[rows * model.NumGroups];
            model.PredictBatch(flat, rows, sequential);

            // Every parallel kernel must be bit-identical to the sequential reference (same decisions,
            // same per-row tree-order accumulation) — not just close.
            foreach (var kernel in new[]
            {
                BoostedTreeModel.BatchKernel.Branchy,
                BoostedTreeModel.BatchKernel.Branchless,
                BoostedTreeModel.BatchKernel.Blocked,
                BoostedTreeModel.BatchKernel.Aos
            })
            {
                var parallel = new float[rows * model.NumGroups];
                model.PredictBatchParallel(flat, rows, parallel, kernel);

                for (var i = 0; i < sequential.Length; i++)
                {
                    Assert.Equal(sequential[i], parallel[i]);
                }
            }
        }

        [Fact]
        public void Prediction_IsZeroAllocation_AfterWarmup()
        {
            var dir = Path.Combine(AppContext.BaseDirectory, "test_fixtures", "xgboost");
            var model = XgboostModelLoader.Load(Path.Combine(dir, "clf_model.json"));
            var io = ReadFixture(Path.Combine(dir, "clf_io.json"));

            const int tiles = 20;
            var features = model.NumFeatures;
            var rows = io.Features.Count * tiles;
            var flat = new float[rows * features];

            for (var r = 0; r < rows; r++)
            {
                io.Features[r % io.Features.Count].CopyTo(flat.AsSpan(r * features, features));
            }

            var output = new float[rows * model.NumGroups];
            var single = new float[model.NumGroups];

            // Warm up: JIT the paths and spawn the OverfitParallel worker pool (a one-time cost).
            model.PredictBatch(flat, rows, output);
            model.PredictBatchParallel(flat, rows, output);
            model.Predict(io.Features[0], single);

            // Per-thread counter: thread-isolated (not polluted by parallel xUnit collections). The
            // calling thread participates in a parallel chunk, so a per-row allocation would surface here;
            // OverfitParallel.For's dispatch is independently proven 0 B/call.
            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < 50; i++)
            {
                model.PredictBatch(flat, rows, output);
                model.PredictBatchParallel(flat, rows, output);
                model.Predict(io.Features[0], single);
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            Assert.Equal(0, allocated);
        }

        [Fact]
        public void Multiclass_ScalarPredict_Throws()
        {
            var dir = Path.Combine(AppContext.BaseDirectory, "test_fixtures", "xgboost");
            var model = XgboostModelLoader.Load(Path.Combine(dir, "multi_model.json"));

            Assert.Throws<OverfitRuntimeException>(() => model.Predict(new float[model.NumFeatures]));
        }

        private static Fixture ReadFixture(string path)
        {
            using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
            var root = doc.RootElement;

            var features = new List<float[]>();

            foreach (var row in root.GetProperty("features").EnumerateArray())
            {
                var values = new List<float>();

                foreach (var v in row.EnumerateArray())
                {
                    values.Add(v.ValueKind == JsonValueKind.Null ? float.NaN : v.GetSingle());
                }

                features.Add(values.ToArray());
            }

            return new Fixture(
                features,
                Read2D(root.GetProperty("margin")),
                Read2D(root.GetProperty("pred")));
        }

        private static float[][] Read2D(JsonElement array)
        {
            var rows = new List<float[]>();

            foreach (var row in array.EnumerateArray())
            {
                var values = new List<float>();

                foreach (var v in row.EnumerateArray())
                {
                    values.Add(v.GetSingle());
                }

                rows.Add(values.ToArray());
            }

            return rows.ToArray();
        }

        private sealed record Fixture(List<float[]> Features, float[][] Margin, float[][] Pred);
    }
}
