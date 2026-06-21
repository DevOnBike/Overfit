// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Trees;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Trees.Diagnostics
{
    /// <summary>
    /// Where the pure-managed predictor lands against the XGBoost reference target (see
    /// <c>Scripts/bench_xgboost.py</c>). Single-threaded so it compares directly to XGBoost's
    /// <c>nthread=1</c> batch number, and it measures the online single-row path (our moat: no
    /// Python / DMatrix marshalling tax). Needs the 300×6 model at <c>C:\xgb\big_model.json</c>
    /// (export: <c>Scripts/bench_xgboost.py</c> config). [LongFact] — diagnostic, skipped by default.
    /// </summary>
    public sealed class XgboostPredictorThroughputTests
    {
        private const string ModelPath = @"C:\xgb\big_model.json";

        private readonly ITestOutputHelper _output;

        public XgboostPredictorThroughputTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        public void ManagedPredictor_Throughput()
        {
            if (!File.Exists(ModelPath))
            {
                _output.WriteLine($"model not found at {ModelPath} — skipping");
                return;
            }

            var model = XgboostModelLoader.Load(ModelPath);
            const int rows = 50_000;
            var features = model.NumFeatures;

            // Deterministic synthetic batch with ~5% missing values (mirrors the reference set).
            var rng = new Random(7);
            var data = new float[rows * features];

            for (var i = 0; i < data.Length; i++)
            {
                data[i] = rng.NextSingle() < 0.05f ? float.NaN : (float)(rng.NextDouble() * 4.0 - 2.0);
            }

            var output = new float[rows * model.NumGroups];

            // ── Batch (single-thread): best-of-N ───────────────────────────────────
            for (var w = 0; w < 3; w++)
            {
                model.PredictBatch(data, rows, output);
            }

            var bestBatchMs = double.MaxValue;

            for (var rep = 0; rep < 10; rep++)
            {
                var sw = Stopwatch.StartNew();
                model.PredictBatch(data, rows, output);
                sw.Stop();
                bestBatchMs = Math.Min(bestBatchMs, sw.Elapsed.TotalMilliseconds);
            }

            var nsPerRow = bestBatchMs * 1e6 / rows;
            var rowsPerSec = rows / (bestBatchMs / 1000.0);

            // ── Batch parallel A/B: branchy vs branchless vs block-of-rows, best-of-N each ──
            var bestBranchyMs = MeasureParallel(model, data, rows, output, BoostedTreeModel.BatchKernel.Branchy);
            var bestBranchlessMs = MeasureParallel(model, data, rows, output, BoostedTreeModel.BatchKernel.Branchless);
            var bestParMs = MeasureParallel(model, data, rows, output, BoostedTreeModel.BatchKernel.Blocked);

            var nsPerRowPar = bestParMs * 1e6 / rows;
            var nsPerRowBranchy = bestBranchyMs * 1e6 / rows;
            var nsPerRowBranchless = bestBranchlessMs * 1e6 / rows;
            var rowsPerSecPar = rows / (bestParMs / 1000.0);
            var speedup = bestBatchMs / bestParMs;
            var blockedGain = bestBranchlessMs / bestParMs;

            // ── Online single-row latency: best-of-N over a full sweep ──────────────
            var single = new float[model.NumGroups];
            var bestOnlineNs = double.MaxValue;

            for (var rep = 0; rep < 5; rep++)
            {
                var sw = Stopwatch.StartNew();

                for (var r = 0; r < rows; r++)
                {
                    model.Predict(new ReadOnlySpan<float>(data, r * features, features), single);
                }

                sw.Stop();
                bestOnlineNs = Math.Min(bestOnlineNs, sw.Elapsed.TotalMilliseconds * 1e6 / rows);
            }

            _output.WriteLine($"model: {model.NumTrees} trees, {model.NumFeatures} features, {model.NumGroups} group(s)  ({OverfitParallel.WorkerCount} workers)");
            _output.WriteLine($"BATCH  single-thread     : {bestBatchMs:F2} ms / {rows:N0} rows  ->  {nsPerRow:F0} ns/row  ({rowsPerSec:N0} rows/s)");
            _output.WriteLine($"BATCH  parallel branchy   : {bestBranchyMs:F2} ms  ->  {nsPerRowBranchy:F0} ns/row");
            _output.WriteLine($"BATCH  parallel branchless: {bestBranchlessMs:F2} ms  ->  {nsPerRowBranchless:F0} ns/row");
            _output.WriteLine($"BATCH  parallel BLOCKED   : {bestParMs:F2} ms  ->  {nsPerRowPar:F0} ns/row  ({rowsPerSecPar:N0} rows/s)   block gain vs branchless {blockedGain:F2}x   total speedup {speedup:F1}x");
            _output.WriteLine($"ONLINE single-row        : {bestOnlineNs:F0} ns/row  ({bestOnlineNs / 1000.0:F2} us/row)");
            _output.WriteLine("XGBoost reference (same 300x6): batch 1-thread 2177 ns/row, all-cores 217 ns/row, online 157 us/row (Python+DMatrix).");

            // Smoke guards — generous; this is a measurement, not a tight regression gate.
            Assert.True(nsPerRow < 20_000, $"batch {nsPerRow:F0} ns/row unexpectedly slow");
            Assert.True(nsPerRowPar <= nsPerRow, $"parallel {nsPerRowPar:F0} ns/row not faster than sequential {nsPerRow:F0}");
            Assert.True(bestOnlineNs < 50_000, $"online {bestOnlineNs:F0} ns/row unexpectedly slow");
        }

        private static double MeasureParallel(
            BoostedTreeModel model,
            float[] data,
            int rows,
            float[] output,
            BoostedTreeModel.BatchKernel kernel)
        {
            for (var w = 0; w < 3; w++)
            {
                model.PredictBatchParallel(data, rows, output, kernel);
            }

            var best = double.MaxValue;

            for (var rep = 0; rep < 10; rep++)
            {
                var sw = Stopwatch.StartNew();
                model.PredictBatchParallel(data, rows, output, kernel);
                sw.Stop();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
            }

            return best;
        }
    }
}
