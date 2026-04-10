// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Monitoring;
using DevOnBike.Overfit.Monitoring.Contracts;
using Xunit;

namespace DevOnBike.Overfit.Tests.Monitoring
{
    public sealed class FeatureImportanceAnalyzerTests
    {
        // -------------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------------

        private const int Dim = 48; // 12 × 4

        private static AnomalyAutoencoder MakeEvalAutoencoder()
        {
            var ae = new AnomalyAutoencoder(Dim, hidden1: 24, hidden2: 12, bottleneckDim: 6);
            // Seed weights with non-zero values so the model distinguishes features
            var rng = new Random(42);
            foreach (var p in ae.Parameters())
            {
                var span = p.Data.AsSpan();
                for (var i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(rng.NextDouble() * 0.4 - 0.2);
                }
            }
            ae.Eval();
            return ae;
        }

        private static List<float[]> MakeVectors(int count, int dim, int seed = 42)
        {
            var rng = new Random(seed);
            var list = new List<float[]>(count);
            for (var i = 0; i < count; i++)
            {
                var v = new float[dim];
                for (var d = 0; d < dim; d++)
                {
                    v[d] = (float)rng.NextDouble();
                }
                list.Add(v);
            }
            return list;
        }

        private static FeatureImportanceAnalyzerConfig FastConfig()
            => new()
            {
                Iterations = 5,
                SamplesPerIteration = 50,
                Seed = 42
            };

        // -------------------------------------------------------------------------
        // Argument validation
        // -------------------------------------------------------------------------

        [Fact]
        public void Analyze_WhenAutoencoderIsNull_ThenThrowsArgumentNullException()
        {
            var analyzer = new FeatureImportanceAnalyzer(FastConfig());
            var vectors = MakeVectors(100, Dim);

            Assert.Throws<ArgumentNullException>(
                () => analyzer.Analyze(null!, vectors));
        }

        [Fact]
        public void Analyze_WhenVectorsIsNull_ThenThrowsArgumentNullException()
        {
            using var ae = MakeEvalAutoencoder();
            var analyzer = new FeatureImportanceAnalyzer(FastConfig());

            Assert.Throws<ArgumentNullException>(
                () => analyzer.Analyze(ae, null!));
        }

        [Fact]
        public void Analyze_WhenVectorsIsEmpty_ThenThrowsArgumentException()
        {
            using var ae = MakeEvalAutoencoder();
            var analyzer = new FeatureImportanceAnalyzer(FastConfig());

            Assert.Throws<ArgumentException>(
                () => analyzer.Analyze(ae, []));
        }

        [Fact]
        public void Analyze_WhenAutoencoderIsInTrainingMode_ThenThrowsInvalidOperationException()
        {
            var ae = new AnomalyAutoencoder(Dim, hidden1: 12, hidden2: 6, bottleneckDim: 3);
            // ae.Train() is the default — do NOT call ae.Eval()
            var analyzer = new FeatureImportanceAnalyzer(FastConfig());
            var vectors = MakeVectors(100, Dim);

            Assert.Throws<InvalidOperationException>(() => analyzer.Analyze(ae, vectors));
            ae.Dispose();
        }

        // -------------------------------------------------------------------------
        // Report structure
        // -------------------------------------------------------------------------

        [Fact]
        public void Analyze_WhenCalled_ThenResultsCountEqualsInputSize()
        {
            using var ae = MakeEvalAutoencoder();
            var analyzer = new FeatureImportanceAnalyzer(FastConfig());
            var vectors = MakeVectors(100, Dim);

            var report = analyzer.Analyze(ae, vectors);

            Assert.Equal(Dim, report.Results.Count);
        }

        [Fact]
        public void Analyze_WhenCalled_ThenConfirmedPlusTentativePlusRejectedEqualsTotal()
        {
            using var ae = MakeEvalAutoencoder();
            var analyzer = new FeatureImportanceAnalyzer(FastConfig());
            var vectors = MakeVectors(100, Dim);

            var report = analyzer.Analyze(ae, vectors);

            Assert.Equal(
                report.Results.Count,
                report.Confirmed.Count + report.Tentative.Count + report.Rejected.Count);
        }

        [Fact]
        public void Analyze_WhenCalled_ThenResultsAreSortedByDescendingImportance()
        {
            using var ae = MakeEvalAutoencoder();
            var analyzer = new FeatureImportanceAnalyzer(FastConfig());
            var vectors = MakeVectors(100, Dim);

            var report = analyzer.Analyze(ae, vectors);

            for (var i = 1; i < report.Results.Count; i++)
            {
                Assert.True(
                    report.Results[i - 1].MeanImportance >= report.Results[i].MeanImportance,
                    $"Results not sorted at index {i}: " +
                    $"{report.Results[i - 1].MeanImportance} < {report.Results[i].MeanImportance}");
            }
        }

        [Fact]
        public void Analyze_WhenCalled_ThenSampleCountMatchesInput()
        {
            using var ae = MakeEvalAutoencoder();
            var analyzer = new FeatureImportanceAnalyzer(FastConfig());
            var vectors = MakeVectors(100, Dim);

            var report = analyzer.Analyze(ae, vectors);

            Assert.Equal(100, report.SampleCount);
        }

        [Fact]
        public void Analyze_WhenCalled_ThenIterationsMatchConfig()
        {
            using var ae = MakeEvalAutoencoder();
            var config = FastConfig() with { Iterations = 7 };
            var analyzer = new FeatureImportanceAnalyzer(config);
            var vectors = MakeVectors(100, Dim);

            var report = analyzer.Analyze(ae, vectors);

            Assert.Equal(7, report.Iterations);
        }

        // -------------------------------------------------------------------------
        // Feature names
        // -------------------------------------------------------------------------

        [Fact]
        public void Analyze_WhenCalled_ThenAllResultsHaveNonEmptyNames()
        {
            using var ae = MakeEvalAutoencoder();
            var analyzer = new FeatureImportanceAnalyzer(FastConfig());
            var vectors = MakeVectors(100, Dim);

            var report = analyzer.Analyze(ae, vectors);

            Assert.All(report.Results, r =>
                Assert.False(string.IsNullOrEmpty(r.Name), $"Index {r.Index} has empty name"));
        }

        [Fact]
        public void Analyze_WhenCalled_ThenFirstResultNameContainsCpuUsageRatio()
        {
            using var ae = MakeEvalAutoencoder();
            var analyzer = new FeatureImportanceAnalyzer(FastConfig());
            var vectors = MakeVectors(100, Dim);

            var report = analyzer.Analyze(ae, vectors);

            // Results are sorted by importance — find index 0 by Index field, not position
            var firstByIndex = report.Results.Single(r => r.Index == 0);
            Assert.Contains("CpuUsageRatio", firstByIndex.Name);
        }

        // -------------------------------------------------------------------------
        // Importance values
        // -------------------------------------------------------------------------

        [Fact]
        public void Analyze_WhenCalled_ThenMeanImportanceIsNonNegative()
        {
            using var ae = MakeEvalAutoencoder();
            var analyzer = new FeatureImportanceAnalyzer(FastConfig());
            var vectors = MakeVectors(100, Dim);

            var report = analyzer.Analyze(ae, vectors);

            Assert.All(report.Results, r =>
                Assert.True(r.MeanImportance >= 0f,
                    $"{r.Name}: MeanImportance={r.MeanImportance} < 0"));
        }

        [Fact]
        public void Analyze_WhenCalled_ThenStdImportanceIsNonNegative()
        {
            using var ae = MakeEvalAutoencoder();
            var analyzer = new FeatureImportanceAnalyzer(FastConfig());
            var vectors = MakeVectors(100, Dim);

            var report = analyzer.Analyze(ae, vectors);

            Assert.All(report.Results, r =>
                Assert.True(r.StdImportance >= 0f,
                    $"{r.Name}: StdImportance={r.StdImportance} < 0"));
        }

        [Fact]
        public void Analyze_WhenCalled_ThenHitRatioIsInRange()
        {
            using var ae = MakeEvalAutoencoder();
            var analyzer = new FeatureImportanceAnalyzer(FastConfig());
            var vectors = MakeVectors(100, Dim);

            var report = analyzer.Analyze(ae, vectors);

            Assert.All(report.Results, r =>
                Assert.True(r.HitRatio >= 0f && r.HitRatio <= 1f,
                    $"{r.Name}: HitRatio={r.HitRatio} out of [0,1]"));
        }

        // -------------------------------------------------------------------------
        // Determinism
        // -------------------------------------------------------------------------

        [Fact]
        public void Analyze_WhenSameSeed_ThenResultsAreDeterministic()
        {
            using var ae = MakeEvalAutoencoder();
            var config = FastConfig() with { Seed = 123 };
            var analyzer = new FeatureImportanceAnalyzer(config);
            var vectors = MakeVectors(100, Dim);

            var report1 = analyzer.Analyze(ae, vectors);
            var report2 = analyzer.Analyze(ae, vectors);

            for (var i = 0; i < report1.Results.Count; i++)
            {
                Assert.Equal(
                    report1.Results[i].MeanImportance,
                    report2.Results[i].MeanImportance,
                    1e-5f);
            }
        }

        // -------------------------------------------------------------------------
        // Cancellation
        // -------------------------------------------------------------------------

        [Fact]
        public void Analyze_WhenCancelledBeforeStart_ThenThrowsOperationCanceledException()
        {
            using var ae = MakeEvalAutoencoder();
            var config = FastConfig() with { Iterations = 100 };
            var analyzer = new FeatureImportanceAnalyzer(config);
            var vectors = MakeVectors(100, Dim);

            using var cts = new CancellationTokenSource();
            cts.Cancel();

            Assert.Throws<OperationCanceledException>(
                () => analyzer.Analyze(ae, vectors, cts.Token));
        }

        // -------------------------------------------------------------------------
        // Config
        // -------------------------------------------------------------------------

        [Fact]
        public void Config_WhenDefaultConfig_ThenIterationsIs20()
            => Assert.Equal(20, new FeatureImportanceAnalyzerConfig().Iterations);

        [Fact]
        public void Config_WhenDefaultConfig_ThenConfirmThresholdIs2()
            => Assert.Equal(2.0f, new FeatureImportanceAnalyzerConfig().ConfirmThreshold);

        [Fact]
        public void Config_WhenDefaultConfig_ThenRejectThresholdIsMinus2()
            => Assert.Equal(-2.0f, new FeatureImportanceAnalyzerConfig().RejectThreshold);
    }
}