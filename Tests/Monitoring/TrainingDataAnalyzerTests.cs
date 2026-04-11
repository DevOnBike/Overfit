// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Data.Contracts;

namespace DevOnBike.Overfit.Tests.Monitoring
{
    // =============================================================================
    // TrainingDataAnalyzerTests
    // =============================================================================

    public sealed class TrainingDataAnalyzerTests
    {
        // -------------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------------

        private static List<float[]> MakeVectors(int count, int dim, float value = 0.5f)
        {
            var rng = new Random(42);
            var list = new List<float[]>(count);
            for (var i = 0; i < count; i++)
            {
                var v = new float[dim];
                for (var d = 0; d < dim; d++)
                {
                    // Add small noise so features are not perfectly constant
                    v[d] = value + (float)(rng.NextDouble() * 0.2 - 0.1);
                }
                list.Add(v);
            }
            return list;
        }

        private static List<float[]> MakeConstantVectors(int count, int dim)
        {
            var list = new List<float[]>(count);
            for (var i = 0; i < count; i++) { list.Add(new float[dim]); } // all zeros
            return list;
        }

        private const int Dim = 48; // 12 × 4

        // -------------------------------------------------------------------------
        // Argument validation
        // -------------------------------------------------------------------------

        [Fact]
        public void Analyze_WhenVectorsIsNull_ThenThrowsArgumentNullException()
            => Assert.Throws<ArgumentNullException>(
                () => new TrainingDataAnalyzer().Analyze(null!));

        [Fact]
        public void Analyze_WhenVectorsIsEmpty_ThenThrowsArgumentException()
            => Assert.Throws<ArgumentException>(
                () => new TrainingDataAnalyzer().Analyze([]));

        // -------------------------------------------------------------------------
        // Sample count
        // -------------------------------------------------------------------------

        [Fact]
        public void Analyze_WhenTooFewSamples_ThenReportHasError()
        {
            var config = new TrainingDataAnalyzerConfig { MinSamples = 200 };
            var analyzer = new TrainingDataAnalyzer(config);
            var vectors = MakeVectors(50, Dim);

            var report = analyzer.Analyze(vectors);

            Assert.False(report.IsViableForTraining);
            Assert.True(report.Errors.Count > 0);
            Assert.Contains("50", report.Errors[0]);
        }

        [Fact]
        public void Analyze_WhenEnoughSamples_ThenIsViableForTraining()
        {
            var vectors = MakeVectors(300, Dim);
            var report = new TrainingDataAnalyzer().Analyze(vectors);

            Assert.True(report.IsViableForTraining);
        }

        [Fact]
        public void Analyze_WhenBetweenMinAndDoubleMin_ThenHasWarning()
        {
            var config = new TrainingDataAnalyzerConfig { MinSamples = 200 };
            var analyzer = new TrainingDataAnalyzer(config);
            var vectors = MakeVectors(250, Dim); // between 200 and 400

            var report = analyzer.Analyze(vectors);

            Assert.True(report.IsViableForTraining);
            Assert.True(report.Warnings.Count > 0);
        }

        // -------------------------------------------------------------------------
        // Non-finite values
        // -------------------------------------------------------------------------

        [Fact]
        public void Analyze_WhenNaNPresent_ThenReportHasError()
        {
            var vectors = MakeVectors(300, Dim);
            vectors[5][3] = float.NaN;

            var report = new TrainingDataAnalyzer().Analyze(vectors);

            Assert.False(report.IsViableForTraining);
            Assert.True(report.TotalNonFiniteCount > 0);
        }

        [Fact]
        public void Analyze_WhenInfPresent_ThenReportHasError()
        {
            var vectors = MakeVectors(300, Dim);
            vectors[10][7] = float.PositiveInfinity;

            var report = new TrainingDataAnalyzer().Analyze(vectors);

            Assert.False(report.IsViableForTraining);
        }

        [Fact]
        public void Analyze_WhenNoNonFinite_ThenTotalNonFiniteCountIsZero()
        {
            var report = new TrainingDataAnalyzer().Analyze(MakeVectors(300, Dim));
            Assert.Equal(0, report.TotalNonFiniteCount);
        }

        // -------------------------------------------------------------------------
        // Constant features
        // -------------------------------------------------------------------------

        [Fact]
        public void Analyze_WhenAllFeaturesConstant_ThenReportHasError()
        {
            var vectors = MakeConstantVectors(300, Dim);
            var report = new TrainingDataAnalyzer().Analyze(vectors);

            Assert.False(report.IsViableForTraining);
            Assert.True(report.ConstantFeatureCount > 0);
        }

        [Fact]
        public void Analyze_WhenSingleFeatureConstant_ThenReportHasWarning()
        {
            var vectors = MakeVectors(300, Dim);
            // Force feature index 0 to be constant
            foreach (var v in vectors) { v[0] = 0f; }

            var report = new TrainingDataAnalyzer().Analyze(vectors);

            Assert.True(report.IsViableForTraining); // only one constant — below 50 % threshold
            Assert.Equal(1, report.ConstantFeatureCount);
            Assert.True(report.Warnings.Any(w => w.Contains("constant")));
        }

        [Fact]
        public void Analyze_WhenFeaturesVary_ThenConstantFeatureCountIsZero()
        {
            var report = new TrainingDataAnalyzer().Analyze(MakeVectors(300, Dim));
            Assert.Equal(0, report.ConstantFeatureCount);
        }

        // -------------------------------------------------------------------------
        // Feature reports
        // -------------------------------------------------------------------------

        [Fact]
        public void Analyze_WhenCalled_ThenFeatureReportsCountEqualsFeatureDimension()
        {
            var vectors = MakeVectors(300, Dim);
            var report = new TrainingDataAnalyzer().Analyze(vectors);

            Assert.Equal(Dim, report.FeatureReports.Count);
            Assert.Equal(Dim, report.FeatureDimension);
        }

        [Fact]
        public void Analyze_WhenCalled_ThenFeatureReportsHaveCorrectIndex()
        {
            var report = new TrainingDataAnalyzer().Analyze(MakeVectors(300, Dim));
            for (var i = 0; i < Dim; i++)
            {
                Assert.Equal(i, report.FeatureReports[i].Index);
            }
        }

        [Fact]
        public void Analyze_WhenCalled_ThenFeatureReportsHaveNames()
        {
            var report = new TrainingDataAnalyzer().Analyze(MakeVectors(300, Dim));
            Assert.All(report.FeatureReports, f =>
                Assert.False(string.IsNullOrEmpty(f.Name)));
        }

        [Fact]
        public void Analyze_WhenCalled_ThenFirstFeatureNameIsCpuUsageRatioMean()
        {
            var report = new TrainingDataAnalyzer().Analyze(MakeVectors(300, Dim));
            Assert.Equal("CpuUsageRatio.mean", report.FeatureReports[0].Name);
        }

        [Fact]
        public void Analyze_WhenCalled_ThenStatisticsAreFinite()
        {
            var report = new TrainingDataAnalyzer().Analyze(MakeVectors(300, Dim));
            Assert.All(report.FeatureReports, f =>
            {
                Assert.True(float.IsFinite(f.Mean), $"{f.Name} mean not finite");
                Assert.True(float.IsFinite(f.Std), $"{f.Name} std not finite");
                Assert.True(f.Std >= 0f, $"{f.Name} std negative");
                Assert.True(f.Min <= f.Max, $"{f.Name} min > max");
            });
        }

        [Fact]
        public void Analyze_WhenCalled_ThenSampleCountMatchesInput()
        {
            var vectors = MakeVectors(300, Dim);
            var report = new TrainingDataAnalyzer().Analyze(vectors);
            Assert.Equal(300, report.SampleCount);
        }

        // -------------------------------------------------------------------------
        // Correlation
        // -------------------------------------------------------------------------

        [Fact]
        public void Analyze_WhenFeaturesIdentical_ThenHighCorrelationPairIsReported()
        {
            var vectors = MakeVectors(300, Dim);
            // Make feature 0 and feature 1 perfectly correlated
            foreach (var v in vectors) { v[1] = v[0]; }

            var report = new TrainingDataAnalyzer().Analyze(vectors);

            Assert.True(report.HighCorrelationPairs.Any(p =>
                (p.FeatureIndexA == 0 && p.FeatureIndexB == 1) ||
                (p.FeatureIndexA == 1 && p.FeatureIndexB == 0)));
        }

        [Fact]
        public void Analyze_WhenFeaturesUncorrelated_ThenNoHighCorrelationPairs()
        {
            var report = new TrainingDataAnalyzer().Analyze(MakeVectors(300, Dim));
            Assert.Empty(report.HighCorrelationPairs);
        }

        [Fact]
        public void Analyze_WhenCorrelationDisabled_ThenHighCorrelationPairsIsEmpty()
        {
            var vectors = MakeVectors(300, Dim);
            foreach (var v in vectors) { v[1] = v[0]; } // would cause correlation pair

            var config = new TrainingDataAnalyzerConfig { ComputeCorrelation = false };
            var analyzer = new TrainingDataAnalyzer(config);
            var report = analyzer.Analyze(vectors);

            Assert.Empty(report.HighCorrelationPairs);
        }
    }

    // =============================================================================
    // HistoricalCsvLoaderTests
    // =============================================================================

    // =============================================================================
    // PrometheusHistoricalSourceTests
    // =============================================================================

}