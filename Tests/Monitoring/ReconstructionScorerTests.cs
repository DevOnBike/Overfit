// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Monitoring;
using Xunit;

namespace DevOnBike.Overfit.Tests.Monitoring
{
    public sealed class ReconstructionScorerTests
    {
        private const float Tolerance = 1e-5f;

        private static void AssertClose(float expected, float actual, string label = "")
            => Assert.True(MathF.Abs(expected - actual) <= Tolerance, $"{label}: expected={expected:F6}, actual={actual:F6}");

        // -------------------------------------------------------------------------
        // ComputeMse — argument validation
        // -------------------------------------------------------------------------

        [Fact]
        public void ComputeMse_WhenSpansHaveDifferentLengths_ThenThrowsArgumentException()
            => Assert.Throws<ArgumentException>(
                () => ReconstructionScorer.ComputeMse([1f, 2f], [1f]));

        [Fact]
        public void ComputeMse_WhenSpansAreEmpty_ThenThrowsArgumentException()
            => Assert.Throws<ArgumentException>(
                () => ReconstructionScorer.ComputeMse([], []));

        // -------------------------------------------------------------------------
        // ComputeMse — correctness
        // -------------------------------------------------------------------------

        [Fact]
        public void ComputeMse_WhenIdenticalSpans_ThenReturnsZero()
        {
            float[] a = [1f, 2f, 3f, 4f];
            AssertClose(0f, ReconstructionScorer.ComputeMse(a, a), "MSE(a,a)=0");
        }

        [Fact]
        public void ComputeMse_WhenKnownValues_ThenReturnsCorrectMse()
        {
            // diff = [1,1,1,1], sumsq = 4, MSE = 4/4 = 1.0
            float[] a = [1f, 2f, 3f, 4f];
            float[] b = [0f, 1f, 2f, 3f];
            AssertClose(1.0f, ReconstructionScorer.ComputeMse(a, b), "MSE");
        }

        [Fact]
        public void ComputeMse_WhenKnownValues_ThenReturnsCorrectMseNonUniform()
        {
            // diff = [0, 2], sumsq = 4, MSE = 4/2 = 2.0
            float[] a = [0f, 2f];
            float[] b = [0f, 0f];
            AssertClose(2.0f, ReconstructionScorer.ComputeMse(a, b), "MSE");
        }

        [Fact]
        public void ComputeMse_WhenSingleElement_ThenReturnsDifferenceSquared()
        {
            // diff = 3, MSE = 9/1 = 9
            float[] a = [5f];
            float[] b = [2f];
            AssertClose(9.0f, ReconstructionScorer.ComputeMse(a, b), "MSE single");
        }

        [Fact]
        public void ComputeMse_IsSymmetric_WhenSwappingInputs()
        {
            float[] a = [1f, 3f, 5f];
            float[] b = [2f, 1f, 7f];
            AssertClose(
                ReconstructionScorer.ComputeMse(a, b),
                ReconstructionScorer.ComputeMse(b, a),
                "MSE(a,b)==MSE(b,a)");
        }

        [Fact]
        public void ComputeMse_WhenAllZeroVectors_ThenReturnsZero()
        {
            float[] a = new float[32]; // zeros
            float[] b = new float[32]; // zeros
            AssertClose(0f, ReconstructionScorer.ComputeMse(a, b), "MSE zeros");
        }

        [Fact]
        public void ComputeMse_WhenLargeVectors_ThenReturnsFiniteResult()
        {
            // Wektory > StackAllocThreshold → ścieżka ArrayPool
            var size = 600;
            var a = Enumerable.Range(0, size).Select(i => (float)i).ToArray();
            var b = Enumerable.Range(0, size).Select(i => (float)i + 1f).ToArray();

            var mse = ReconstructionScorer.ComputeMse(a, b);

            Assert.True(float.IsFinite(mse), $"Non-finite MSE: {mse}");
            AssertClose(1.0f, mse, "MSE large vectors (all diffs = 1)");
        }

        // -------------------------------------------------------------------------
        // ComputeScore
        // -------------------------------------------------------------------------

        [Fact]
        public void ComputeScore_WhenMseIsZero_ThenReturnsZero()
        {
            var scorer = new ReconstructionScorer();
            scorer.Calibrate([0.5f], percentile: 1.0f); // threshold = 0.5

            AssertClose(0f, scorer.ComputeScore(0f), "score=0");
        }

        [Fact]
        public void ComputeScore_WhenMseEqualsThreshold_ThenReturnsOne()
        {
            var scorer = new ReconstructionScorer();
            scorer.Calibrate([0.5f], percentile: 1.0f); // threshold = 0.5

            AssertClose(1.0f, scorer.ComputeScore(0.5f), "score=1 at threshold");
        }

        [Fact]
        public void ComputeScore_WhenMseExceedsThreshold_ThenReturnsClamped1()
        {
            var scorer = new ReconstructionScorer();
            scorer.Calibrate([0.5f], percentile: 1.0f);

            // MSE >> threshold — musi być clamped do 1, nie > 1
            AssertClose(1.0f, scorer.ComputeScore(99f), "clamped to 1");
        }

        [Fact]
        public void ComputeScore_WhenMseIsHalfThreshold_ThenReturns0Point5()
        {
            var scorer = new ReconstructionScorer();
            scorer.Calibrate([1.0f], percentile: 1.0f); // threshold = 1.0

            AssertClose(0.5f, scorer.ComputeScore(0.5f), "score=0.5");
        }

        // -------------------------------------------------------------------------
        // Score — integracja ComputeMse + ComputeScore
        // -------------------------------------------------------------------------

        [Fact]
        public void Score_WhenIdenticalVectors_ThenReturnsZero()
        {
            var scorer = new ReconstructionScorer();
            scorer.Calibrate([1.0f], percentile: 1.0f);
            float[] v = [1f, 2f, 3f, 4f];

            AssertClose(0f, scorer.Score(v, v), "score identical");
        }

        [Fact]
        public void Score_WhenKnownMse_ThenMatchesComputeScoreOfComputeMse()
        {
            var scorer = new ReconstructionScorer();
            scorer.Calibrate([2.0f], percentile: 1.0f); // threshold=2
            float[] a = [0f, 2f];
            float[] b = [0f, 0f]; // MSE = 2.0

            var directScore = scorer.Score(a, b);
            var separateScore = scorer.ComputeScore(ReconstructionScorer.ComputeMse(a, b));

            AssertClose(directScore, separateScore, "Score == ComputeScore(ComputeMse)");
        }

        [Fact]
        public void Score_WhenUncalibrated_ThenDefaultThresholdIs1()
        {
            var scorer = new ReconstructionScorer();
            // threshold domyślnie 1.0 — MSE=0.3 → score=0.3
            float[] a = [0f, 0f];
            float[] b = [0f, MathF.Sqrt(0.6f)]; // MSE = 0.6/2 = 0.3

            AssertClose(0.3f, scorer.Score(a, b), "uncalibrated default threshold=1");
        }

        // -------------------------------------------------------------------------
        // Calibrate — argument validation
        // -------------------------------------------------------------------------

        [Fact]
        public void Calibrate_WhenEmptySpan_ThenThrowsArgumentException()
            => Assert.Throws<ArgumentException>(
                () => new ReconstructionScorer().Calibrate([]));

        [Fact]
        public void Calibrate_WhenPercentileIsZero_ThenThrowsArgumentOutOfRange()
            => Assert.Throws<ArgumentOutOfRangeException>(
                () => new ReconstructionScorer().Calibrate([1f], percentile: 0f));

        [Fact]
        public void Calibrate_WhenPercentileExceeds1_ThenThrowsArgumentOutOfRange()
            => Assert.Throws<ArgumentOutOfRangeException>(
                () => new ReconstructionScorer().Calibrate([1f], percentile: 1.1f));

        // -------------------------------------------------------------------------
        // Calibrate — correctness
        // -------------------------------------------------------------------------

        [Fact]
        public void Calibrate_WhenCalled_ThenIsCalibratedIsTrue()
        {
            var scorer = new ReconstructionScorer();
            scorer.Calibrate([0.1f, 0.2f, 0.3f]);
            Assert.True(scorer.IsCalibrated);
        }

        [Fact]
        public void Calibrate_WhenSingleValue_ThenThresholdEqualsThatValue()
        {
            var scorer = new ReconstructionScorer();
            scorer.Calibrate([0.42f]);
            AssertClose(0.42f, scorer.Threshold, "single value threshold");
        }

        [Fact]
        public void Calibrate_WhenAllValuesIdentical_ThenThresholdEqualsThatValue()
        {
            var scorer = new ReconstructionScorer();
            scorer.Calibrate([0.5f, 0.5f, 0.5f, 0.5f]);
            AssertClose(0.5f, scorer.Threshold, "identical values threshold");
        }

        [Fact]
        public void Calibrate_WhenP99With100Values_ThenThresholdIs99thElement()
        {
            // 100 wartości [1..100] — p99 nearest-rank: ceil(0.99×100)-1=98 → sorted[98]=99
            var mse = Enumerable.Range(1, 100).Select(i => (float)i).ToArray();
            var scorer = new ReconstructionScorer();
            scorer.Calibrate(mse, percentile: 0.99f);
            AssertClose(99f, scorer.Threshold, "p99 of 1..100");
        }

        [Fact]
        public void Calibrate_WhenP100With100Values_ThenThresholdIsMaxValue()
        {
            var mse = Enumerable.Range(1, 100).Select(i => (float)i).ToArray();
            var scorer = new ReconstructionScorer();
            scorer.Calibrate(mse, percentile: 1.0f);
            AssertClose(100f, scorer.Threshold, "p100 = max");
        }

        [Fact]
        public void Calibrate_WhenAllValuesZero_ThenThresholdIsFloatEpsilon()
        {
            var scorer = new ReconstructionScorer();
            scorer.Calibrate([0f, 0f, 0f]);
            Assert.True(scorer.Threshold > 0f, "threshold > 0 even when all MSE=0");
            Assert.Equal(float.Epsilon, scorer.Threshold);
        }

        [Fact]
        public void Calibrate_WhenUnsortedValues_ThenThresholdIsCorrectPercentile()
        {
            // Wartości celowo w złej kolejności — sortowanie wewnętrzne
            float[] mse = [5f, 1f, 3f, 2f, 4f]; // sorted: [1,2,3,4,5]
            var scorer = new ReconstructionScorer();
            scorer.Calibrate(mse, percentile: 1.0f); // p100 = max = 5
            AssertClose(5f, scorer.Threshold, "p100 of unsorted values");
        }

        [Fact]
        public void Calibrate_WhenRecalibrated_ThenThresholdUpdates()
        {
            var scorer = new ReconstructionScorer();
            scorer.Calibrate([1.0f]);
            scorer.Calibrate([2.0f]); // rekalibracja
            AssertClose(2.0f, scorer.Threshold, "recalibration updates threshold");
        }

        // -------------------------------------------------------------------------
        // CalibrateFromModel
        // -------------------------------------------------------------------------

        [Fact]
        public void CalibrateFromModel_WhenNormalDataProvided_ThenIsCalibratedIsTrue()
        {
            using var autoencoder = new AnomalyAutoencoder(inputSize: 8, hidden1: 4, hidden2: 2, bottleneckDim: 2);
            autoencoder.Eval();

            var normalData = Enumerable.Range(0, 20)
                .Select(_ => Enumerable.Range(0, 8).Select(i => (float)i * 0.1f).ToArray())
                .ToList();

            var scorer = new ReconstructionScorer();
            scorer.CalibrateFromModel(autoencoder, normalData);

            Assert.True(scorer.IsCalibrated);
            Assert.True(scorer.Threshold > 0f);
        }

        [Fact]
        public void CalibrateFromModel_WhenEmptyData_ThenThrowsInvalidOperationException()
        {
            using var autoencoder = new AnomalyAutoencoder(inputSize: 8, hidden1: 4, hidden2: 2, bottleneckDim: 2);
            autoencoder.Eval();

            var scorer = new ReconstructionScorer();

            Assert.Throws<InvalidOperationException>(
                () => scorer.CalibrateFromModel(autoencoder, []));
        }

        [Fact]
        public void CalibrateFromModel_WhenWrongFeatureLength_ThenThrowsArgumentException()
        {
            using var autoencoder = new AnomalyAutoencoder(inputSize: 8, hidden1: 4, hidden2: 2, bottleneckDim: 2);
            autoencoder.Eval();

            // Wektor za krótki
            var wrongData = new List<float[]> { new float[4] };
            var scorer = new ReconstructionScorer();

            Assert.Throws<ArgumentException>(
                () => scorer.CalibrateFromModel(autoencoder, wrongData));
        }

        [Fact]
        public void CalibrateFromModel_WhenAutoencoderIsNull_ThenThrowsArgumentNullException()
        {
            var scorer = new ReconstructionScorer();

            Assert.Throws<ArgumentNullException>(
                () => scorer.CalibrateFromModel(null!, []));
        }

        // -------------------------------------------------------------------------
        // Save / Load
        // -------------------------------------------------------------------------

        [Fact]
        public void SaveLoad_WhenRoundtripped_ThenThresholdIsPreserved()
        {
            var scorer = new ReconstructionScorer();
            scorer.Calibrate([0.1f, 0.2f, 0.5f, 0.8f, 1.0f]);
            var originalThreshold = scorer.Threshold;

            using var ms = new MemoryStream();
            using (var bw = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
            {
                scorer.Save(bw);
            }

            ms.Position = 0;
            var loaded = new ReconstructionScorer();
            using (var br = new BinaryReader(ms))
            {
                loaded.Load(br);
            }

            AssertClose(originalThreshold, loaded.Threshold, "loaded threshold");
        }

        [Fact]
        public void SaveLoad_WhenRoundtripped_ThenIsCalibratedIsPreserved()
        {
            var scorer = new ReconstructionScorer();
            scorer.Calibrate([1.0f]);

            using var ms = new MemoryStream();
            using (var bw = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
            {
                scorer.Save(bw);
            }

            ms.Position = 0;
            var loaded = new ReconstructionScorer();
            using (var br = new BinaryReader(ms))
            {
                loaded.Load(br);
            }

            Assert.True(loaded.IsCalibrated);
        }

        [Fact]
        public void Save_WhenPathProvided_ThenFileIsCreatedWithContent()
        {
            var scorer = new ReconstructionScorer();
            scorer.Calibrate([0.3f]);
            var path = Path.GetTempFileName();

            try
            {
                scorer.Save(path);
                Assert.True(File.Exists(path));
                Assert.True(new FileInfo(path).Length > 0);
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Fact]
        public void Load_WhenFileDoesNotExist_ThenThrowsFileNotFoundException()
            => Assert.Throws<FileNotFoundException>(
                () => new ReconstructionScorer().Load("/tmp/nieistniejacy_scorer.bin"));

        // -------------------------------------------------------------------------
        // IsCalibrated default
        // -------------------------------------------------------------------------

        [Fact]
        public void IsCalibrated_WhenNotCalibrated_ThenIsFalse()
            => Assert.False(new ReconstructionScorer().IsCalibrated);

        [Fact]
        public void Threshold_WhenNotCalibrated_ThenDefaultIs1()
            => AssertClose(1.0f, new ReconstructionScorer().Threshold, "default threshold");
    }
}