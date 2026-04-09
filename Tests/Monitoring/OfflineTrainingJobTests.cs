// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Monitoring;

namespace DevOnBike.Overfit.Tests.Monitoring
{
    public sealed class OfflineTrainingJobTests
    {
        // -------------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------------

        private const int InputSize = 8; // small size for fast tests

        private static AnomalyAutoencoder MakeAutoencoder()
            => new(inputSize: InputSize, hidden1: 4, hidden2: 2, bottleneckDim: 2);

        private static List<float[]> MakeTrainingData(int count, int seed = 42)
        {
            var rng = new Random(seed);
            var data = new List<float[]>(count);

            for (var i = 0; i < count; i++)
            {
                var sample = new float[InputSize];
                for (var f = 0; f < InputSize; f++)
                {
                    sample[f] = (float)rng.NextDouble() * 0.5f; // values in [0, 0.5] — "normal"
                }
                data.Add(sample);
            }

            return data;
        }

        private static OfflineTrainingJob MakeJob(int epochs = 3)
            => new(new OfflineTrainingConfig
            {
                Epochs = epochs,
                LearningRate = 1e-3f,
                Seed = 42,
                ShuffleEachEpoch = false // deterministic for tests
            });

        // -------------------------------------------------------------------------
        // OfflineTrainingConfig defaults
        // -------------------------------------------------------------------------

        [Fact]
        public void Config_WhenDefaultConfig_ThenEpochsIs50()
            => Assert.Equal(50, new OfflineTrainingConfig().Epochs);

        [Fact]
        public void Config_WhenDefaultConfig_ThenLearningRateIs1e3()
            => Assert.Equal(1e-3f, new OfflineTrainingConfig().LearningRate);

        [Fact]
        public void Config_WhenDefaultConfig_ThenCalibrationPercentileIs099()
            => Assert.Equal(0.99f, new OfflineTrainingConfig().CalibrationPercentile);

        [Fact]
        public void Config_WhenDefaultConfig_ThenShuffleEachEpochIsTrue()
            => Assert.True(new OfflineTrainingConfig().ShuffleEachEpoch);

        [Fact]
        public void Config_WhenDefaultConfig_ThenSeedIsNull()
            => Assert.True(new OfflineTrainingConfig().Seed is null);

        // -------------------------------------------------------------------------
        // Run — argument validation
        // -------------------------------------------------------------------------

        [Fact]
        public void Run_WhenAutoencoderIsNull_ThenThrowsArgumentNullException()
        {
            var job = MakeJob();
            var scorer = new ReconstructionScorer();
            var data = MakeTrainingData(5);

            Assert.Throws<ArgumentNullException>(() => job.Run(null!, scorer, data));
        }

        [Fact]
        public void Run_WhenScorerIsNull_ThenThrowsArgumentNullException()
        {
            var job = MakeJob();
            using var autoencoder = MakeAutoencoder();
            var data = MakeTrainingData(5);

            Assert.Throws<ArgumentNullException>(() => job.Run(autoencoder, null!, data));
        }

        [Fact]
        public void Run_WhenTrainingDataIsNull_ThenThrowsArgumentNullException()
        {
            var job = MakeJob();
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();

            Assert.Throws<ArgumentNullException>(() => job.Run(autoencoder, scorer, null!));
        }

        [Fact]
        public void Run_WhenTrainingDataIsEmpty_ThenThrowsArgumentException()
        {
            var job = MakeJob();
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();

            Assert.Throws<ArgumentException>(() => job.Run(autoencoder, scorer, []));
        }

        [Fact]
        public void Run_WhenSampleHasWrongLength_ThenThrowsArgumentException()
        {
            var job = MakeJob();
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();
            var data = new List<float[]>
            {
                new float[InputSize - 1]
            }; // wrong size

            Assert.Throws<ArgumentException>(() => job.Run(autoencoder, scorer, data));
        }

        [Fact]
        public void Run_WhenOnlyOneSampleIsWrongLength_ThenThrowsArgumentException()
        {
            var job = MakeJob();
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();
            var data = MakeTrainingData(4);
            data.Add(new float[InputSize + 5]); // bad sample at the end

            Assert.Throws<ArgumentException>(() => job.Run(autoencoder, scorer, data));
        }

        // -------------------------------------------------------------------------
        // Run — post-conditions
        // -------------------------------------------------------------------------

        [Fact]
        public void Run_WhenCompleted_ThenScorerIsCalibrated()
        {
            var job = MakeJob();
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();
            var data = MakeTrainingData(10);

            job.Run(autoencoder, scorer, data);

            Assert.True(scorer.IsCalibrated);
        }

        [Fact]
        public void Run_WhenCompleted_ThenAutoencoderIsInEvalMode()
        {
            var job = MakeJob();
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();
            var data = MakeTrainingData(10);

            job.Run(autoencoder, scorer, data);

            Assert.False(autoencoder.IsTraining);
        }

        [Fact]
        public void Run_WhenCompleted_ThenEpochLossesLengthEqualsEpochs()
        {
            const int epochs = 5;
            var job = MakeJob(epochs);
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();
            var data = MakeTrainingData(10);

            var result = job.Run(autoencoder, scorer, data);

            Assert.Equal(epochs, result.EpochLosses.Length);
        }

        [Fact]
        public void Run_WhenCompleted_ThenAllEpochLossesAreFinite()
        {
            var job = MakeJob();
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();
            var data = MakeTrainingData(10);

            var result = job.Run(autoencoder, scorer, data);

            foreach (var loss in result.EpochLosses)
            {
                Assert.True(float.IsFinite(loss), $"Non-finite epoch loss: {loss}");
            }
        }

        [Fact]
        public void Run_WhenCompleted_ThenAllEpochLossesAreNonNegative()
        {
            var job = MakeJob();
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();
            var data = MakeTrainingData(10);

            var result = job.Run(autoencoder, scorer, data);

            foreach (var loss in result.EpochLosses)
            {
                Assert.True(loss >= 0f, $"Negative epoch loss: {loss}");
            }
        }
        [Fact]
        public void Run_WhenCompleted_ThenDurationIsPositive()
        {
            var job = MakeJob();
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();
            var data = MakeTrainingData(10);

            var result = job.Run(autoencoder, scorer, data);

            Assert.True(result.Duration > TimeSpan.Zero);
        }

        [Fact]
        public void Run_WhenCompleted_ThenFinalThresholdMatchesScorerThreshold()
        {
            var job = MakeJob();
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();
            var data = MakeTrainingData(10);

            var result = job.Run(autoencoder, scorer, data);

            Assert.Equal(scorer.Threshold, result.FinalThreshold);
        }

        [Fact]
        public void Run_WhenCompleted_ThenFinalThresholdIsPositive()
        {
            var job = MakeJob();
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();
            var data = MakeTrainingData(10);

            var result = job.Run(autoencoder, scorer, data);

            Assert.True(result.FinalThreshold > 0f);
        }

        [Fact]
        public void Run_WhenSingleSample_ThenCompletesWithoutThrowing()
        {
            var job = MakeJob(epochs: 2);
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();
            var data = MakeTrainingData(1); // edge case: single sample

            var result = job.Run(autoencoder, scorer, data);

            Assert.Equal(2, result.EpochLosses.Length);
            Assert.True(scorer.IsCalibrated);
        }

        // -------------------------------------------------------------------------
        // Run — training data is not mutated
        // -------------------------------------------------------------------------

        [Fact]
        public void Run_WhenCompleted_ThenTrainingDataIsNotMutated()
        {
            var job = MakeJob();
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();
            var data = MakeTrainingData(5);
            var copies = data.Select(s => s.ToArray()).ToList();

            job.Run(autoencoder, scorer, data);

            for (var i = 0; i < data.Count; i++)
            {
                Assert.Equal(copies[i], data[i]);
            }
        }

        // -------------------------------------------------------------------------
        // Run — progress reporting
        // -------------------------------------------------------------------------

        [Fact]
        public void Run_WhenProgressProvided_ThenReportsProgressForEachEpoch()
        {
            const int epochs = 4;
            var job = MakeJob(epochs);
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();
            var data = MakeTrainingData(5);
            var reports = new List<TrainingProgress>();

            job.Run(autoencoder, scorer, data, progress: new Progress<TrainingProgress>(p => {
                lock (reports) { reports.Add(p); }
            }));

            // Progress is reported synchronously (blocking Progress<T> calls Report immediately)
            // In practice IProgress<T> may be async — assert at least one report
            // For synchronous IProgress stub, assert exact count
            Assert.Equal(epochs, reports.Count);
        }

        [Fact]
        public void Run_WhenProgressProvided_ThenEpochNumbersAre1Based()
        {
            const int epochs = 3;
            var job = MakeJob(epochs);
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();
            var data = MakeTrainingData(5);
            var epochNumbers = new List<int>();

            var progressSink = new SynchronousProgress<TrainingProgress>(p =>
                epochNumbers.Add(p.Epoch));

            job.Run(autoencoder, scorer, data, progress: progressSink);

            Assert.Equal([1, 2, 3], epochNumbers);
        }

        [Fact]
        public void Run_WhenProgressProvided_ThenTotalEpochsMatchesConfig()
        {
            const int epochs = 3;
            var job = MakeJob(epochs);
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();
            var data = MakeTrainingData(5);
            var totalEpochs = new List<int>();

            var progressSink = new SynchronousProgress<TrainingProgress>(p =>
                totalEpochs.Add(p.TotalEpochs));

            job.Run(autoencoder, scorer, data, progress: progressSink);

            foreach (var t in totalEpochs) { Assert.Equal(epochs, t); }
        }

        [Fact]
        public void Run_WhenProgressProvided_ThenProgressPctIncreasesMonotonically()
        {
            const int epochs = 4;
            var job = MakeJob(epochs);
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();
            var data = MakeTrainingData(5);
            var pcts = new List<float>();

            var progressSink = new SynchronousProgress<TrainingProgress>(p =>
                pcts.Add(p.ProgressPct));

            job.Run(autoencoder, scorer, data, progress: progressSink);

            for (var i = 1; i < pcts.Count; i++)
            {
                Assert.True(pcts[i] > pcts[i - 1],
                $"ProgressPct did not increase: {pcts[i - 1]} → {pcts[i]}");
            }
        }

        // -------------------------------------------------------------------------
        // Run — cancellation
        // -------------------------------------------------------------------------

        [Fact]
        public void Run_WhenCancelledBeforeStart_ThenThrowsOperationCanceledException()
        {
            var job = MakeJob(epochs: 10);
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();
            var data = MakeTrainingData(5);

            using var cts = new CancellationTokenSource();
            cts.Cancel(); // cancel before Run

            Assert.Throws<OperationCanceledException>(
            () => job.Run(autoencoder, scorer, data, ct: cts.Token));
        }

        [Fact]
        public void Run_WhenCancelledDuringTraining_ThenThrowsOperationCanceledException()
        {
            var job = MakeJob(epochs: 100);
            using var autoencoder = MakeAutoencoder();
            var scorer = new ReconstructionScorer();
            var data = MakeTrainingData(5);

            using var cts = new CancellationTokenSource();

            // Cancel after the first epoch report
            var sink = new SynchronousProgress<TrainingProgress>(_ => cts.Cancel());

            Assert.Throws<OperationCanceledException>(
            () => job.Run(autoencoder, scorer, data, progress: sink, ct: cts.Token));
        }

        // -------------------------------------------------------------------------
        // OfflineTrainingResult — computed properties
        // -------------------------------------------------------------------------

        [Fact]
        public void Result_WhenEpochLossesProvided_ThenInitialLossIsFirstElement()
        {
            var result = new OfflineTrainingResult
            {
                EpochLosses = [3f, 2f, 1f]
            };
            Assert.Equal(3f, result.InitialLoss);
        }

        [Fact]
        public void Result_WhenEpochLossesProvided_ThenFinalLossIsLastElement()
        {
            var result = new OfflineTrainingResult
            {
                EpochLosses = [3f, 2f, 1f]
            };
            Assert.Equal(1f, result.FinalLoss);
        }

        [Fact]
        public void Result_WhenLossDecreases_ThenLossReductionIsPositive()
        {
            var result = new OfflineTrainingResult
            {
                EpochLosses = [4f, 2f, 1f]
            };
            Assert.True(result.LossReduction > 0f, $"LossReduction={result.LossReduction}");
        }

        [Fact]
        public void Result_WhenLossIncreases_ThenLossReductionIsNegative()
        {
            var result = new OfflineTrainingResult
            {
                EpochLosses = [1f, 2f, 4f]
            };
            Assert.True(result.LossReduction < 0f, $"LossReduction={result.LossReduction}");
        }

        [Fact]
        public void Result_WhenEpochLossesIsEmpty_ThenInitialLossIsZero()
        {
            var result = new OfflineTrainingResult
            {
                EpochLosses = []
            };
            Assert.Equal(0f, result.InitialLoss);
            Assert.Equal(0f, result.FinalLoss);
            Assert.Equal(0f, result.LossReduction);
        }

        // -------------------------------------------------------------------------
        // TrainingProgress — ProgressPct
        // -------------------------------------------------------------------------

        [Theory]
        [InlineData(1, 4, 25f)]
        [InlineData(2, 4, 50f)]
        [InlineData(4, 4, 100f)]
        [InlineData(1, 10, 10f)]
        public void TrainingProgress_WhenEpochAndTotalSet_ThenProgressPctIsCorrect(
            int epoch, int total, float expectedPct)
        {
            var p = new TrainingProgress
            {
                Epoch = epoch,
                TotalEpochs = total
            };
            Assert.True(MathF.Abs(expectedPct - p.ProgressPct) < 0.001f,
            $"ProgressPct: expected {expectedPct}, got {p.ProgressPct}");
        }

        // -------------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------------

        /// <summary>
        /// Synchronous IProgress&lt;T&gt; wrapper — calls the action immediately on Report.
        /// Necessary because System.Progress&lt;T&gt; posts to SynchronizationContext.
        /// </summary>
        private sealed class SynchronousProgress<T>(Action<T> handler) : IProgress<T>
        {
            public void Report(T value) => handler(value);
        }
    }
}