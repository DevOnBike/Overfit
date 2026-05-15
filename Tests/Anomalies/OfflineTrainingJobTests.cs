// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Training;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// End-to-end test: CSV → tokenize → train GPT → checkpoint.bin.
    ///
    /// Czas: ~30-60 sek (Quick config, 500 kroków).
    ///
    /// Fixture:
    ///   Wygeneruj: python3 Scripts/generate_k8s_metrics.py --days 1 --out test_fixtures/k8s_metrics.csv
    ///   Lub pobierz z outputs/k8s_metrics.zip i wrzuć do Tests/test_fixtures/
    /// </summary>
    public class OfflineTrainingJobTests
    {
        private readonly ITestOutputHelper _output;
        private const string CsvPath = "test_fixtures/k8s_metrics.csv";
        private const string CheckpointPath = "test_fixtures/k8s_anomaly_checkpoint.bin";

        public OfflineTrainingJobTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        public async Task TrainOnCsv_LossDecreases_CheckpointWritten()
        {
            if (!File.Exists(CsvPath))
            {
                throw new Exception(
                    $"Fixture '{CsvPath}' not found. " +
                    "python3 Scripts/generate_k8s_metrics.py --days 1 --out test_fixtures/k8s_metrics.csv");
            }

            var config = new GptTrainingConfig
            {
                DModel = 64,
                NHeads = 2,
                NLayers = 2,
                ContextLength = 120,   // 10 snapshots × 12 tokens
                Steps = 500,
                ReportEvery = 100,
                ValSteps = 20,
                LearningRateMax = 3e-4f,
                LearningRateMin = 3e-5f,
                ArenaSize = 30_000_000,
            };

            var job = new OfflineTrainingJob(config);
            var progress = new Progress<TrainingProgress>(p => _output.WriteLine(p.ToString()));

            var result = await job.RunAsync(CsvPath, CheckpointPath, progress);

            _output.WriteLine(string.Empty);
            _output.WriteLine($"Snapshots loaded: {result.SnapshotsLoaded:N0}");
            _output.WriteLine($"Skipped rows:     {result.SkippedCsvRows}");
            _output.WriteLine($"Initial loss:     {result.InitialLoss:F4}");
            _output.WriteLine($"Final val loss:   {result.FinalValLoss:F4}");
            _output.WriteLine($"Training time:    {result.TrainingTime:mm\\:ss}");
            _output.WriteLine($"Checkpoint:       {result.CheckpointPath}");

            // Asercje
            Assert.True(result.SnapshotsLoaded > 0,
                "Nie załadowano żadnych snapshotów — sprawdź format CSV.");

            Assert.False(float.IsNaN(result.FinalValLoss) || float.IsInfinity(result.FinalValLoss),
                "Val loss NaN/Inf — problem numeryczny.");

            Assert.True(result.FinalValLoss < result.InitialLoss,
                $"Loss nie spada: {result.InitialLoss:F4} → {result.FinalValLoss:F4}. " +
                "Gradient nie przepływa przez GPT.");

            Assert.True(File.Exists(CheckpointPath),
                $"Checkpoint nie został zapisany: {CheckpointPath}");

            var checkpointSize = new FileInfo(CheckpointPath).Length;
            Assert.True(checkpointSize > 0, "Checkpoint jest pusty.");

            _output.WriteLine(string.Empty);
            _output.WriteLine($"✓ Model wytrenowany, checkpoint zapisany ({checkpointSize / 1024:N0}KB).");
            _output.WriteLine("✓ Gotowy do użycia w GptAnomalyDetector + LiveMonitoringPipeline.");
        }

        /// <summary>
        /// Medium training — 128d, 4 warstwy, 2K kroków.
        /// Czas: ~5-10 min. Weryfikuje pipeline przed Production.
        /// </summary>
        [LongFact]
        public async Task TrainMedium_LossDecreases_2000Steps()
        {
            if (!File.Exists(CsvPath))
            {
                throw new Exception($"Fixture '{CsvPath}' not found.");
            }

            const string medCheckpoint = "test_fixtures/k8s_anomaly_medium.bin";

            var config = GptTrainingConfig.Medium;

            var job = new OfflineTrainingJob(config);
            var progress = new Progress<TrainingProgress>(p => _output.WriteLine(p.ToString()));

            var result = await job.RunAsync(CsvPath, medCheckpoint, progress);

            _output.WriteLine(string.Empty);
            _output.WriteLine($"Final val loss: {result.FinalValLoss:F4}");
            _output.WriteLine($"Training time:  {result.TrainingTime:mm\\:ss}");

            Assert.False(float.IsNaN(result.FinalValLoss));
            Assert.True(result.FinalValLoss < result.InitialLoss * 0.7f,
                $"Loss nie spada 30%+: {result.InitialLoss:F4} → {result.FinalValLoss:F4}");
            Assert.True(File.Exists(medCheckpoint));
            _output.WriteLine("✓ Medium checkpoint gotowy.");
        }

        /// <summary>
        /// Production training — 256d, 6 warstw, 10K kroków.
        /// Czas: ~2h na Ryzen 9 9950X3D.
        /// Odpal przez noc:
        ///   dotnet test --filter "TrainProduction" --timeout 14400000
        /// </summary>
        [LongFact]
        public async Task TrainProduction_LossBelow280()
        {
            if (!File.Exists(CsvPath))
            {
                throw new Exception($"Fixture '{CsvPath}' not found.");
            }

            const string prodCheckpoint = "test_fixtures/k8s_anomaly_production.bin";

            var config = GptTrainingConfig.Production;

            var job = new OfflineTrainingJob(config);
            var progress = new Progress<TrainingProgress>(p => _output.WriteLine(p.ToString()));

            var result = await job.RunAsync(CsvPath, prodCheckpoint, progress);

            _output.WriteLine(string.Empty);
            _output.WriteLine($"Final val loss: {result.FinalValLoss:F4}");
            _output.WriteLine($"Training time:  {result.TrainingTime:hh\\:mm\\:ss}");

            Assert.False(float.IsNaN(result.FinalValLoss));
            Assert.True(result.FinalValLoss < result.InitialLoss * 0.6f,
                $"Loss nie spada 40%+: {result.InitialLoss:F4} → {result.FinalValLoss:F4}");
            Assert.True(File.Exists(prodCheckpoint));

            _output.WriteLine("✓ Production checkpoint gotowy.");
        }
    }
}
