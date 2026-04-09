// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Monitoring;
using Xunit;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Monitoring
{
    /// <summary>
    /// Full end-to-end integration test for the anomaly detection pipeline.
    ///
    /// Flow:
    ///   1. Generate synthetic historical metrics representing 8 hours of normal traffic
    ///   2. Build feature vectors by sliding a 6-sample window over the history
    ///   3. Train AnomalyAutoencoder + calibrate ReconstructionScorer
    ///   4. Assert training converged (loss decreased)
    ///   5. Score normal samples — expect low anomaly scores
    ///   6. Score injected anomaly scenarios — expect high anomaly scores
    ///   7. Round-trip model through ModelPersistence — scores must be identical
    /// </summary>
    public sealed class AnomalyDetectionIntegrationTests(ITestOutputHelper output)
    {
        // -------------------------------------------------------------------------
        // Constants
        // -------------------------------------------------------------------------

        // Feature layout in MetricSnapshot (must match MetricSnapshot.WriteFeatureVector order)
        private const int F_CPU = 0;
        private const int F_MEMORY = 1;
        private const int F_LATENCY_P95 = 2;
        private const int F_RPS = 3;
        private const int F_ERROR_RATE = 4;
        private const int F_GC_PAUSE = 5;
        private const int F_THREADPOOL = 6;
        private const int F_HEAP = 7;

        private const int FeatureCount = MetricSnapshot.FeatureCount; // 8
        private const int WindowSize = 6;   // 60 s at 10 s scrape
        private const int StepSize = 1;
        private const int InputSize = FeatureCount * FeatureExtractor.StatsPerFeature; // 32

        // Training data: ~8 hours at 10 s interval = 2880 raw samples → 2875 windows
        private const int RawSamples = 2880;

        // -------------------------------------------------------------------------
        // Synthetic data generators
        // -------------------------------------------------------------------------

        /// <summary>
        /// Generates realistic normal-traffic metric snapshots.
        ///
        /// Profiles modelled:
        ///   CPU:          0.15–0.45 with slight daily sinusoidal wave + white noise
        ///   Memory:       200–400 MB slow drift + small oscillation
        ///   Latency p95:  80–150 ms with occasional 20 ms spikes (99th percentile events)
        ///   RPS:          50–200 r/s, higher during "business hours" simulation
        ///   Error rate:   0–0.5 % — sporadic, never sustained
        ///   GC pause:     2–15 ms, correlated with heap size
        ///   ThreadPool:   5–25 items
        ///   Heap:         150–350 MB slow growth then GC collection
        /// </summary>
        private static List<MetricSnapshot> GenerateNormalHistory(int sampleCount, int seed = 42)
        {
            var rng = new Random(seed);
            var snapshots = new List<MetricSnapshot>(sampleCount);
            var heap = 200_000_000f; // start at 200 MB

            for (var i = 0; i < sampleCount; i++)
            {
                // Simulate slow heap growth with periodic GC collections
                heap += rng.NextSingle() * 500_000f; // grow ~0.5 MB per step
                if (heap > 350_000_000f)
                {
                    heap = 160_000_000f + rng.NextSingle() * 20_000_000f; // GC collect
                }

                var timeOfDay = (float)Math.Sin(2 * Math.PI * i / (6 * 60)); // 1 cycle per hour

                snapshots.Add(new MetricSnapshot
                {
                    Timestamp = DateTime.UtcNow.AddSeconds(-sampleCount + i),
                    PodName = "integration-test-pod",
                    CpuUsage = Clamp(0.30f + 0.08f * timeOfDay + Noise(rng, 0.04f), 0.05f, 0.60f),
                    MemoryBytes = Clamp(280_000_000f + 60_000_000f * timeOfDay + Noise(rng, 5_000_000f), 150_000_000f, 420_000_000f),
                    RequestLatencyP95 = Clamp(110f + 20f * timeOfDay + Noise(rng, 15f) + (rng.NextDouble() < 0.01 ? 40f : 0f), 60f, 200f),
                    RequestsPerSecond = Clamp(120f + 50f * timeOfDay + Noise(rng, 20f), 30f, 250f),
                    ErrorRate = Clamp(Noise(rng, 0.002f), 0f, 0.008f),
                    GcPauseMs = Clamp(5f + heap / 50_000_000f + Noise(rng, 2f), 1f, 30f),
                    ThreadPoolQueue = Clamp(12f + 5f * timeOfDay + Noise(rng, 4f), 1f, 40f),
                    HeapBytes = heap
                });
            }

            return snapshots;
        }

        /// <summary>
        /// Generates anomalous snapshots simulating three real-world failure scenarios.
        /// Each scenario produces WindowSize snapshots (enough for one inference window).
        /// </summary>
        private static IReadOnlyDictionary<string, List<MetricSnapshot>> GenerateAnomalyScenarios(int seed = 99)
        {
            var rng = new Random(seed);

            return new Dictionary<string, List<MetricSnapshot>>
            {
                // Scenario 1: Memory leak — heap grows past threshold, GC struggles
                ["memory_leak"] = Enumerable.Range(0, WindowSize).Select(i =>
                    new MetricSnapshot
                    {
                        Timestamp = DateTime.UtcNow.AddSeconds(i * 10),
                        PodName = "integration-test-pod",
                        CpuUsage = Clamp(0.55f + i * 0.04f + Noise(rng, 0.02f), 0f, 1f),
                        MemoryBytes = 700_000_000f + i * 50_000_000f, // 700 MB → 950 MB
                        RequestLatencyP95 = 300f + i * 40f + Noise(rng, 20f), // latency degrading
                        RequestsPerSecond = Clamp(100f - i * 10f + Noise(rng, 5f), 0f, 250f),
                        ErrorRate = Clamp(0.05f + i * 0.02f + Noise(rng, 0.01f), 0f, 1f),
                        GcPauseMs = 80f + i * 20f + Noise(rng, 10f), // GC thrashing
                        ThreadPoolQueue = 60f + i * 15f + Noise(rng, 5f),  // queue backing up
                        HeapBytes = 700_000_000f + i * 50_000_000f
                    }).ToList(),

                // Scenario 2: Traffic spike + error cascade — RPS spike, error rate explodes
                ["traffic_spike_errors"] = Enumerable.Range(0, WindowSize).Select(i =>
                    new MetricSnapshot
                    {
                        Timestamp = DateTime.UtcNow.AddSeconds(i * 10),
                        PodName = "integration-test-pod",
                        CpuUsage = Clamp(0.90f + Noise(rng, 0.03f), 0f, 1f), // pegged
                        MemoryBytes = 380_000_000f + Noise(rng, 10_000_000f),
                        RequestLatencyP95 = 800f + i * 100f + Noise(rng, 50f), // latency explosion
                        RequestsPerSecond = 500f + Noise(rng, 30f),            // traffic 4× normal
                        ErrorRate = Clamp(0.30f + i * 0.05f + Noise(rng, 0.02f), 0f, 1f), // 30–60% errors
                        GcPauseMs = 12f + Noise(rng, 3f),
                        ThreadPoolQueue = 200f + i * 30f + Noise(rng, 15f), // threadpool saturated
                        HeapBytes = 280_000_000f + Noise(rng, 5_000_000f)
                    }).ToList(),

                // Scenario 3: CPU throttling — pod CPU-throttled, latency increases, RPS drops
                ["cpu_throttle"] = Enumerable.Range(0, WindowSize).Select(i =>
                    new MetricSnapshot
                    {
                        Timestamp = DateTime.UtcNow.AddSeconds(i * 10),
                        PodName = "integration-test-pod",
                        CpuUsage = Clamp(0.98f + Noise(rng, 0.01f), 0f, 1f), // throttle ceiling
                        MemoryBytes = 260_000_000f + Noise(rng, 5_000_000f),
                        RequestLatencyP95 = 500f + i * 60f + Noise(rng, 30f),
                        RequestsPerSecond = Clamp(20f - i * 2f + Noise(rng, 3f), 0f, 250f), // starved
                        ErrorRate = Clamp(0.08f + Noise(rng, 0.01f), 0f, 1f),
                        GcPauseMs = 8f + Noise(rng, 3f),
                        ThreadPoolQueue = 80f + i * 10f + Noise(rng, 5f),
                        HeapBytes = 230_000_000f + Noise(rng, 5_000_000f)
                    }).ToList()
            };
        }

        // -------------------------------------------------------------------------
        // Feature extraction helpers
        // -------------------------------------------------------------------------

        /// <summary>
        /// Converts raw MetricSnapshot history into training feature vectors
        /// by sliding a window of size <see cref="WindowSize"/> with step 1.
        /// This is the same transformation applied at inference time.
        /// </summary>
        private static List<float[]> ExtractTrainingVectors(IReadOnlyList<MetricSnapshot> snapshots)
        {
            var buffer = new SlidingWindowBuffer(WindowSize, StepSize, FeatureCount);
            var windowScratch = new float[buffer.WindowFloats];
            var statsScratch = new float[FeatureExtractor.OutputSize(FeatureCount)];
            var vectors = new List<float[]>(snapshots.Count);

            foreach (var snap in snapshots)
            {
                buffer.Add(in snap);

                if (!buffer.TryGetWindow(windowScratch, out _))
                {
                    continue;
                }

                FeatureExtractor.Extract(windowScratch, WindowSize, FeatureCount, statsScratch);

                var vec = new float[InputSize];
                statsScratch.CopyTo(vec, 0);
                vectors.Add(vec);
            }

            buffer.Dispose();
            return vectors;
        }

        /// <summary>
        /// Extracts a single feature vector from a list of WindowSize snapshots.
        /// Used for scoring individual anomaly scenarios.
        /// </summary>
        private static float[] ExtractSingleVector(IReadOnlyList<MetricSnapshot> snapshots)
        {
            var vectors = ExtractTrainingVectors(snapshots);

            if (vectors.Count == 0)
            {
                throw new InvalidOperationException(
                    $"Expected at least 1 window from {snapshots.Count} snapshots " +
                    $"(WindowSize={WindowSize}). Provide at least {WindowSize} snapshots.");
            }

            return vectors[^1]; // last window = all snapshots included
        }

        // -------------------------------------------------------------------------
        // Integration test
        // -------------------------------------------------------------------------

        [Fact(Skip = "aaa")]
        public void TrainOnNormalHistory_WhenAnomalyInjected_ThenScoreIsHigherThanNormal()
        {
            // ----------------------------------------------------------------
            // STEP 1 — Generate 8 hours of normal synthetic traffic
            // ----------------------------------------------------------------
            output.WriteLine("=== STEP 1: Generating synthetic historical data ===");

            var normalHistory = GenerateNormalHistory(RawSamples, seed: 42);
            var trainingVectors = ExtractTrainingVectors(normalHistory);

            output.WriteLine($"Raw samples:       {normalHistory.Count}");
            output.WriteLine($"Training windows:  {trainingVectors.Count}");
            output.WriteLine($"Feature vector dim: {InputSize} (={FeatureCount} features × {FeatureExtractor.StatsPerFeature} stats)");

            Assert.True(trainingVectors.Count > 100,
                $"Expected >100 training windows, got {trainingVectors.Count}");

            // Normalize to [0, 1] per dimension — raw values (MemoryBytes ~280M,
            // HeapBytes ~200M) cause gradient explosion without this step.
            // OfflineTrainingJob docs: "Training data is expected to be already normalised."
            var normalizer = MinMaxNormalizer.Fit(trainingVectors);
            var normalizedVectors = normalizer.TransformAll(trainingVectors);

            output.WriteLine($"Normalization: fitted on {normalizedVectors.Count} vectors ({InputSize} dims)");

            // ----------------------------------------------------------------
            // STEP 2 — Build and train the autoencoder
            // ----------------------------------------------------------------
            output.WriteLine("\n=== STEP 2: Training AnomalyAutoencoder ===");

            using var autoencoder = new AnomalyAutoencoder(
                inputSize: InputSize,   // 32
                hidden1: 16,
                hidden2: 8,
                bottleneckDim: 4);

            var scorer = new ReconstructionScorer();

            var config = new OfflineTrainingConfig
            {
                Epochs = 60,
                LearningRate = 5e-4f,
                CalibrationPercentile = 0.99f,
                ShuffleEachEpoch = true,
                Seed = 42
            };

            var progressLog = new List<string>();
            var progress = new Progress<TrainingProgress>(p =>
            {
                if (p.Epoch % 10 == 0 || p.Epoch == 1)
                {
                    var msg = $"  Epoch {p.Epoch,3}/{p.TotalEpochs} — loss: {p.EpochLoss:F6}";
                    progressLog.Add(msg);
                }
            });

            var job = new OfflineTrainingJob(config);
            var result = job.Run(autoencoder, scorer, normalizedVectors, progress);

            // Flush progress (Progress<T> posts to sync context — collect via log)
            foreach (var line in progressLog) { output.WriteLine(line); }

            output.WriteLine($"\nInitial loss:   {result.InitialLoss:F6}");
            output.WriteLine($"Final loss:     {result.FinalLoss:F6}");
            output.WriteLine($"Loss reduction: {result.LossReduction * 100:F1}%");
            output.WriteLine($"Threshold p99:  {scorer.Threshold:F6}");
            output.WriteLine($"Duration:       {result.Duration.TotalSeconds:F1}s");

            // ----------------------------------------------------------------
            // STEP 3 — Assert training converged
            // ----------------------------------------------------------------
            output.WriteLine("\n=== STEP 3: Checking convergence ===");

            Assert.True(result.LossReduction > 0,
                $"Expected positive loss reduction, got {result.LossReduction:F4}. " +
                $"Initial={result.InitialLoss:F6}, Final={result.FinalLoss:F6}");

            Assert.True(scorer.IsCalibrated,
                "Scorer should be calibrated after OfflineTrainingJob.Run");

            Assert.True(scorer.Threshold > 0f,
                $"Threshold must be positive, got {scorer.Threshold}");

            output.WriteLine($"  Loss reduced:    {result.LossReduction * 100:F1}% ✓");
            output.WriteLine($"  IsCalibrated:    {scorer.IsCalibrated} ✓");
            output.WriteLine($"  Threshold:       {scorer.Threshold:F6} ✓");

            // ----------------------------------------------------------------
            // STEP 4 — Score normal samples (should be low)
            // ----------------------------------------------------------------
            output.WriteLine("\n=== STEP 4: Scoring normal samples ===");

            // Use last 30 normal windows (not in the majority of training)
            var normalSamples = trainingVectors
                .Skip(trainingVectors.Count - 30)
                .Select(normalizer.Transform)
                .ToList();

            var reconstruction = new float[InputSize];
            var normalScores = normalSamples.Select(features =>
            {
                autoencoder.Reconstruct(features, reconstruction);
                var mse = ReconstructionScorer.ComputeMse(features, reconstruction);
                return scorer.ComputeScore(mse);
            }).ToList();

            var avgNormalScore = normalScores.Average();
            var maxNormalScore = normalScores.Max();

            output.WriteLine($"  Normal samples scored: {normalScores.Count}");
            output.WriteLine($"  Avg score:  {avgNormalScore:F4}");
            output.WriteLine($"  Max score:  {maxNormalScore:F4}");

            Assert.True(avgNormalScore < 0.5f,
                $"Normal data should score below 0.5 on average, got {avgNormalScore:F4}");

            // ----------------------------------------------------------------
            // STEP 5 — Score anomaly scenarios (should be high)
            // ----------------------------------------------------------------
            output.WriteLine("\n=== STEP 5: Scoring anomaly scenarios ===");

            var anomalyScenarios = GenerateAnomalyScenarios(seed: 99);

            foreach (var (scenarioName, snapshots) in anomalyScenarios)
            {
                var features = normalizer.Transform(ExtractSingleVector(snapshots));
                autoencoder.Reconstruct(features, reconstruction);
                var mse = ReconstructionScorer.ComputeMse(features, reconstruction);
                var score = scorer.ComputeScore(mse);

                output.WriteLine($"  [{scenarioName,-28}] score={score:F4}  mse={mse:F6}");

                Assert.True(score > avgNormalScore,
                    $"Anomaly '{scenarioName}' score {score:F4} should exceed " +
                    $"avg normal score {avgNormalScore:F4}");
            }

            // ----------------------------------------------------------------
            // STEP 6 — ModelPersistence round-trip
            // ----------------------------------------------------------------
            output.WriteLine("\n=== STEP 6: ModelPersistence round-trip ===");

            var testFeatures = normalizer.Transform(trainingVectors[trainingVectors.Count / 2]);
            var reconBefore = new float[InputSize];
            var reconAfter = new float[InputSize];

            autoencoder.Reconstruct(testFeatures, reconBefore);
            var scoreBefore = scorer.ComputeScore(
                ReconstructionScorer.ComputeMse(testFeatures, reconBefore));

            using var ms = new MemoryStream();
            using (var bw = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
            {
                ModelPersistence.Save(bw, autoencoder, scorer, label: "integration-test-v1");
            }

            ms.Position = 0;
            using var br = new BinaryReader(ms, System.Text.Encoding.UTF8);
            var (loadedAe, loadedScorer) = ModelPersistence.Load(br);

            using (loadedAe)
            {
                loadedAe.Eval();
                loadedAe.Reconstruct(testFeatures, reconAfter);
                var scoreAfter = loadedScorer.ComputeScore(
                    ReconstructionScorer.ComputeMse(testFeatures, reconAfter));

                output.WriteLine($"  Bundle size:       {ms.Length / 1024} KB");
                output.WriteLine($"  Score before save: {scoreBefore:F6}");
                output.WriteLine($"  Score after load:  {scoreAfter:F6}");
                output.WriteLine($"  Threshold match:   {MathF.Abs(scorer.Threshold - loadedScorer.Threshold) < 1e-5f}");

                Assert.True(MathF.Abs(scoreBefore - scoreAfter) < 1e-4f,
                    $"Score changed after save/load: {scoreBefore:F6} → {scoreAfter:F6}");

                Assert.True(MathF.Abs(scorer.Threshold - loadedScorer.Threshold) < 1e-5f,
                    "Threshold changed after save/load");
            }

            output.WriteLine("\n=== ALL STEPS PASSED ===");
        }

        // -------------------------------------------------------------------------
        // Normalization helpers
        // -------------------------------------------------------------------------

        private sealed class MinMaxNormalizer
        {
            private readonly float[] _min;
            private readonly float[] _max;
            private readonly int _dim;

            private MinMaxNormalizer(float[] min, float[] max)
            {
                _min = min;
                _max = max;
                _dim = min.Length;
            }

            /// <summary>
            /// Computes per-dimension min/max from the training set.
            /// Call once on training data, then reuse the returned instance
            /// to normalize both training and inference vectors consistently.
            /// </summary>
            public static MinMaxNormalizer Fit(IReadOnlyList<float[]> vectors)
            {
                var dim = vectors[0].Length;
                var min = new float[dim];
                var max = new float[dim];

                for (var i = 0; i < dim; i++) { min[i] = float.MaxValue; max[i] = float.MinValue; }

                foreach (var v in vectors)
                {
                    for (var i = 0; i < dim; i++)
                    {
                        if (v[i] < min[i]) { min[i] = v[i]; }
                        if (v[i] > max[i]) { max[i] = v[i]; }
                    }
                }

                return new MinMaxNormalizer(min, max);
            }

            /// <summary>Normalizes vector to [0, 1] per dimension using training-set min/max.</summary>
            public float[] Transform(float[] vector)
            {
                var result = new float[_dim];
                for (var i = 0; i < _dim; i++)
                {
                    var range = _max[i] - _min[i];
                    result[i] = range > 1e-8f
                        ? Math.Clamp((vector[i] - _min[i]) / range, 0f, 2f) // clamp: anomalies may exceed 1
                        : 0f;
                }
                return result;
            }

            public List<float[]> TransformAll(IReadOnlyList<float[]> vectors)
            {
                var result = new List<float[]>(vectors.Count);
                foreach (var v in vectors) { result.Add(Transform(v)); }
                return result;
            }
        }
        // -------------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------------

        private static float Noise(Random rng, float amplitude)
            => (rng.NextSingle() * 2f - 1f) * amplitude;

        private static float Clamp(float v, float min, float max)
            => v < min ? min : v > max ? max : v;

    }
}