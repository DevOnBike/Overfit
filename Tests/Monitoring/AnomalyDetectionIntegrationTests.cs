// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Monitoring;
using DevOnBike.Overfit.Monitoring.Contracts;
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
        // =========================================================================
        // [FACT]
        // =========================================================================

        [Fact]
        public void TrainOnNormalHistory_WhenAnomalyInjected_ThenScoreIsHigherThanNormal()
        {
            // ----------------------------------------------------------------
            // STEP 1 — Generate 8 hours of normal synthetic traffic
            // ----------------------------------------------------------------
            output.WriteLine("=== STEP 1: Generating synthetic historical data ===");

            var normalHistory = GenerateNormalHistory(RawSamples, seed: 42);
            var trainingVectors = ExtractFeatureVectors(normalHistory);

            output.WriteLine($"Raw samples:        {normalHistory.Count}");
            output.WriteLine($"Training windows:   {trainingVectors.Count}");
            output.WriteLine($"Feature vector dim: {InputSize} (={FeatureCount} features × {FeatureExtractor.StatsPerFeature} stats)");

            Assert.True(trainingVectors.Count > 100, $"Expected >100 training windows, got {trainingVectors.Count}");

            // Normalize to [0, 1] per dimension — raw values (MemoryBytes ~280M,
            // HeapBytes ~200M) cause gradient explosion without this step.
            // OfflineTrainingJob contract: "Training data is expected to be already normalised."
            var normalizer = MinMaxNormalizer.Fit(trainingVectors);
            var normalizedVectors = normalizer.TransformAll(trainingVectors);

            output.WriteLine($"Normalization: fitted on {normalizedVectors.Count} vectors ({InputSize} dims)");

            // ----------------------------------------------------------------
            // STEP 2 — Build and train the autoencoder
            // ----------------------------------------------------------------
            output.WriteLine("\n=== STEP 2: Training AnomalyAutoencoder ===");

            using var autoencoder = new AnomalyAutoencoder(
                inputSize: InputSize,
                hidden1: 24,
                hidden2: 12,
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
                    progressLog.Add($"  Epoch {p.Epoch,3}/{p.TotalEpochs} — loss: {p.EpochLoss:F6}");
                }
            });

            var result = new OfflineTrainingJob(config).Run(autoencoder, scorer, normalizedVectors, progress);

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

            output.WriteLine($"  Loss reduced: {result.LossReduction * 100:F1}% ✓");
            output.WriteLine($"  Calibrated:   {scorer.IsCalibrated} ✓");
            output.WriteLine($"  Threshold:    {scorer.Threshold:F6} ✓");

            // ----------------------------------------------------------------
            // STEP 4 — Score normal samples (should be low)
            // ----------------------------------------------------------------
            output.WriteLine("\n=== STEP 4: Scoring normal samples ===");

            var reconstruction = new float[InputSize];
            var normalScores = trainingVectors
                .Skip(trainingVectors.Count - 30)
                .Select(normalizer.Transform)
                .Select(features =>
                {
                    autoencoder.Reconstruct(features, reconstruction);
                    return scorer.ComputeScore(ReconstructionScorer.ComputeMse(features, reconstruction));
                })
                .ToList();

            var avgNormalScore = normalScores.Average();

            output.WriteLine($"  Normal samples scored: {normalScores.Count}");
            output.WriteLine($"  Avg score: {avgNormalScore:F4}");
            output.WriteLine($"  Max score: {normalScores.Max():F4}");

            Assert.True(avgNormalScore < 0.5f,
                $"Normal data should score below 0.5 on average, got {avgNormalScore:F4}");

            // ----------------------------------------------------------------
            // STEP 5 — Score anomaly scenarios (should be high)
            // ----------------------------------------------------------------
            output.WriteLine("\n=== STEP 5: Scoring anomaly scenarios ===");

            foreach (var (scenarioName, snapshots) in GenerateAnomalyScenarios(seed: 99))
            {
                var features = normalizer.Transform(ExtractLastVector(snapshots));
                autoencoder.Reconstruct(features, reconstruction);
                var mse = ReconstructionScorer.ComputeMse(features, reconstruction);
                var score = scorer.ComputeScore(mse);

                output.WriteLine($"  [{scenarioName,-28}] score={score:F4}  mse={mse:F6}");

                Assert.True(score > avgNormalScore,
                    $"Anomaly '{scenarioName}' score {score:F4} should exceed avg normal {avgNormalScore:F4}");
            }

            // ----------------------------------------------------------------
            // STEP 6 — ModelPersistence round-trip
            // ----------------------------------------------------------------
            output.WriteLine("\n=== STEP 6: ModelPersistence round-trip ===");

            var testFeatures = normalizer.Transform(trainingVectors[trainingVectors.Count / 2]);
            var reconBefore = new float[InputSize];
            var reconAfter = new float[InputSize];

            autoencoder.Reconstruct(testFeatures, reconBefore);
            var scoreBefore = scorer.ComputeScore(ReconstructionScorer.ComputeMse(testFeatures, reconBefore));

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

        // =========================================================================
        // Constants
        // =========================================================================

        private const int FeatureCount = MetricSnapshot.FeatureCount;                     // 8
        private const int WindowSize = 6;                                                // 60 s at 10 s scrape
        private const int StepSize = 1;
        private const int InputSize = FeatureCount * FeatureExtractor.StatsPerFeature;  // 32
        private const int RawSamples = 2880;                                             // ~8 h at 10 s

        // =========================================================================
        // Data generators
        // =========================================================================

        /// <summary>
        /// Generates realistic normal-traffic metric snapshots.
        ///
        /// Profiles modelled:
        ///   CPU:          0.15–0.45 with slight daily sinusoidal wave + white noise
        ///   Memory working set: 200–400 MB slow drift + small oscillation
        ///   Latency p50/p95/p99: 60/110/180 ms with occasional spikes (99th percentile events)
        ///   RPS:          50–200 r/s, higher during "business hours" simulation
        ///   Error rate:   0–0.5 % — sporadic, never sustained
        ///   GC pause ratio: 0.01–0.05, correlated with heap size
        ///   ThreadPool:   5–25 items
        ///   GC Gen2 heap: 50–200 MB slow growth then GC collection
        /// </summary>
        private static List<MetricSnapshot> GenerateNormalHistory(int sampleCount, int seed = 42)
        {
            var rng = new Random(seed);
            var snapshots = new List<MetricSnapshot>(sampleCount);
            var heap = 200_000_000f;

            for (var i = 0; i < sampleCount; i++)
            {
                heap += rng.NextSingle() * 500_000f;
                if (heap > 350_000_000f)
                {
                    heap = 160_000_000f + rng.NextSingle() * 20_000_000f;
                }

                var tod = (float)Math.Sin(2 * Math.PI * i / (6 * 60)); // sinusoidal daily wave

                snapshots.Add(new MetricSnapshot
                {
                    Timestamp = DateTime.UtcNow.AddSeconds(-sampleCount + i),
                    PodName = "integration-test-pod",
                    CpuUsageRatio = Clamp(0.30f + 0.08f * tod + Noise(rng, 0.04f), 0.05f, 0.60f),
                    CpuThrottleRatio = Clamp(Noise(rng, 0.01f), 0f, 0.15f),
                    MemoryWorkingSetBytes = Clamp(280_000_000f + 60_000_000f * tod + Noise(rng, 5_000_000f), 150_000_000f, 420_000_000f),
                    OomEventsRate = 0f,
                    LatencyP50Ms = Clamp(60f + 10f * tod + Noise(rng, 8f), 30f, 120f),
                    LatencyP95Ms = Clamp(110f + 20f * tod + Noise(rng, 15f) + (rng.NextDouble() < 0.01 ? 40f : 0f), 60f, 200f),
                    LatencyP99Ms = Clamp(180f + 30f * tod + Noise(rng, 25f) + (rng.NextDouble() < 0.01 ? 80f : 0f), 90f, 350f),
                    RequestsPerSecond = Clamp(120f + 50f * tod + Noise(rng, 20f), 30f, 250f),
                    ErrorRate = Clamp(Noise(rng, 0.002f), 0f, 0.008f),
                    GcGen2HeapBytes = heap,
                    GcPauseRatio = Clamp((5f + heap / 50_000_000f) / 1000f + Noise(rng, 0.001f), 0f, 0.05f),
                    ThreadPoolQueueLength = Clamp(12f + 5f * tod + Noise(rng, 4f), 1f, 40f)
                });
            }

            return snapshots;
        }

        /// <summary>
        /// Generates anomalous snapshots simulating three real-world failure scenarios.
        /// Each scenario produces <see cref="WindowSize"/> snapshots (one inference window).
        /// </summary>
        private static IReadOnlyDictionary<string, List<MetricSnapshot>> GenerateAnomalyScenarios(int seed = 99)
        {
            var rng = new Random(seed);

            return new Dictionary<string, List<MetricSnapshot>>
            {
                // Memory leak — heap 700 MB → 950 MB, GC thrashing, latency degrading
                ["memory_leak"] = Enumerable.Range(0, WindowSize).Select(i => new MetricSnapshot
                {
                    Timestamp = DateTime.UtcNow.AddSeconds(i * 10),
                    PodName = "integration-test-pod",
                    CpuUsageRatio = Clamp(0.55f + i * 0.04f + Noise(rng, 0.02f), 0f, 1f),
                    CpuThrottleRatio = Clamp(0.20f + i * 0.05f + Noise(rng, 0.02f), 0f, 1f),
                    MemoryWorkingSetBytes = 700_000_000f + i * 50_000_000f,
                    OomEventsRate = i >= 4 ? 0.1f : 0f,
                    LatencyP50Ms = 200f + i * 30f + Noise(rng, 15f),
                    LatencyP95Ms = 300f + i * 40f + Noise(rng, 20f),
                    LatencyP99Ms = 500f + i * 80f + Noise(rng, 40f),
                    RequestsPerSecond = Clamp(100f - i * 10f + Noise(rng, 5f), 0f, 250f),
                    ErrorRate = Clamp(0.05f + i * 0.02f + Noise(rng, 0.01f), 0f, 1f),
                    GcGen2HeapBytes = 700_000_000f + i * 50_000_000f,
                    GcPauseRatio = Clamp(0.08f + i * 0.02f + Noise(rng, 0.01f), 0f, 1f),
                    ThreadPoolQueueLength = 60f + i * 15f + Noise(rng, 5f)
                }).ToList(),

                // Traffic spike + error cascade — CPU pegged, 500 RPS, 30–60 % errors
                ["traffic_spike_errors"] = Enumerable.Range(0, WindowSize).Select(i => new MetricSnapshot
                {
                    Timestamp = DateTime.UtcNow.AddSeconds(i * 10),
                    PodName = "integration-test-pod",
                    CpuUsageRatio = Clamp(0.90f + Noise(rng, 0.03f), 0f, 1f),
                    CpuThrottleRatio = Clamp(0.70f + Noise(rng, 0.05f), 0f, 1f),
                    MemoryWorkingSetBytes = 380_000_000f + Noise(rng, 10_000_000f),
                    OomEventsRate = 0f,
                    LatencyP50Ms = 400f + i * 50f + Noise(rng, 30f),
                    LatencyP95Ms = 800f + i * 100f + Noise(rng, 50f),
                    LatencyP99Ms = 1500f + i * 200f + Noise(rng, 100f),
                    RequestsPerSecond = 500f + Noise(rng, 30f),
                    ErrorRate = Clamp(0.30f + i * 0.05f + Noise(rng, 0.02f), 0f, 1f),
                    GcGen2HeapBytes = 280_000_000f + Noise(rng, 5_000_000f),
                    GcPauseRatio = Clamp(0.03f + Noise(rng, 0.005f), 0f, 1f),
                    ThreadPoolQueueLength = 200f + i * 30f + Noise(rng, 15f)
                }).ToList(),

                // CPU throttling — throttled to limit, latency rising, RPS starved
                ["cpu_throttle"] = Enumerable.Range(0, WindowSize).Select(i => new MetricSnapshot
                {
                    Timestamp = DateTime.UtcNow.AddSeconds(i * 10),
                    PodName = "integration-test-pod",
                    CpuUsageRatio = Clamp(0.98f + Noise(rng, 0.01f), 0f, 1f),
                    CpuThrottleRatio = Clamp(0.85f + i * 0.02f + Noise(rng, 0.02f), 0f, 1f),
                    MemoryWorkingSetBytes = 260_000_000f + Noise(rng, 5_000_000f),
                    OomEventsRate = 0f,
                    LatencyP50Ms = 250f + i * 30f + Noise(rng, 20f),
                    LatencyP95Ms = 500f + i * 60f + Noise(rng, 30f),
                    LatencyP99Ms = 900f + i * 100f + Noise(rng, 60f),
                    RequestsPerSecond = Clamp(20f - i * 2f + Noise(rng, 3f), 0f, 250f),
                    ErrorRate = Clamp(0.08f + Noise(rng, 0.01f), 0f, 1f),
                    GcGen2HeapBytes = 230_000_000f + Noise(rng, 5_000_000f),
                    GcPauseRatio = Clamp(0.02f + Noise(rng, 0.005f), 0f, 1f),
                    ThreadPoolQueueLength = 80f + i * 10f + Noise(rng, 5f)
                }).ToList()
            };
        }

        // =========================================================================
        // Feature extraction
        // =========================================================================

        /// <summary>
        /// Slides a window over the snapshot history and extracts one feature vector
        /// per completed window — identical to the transformation used at inference time.
        /// </summary>
        private static List<float[]> ExtractFeatureVectors(IReadOnlyList<MetricSnapshot> snapshots)
        {
            using var buffer = new SlidingWindowBuffer(WindowSize, StepSize, FeatureCount);
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

            return vectors;
        }

        /// <summary>
        /// Returns the last feature vector from a short snapshot list.
        /// Used to score individual anomaly scenarios (each has exactly <see cref="WindowSize"/> snapshots).
        /// </summary>
        private static float[] ExtractLastVector(IReadOnlyList<MetricSnapshot> snapshots)
        {
            var vectors = ExtractFeatureVectors(snapshots);

            if (vectors.Count == 0)
            {
                throw new InvalidOperationException(
                    $"No window produced from {snapshots.Count} snapshots " +
                    $"(need at least WindowSize={WindowSize}).");
            }

            return vectors[^1];
        }

        // =========================================================================
        // MinMaxNormalizer
        // =========================================================================

        /// <summary>
        /// Per-dimension min-max normalization fitted on training data.
        ///
        /// Raw metric values (MemoryBytes ~280M, HeapBytes ~200M) produce enormous MSE
        /// gradients that prevent autoencoder convergence. Normalizing to [0, 1] before
        /// training keeps loss in a stable range.
        ///
        /// Anomaly values that exceed the training range are clamped to [0, 2] rather than
        /// [0, 1] so the autoencoder still sees they are out-of-distribution.
        ///
        /// Always use the same normalizer instance for both training and inference:
        /// <code>
        ///   var normalizer = MinMaxNormalizer.Fit(trainingVectors);
        ///   var normalized = normalizer.TransformAll(trainingVectors);  // training
        ///   var score      = scorer.Score(normalizer.Transform(liveVector), recon);  // inference
        /// </code>
        /// </summary>
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

            /// <summary>Computes per-dimension min/max from the training set. Call on training data only.</summary>
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

            /// <summary>
            /// Normalizes a single vector using training-set min/max.
            /// Values outside training range are clamped to [0, 2] — anomalies remain distinguishable.
            /// </summary>
            public float[] Transform(float[] vector)
            {
                var result = new float[_dim];
                for (var i = 0; i < _dim; i++)
                {
                    var range = _max[i] - _min[i];
                    result[i] = range > 1e-8f
                        ? Math.Clamp((vector[i] - _min[i]) / range, 0f, 2f)
                        : 0f;
                }
                return result;
            }

            /// <summary>Normalizes a collection of vectors. Returns a new list; originals are unchanged.</summary>
            public List<float[]> TransformAll(IReadOnlyList<float[]> vectors)
            {
                var result = new List<float[]>(vectors.Count);
                foreach (var v in vectors) { result.Add(Transform(v)); }
                return result;
            }
        }

        // =========================================================================
        // Math helpers
        // =========================================================================

        private static float Noise(Random rng, float amplitude)
            => (rng.NextSingle() * 2f - 1f) * amplitude;

        private static float Clamp(float v, float min, float max)
            => v < min ? min : v > max ? max : v;
    }
}