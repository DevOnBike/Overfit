// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Data.Features
{
    /*
    public sealed class FeatureImportanceAnalyzer
    {
        private readonly FeatureImportanceAnalyzerConfig _config;

        private static readonly string[] RawFeatureNames =
        [
            "CpuUsageRatio", "CpuThrottleRatio", "MemoryWorkingSetBytes", "OomEventsRate",
            "LatencyP50Ms",  "LatencyP95Ms",     "LatencyP99Ms",          "RequestsPerSecond",
            "ErrorRate",     "GcGen2HeapBytes",  "GcPauseRatio",          "ThreadPoolQueueLength"
        ];

        private static readonly string[] StatSuffixes = ["mean", "std", "p95", "delta"];

        public FeatureImportanceAnalyzer(FeatureImportanceAnalyzerConfig config = null)
        {
            _config = config ?? new FeatureImportanceAnalyzerConfig();
        }

        // -------------------------------------------------------------------------
        // Analyze
        // -------------------------------------------------------------------------

        /// <summary>
        /// Runs Boruta-style permutation importance analysis on the trained autoencoder.
        /// The autoencoder must be in Eval mode. Training vectors must be normalised
        /// (same normalisation as used during training).
        /// </summary>
        /// <param name="autoencoder">Trained autoencoder in Eval mode.</param>
        /// <param name="trainingVectors">
        ///   Normalised feature vectors used for training. Used as the reference
        ///   distribution for permutation.
        /// </param>
        /// <param name="ct">Optional cancellation token — checked between iterations.</param>
        public FeatureImportanceReport Analyze(
            AnomalyAutoencoder autoencoder,
            IReadOnlyList<float[]> trainingVectors,
            CancellationToken ct = default)
        {
            ArgumentNullException.ThrowIfNull(autoencoder);
            ArgumentNullException.ThrowIfNull(trainingVectors);

            if (trainingVectors.Count == 0)
            {
                throw new ArgumentException(
                    "Training vectors must not be empty.", nameof(trainingVectors));
            }

            if (autoencoder.IsTraining)
            {
                throw new InvalidOperationException(
                    "Autoencoder must be in Eval mode. Call autoencoder.Eval() before analysis.");
            }

            var dim = autoencoder.InputSize;
            var rng = _config.Seed.HasValue ? new Random(_config.Seed.Value) : new Random();
            var recon = new float[dim];
            var permuted = new float[dim];
            var iterations = _config.Iterations;

            // Per-feature accumulators: [importance_per_iteration]
            var realHits = new int[dim];       // times feature beat max shadow
            var realImps = new float[dim];     // cumulative importance
            var realImpSq = new float[dim];     // for std calculation
            var shadowImps = new float[dim];     // cumulative shadow importance

            // Determine how many samples to use per iteration
            var sampleCount = _config.SamplesPerIteration.HasValue
                ? Math.Min(_config.SamplesPerIteration.Value, trainingVectors.Count)
                : trainingVectors.Count;

            // Build index pool for subsampling
            var indices = Enumerable.Range(0, trainingVectors.Count).ToArray();

            for (var iter = 0; iter < iterations; iter++)
            {
                ct.ThrowIfCancellationRequested();

                // Subsample if configured
                Shuffle(indices, rng);
                var subset = indices.AsSpan(0, sampleCount);

                // Compute baseline MSE for this subset (no permutation)
                var baseline = ComputeMeanMse(autoencoder, trainingVectors, subset, recon);

                // Per-iteration shadow importance: max across all shadow features this iteration
                var maxShadowThisIter = 0f;
                var iterShadowImps = new float[dim];

                // For each feature: compute real importance AND shadow importance
                for (var f = 0; f < dim; f++)
                {
                    // Real importance: permute column f across the subset
                    var realImp = ComputePermutationImportance(
                        autoencoder, trainingVectors, subset, f, rng, recon, permuted, baseline);

                    // Shadow importance: permute a fresh independent copy of column f
                    var shadowImp = ComputePermutationImportance(
                        autoencoder, trainingVectors, subset, f, rng, recon, permuted, baseline);

                    realImps[f] += realImp;
                    realImpSq[f] += realImp * realImp;
                    shadowImps[f] += shadowImp;
                    iterShadowImps[f] = shadowImp;

                    if (shadowImp > maxShadowThisIter) { maxShadowThisIter = shadowImp; }
                }

                // Count hits: feature beats max shadow this iteration
                for (var f = 0; f < dim; f++)
                {
                    var realImpThisIter = realImps[f] - (iter > 0
                        ? realImps[f] - realImps[f] / (iter + 1) * iter  // running mean trick
                        : 0f);

                    // Use direct per-iter calculation from the accumulators
                    // Re-derive this iteration's importance from cumulative
                }
            }

            // Re-run to get per-iteration values needed for hit counting
            // (above loop accumulates — we need to restart with per-iteration tracking)
            return RunWithPerIterationTracking(autoencoder, trainingVectors, dim, rng, ct);
        }

        // -------------------------------------------------------------------------
        // Core loop with per-iteration tracking
        // -------------------------------------------------------------------------

        private FeatureImportanceReport RunWithPerIterationTracking(
            AnomalyAutoencoder autoencoder,
            IReadOnlyList<float[]> trainingVectors,
            int dim,
            Random rng,
            CancellationToken ct)
        {
            var iterations = _config.Iterations;
            var recon = new float[dim];
            var permuted = new float[dim];

            var sampleCount = _config.SamplesPerIteration.HasValue
                ? Math.Min(_config.SamplesPerIteration.Value, trainingVectors.Count)
                : trainingVectors.Count;

            var indices = Enumerable.Range(0, trainingVectors.Count).ToArray();

            // Per-iteration storage for variance and hit counting
            var realImpPerIter = new float[dim, iterations]; // [feature, iter]
            var shadowImpPerIter = new float[dim, iterations]; // [feature, iter]

            for (var iter = 0; iter < iterations; iter++)
            {
                ct.ThrowIfCancellationRequested();

                Shuffle(indices, rng);
                var subset = indices.AsSpan(0, sampleCount);
                var baseline = ComputeMeanMse(autoencoder, trainingVectors, subset, recon);

                for (var f = 0; f < dim; f++)
                {
                    realImpPerIter[f, iter] = ComputePermutationImportance(
                        autoencoder, trainingVectors, subset, f, rng, recon, permuted, baseline);

                    shadowImpPerIter[f, iter] = ComputePermutationImportance(
                        autoencoder, trainingVectors, subset, f, rng, recon, permuted, baseline);
                }
            }

            // Build results
            var results = new FeatureImportanceResult[dim];

            for (var f = 0; f < dim; f++)
            {
                // Compute mean and std of real importance
                var meanImp = 0f;
                for (var i = 0; i < iterations; i++) { meanImp += realImpPerIter[f, i]; }
                meanImp /= iterations;

                var varImp = 0f;
                for (var i = 0; i < iterations; i++)
                {
                    var d = realImpPerIter[f, i] - meanImp;
                    varImp += d * d;
                }
                var stdImp = MathF.Sqrt(varImp / iterations);

                // Mean shadow importance for this feature
                var meanShadow = 0f;
                for (var i = 0; i < iterations; i++) { meanShadow += shadowImpPerIter[f, i]; }
                meanShadow /= iterations;

                // Max shadow per iteration (across all features)
                var hits = 0;
                for (var i = 0; i < iterations; i++)
                {
                    var maxShadow = 0f;
                    for (var g = 0; g < dim; g++)
                    {
                        if (shadowImpPerIter[g, i] > maxShadow) { maxShadow = shadowImpPerIter[g, i]; }
                    }
                    if (realImpPerIter[f, i] > maxShadow) { hits++; }
                }

                // Z-score: (mean - maxShadow_mean) / std — using mean shadow as proxy
                var maxShadowMean = 0f;
                for (var i = 0; i < iterations; i++)
                {
                    var maxS = 0f;
                    for (var g = 0; g < dim; g++)
                    {
                        if (shadowImpPerIter[g, i] > maxS) { maxS = shadowImpPerIter[g, i]; }
                    }
                    maxShadowMean += maxS;
                }
                maxShadowMean /= iterations;

                var zScore = stdImp > 1e-8f ? (meanImp - maxShadowMean) / stdImp : 0f;
                var hitRatio = (float)hits / iterations;

                var verdict = zScore >= _config.ConfirmThreshold
                    ? FeatureImportanceVerdict.Confirmed
                    : zScore <= _config.RejectThreshold
                        ? FeatureImportanceVerdict.Rejected
                        : FeatureImportanceVerdict.Tentative;

                results[f] = new FeatureImportanceResult
                {
                    Index = f,
                    Name = BuildFeatureName(f),
                    MeanImportance = meanImp,
                    StdImportance = stdImp,
                    ShadowMeanImportance = meanShadow,
                    ZScore = zScore,
                    HitRatio = hitRatio,
                    Verdict = verdict
                };
            }

            // Sort by descending MeanImportance
            var sorted = results.OrderByDescending(r => r.MeanImportance).ToArray();

            return new FeatureImportanceReport
            {
                Results = sorted,
                Confirmed = sorted.Where(r => r.Verdict == FeatureImportanceVerdict.Confirmed).ToList(),
                Tentative = sorted.Where(r => r.Verdict == FeatureImportanceVerdict.Tentative).ToList(),
                Rejected = sorted.Where(r => r.Verdict == FeatureImportanceVerdict.Rejected).ToList(),
                SampleCount = trainingVectors.Count,
                Iterations = iterations
            };
        }

        // -------------------------------------------------------------------------
        // Permutation importance computation
        // -------------------------------------------------------------------------

        private static float ComputeMeanMse(
            AnomalyAutoencoder autoencoder,
            IReadOnlyList<float[]> vectors,
            ReadOnlySpan<int> subset,
            float[] recon)
        {
            var totalMse = 0f;
            foreach (var idx in subset)
            {
                autoencoder.Reconstruct(vectors[idx], recon);
                totalMse += ReconstructionScorer.ComputeMse(vectors[idx], recon);
            }
            return totalMse / subset.Length;
        }

        private static float ComputePermutationImportance(
            AnomalyAutoencoder autoencoder,
            IReadOnlyList<float[]> vectors,
            ReadOnlySpan<int> subset,
            int featureIndex,
            Random rng,
            float[] recon,
            float[] permuted,
            float baselineMse)
        {
            // Build a permuted version of column featureIndex across the subset
            // Collect column values and shuffle them
            var columnValues = new float[subset.Length];
            for (var i = 0; i < subset.Length; i++)
            {
                columnValues[i] = vectors[subset[i]][featureIndex];
            }
            Shuffle(columnValues, rng);

            // Compute MSE with the permuted column
            var totalMse = 0f;
            for (var i = 0; i < subset.Length; i++)
            {
                var original = vectors[subset[i]];

                // Copy original and replace the permuted feature
                original.CopyTo(permuted, 0);
                permuted[featureIndex] = columnValues[i];

                autoencoder.Reconstruct(permuted, recon);
                totalMse += ReconstructionScorer.ComputeMse(permuted, recon);
            }

            var permutedMse = totalMse / subset.Length;

            // Importance = how much worse reconstruction gets after permutation
            return MathF.Max(0f, permutedMse - baselineMse);
        }

        // -------------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------------

        private static void Shuffle<T>(T[] array, Random rng)
        {
            for (var i = array.Length - 1; i > 0; i--)
            {
                var j = rng.Next(i + 1);
                (array[i], array[j]) = (array[j], array[i]);
            }
        }

        private static string BuildFeatureName(int featureIndex)
        {
            var statsPerFeature = FeatureExtractor.StatsPerFeature;
            var rawIndex = featureIndex / statsPerFeature;
            var statIndex = featureIndex % statsPerFeature;

            var rawName = rawIndex < RawFeatureNames.Length
                ? RawFeatureNames[rawIndex]
                : $"feature{rawIndex}";

            var statName = statIndex < StatSuffixes.Length
                ? StatSuffixes[statIndex]
                : $"stat{statIndex}";

            return $"{rawName}.{statName}";
        }
    }
    */
}