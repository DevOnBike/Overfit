// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using System.Numerics.Tensors;
using DevOnBike.Overfit.Data.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// Validates and analyses training feature vectors before passing them to
    /// <see cref="OfflineTrainingJob"/>.
    ///
    /// Catches problems that would cause silent training failure:
    ///   - Too few samples for generalisation
    ///   - NaN / Inf values that cause gradient explosion
    ///   - Constant features the model cannot learn from
    ///   - Highly correlated features that waste model capacity
    ///
    /// Usage:
    /// <code>
    ///   var analyzer = new TrainingDataAnalyzer();
    ///   var report   = analyzer.Analyze(trainingVectors);
    ///
    ///   foreach (var warning in report.Warnings) logger.LogWarning(warning);
    ///   foreach (var error   in report.Errors)   logger.LogError(error);
    ///
    ///   if (!report.IsViableForTraining)
    ///       throw new InvalidOperationException("Training data not viable.");
    ///
    ///   var result = new OfflineTrainingJob(config).Run(autoencoder, scorer, trainingVectors);
    /// </code>
    /// </summary>
    public sealed class TrainingDataAnalyzer
    {
        private readonly TrainingDataAnalyzerConfig _config;

        // Feature names aligned with MetricSnapshot.WriteFeatureVector order.
        // Each raw metric produces StatsPerFeature=4 stats: mean, std, p95, delta.
        private static readonly string[] RawFeatureNames =
        [
            "CpuUsageRatio", "CpuThrottleRatio", "MemoryWorkingSetBytes", "OomEventsRate",
            "LatencyP50Ms",  "LatencyP95Ms",     "LatencyP99Ms",          "RequestsPerSecond",
            "ErrorRate",     "GcGen2HeapBytes",  "GcPauseRatio",          "ThreadPoolQueueLength"
        ];

        private static readonly string[] StatSuffixes = ["mean", "std", "p95", "delta"];

        public TrainingDataAnalyzer(TrainingDataAnalyzerConfig config = null)
        {
            _config = config ?? new TrainingDataAnalyzerConfig();
        }

        // -------------------------------------------------------------------------
        // Analyze
        // -------------------------------------------------------------------------

        /// <summary>
        /// Runs the full analysis pipeline on the provided training vectors.
        /// </summary>
        /// <param name="vectors">
        ///   Normalised feature vectors — output of <see cref="FeatureExtractor"/> after
        ///   <see cref="MinMaxNormalizer"/> (or equivalent). Raw un-normalised values will
        ///   produce misleading CV and correlation results.
        /// </param>
        /// <exception cref="ArgumentNullException">When vectors is null.</exception>
        /// <exception cref="ArgumentException">When vectors is empty or vectors have length 0.</exception>
        public TrainingDataReport Analyze(IReadOnlyList<float[]> vectors)
        {
            ArgumentNullException.ThrowIfNull(vectors);

            if (vectors.Count == 0)
            {
                throw new ArgumentException("Training vectors must not be empty.", nameof(vectors));
            }

            var dim = vectors[0].Length;
            var errors = new List<string>();
            var warnings = new List<string>();

            // ---- Hard check: non-finite values ----
            var nonFiniteCounts = CountNonFinite(vectors, dim);
            var totalNonFinite = nonFiniteCounts.Sum();

            if (totalNonFinite > 0)
            {
                errors.Add(
                    $"Training data contains {totalNonFinite} NaN/Inf values across " +
                    $"{nonFiniteCounts.Count(c => c > 0)} feature dimensions. " +
                    $"Normalise the data and remove bad samples before training.");
            }

            // ---- Hard check: sample count ----
            if (vectors.Count < _config.MinSamples)
            {
                errors.Add(
                    $"Only {vectors.Count} training samples — minimum required is " +
                    $"{_config.MinSamples}. Collect more historical data before training.");
            }

            // ---- Per-feature statistics ----
            var featureReports = BuildFeatureReports(vectors, dim, nonFiniteCounts);

            // ---- Hard check: too many constant features ----
            var constantCount = featureReports.Count(f => f.IsConstant);
            var constantFraction = (float)constantCount / dim;

            if (constantFraction > _config.MaxConstantFeatureFraction)
            {
                errors.Add(
                    $"{constantCount}/{dim} features are effectively constant (CV < " +
                    $"{_config.ConstantFeatureThreshold:F3}). The model cannot learn " +
                    $"from constant inputs. Check that metrics are being scraped correctly.");
            }
            else if (constantCount > 0)
            {
                var names = featureReports
                    .Where(f => f.IsConstant)
                    .Select(f => f.Name);
                warnings.Add(
                    $"{constantCount} constant features (consider removing): " +
                    string.Join(", ", names));
            }

            // ---- Warning: low sample count ----
            if (vectors.Count < _config.MinSamples * 2 && vectors.Count >= _config.MinSamples)
            {
                warnings.Add(
                    $"Only {vectors.Count} training samples. " +
                    $"More data (≥{_config.MinSamples * 2}) improves generalisation.");
            }

            // ---- Correlation matrix ----
            var highCorrPairs = new List<CorrelationPair>();

            if (_config.ComputeCorrelation && totalNonFinite == 0)
            {
                highCorrPairs = FindHighCorrelationPairs(vectors, dim, featureReports);

                if (highCorrPairs.Count > 0)
                {
                    var pairs = highCorrPairs
                        .Select(p => $"{p.FeatureNameA} ↔ {p.FeatureNameB} (r={p.Correlation:F2})");
                    warnings.Add(
                        $"{highCorrPairs.Count} highly correlated feature pairs " +
                        $"(|r| > {_config.HighCorrelationThreshold:F2}) — " +
                        $"redundant information: " + string.Join(", ", pairs));
                }
            }

            return new TrainingDataReport
            {
                SampleCount = vectors.Count,
                FeatureDimension = dim,
                FeatureReports = featureReports,
                HighCorrelationPairs = highCorrPairs,
                Warnings = warnings,
                Errors = errors
            };
        }

        // -------------------------------------------------------------------------
        // Private — per-feature stats
        // -------------------------------------------------------------------------

        private IReadOnlyList<FeatureReport> BuildFeatureReports(
            IReadOnlyList<float[]> vectors,
            int dim,
            int[] nonFiniteCounts)
        {
            var reports = new FeatureReport[dim];

            for (var f = 0; f < dim; f++)
            {
                var values = new float[vectors.Count];
                for (var s = 0; s < vectors.Count; s++) { values[s] = vectors[s][f]; }

                var mean = TensorPrimitives.Sum(values) / values.Length;
                var std = ComputeStd(values, mean);
                var cv = std / (MathF.Abs(mean) + 1e-8f);
                var min = TensorPrimitives.Min(values);
                var max = TensorPrimitives.Max(values);

                reports[f] = new FeatureReport
                {
                    Index = f,
                    Name = BuildFeatureName(f),
                    Mean = mean,
                    Std = std,
                    Min = min,
                    Max = max,
                    CoefficientOfVariation = cv,
                    IsConstant = cv < _config.ConstantFeatureThreshold,
                    NonFiniteCount = nonFiniteCounts[f]
                };
            }

            return reports;
        }

        private static float ComputeStd(float[] values, float mean)
        {
            var sumSq = 0f;
            
            foreach (var v in values)
            {
                sumSq += (v - mean) * (v - mean);
            }
            
            return MathF.Sqrt(sumSq / values.Length);
        }

        // -------------------------------------------------------------------------
        // Private — correlation
        // -------------------------------------------------------------------------

        private List<CorrelationPair> FindHighCorrelationPairs(
            IReadOnlyList<float[]> vectors,
            int dim,
            IReadOnlyList<FeatureReport> reports)
        {
            var pairs = new List<CorrelationPair>();
            var n = vectors.Count;

            // Extract columns once — avoids re-iterating vectors for every pair
            var cols = new float[dim][];
            for (var f = 0; f < dim; f++)
            {
                cols[f] = new float[n];
                for (var s = 0; s < n; s++) { cols[f][s] = vectors[s][f]; }
            }

            for (var i = 0; i < dim; i++)
            {
                for (var j = i + 1; j < dim; j++)
                {
                    var r = PearsonCorrelation(cols[i], cols[j]);

                    if (MathF.Abs(r) >= _config.HighCorrelationThreshold)
                    {
                        pairs.Add(new CorrelationPair
                        {
                            FeatureIndexA = i,
                            FeatureIndexB = j,
                            FeatureNameA = reports[i].Name,
                            FeatureNameB = reports[j].Name,
                            Correlation = r
                        });
                    }
                }
            }

            return pairs;
        }

        private static float PearsonCorrelation(float[] x, float[] y)
        {
            var n = x.Length;
            var meanX = TensorPrimitives.Sum(x) / n;
            var meanY = TensorPrimitives.Sum(y) / n;

            var cov = 0f;
            var varX = 0f;
            var varY = 0f;

            for (var i = 0; i < n; i++)
            {
                var dx = x[i] - meanX;
                var dy = y[i] - meanY;
                cov += dx * dy;
                varX += dx * dx;
                varY += dy * dy;
            }

            var denom = MathF.Sqrt(varX * varY);
            
            return denom < 1e-8f ? 0f : cov / denom;
        }

        // -------------------------------------------------------------------------
        // Private — helpers
        // -------------------------------------------------------------------------

        private static int[] CountNonFinite(IReadOnlyList<float[]> vectors, int dim)
        {
            var counts = new int[dim];
            
            foreach (var v in vectors)
            {
                for (var f = 0; f < dim && f < v.Length; f++)
                {
                    if (!float.IsFinite(v[f]))
                    {
                        counts[f]++;
                    }
                }
            }
            
            return counts;
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
}