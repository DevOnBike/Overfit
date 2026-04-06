// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Contracts;

namespace DevOnBike.Overfit.Data.Prepare
{
    /// <summary>
    /// Filters out columns with strong linear correlation (Pearson).
    /// When two features are correlated above the threshold, the layer keeps the one with higher correlation to the Target (if available) or the first one encountered.
    /// </summary>
    public sealed class CorrelationFilterLayer : IDataLayer
    {
        private readonly float _threshold;
        private readonly DropStrategy _strategy;

        private int[] _keptIndices;
        private bool _fitted;

        /// <param name="threshold">
        /// Correlation threshold (absolute value |r|). Default is 0.95.
        /// 0.98 = aggressive filtration (only nearly identical features).
        /// 0.85 = mild filtration (broader removal of multicollinearity).
        /// </param>
        /// <param name="strategy">
        /// Selection strategy for dropping a feature from a correlated pair.
        /// KeepFirst: retains the feature with the lower index (fast, deterministic).
        /// KeepHigherTargetCorrelation: retains the feature more strongly correlated with the Target 
        /// (slower — requires N additional Pearson calculations, but provides better selection).
        /// </param>
        public CorrelationFilterLayer(
            float threshold = 0.95f,
            DropStrategy strategy = DropStrategy.KeepHigherTargetCorrelation)
        {
            if (threshold is <= 0f or > 1f)
            {
                throw new ArgumentOutOfRangeException(nameof(threshold), "Correlation threshold must be in the range (0, 1].");
            }

            _threshold = threshold;
            _strategy = strategy;
        }

        public PipelineContext Process(PipelineContext context)
        {
            var rows = context.Features.GetDim(0);
            var cols = context.Features.GetDim(1);

            if (rows < 3 || cols <= 1)
            {
                return context;
            }

            if (!_fitted)
            {
                _keptIndices = Fit(context.Features, context.Targets, rows, cols);
                _fitted = true;
            }

            if (_keptIndices.Length == cols)
            {
                return context;
            }

            var filtered = ExtractColumns(context.Features, _keptIndices, rows);

            context.Features.Dispose();
            context.Features = filtered;

            return context;
        }

        private int[] Fit(FastTensor<float> features, FastTensor<float> targets, int rows, int cols)
        {
            var featureSpan = features.AsReadOnlySpan();
            var targetSpan = targets.AsReadOnlySpan();

            using var sums = new FastBuffer<double>(cols);
            using var sumsSq = new FastBuffer<double>(cols);
            using var means = new FastBuffer<double>(cols);

            var sumSpan = sums.AsSpan();
            var sumSqSpan = sumsSq.AsSpan();
            var meanSpan = means.AsSpan();

            for (var c = 0; c < cols; c++)
            {
                double sum = 0;
                double sumSq = 0;

                for (var r = 0; r < rows; r++)
                {
                    double val = featureSpan[r * cols + c];
                    sum += val;
                    sumSq += val * val;
                }

                sumSpan[c] = sum;
                sumSqSpan[c] = sumSq;
                meanSpan[c] = sum / rows;
            }

            float[] targetCorrelations = null;

            if (_strategy == DropStrategy.KeepHigherTargetCorrelation)
            {
                targetCorrelations = ComputeTargetCorrelations(featureSpan, targetSpan, sumSpan, sumSqSpan, rows, cols);
            }

            var dropped = new HashSet<int>();

            for (var i = 0; i < cols; i++)
            {
                if (dropped.Contains(i))
                {
                    continue;
                }

                for (var j = i + 1; j < cols; j++)
                {
                    if (dropped.Contains(j))
                    {
                        continue;
                    }

                    var r = CalculatePearsonFast(
                        featureSpan, i, j, rows, cols,
                        sumSpan, sumSqSpan);

                    if (MathF.Abs(r) < _threshold)
                    {
                        continue;
                    }

                    var dropIdx = ChooseColumnToDrop(i, j, targetCorrelations);
                    dropped.Add(dropIdx);
                }
            }

            var kept = new List<int>(cols - dropped.Count);

            for (var c = 0; c < cols; c++)
            {
                if (!dropped.Contains(c))
                {
                    kept.Add(c);
                }
            }

            return kept.ToArray();
        }

        /// <summary>
        /// Optimized Pearson calculation using precomputed sums.
        /// </summary>
        private float CalculatePearsonFast(
            ReadOnlySpan<float> span, int colA, int colB,
            int rows, int cols,
            Span<double> sums, Span<double> sumsSq)
        {
            double sumAB = 0;

            for (var r = 0; r < rows; r++)
            {
                sumAB += (double)span[r * cols + colA] * span[r * cols + colB];
            }

            var num = (rows * sumAB) - (sums[colA] * sums[colB]);
            var denA = (rows * sumsSq[colA]) - (sums[colA] * sums[colA]);
            var denB = (rows * sumsSq[colB]) - (sums[colB] * sums[colB]);
            var den = Math.Sqrt(denA * denB);

            if (den == 0)
            {
                return 0f;
            }

            return (float)(num / den);
        }

        /// <summary>
        /// Calculates the correlation of each feature with the Target.
        /// Features with higher correlation are considered more valuable for prediction.
        /// </summary>
        private float[] ComputeTargetCorrelations(
            ReadOnlySpan<float> featureSpan,
            ReadOnlySpan<float> targetSpan,
            Span<double> featureSums,
            Span<double> featureSumsSq,
            int rows, int cols)
        {
            var result = new float[cols];

            // Target statistics.
            double tSum = 0;
            double tSumSq = 0;

            for (var r = 0; r < rows; r++)
            {
                double val = targetSpan[r];
                tSum += val;
                tSumSq += val * val;
            }

            for (var c = 0; c < cols; c++)
            {
                double sumFT = 0;

                for (var r = 0; r < rows; r++)
                {
                    sumFT += (double)featureSpan[r * cols + c] * targetSpan[r];
                }

                var num = (rows * sumFT) - (featureSums[c] * tSum);
                var denF = (rows * featureSumsSq[c]) - (featureSums[c] * featureSums[c]);
                var denT = (rows * tSumSq) - (tSum * tSum);
                var den = Math.Sqrt(denF * denT);

                result[c] = den == 0 ? 0f : (float)(num / den);
            }

            return result;
        }

        /// <summary>
        /// Selects the column to drop from a correlated pair.
        /// </summary>
        private int ChooseColumnToDrop(int colA, int colB, float[] targetCorrelations)
        {
            if (_strategy == DropStrategy.KeepFirst)
            {
                return colB;
            }

            var corrA = MathF.Abs(targetCorrelations[colA]);
            var corrB = MathF.Abs(targetCorrelations[colB]);

            return corrA >= corrB ? colB : colA;
        }

        /// <summary>
        /// Extracts selected columns into a new contiguous FastTensor.
        /// </summary>
        private FastTensor<float> ExtractColumns(FastTensor<float> src, int[] indices, int rows)
        {
            var oldCols = src.GetDim(1);
            var newCols = indices.Length;

            var result = new FastTensor<float>(rows, newCols);
            var srcSpan = src.AsReadOnlySpan();
            var dstSpan = result.AsSpan();

            for (var r = 0; r < rows; r++)
            {
                var srcOffset = r * oldCols;
                var dstOffset = r * newCols;

                for (var c = 0; c < newCols; c++)
                {
                    dstSpan[dstOffset + c] = srcSpan[srcOffset + indices[c]];
                }
            }

            return result;
        }

        /// <summary>
        /// Resets persisted column indices, forcing a re-Fit on the next Process call.
        /// </summary>
        public void Reset()
        {
            _keptIndices = null;
            _fitted = false;
        }
    }

}