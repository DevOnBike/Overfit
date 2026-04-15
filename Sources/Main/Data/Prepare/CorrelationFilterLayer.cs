// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.

using DevOnBike.Overfit.Data.Abstractions;
using DevOnBike.Overfit.Data.Contracts;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Data.Prepare
{
    public sealed class CorrelationFilterLayer : IDataLayer
    {
        private readonly DropStrategy _strategy;
        private readonly float _threshold;
        private bool _fitted;
        private int[] _keptIndices;

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
            var rows = context.Features.GetView().GetDim(0);
            var cols = context.Features.GetView().GetDim(1);

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
            var featureSpan = features.GetView().AsReadOnlySpan();
            var targetSpan = targets.GetView().AsReadOnlySpan();

            using var sums = new PooledBuffer<double>(cols, clearMemory: true);
            using var sumsSq = new PooledBuffer<double>(cols, clearMemory: true);
            using var means = new PooledBuffer<double>(cols, clearMemory: true);

            var sumSpan = sums.Span;
            var sumSqSpan = sumsSq.Span;
            var meanSpan = means.Span;

            for (var c = 0; c < cols; c++)
            {
                double sum = 0; double sumSq = 0;
                for (var r = 0; r < rows; r++)
                {
                    double val = featureSpan[r * cols + c];
                    sum += val; sumSq += val * val;
                }
                sumSpan[c] = sum; sumSqSpan[c] = sumSq; meanSpan[c] = sum / rows;
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

                    var r = CalculatePearsonFast(featureSpan, i, j, rows, cols, sumSpan, sumSqSpan);

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

        private float CalculatePearsonFast(ReadOnlySpan<float> span, int colA, int colB, int rows, int cols, Span<double> sums, Span<double> sumsSq)
        {
            double sumAB = 0;
            for (var r = 0; r < rows; r++)
            {
                sumAB += (double)span[r * cols + colA] * span[r * cols + colB];
            }

            var num = rows * sumAB - sums[colA] * sums[colB];
            var denA = rows * sumsSq[colA] - sums[colA] * sums[colA];
            var denB = rows * sumsSq[colB] - sums[colB] * sums[colB];
            var den = Math.Sqrt(denA * denB);

            if (den == 0)
            {
                return 0f;
            }

            return (float)(num / den);
        }

        private float[] ComputeTargetCorrelations(ReadOnlySpan<float> featureSpan, ReadOnlySpan<float> targetSpan, Span<double> featureSums, Span<double> featureSumsSq, int rows, int cols)
        {
            var result = new float[cols];

            double tSum = 0; double tSumSq = 0;
            for (var r = 0; r < rows; r++)
            {
                double val = targetSpan[r];
                tSum += val; tSumSq += val * val;
            }

            for (var c = 0; c < cols; c++)
            {
                double sumFT = 0;
                for (var r = 0; r < rows; r++)
                {
                    sumFT += (double)featureSpan[r * cols + c] * targetSpan[r];
                }

                var num = rows * sumFT - featureSums[c] * tSum;
                var denF = rows * featureSumsSq[c] - featureSums[c] * featureSums[c];
                var denT = rows * tSumSq - tSum * tSum;
                var den = Math.Sqrt(denF * denT);

                result[c] = den == 0 ? 0f : (float)(num / den);
            }

            return result;
        }

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

        private FastTensor<float> ExtractColumns(FastTensor<float> src, int[] indices, int rows)
        {
            var oldCols = src.GetView().GetDim(1);
            var newCols = indices.Length;

            var result = new FastTensor<float>(rows, newCols, clearMemory: false);
            var srcSpan = src.GetView().AsReadOnlySpan();
            var dstSpan = result.GetView().AsSpan();

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

        public void Reset()
        {
            _keptIndices = null;
            _fitted = false;
        }
    }
}