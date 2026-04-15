// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Data.Abstractions;
using DevOnBike.Overfit.Data.Contracts;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Data.Prepare
{
    public sealed class BorutaSelectionLayer : IDataLayer
    {
        private readonly float _confirmationRatio;
        private readonly int _maxDepth;
        private readonly int _numIterations;
        private readonly int _numTrees;

        public BorutaSelectionLayer(
            int iterations = 20,
            int numTrees = 100,
            int maxDepth = 8,
            float confirmationRatio = 0.5f)
        {
            if (iterations < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(iterations), "Iterations must be >= 1.");
            }
            if (confirmationRatio is <= 0f or >= 1f)
            {
                throw new ArgumentOutOfRangeException(nameof(confirmationRatio), "Confirmation ratio must be in the range (0, 1).");
            }

            _numIterations = iterations; _numTrees = numTrees; _maxDepth = maxDepth; _confirmationRatio = confirmationRatio;
        }

        public PipelineContext Process(PipelineContext context)
        {
            var rows = context.Features.GetView().GetDim(0);
            var cols = context.Features.GetView().GetDim(1);

            if (cols == 0 || rows == 0)
            {
                return context;
            }

            using var hitCounts = new PooledBuffer<int>(cols, clearMemory: true);

            for (var iter = 0; iter < _numIterations; iter++)
            {
                using var extendedFeatures = CreateShadowDataset(context.Features);

                var forest = new FastRandomForest(_numTrees, _maxDepth);
                var importance = forest.TrainAndGetImportance(extendedFeatures, context.Targets);
                var shadowMax = float.MinValue;

                for (var i = cols; i < cols * 2; i++)
                {
                    if (importance[i] > shadowMax)
                    {
                        shadowMax = importance[i];
                    }
                }

                for (var i = 0; i < cols; i++)
                {
                    if (importance[i] > shadowMax)
                    {
                        hitCounts.Span[i]++;
                    }
                }
            }

            var threshold = (int)(_numIterations * _confirmationRatio);
            var keptIndices = new List<int>(cols);
            var hitSpan = hitCounts.Span;

            for (var i = 0; i < cols; i++)
            {
                if (hitSpan[i] > threshold)
                {
                    keptIndices.Add(i);
                }
            }

            if (keptIndices.Count == 0 || keptIndices.Count == cols)
            {
                return context;
            }

            var filteredFeatures = ExtractSelectedColumns(context.Features, keptIndices);

            context.Features.Dispose();

            return new PipelineContext(filteredFeatures, context.Targets);
        }

        private FastTensor<float> CreateShadowDataset(FastTensor<float> original)
        {
            var rows = original.GetView().GetDim(0);
            var cols = original.GetView().GetDim(1);
            var extendedCols = cols * 2;
            var extended = new FastTensor<float>(rows, extendedCols, clearMemory: false);

            var srcSpan = original.GetView().AsReadOnlySpan();
            var dstSpan = extended.GetView().AsSpan();

            for (var r = 0; r < rows; r++)
            {
                srcSpan.Slice(r * cols, cols).CopyTo(dstSpan.Slice(r * extendedCols, cols));
            }

            using var shadowBuffer = new PooledBuffer<float>(rows);

            for (var c = 0; c < cols; c++)
            {
                var shadowSpan = shadowBuffer.Span;

                for (var r = 0; r < rows; r++)
                {
                    shadowSpan[r] = srcSpan[r * cols + c];
                }

                for (var i = rows - 1; i > 0; i--)
                {
                    var j = Random.Shared.Next(i + 1);
                    (shadowSpan[i], shadowSpan[j]) = (shadowSpan[j], shadowSpan[i]);
                }

                for (var r = 0; r < rows; r++)
                {
                    dstSpan[r * extendedCols + cols + c] = shadowSpan[r];
                }
            }

            return extended;
        }

        private FastTensor<float> ExtractSelectedColumns(FastTensor<float> src, List<int> indices)
        {
            var rows = src.GetView().GetDim(0);
            var oldCols = src.GetView().GetDim(1);
            var newCols = indices.Count;

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
    }
}