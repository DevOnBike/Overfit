// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.

using System.Linq;
using DevOnBike.Overfit.Data.Abstractions;
using DevOnBike.Overfit.Data.Contracts;
using DevOnBike.Overfit.Statistical;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Data.Prepare
{
    public sealed class ShapSelectionLayer : IDataLayer
    {
        private readonly int _targetFeatureCount;
        private readonly float _minImportanceThreshold;
        private readonly int _numTrees;
        private readonly int _maxDepth;
        private bool _fitted;
        private int[] _keptIndices;

        public ShapSelectionLayer(
            int targetFeatureCount = 0,
            float minImportanceThreshold = 0.01f,
            int numTrees = 100,
            int maxDepth = 8)
        {
            _targetFeatureCount = targetFeatureCount;
            _minImportanceThreshold = minImportanceThreshold;
            _numTrees = numTrees;
            _maxDepth = maxDepth;
        }

        public PipelineContext Process(PipelineContext context)
        {
            var rows = context.Features.GetView().GetDim(0);
            var cols = context.Features.GetView().GetDim(1);

            if (cols == 0 || rows < 10)
            {
                return context;
            }

            if (!_fitted)
            {
                _keptIndices = Fit(context.Features, context.Targets);
                _fitted = true;
            }

            if (_keptIndices == null || _keptIndices.Length == cols)
            {
                return context;
            }

            var filteredFeatures = ExtractColumns(context.Features, _keptIndices);
            context.Features.Dispose();

            return new PipelineContext(filteredFeatures, context.Targets);
        }

        private int[] Fit(FastTensor<float> features, FastTensor<float> targets)
        {
            var cols = features.GetView().GetDim(1);

            using var forest = new FastRandomForest(_numTrees, _maxDepth);
            forest.Train(features, targets);

            var background = new float[cols];
            CalculateMeans(features, background);

            using var shap = new ShapKernel(modelFunc: forest.Predict, background: background, numSamples: 512);

            var analyzer = new GlobalShapAnalyzer(shap, cols);
            var importanceRanking = analyzer.AnalyzeImportance(features);

            var selected = _targetFeatureCount > 0 ? importanceRanking.Take(_targetFeatureCount) : importanceRanking.Where(x => x.ImportanceScore >= _minImportanceThreshold);
            var kept = selected.Select(x => x.FeatureIndex).OrderBy(x => x).ToArray();

            return kept.Length == 0 ? null : kept;
        }

        private void CalculateMeans(FastTensor<float> tensor, float[] output)
        {
            var rows = tensor.GetView().GetDim(0);
            var cols = tensor.GetView().GetDim(1);
            var span = tensor.GetView().AsReadOnlySpan();

            for (var c = 0; c < cols; c++)
            {
                float sum = 0;
                for (var r = 0; r < rows; r++)
                {
                    sum += span[r * cols + c];
                }
                output[c] = sum / rows;
            }
        }

        private FastTensor<float> ExtractColumns(FastTensor<float> src, int[] indices)
        {
            var rows = src.GetView().GetDim(0);
            var oldCols = src.GetView().GetDim(1);
            var newCols = indices.Length;

            var result = new FastTensor<float>(rows, newCols, clearMemory: false);
            var srcSpan = src.GetView().AsReadOnlySpan();
            var dstSpan = result.GetView().AsSpan();

            for (var r = 0; r < rows; r++)
            {
                for (var c = 0; c < newCols; c++)
                {
                    dstSpan[r * newCols + c] = srcSpan[r * oldCols + indices[c]];
                }
            }

            return result;
        }

        public void Reset()
        {
            _fitted = false;
            _keptIndices = null;
        }
    }
}