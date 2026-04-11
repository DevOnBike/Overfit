// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Abstractions;
using DevOnBike.Overfit.Data.Contracts;
using DevOnBike.Overfit.Statistical;

namespace DevOnBike.Overfit.Data.Prepare
{
    /// <summary>
    /// Warstwa selekcji cech oparta na wartościach SHAP.
    /// Identyfikuje najbardziej istotne cechy przy użyciu modelu FastRandomForest 
    /// i silnika UniversalKernelShap.
    /// </summary>
    public sealed class ShapSelectionLayer : IDataLayer
    {
        private readonly int _targetFeatureCount;
        private readonly float _minImportanceThreshold;
        private readonly int _numTrees;
        private readonly int _maxDepth;
        private bool _fitted;
        private int[] _keptIndices;

        /// <param name="targetFeatureCount">Docelowa liczba cech (wybiera top N). 0 = użyj progu istotności.</param>
        /// <param name="minImportanceThreshold">Minimalna średnia wartość |SHAP|, aby zachować cechę.</param>
        /// <param name="numTrees">Liczba drzew w lesie używanym do analizy.</param>
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
            var rows = context.Features.GetDim(0);
            var cols = context.Features.GetDim(1);

            if (cols == 0 || rows < 10)
            {
                return context;
            }

            // Faza Fit: Wyznaczamy istotne cechy tylko raz (na danych treningowych)
            if (!_fitted)
            {
                _keptIndices = Fit(context.Features, context.Targets);
                _fitted = true;
            }

            // Jeśli nie wybrano żadnych cech lub wszystkie są ważne, nic nie zmieniaj
            if (_keptIndices == null || _keptIndices.Length == cols)
            {
                return context;
            }

            // Faza Transform: Filtrowanie kolumn
            var filteredFeatures = ExtractColumns(context.Features, _keptIndices);
            context.Features.Dispose();

            return new PipelineContext(filteredFeatures, context.Targets);
        }

        private int[] Fit(FastTensor<float> features, FastTensor<float> targets)
        {
            var rows = features.GetDim(0);
            var cols = features.GetDim(1);

            // 1. Trenujemy model bazowy (Random Forest)
            using var forest = new FastRandomForest(_numTrees, _maxDepth);
            
            forest.Train(features, targets);

            // 2. Przygotowujemy SHAP (tło = średnie ze zbioru)
            var background = new float[cols];
            CalculateMeans(features, background);

            using var shap = new ShapKernel(modelFunc: forest.Predict, background: background, numSamples: 512);

            // 3. Analiza Globalna
            var analyzer = new GlobalShapAnalyzer(shap, cols);
            var importanceRanking = analyzer.AnalyzeImportance(features);

            // 4. Selekcja indeksów

            var selected = _targetFeatureCount > 0 ? importanceRanking.Take(_targetFeatureCount) : importanceRanking.Where(x => x.ImportanceScore >= _minImportanceThreshold);
            var kept = selected.Select(x => x.FeatureIndex).OrderBy(x => x).ToArray();
            
            return kept.Length == 0 ? null : kept;
        }

        private void CalculateMeans(FastTensor<float> tensor, float[] output)
        {
            var rows = tensor.GetDim(0);
            var cols = tensor.GetDim(1);
            var span = tensor.AsReadOnlySpan();

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
            var rows = src.GetDim(0);
            var oldCols = src.GetDim(1);
            var newCols = indices.Length;

            var result = new FastTensor<float>(rows, newCols);
            var srcSpan = src.AsReadOnlySpan();
            var dstSpan = result.AsSpan();

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