// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using DevOnBike.Overfit.Data.Contracts;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Statistical
{
    public sealed class GlobalShapAnalyzer
    {
        private readonly ShapKernel _shap;
        private readonly int _featureCount;

        public GlobalShapAnalyzer(ShapKernel shap, int featureCount)
        {
            _shap = shap;
            _featureCount = featureCount;
        }

        public List<FeatureImportance> AnalyzeImportance(FastTensor<float> trainingData)
        {
            // ZMIANA: Pobieramy bezalokacyjny widok na dane
            var view = trainingData.GetView();
            var rows = view.GetDim(0);
            var span = view.AsReadOnlySpan();

            var globalImportance = new float[_featureCount];
            var shapBuffer = new float[_featureCount];
            var rowBuffer = new float[_featureCount];

            for (var r = 0; r < rows; r++)
            {
                // ZMIANA: Używamy span z widoku
                span.Slice(r * _featureCount, _featureCount).CopyTo(rowBuffer);
                _shap.Explain(rowBuffer, shapBuffer);

                for (var f = 0; f < _featureCount; f++)
                {
                    globalImportance[f] += MathF.Abs(shapBuffer[f]);
                }
            }

            var results = new List<FeatureImportance>();
            for (var f = 0; f < _featureCount; f++)
            {
                results.Add(new FeatureImportance
                {
                    FeatureIndex = f,
                    ImportanceScore = globalImportance[f] / rows
                });
            }

            return results.OrderByDescending(x => x.ImportanceScore).ToList();
        }
    }
}