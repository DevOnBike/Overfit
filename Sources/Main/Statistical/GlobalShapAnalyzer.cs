// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Contracts;

namespace DevOnBike.Overfit.Statistical
{
    public sealed class GlobalShapAnalyzer
    {
        private readonly UniversalKernelShap _shap;
        private readonly int _featureCount;

        public GlobalShapAnalyzer(UniversalKernelShap shap, int featureCount)
        {
            _shap = shap;
            _featureCount = featureCount;
        }

        public List<FeatureImportance> AnalyzeImportance(FastTensor<float> trainingData)
        {
            int rows = trainingData.GetDim(0);
            float[] globalImportance = new float[_featureCount];
            float[] shapBuffer = new float[_featureCount];
            float[] rowBuffer = new float[_featureCount];

            for (int r = 0; r < rows; r++)
            {
                trainingData.AsReadOnlySpan().Slice(r * _featureCount, _featureCount).CopyTo(rowBuffer);
                _shap.Explain(rowBuffer, shapBuffer);

                for (int f = 0; f < _featureCount; f++)
                {
                    globalImportance[f] += MathF.Abs(shapBuffer[f]);
                }
            }

            var results = new List<FeatureImportance>();
            for (int f = 0; f < _featureCount; f++)
            {
                results.Add(new FeatureImportance { FeatureIndex = f, ImportanceScore = globalImportance[f] / rows });
            }

            return results.OrderByDescending(x => x.ImportanceScore).ToList();
        }
    }
}
