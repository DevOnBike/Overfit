// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.

using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Statistical;
using DevOnBike.Overfit.Core;
using Xunit;
using System;

namespace DevOnBike.Overfit.Tests
{
    public class ShapCpuIntegrationTests : IDisposable
    {
        private readonly K8sAnomalyDetector _detector;
        private readonly int _featureCount;
        private readonly float[] _background;

        public ShapCpuIntegrationTests()
        {
            _featureCount = MetricSnapshot.FeatureCount * 4;
            _detector = new K8sAnomalyDetector(_featureCount);
            _background = new float[_featureCount];
            SetupHarshHmmForCpuDetection();
        }

        [Fact]
        public void Shap_ShouldCorrectlyAttributeAnomalyToCpu()
        {
            // 1. ARRANGE: Drastyczna anomalia 10 sigma
            var anomalousInstance = new float[_featureCount];
            anomalousInstance[0] = 10.0f; // CpuUsageRatio Mean
            anomalousInstance[2] = 10.0f; // CpuUsageRatio P95

            // 2. ACT: Uruchamiamy jądro SHAP (4096 próbek dla pełnej stabilności)
            using var shap = new ShapKernel(
                modelFunc: _detector.ScoreWindow,
                background: _background,
                numSamples: 4096);

            var shapValues = new float[_featureCount];
            shap.Explain(anomalousInstance, shapValues);

            // 3. ASSERT: 
            // Przy wariancji 1.0 wpływ winnej cechy to ok. -50.
            var cpuMeanShap = shapValues[0];
            var cpuP95Shap = shapValues[2];

            Assert.True(cpuMeanShap < -30f, $"CpuMean powinien być < -30, jest: {cpuMeanShap}");
            Assert.True(cpuP95Shap < -30f, $"CpuP95 powinien być < -30, jest: {cpuP95Shap}");

            // Sprawdzenie sumy (Aksjomat Efektywności)
            float totalShap = 0;
            for (var i = 0; i < _featureCount; i++)
            {
                totalShap += shapValues[i];
            }

            var expectedDiff = _detector.ScoreWindow(anomalousInstance) - _detector.ScoreWindow(_background);
            Assert.Equal(expectedDiff, totalShap, precision: 1);
        }

        private void SetupHarshHmmForCpuDetection()
        {
            float[] initialProbs = [1.0f, 0.0f, 0.0f];
            float[] transitionMatrix = [1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f];
            var means = new float[3 * _featureCount];

            var covariances = new FastTensor<float>[3];
            for (var i = 0; i < 3; i++)
            {
                covariances[i] = new FastTensor<float>(_featureCount, _featureCount, clearMemory: true);
                for (var j = 0; j < _featureCount; j++)
                {
                    covariances[i].GetView()[j, j] = 1.0f;
                }
            }

            _detector.LoadParameters(initialProbs, transitionMatrix, means, covariances);

            foreach (var t in covariances)
            {
                t.Dispose();
            }
        }

        public void Dispose() => _detector?.Dispose();
    }
}