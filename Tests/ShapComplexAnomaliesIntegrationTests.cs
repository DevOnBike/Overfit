// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.

using System;
using Xunit;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Statistical;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Tests
{
    public class ShapComplexAnomaliesIntegrationTests : IDisposable
    {
        private readonly K8sAnomalyDetector _detector;
        private readonly int _featureCount;
        private readonly float[] _normalizedBackground;

        public ShapComplexAnomaliesIntegrationTests()
        {
            _featureCount = MetricSnapshot.FeatureCount * 4; // 48 cech
            _detector = new K8sAnomalyDetector(_featureCount);

            // 1. ARRANGE: Tło (Background) reprezentuje stan lekko podwyższonego szumu.
            // Ustawiamy wszystko na 0.0 (idealnie), poza ErrorRate.
            _normalizedBackground = new float[_featureCount];

            // Indeks 32 to ErrorRate Mean. 
            // Ustawiamy tło na 1.0 (czyli 1 odchylenie standardowe od normy).
            _normalizedBackground[32] = 1.0f;

            SetupNormalizedHmm();
        }

        [Fact]
        public void Shap_ShouldPartitionBlame_InCascadeFailure()
        {
            // 2. ARRANGE: Instancja z awarią (wartości znormalizowane)
            var anomalousInstance = new float[_featureCount];

            // CPU i Latencja są bardzo wysokie (odchylenie o 5 sigma)
            anomalousInstance[0] = 5.0f;  // CpuUsageRatio Mean
            anomalousInstance[20] = 5.0f; // LatencyP95Ms Mean

            // KLUCZ: ErrorRate w instancji jest IDEALNY (0.0).
            // Ponieważ background wynosił 1.0, zmiana na 0.0 jest "poprawą",
            // co SHAP musi wykazać jako wartość dodatnią.
            anomalousInstance[32] = 0.0f;

            using var shap = new UniversalKernelShap(
                modelFunc: (span) => _detector.ScoreWindow(span),
                background: _normalizedBackground,
                numSamples: 1024
            );

            Span<float> shapValues = stackalloc float[_featureCount];

            // 3. ACT
            shap.Explain(anomalousInstance, shapValues);

            // 4. ASSERT
            float cpuShap = shapValues[0];
            float latencyShap = shapValues[20];
            float errorRateShap = shapValues[32];

            // Winowajcy muszą mieć ujemny wpływ (pogarszają wynik względem tła)
            Assert.True(cpuShap < -1.0f, $"CPU powinno pogarszać wynik. Jest: {cpuShap}");
            Assert.True(latencyShap < -1.0f, $"Latency powinno pogarszać wynik. Jest: {latencyShap}");

            // Sukces: ErrorRate jest bliżej średniej (0.0) niż tło (1.0).
            // SHAP musi być dodatni.
            Assert.True(errorRateShap > 0.05f, $"ErrorRate powinien pomagać (>0), a ma: {errorRateShap}");

            // Weryfikacja Aksjomatu Efektywności
            float totalShap = 0;
            for (int i = 0; i < _featureCount; i++) totalShap += shapValues[i];

            float modelDiff = _detector.ScoreWindow(anomalousInstance) - _detector.ScoreWindow(_normalizedBackground);
            Assert.Equal(modelDiff, totalShap, precision: 1);
        }

        private void SetupNormalizedHmm()
        {
            float[] initialProbs = { 1.0f, 0.0f, 0.0f };
            float[] transitionMatrix = { 1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f };

            // W modelu znormalizowanym średnie zawsze wynoszą 0.0
            var means = new float[3 * _featureCount];

            var covariances = new FastMatrix<float>[3];
            for (int i = 0; i < 3; i++)
            {
                covariances[i] = new FastMatrix<float>(_featureCount, _featureCount);
                // Wariancja 1.0 dla wszystkich cech (znormalizowane dane)
                for (int j = 0; j < _featureCount; j++) covariances[i][j, j] = 1.0f;
            }

            _detector.LoadParameters(initialProbs, transitionMatrix, means, covariances);
            foreach (var m in covariances) m.Dispose();
        }

        public void Dispose() => _detector?.Dispose();
    }
}