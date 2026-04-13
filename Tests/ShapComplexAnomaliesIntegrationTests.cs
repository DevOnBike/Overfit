// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Statistical;
using DevOnBike.Overfit.Core;
using Xunit;
using System;

namespace DevOnBike.Overfit.Tests
{
    public class ShapComplexAnomaliesIntegrationTests : IDisposable
    {
        private readonly K8sAnomalyDetector _detector;
        private readonly int _featureCount;
        private readonly float[] _normalizedBackground;

        public ShapComplexAnomaliesIntegrationTests()
        {
            _featureCount = MetricSnapshot.FeatureCount * 4; // 48 cech (12 metryk * 4 statystyki)
            _detector = new K8sAnomalyDetector(_featureCount);

            // 1. ARRANGE: Tło (Background) reprezentuje stan lekko podwyższonego szumu.
            _normalizedBackground = new float[_featureCount];

            // Indeks 32 to ErrorRate Mean. 
            // Ustawiamy tło na 1.0 (czyli 1 odchylenie standardowe od normy w świecie znormalizowanym).
            _normalizedBackground[32] = 1.0f;

            SetupNormalizedHmm();
        }

        [Fact]
        public void Shap_ShouldPartitionBlame_InCascadeFailure()
        {
            // 2. ARRANGE: Instancja z awarią (w stosunku do tła)
            var anomalousInstance = new float[_featureCount];
            _normalizedBackground.CopyTo(anomalousInstance, 0);

            // Wartości dobrane tak, by model widział różnicę, ale nie wpadał w nasycenie (underflow)
            anomalousInstance[0] = 2.0f;   // CpuUsage Mean rośnie
            anomalousInstance[20] = 5.0f;  // Latency Mean rośnie mocniej (główna anomalia)
            anomalousInstance[32] = 3.0f;  // ErrorRate Mean rośnie umiarkowanie

            // 3. ACT: Wyjaśniamy zjawisko przy użyciu KernelSHAP
            using var shap = new ShapKernel(
                modelFunc: _detector.ScoreWindow,
                background: _normalizedBackground,
                numSamples: 2048);

            var shapValues = new float[_featureCount];
            shap.Explain(anomalousInstance, shapValues);

            // 4. ASSERT: Analiza wyników SHAP
            var cpuShap = shapValues[0];
            var latencyShap = shapValues[20];
            var errorRateShap = shapValues[32];

            // Weryfikacja kierunku wpływu: anomalie muszą obniżać Score (wartości SHAP ujemne)
            Assert.True(cpuShap < 0f, $"CPU powinno obniżać score, a ma: {cpuShap}");
            Assert.True(latencyShap < 0f, $"Latency powinno obniżać score, a ma: {latencyShap}");
            Assert.True(errorRateShap < 0f, $"ErrorRate powinno obniżać score, a ma: {errorRateShap}");

            // Weryfikacja rangi: Latency (5.0) jest gorsze statystycznie od CPU (2.0)
            // Im bardziej ujemna wartość SHAP, tym większa "wina" za spadek prawdopodobieństwa.
            Assert.True(latencyShap < cpuShap,
                $"Latency (5.0) powinno być uznane za większą anomalię niż CPU (2.0). " +
                $"SHAP Latency: {latencyShap}, SHAP CPU: {cpuShap}");

            // Weryfikacja Aksjomatu Efektywności SHAP: suma wpływów == zmiana wyjścia modelu
            float totalShap = 0;
            for (var i = 0; i < _featureCount; i++)
            {
                totalShap += shapValues[i];
            }

            var modelDiff = _detector.ScoreWindow(anomalousInstance) - _detector.ScoreWindow(_normalizedBackground);

            // Sprawdzamy równość z tolerancją wynikającą z próbkowania (numSamples)
            Assert.Equal(modelDiff, totalShap, precision: 1);
        }

        private void SetupNormalizedHmm()
        {
            // Model deterministyczny: startuje i zostaje w stanie Normal (idx 0)
            float[] initialProbs = [1.0f, 0.0f, 0.0f];
            float[] transitionMatrix = [1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f];

            // Średnie ustawione na 0.0 dla wszystkich stanów
            var means = new float[3 * _featureCount];

            var covariances = new FastTensor<float>[3];
            for (var i = 0; i < 3; i++)
            {
                // Inicjalizacja z wyczyszczoną pamięcią (macierz diagonalna)
                covariances[i] = new FastTensor<float>(_featureCount, _featureCount, clearMemory: true);

                for (var j = 0; j < _featureCount; j++)
                {
                    // Ustawiamy wariancję na 2.0. Rozszerza to "pole widzenia" modelu Gausowskiego,
                    // pozwalając mu rozróżniać duże odchylenia bez wpadania w zero maszynowe (Underflow).
                    covariances[i].GetView()[j, j] = 2.0f;
                }
            }

            // Załadowanie parametrów do detektora
            _detector.LoadParameters(initialProbs, transitionMatrix, means, covariances);

            // Zwolnienie tymczasowych tensorów po załadowaniu (HMM tworzy własne dekompozycje Cholesky'ego)
            foreach (var t in covariances)
            {
                t.Dispose();
            }
        }

        public void Dispose()
        {
            _detector?.Dispose(); //
        }
    }
}