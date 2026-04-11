// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Statistical;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// Test integracyjny weryfikujący poprawność silnika SHAP.
    /// Sprawdza, czy algorytm poprawnie przypisuje "winę" za anomalię do konkretnych metryk CPU.
    /// </summary>
    public class ShapCpuIntegrationTests : IDisposable
    {
        private readonly K8sAnomalyDetector _detector;
        private readonly int _featureCount;
        private readonly float[] _background;

        public ShapCpuIntegrationTests()
        {
            // 1. ARRANGE: Inicjalizacja (12 metryk * 4 statystyki = 48 cech)
            _featureCount = MetricSnapshot.FeatureCount * 4;
            _detector = new K8sAnomalyDetector(_featureCount);

            // Tło (background) to idealny stan "zero"
            _background = new float[_featureCount];

            // Konfiguracja modelu pod test deterministyczny
            SetupHarshHmmForCpuDetection();
        }

        [Fact]
        public void Shap_ShouldIdentify_CpuUsage_AsMajorAnomalyDriver()
        {
            // 1. ARRANGE: Tworzymy instancję z drastyczną anomalią na CPU
            var anomalousInstance = new float[_featureCount];

            // Indeks 0: CpuUsageRatio Mean. 
            // Skok z 0.0 na 0.95 przy wariancji 0.01 wygeneruje ogromną karę log-likelihood.
            anomalousInstance[0] = 0.95f;

            // Uruchamiamy generyczny eksplainer KernelSHAP
            using var shap = new ShapKernel(
                modelFunc: (span) => _detector.ScoreWindow(span),
                background: _background,
                numSamples: 1024 // Zwiększona liczba próbek dla stabilności regresji
            );

            Span<float> shapValues = stackalloc float[_featureCount];

            // 2. ACT: Obliczamy wartości SHAP
            shap.Explain(anomalousInstance, shapValues);

            // 3. ASSERT: Weryfikacja matematyczna
            var cpuUsageContribution = shapValues[0];
            var unrelatedMetricContribution = shapValues[20]; // np. jakaś metryka RAM/Net

            // Przy tej konfiguracji HMM, spadek Log-Likelihood powinien wynosić ok. -45.
            // SHAP musi to "rozliczyć" przede wszystkim na cechę 0.
            Assert.True(cpuUsageContribution < -10.0f,
                $"BŁĄD: SHAP nie wykrył dominacji CPU. Wartość: {cpuUsageContribution}. " +
                "Sprawdź czy UniversalKernelShap poprawnie rozwiązuje regresję.");

            Assert.True(MathF.Abs(unrelatedMetricContribution) < 1.0f,
                $"BŁĄD: Cecha niepowiązana ma zbyt duży wpływ: {unrelatedMetricContribution}");
        }

        /// <summary>
        /// Konfiguruje model HMM tak, aby był ekstremalnie czuły na zmiany i deterministyczny.
        /// </summary>
        private void SetupHarshHmmForCpuDetection()
        {
            // Prawdopodobieństwa początkowe: zawsze startujemy w Normal
            float[] initialProbs = [1.0f, 0.0f, 0.0f];

            // Macierz przejść: wymuszamy pozostanie w stanie Normal, 
            // aby model nie "uciekł" do stanu Failure (co zamaskowałoby anomalię w SHAP)
            float[] transitionMatrix =
            [
                1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 1.0f
            ];

            var means = new float[3 * _featureCount];
            // Stan Normalny oczekuje wartości 0.0 dla wszystkich metryk
            // (czyli dokładnie tyle, ile mamy w _background)
            for (var i = 0; i < means.Length; i++) means[i] = 0.0f;

            var covariances = new FastMatrix<float>[3];
            for (var i = 0; i < 3; i++)
            {
                covariances[i] = new FastMatrix<float>(_featureCount, _featureCount);
                for (var j = 0; j < _featureCount; j++)
                {
                    // Niska wariancja (0.01) sprawia, że odchylenie o 0.95 
                    // staje się "matematyczną katastrofą" dla modelu.
                    covariances[i][j, j] = 0.01f;
                }
            }

            _detector.LoadParameters(initialProbs, transitionMatrix, means, covariances);

            foreach (var m in covariances) m.Dispose();
        }

        public void Dispose() => _detector?.Dispose();
    }
}