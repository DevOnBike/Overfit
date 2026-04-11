// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.

using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Statistical;
using DevOnBike.Overfit.Data.Prepare;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Tests
{
    public class K8sPipelineIntegrationTests : IDisposable
    {
        private readonly K8sAnomalyDetector _detector;
        private readonly SlidingWindowBuffer _buffer;
        private readonly ScalerParams _scalerParams;
        private readonly int _featureVectorSize;

        public K8sPipelineIntegrationTests()
        {
            // 1. ARRANGE: Inicjalizacja pod MetricSnapshot.FeatureCount (12)
            int windowSize = 10;
            _buffer = new SlidingWindowBuffer(windowSize: windowSize, stepSize: 1, featureCount: MetricSnapshot.FeatureCount);

            // 12 metryk * 4 statystyki = 48
            _featureVectorSize = FeatureExtractor.OutputSize(MetricSnapshot.FeatureCount);
            _detector = new K8sAnomalyDetector(_featureVectorSize);

            // Parametry skalera (Neutralne dla testu)
            _scalerParams = new ScalerParams
            {
                Medians = new float[_featureVectorSize],
                Iqrs = new float[_featureVectorSize]
            };
            Array.Fill(_scalerParams.Iqrs, 1.0f);

            SetupDeterministicHmm();
        }

        [Fact]
        public void Pipeline_ShouldClassify_NormalTraffic_AsStateNormal()
        {
            // 1. ARRANGE: Zdrowe metryki zgodnie z nowym kontraktem
            for (int i = 0; i < _buffer.WindowSize; i++)
            {
                _buffer.Add(new MetricSnapshot
                {
                    Timestamp = DateTime.UtcNow.AddSeconds(i),
                    PodName = "test-pod",
                    CpuUsageRatio = 0.1f,           //
                    MemoryWorkingSetBytes = 100 * 1024 * 1024,
                    LatencyP95Ms = 20f,             //
                    RequestsPerSecond = 100f,       //
                    ErrorRate = 0.0f                //
                });
            }

            Span<float> windowScratch = stackalloc float[_buffer.WindowFloats];
            Span<float> featuresScratch = stackalloc float[_featureVectorSize];

            // 2. ACT
            bool extracted = FeatureExtractor.TryExtract(_buffer, windowScratch, featuresScratch, out _);
            ScaleFeatures(featuresScratch);

            // Poprawione wywołanie: tylko jeden argument
            int predictedRegime = _detector.GetCurrentRegime(featuresScratch);

            // 3. ASSERT
            Assert.True(extracted);
            Assert.Equal(K8sAnomalyDetector.StateNormal, predictedRegime);
        }

        [Fact]
        public void Pipeline_ShouldClassify_AnomalousTraffic_AsStateFailure()
        {
            // 1. ARRANGE: Symulacja awarii
            for (int i = 0; i < _buffer.WindowSize; i++)
            {
                _buffer.Add(new MetricSnapshot
                {
                    Timestamp = DateTime.UtcNow.AddSeconds(i),
                    PodName = "test-pod",
                    CpuUsageRatio = 0.95f,
                    LatencyP95Ms = 5000f,
                    ErrorRate = 0.5f,
                    ThreadPoolQueueLength = 100f
                });
            }

            Span<float> windowScratch = stackalloc float[_buffer.WindowFloats];
            Span<float> featuresScratch = stackalloc float[_featureVectorSize];

            // 2. ACT
            FeatureExtractor.TryExtract(_buffer, windowScratch, featuresScratch, out _);
            ScaleFeatures(featuresScratch);

            // Poprawione wywołanie
            int predictedRegime = _detector.GetCurrentRegime(featuresScratch);

            // 3. ASSERT
            Assert.Equal(K8sAnomalyDetector.StateFailure, predictedRegime);
        }

        private void ScaleFeatures(Span<float> features)
        {
            for (int i = 0; i < features.Length; i++)
            {
                features[i] = (features[i] - _scalerParams.Medians[i]) / _scalerParams.Iqrs[i];
            }
        }

        private void SetupDeterministicHmm()
        {
            // ZMIANA 1: Dajemy równe szanse na start (0.33), 
            // aby logarytm nie wynosił -Infinity i pozwalał danym "przemówić".
            float[] initialProbs = { 0.33f, 0.33f, 0.33f };

            float[] transitionMatrix = {
        0.80f, 0.15f, 0.05f,
        0.10f, 0.80f, 0.10f,
        0.05f, 0.15f, 0.80f
    };

            var means = new float[3 * _featureVectorSize];

            // --- StateNormal ---
            means[K8sAnomalyDetector.StateNormal * _featureVectorSize + (0 * 4)] = 0.1f;  // CpuUsageRatio
            means[K8sAnomalyDetector.StateNormal * _featureVectorSize + (5 * 4)] = 20f;   // LatencyP95Ms

            // --- StateFailure ---
            // ZMIANA 2: Musimy ustawić średnie dla WSZYSTKICH cech, które testujemy jako anomalne.
            // Jeśli w teście dajesz ErrorRate 0.5 i ThreadPool 100, a tutaj zostawisz 0, 
            // to model uzna to za ogromną anomalię nawet wewnątrz stanu Failure!
            means[K8sAnomalyDetector.StateFailure * _featureVectorSize + (0 * 4)] = 0.95f; // CpuUsageRatio
            means[K8sAnomalyDetector.StateFailure * _featureVectorSize + (5 * 4)] = 5000f; // LatencyP95Ms
            means[K8sAnomalyDetector.StateFailure * _featureVectorSize + (8 * 4)] = 0.5f;  // ErrorRate (indeks 8 * 4)
            means[K8sAnomalyDetector.StateFailure * _featureVectorSize + (11 * 4)] = 100f; // ThreadPool (indeks 11 * 4)

            var covariances = new FastMatrix<float>[3];
            for (int i = 0; i < 3; i++)
            {
                covariances[i] = new FastMatrix<float>(_featureVectorSize, _featureVectorSize);
                // ZMIANA 3: Zwiększamy wariancję (np. do 100.0), aby model był mniej "sztywny" 
                // i lepiej tolerował różnice numeryczne w testach.
                for (int j = 0; j < _featureVectorSize; j++)
                {
                    covariances[i][j, j] = 100.0f;
                }
            }

            _detector.LoadParameters(initialProbs, transitionMatrix, means, covariances);

            foreach (var matrix in covariances) matrix.Dispose();
        }

        public void Dispose()
        {
            _detector?.Dispose();
            _buffer?.Dispose();
        }
    }
}