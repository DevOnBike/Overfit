// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Statistical;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Tests
{
    public class K8sPipelineIntegrationTests : IDisposable
    {
        private readonly K8sAnomalyDetector _detector;
        private readonly SlidingWindowBuffer _buffer;
        private readonly int _featureVectorSize;
        private readonly int _windowSize = 10;

        public K8sPipelineIntegrationTests()
        {
            _buffer = new SlidingWindowBuffer(windowSize: _windowSize, featureCount: MetricSnapshot.FeatureCount, stepSize: 1);

            _featureVectorSize = FeatureExtractor.OutputSize(MetricSnapshot.FeatureCount);
            _detector = new K8sAnomalyDetector(_featureVectorSize);

            var initialProbs = new float[3] { 0.8f, 0.15f, 0.05f };
            var transitionMatrix = new float[9]
            {
                0.9f, 0.08f, 0.02f,
                0.1f, 0.8f,  0.1f,
                0.05f, 0.1f, 0.85f
            };

            var means = new float[3 * _featureVectorSize];
            // Ustawiamy progi dla stanu awarii (StateFailure)
            // Zakładamy, że FeatureExtractor produkuje 4 statystyki na metrykę (Mean, Min, Max, P95)
            means[K8sAnomalyDetector.StateFailure * _featureVectorSize + 0 * 4] = 0.95f; // CPU Mean
            means[K8sAnomalyDetector.StateFailure * _featureVectorSize + 5 * 4] = 5000f; // Latency Mean
            means[K8sAnomalyDetector.StateFailure * _featureVectorSize + 8 * 4] = 0.5f;  // ErrorRate Mean
            means[K8sAnomalyDetector.StateFailure * _featureVectorSize + 11 * 4] = 100f; // ThreadPool Mean

            var covariances = new FastTensor<float>[3];
            for (var i = 0; i < 3; i++)
            {
                covariances[i] = new FastTensor<float>(_featureVectorSize, _featureVectorSize, clearMemory: true);
                for (var j = 0; j < _featureVectorSize; j++)
                {
                    covariances[i].GetView()[j, j] = 100.0f;
                }
            }

            _detector.LoadParameters(initialProbs, transitionMatrix, means, covariances);

            foreach (var cov in covariances)
            {
                cov.Dispose();
            }
        }

        [Fact]
        public void NormalTraffic_ShouldRemainInNormalState()
        {
            var dt = DateTime.UtcNow;
            for (var i = 0; i < _windowSize; i++)
            {
                var metric = CreateMockMetric(cpu: 0.1f, memory: 50, latency: 20f, errRate: 0.01f, threadPool: 5);
                _buffer.Add(metric, dt.AddSeconds(i * 15));
            }

            // 1. Pobieramy surowe okno
            Span<float> rawWindow = stackalloc float[_windowSize * MetricSnapshot.FeatureCount];
            Assert.True(_buffer.TryGetWindow(rawWindow, out _));

            // 2. Ekstrahujemy cechy - POPRAWIONA KOLEJNOŚĆ ARGUMENTÓW
            Span<float> windowFeatures = stackalloc float[_featureVectorSize];
            FeatureExtractor.Extract(rawWindow, _windowSize, MetricSnapshot.FeatureCount, windowFeatures);

            var state = _detector.GetCurrentRegime(windowFeatures);
            Assert.Equal(K8sAnomalyDetector.StateNormal, state);
        }

        [Fact]
        public void HighCpuAndLatency_ShouldTriggerFailureState()
        {
            var dt = DateTime.UtcNow;
            for (var i = 0; i < _windowSize; i++)
            {
                var metric = CreateMockMetric(cpu: 0.99f, memory: 90, latency: 6000f, errRate: 0.6f, threadPool: 120);
                _buffer.Add(metric, dt.AddSeconds(i * 15));
            }

            // 1. Pobieramy surowe okno
            Span<float> rawWindow = stackalloc float[_windowSize * MetricSnapshot.FeatureCount];
            Assert.True(_buffer.TryGetWindow(rawWindow, out _));

            // 2. Ekstrahujemy cechy - POPRAWIONA KOLEJNOŚĆ ARGUMENTÓW
            Span<float> windowFeatures = stackalloc float[_featureVectorSize];
            FeatureExtractor.Extract(rawWindow, _windowSize, MetricSnapshot.FeatureCount, windowFeatures);

            var state = _detector.GetCurrentRegime(windowFeatures);
            Assert.Equal(K8sAnomalyDetector.StateFailure, state);
        }

        private float[] CreateMockMetric(float cpu, float memory, float latency, float errRate, float threadPool)
        {
            var m = new float[MetricSnapshot.FeatureCount];
            m[0] = cpu;
            m[1] = memory;
            m[5] = latency;
            m[8] = errRate;
            m[11] = threadPool;
            return m;
        }

        public void Dispose()
        {
            _buffer?.Dispose();
            _detector?.Dispose();
        }
    }
}