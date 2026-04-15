// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Statistical
{
    public sealed class K8sAnomalyDetector : IDisposable
    {
        private readonly GaussianHMM _hmm;

        public const int StateNormal = 0;
        public const int StateHighLoad = 1;
        public const int StateFailure = 2;

        public int FeatureCount { get; }

        public K8sAnomalyDetector(int featureCount)
        {
            FeatureCount = featureCount;
            _hmm = new GaussianHMM(stateCount: 3, featureCount: featureCount);
        }

        public void LoadParameters(
            ReadOnlySpan<float> initialProbs,
            ReadOnlySpan<float> transitionMatrix,
            ReadOnlySpan<float> means,
            FastTensor<float>[] covariances)
        {
            _hmm.SetModel(initialProbs, transitionMatrix, means, covariances);
        }

        public float ScoreWindow(ReadOnlySpan<float> windowFeatures)
        {
            return _hmm.ScoreSequence(timeSteps: 1, windowFeatures);
        }

        public int GetCurrentRegime(ReadOnlySpan<float> windowFeatures)
        {
            // Nowy zoptymalizowany bufor na stosie
            using var statesBuffer = new PooledBuffer<int>(1);

            _hmm.DecodeViterbi(timeSteps: 1, windowFeatures, statesBuffer.Span);

            return statesBuffer.Span[0];
        }

        public void Dispose()
        {
            _hmm?.Dispose();
        }
    }
}