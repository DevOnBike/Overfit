// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Statistical
{
    public class K8sAnomalyDetector : IDisposable
    {
        private readonly GaussianHMM _hmm;

        public const int StateNormal = 0;
        public const int StateHighLoad = 1;
        public const int StateFailure = 2;

        public int FeatureCount { get; }

        public K8sAnomalyDetector(int featureCount)
        {
            FeatureCount = featureCount;
            // Inicjalizacja HMM: 3 ukryte stany, liczba cech zgodna z FeatureExtractor (domyślnie 48)
            _hmm = new GaussianHMM(stateCount: 3, featureCount: featureCount);
        }

        /// <summary>
        /// Inicjuje model z wytrenowanymi parametrami.
        /// </summary>
        public void LoadParameters(
            ReadOnlySpan<float> initialProbs,
            ReadOnlySpan<float> transitionMatrix,
            ReadOnlySpan<float> means,
            FastMatrix<float>[] covariances)
        {
            _hmm.SetModel(initialProbs, transitionMatrix, means, covariances);
        }

        /// <summary>
        /// Zwraca Log-Likelihood dla zagregowanego okna statystyk.
        /// Traktuje wejście jako pojedynczą obserwację (T=1).
        /// </summary>
        public float ScoreWindow(ReadOnlySpan<float> windowFeatures)
        {
            // Przekazujemy timeSteps: 1, ponieważ windowFeatures to jeden zagregowany wektor
            return _hmm.ScoreSequence(timeSteps: 1, windowFeatures);
        }

        /// <summary>
        /// Zgadywanie w czasie rzeczywistym: "W jakim stanie aktualnie jest Pod?"
        /// Wykorzystuje algorytm Viterbiego dla pojedynczej obserwacji (T=1).
        /// </summary>
        public int GetCurrentRegime(ReadOnlySpan<float> windowFeatures)
        {
            // Dla pojedynczego wektora cech wystarczy bufor o rozmiarze 1
            var statesBuffer = ArrayPool<int>.Shared.Rent(1);
            try
            {
                // Przekazujemy timeSteps: 1. 
                // windowFeatures musi mieć długość dokładnie równą FeatureCount.
                _hmm.DecodeViterbi(timeSteps: 1, windowFeatures, statesBuffer);

                return statesBuffer[0];
            }
            finally
            {
                ArrayPool<int>.Shared.Return(statesBuffer);
            }
        }

        public void Dispose() => _hmm?.Dispose();
    }
}