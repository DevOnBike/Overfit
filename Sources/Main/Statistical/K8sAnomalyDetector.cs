// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Statistical;

namespace DevOnBike.Overfit.Monitoring
{
    public class K8sAnomalyDetector : IDisposable
    {
        private readonly GaussianHMM _hmm;

        public const int StateNormal = 0;
        public const int StateHighLoad = 1;
        public const int StateFailure = 2;

        public K8sAnomalyDetector(int featureCount)
        {
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
        /// Zwraca Log-Likelihood dla całego okna. Jeśli wartość jest nienaturalnie niska,
        /// system doświadcza anomalii (np. zachowanie wykraczające poza wszystkie 3 wyuczone stany).
        /// </summary>
        public float ScoreWindow(ReadOnlySpan<float> windowFeatures, int windowSize)
        {
            return _hmm.ScoreSequence(windowSize, windowFeatures);
        }

        /// <summary>
        /// Odpowiada na pytanie: "W jakim aktualnie reżimie znajduje się usługa?"
        /// (Na podstawie ostatnich 'windowSize' obserwacji).
        /// </summary>
        public int GetCurrentRegime(ReadOnlySpan<float> windowFeatures, int windowSize)
        {
            int[] statesBuffer = ArrayPool<int>.Shared.Rent(windowSize);
            try
            {
                _hmm.DecodeViterbi(windowSize, windowFeatures, statesBuffer);
                // Wynik z ostatniego kroku czasowego
                return statesBuffer[windowSize - 1];
            }
            finally
            {
                ArrayPool<int>.Shared.Return(statesBuffer);
            }
        }

        public void Dispose()
        {
            _hmm?.Dispose();
        }
    }
}