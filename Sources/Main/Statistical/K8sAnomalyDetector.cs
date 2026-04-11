using DevOnBike.Overfit.Statistical;
using System.Buffers;

namespace DevOnBike.Overfit.Monitoring
{
    public class K8sAnomalyDetector : IDisposable
    {
        private readonly GaussianHMM _hmm;

        // Zdefiniowane z góry stany, dla jasności
        public const int StateNormal = 0;
        public const int StateHighLoad = 1;
        public const int StateFailure = 2;

        public K8sAnomalyDetector()
        {
            // 3 ukryte stany, 8 cech (np. CPU, RAM, Latency)
            _hmm = new GaussianHMM(stateCount: 3, featureCount: 8);

            // W prawdziwym projekcie pobierasz te parametry z pliku .json po treningu Baum-Welch.
            // Tutaj inicjujemy "wiedzą ekspercką" z palca (tzw. priory):

            float[] initialProbs = { 0.9f, 0.1f, 0.0f }; // Prawie zawsze startuje jako normalny

            float[] transitionMatrix = {
                // To Normal, To HighLoad, To Failure
                0.90f, 0.09f, 0.01f, // Z Normalnego
                0.10f, 0.85f, 0.05f, // Z HighLoad
                0.00f, 0.20f, 0.80f  // Z Failure
            };

            // Średnie (Means) - jakie wartości przyjmuje tych 8 cech w każdym stanie
            float[] means = new float[3 * 8];
            // Tu ładujesz swoje znormalizowane średnie...

            // Wariancje (Variances) - jak bardzo mogą wibrować metryki
            float[] variances = new float[3 * 8];
            Array.Fill(variances, 1.0f); // Zakładamy znormalizowane wariancje

            _hmm.SetModel(initialProbs, transitionMatrix, means, variances);
        }

        /// <summary>
        /// Zwraca Log-Likelihood dla okna czasowego (np. 60 sekund).
        /// Wynik bliski zeru = wszystko OK. Silnie ujemny (np. -150) = ogromna anomalia.
        /// </summary>
        public float EvaluateWindow(ReadOnlySpan<float> windowFeatures, int windowSize)
        {
            return _hmm.ScoreSequence(windowSize, windowFeatures);
        }

        /// <summary>
        /// Zgadywanie w czasie rzeczywistym: "W jakim stanie aktualnie jest Pod?"
        /// </summary>
        public int GetCurrentRegime(ReadOnlySpan<float> windowFeatures, int windowSize)
        {
            int[] statesBuffer = ArrayPool<int>.Shared.Rent(windowSize);
            try
            {
                _hmm.DecodeViterbi(windowSize, windowFeatures, statesBuffer);
                // Zwracamy stan z OSTATNIEGO kroku w oknie
                return statesBuffer[windowSize - 1];
            }
            finally
            {
                ArrayPool<int>.Shared.Return(statesBuffer);
            }
        }

        public void Dispose() => _hmm.Dispose();
    }
}