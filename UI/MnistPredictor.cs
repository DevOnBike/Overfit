using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;
using System.IO;

namespace DevOnBike.Overfit.UI
{
    public sealed class MnistPredictor : IDisposable
    {
        private readonly ConvLayer _conv1;
        private readonly BatchNorm1D _bn1;
        private readonly LinearLayer _fc1;
        private readonly Sequential _weightsContainer;

        public MnistPredictor(string modelPath)
        {
            // 1. Definicja architektury (MUSI BYĆ IDENTYCZNA JAK W TRENINGU)
            // Wejście 28x28 -> Conv 3x3 (8 filtrów) -> ReLU -> MaxPool 2x2 -> BN -> FC
            _conv1 = new ConvLayer(inChannels: 1, outChannels: 8, h: 28, w: 28, kSize: 3);
            _bn1 = new BatchNorm1D(1352); // 8 * 13 * 13 = 1352
            _fc1 = new LinearLayer(1352, 10);

            // 2. Sklejamy je w kontener do masowego ładowania wag
            _weightsContainer = new Sequential(_conv1, _bn1, _fc1);

            // 3. Ładowanie wag z pliku wyeksportowanego przez testy
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"Brak pliku modelu: {modelPath}");

            using (var fs = new FileStream(modelPath, FileMode.Open))
            using (var br = new BinaryReader(fs))
            {
                _weightsContainer.Load(br);
            }

            // 4. Przestawiamy całą sieć w tryb WNIOSKOWANIA (Eval)
            // Wyłącza to m.in. aktualizację statystyk w BatchNorm
            _weightsContainer.Eval();

            // 5. Inicjalizacja grafu dla wątku UI, jeśli jeszcze nie istnieje
            if (ComputationGraph.Active == null)
            {
                ComputationGraph.Active = new ComputationGraph();
            }
        }

        public int Predict(double[] pixelData)
        {
            if (pixelData == null || pixelData.Length != 784)
                throw new ArgumentException("Niepoprawne dane wejściowe. Oczekiwano 784 pikseli.");

            // KRYTYCZNE: Wyłączamy nagrywanie operacji na taśmę. 
            // Zapobiega to wyciekom pamięci i błędom ObjectDisposedException przy Dispose() węzłów
            ComputationGraph.Active.IsRecording = false;

            try
            {
                // Resetujemy licznik operacji, aby zachować Zero-Alloc
                ComputationGraph.Active.Reset();

                using var inputMat = new FastMatrix<double>(1, 784);
                pixelData.CopyTo(inputMat.AsSpan());

                // Wejście nie wymaga gradientów
                using var input = new AutogradNode(inputMat, requiresGrad: false);

                // --- MANUALNY FORWARD PASS (Zgodny z MnistTrainingTests) ---
                // Nie używamy _weightsContainer.Forward(), ponieważ ReLU i MaxPool 
                // są wywoływane jako statyczne metody TensorMath

                using var h1 = _conv1.Forward(input);
                using var a1 = TensorMath.ReLU(h1);
                using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);

                using var bnOut = _bn1.Forward(p1);
                using var output = _fc1.Forward(bnOut);

                // Zwracamy indeks o najwyższym prawdopodobieństwie
                return output.Data.ArgMax();
            }
            finally
            {
                // Przywracamy nagrywanie (bezpieczeństwo dla ewentualnych testów w tym samym procesie)
                ComputationGraph.Active.IsRecording = true;
            }
        }

        public void Dispose()
        {
            // Zwalniamy bufory SIMD wszystkich warstw
            _weightsContainer?.Dispose();
        }
    }
}