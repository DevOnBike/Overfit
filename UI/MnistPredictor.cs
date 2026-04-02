using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;
using System.IO;
using System;

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
            // 1. Definicja architektury (Musi być identyczna jak w MnistTrainingTests)
            // Wejście [1, 1, 28, 28] -> Conv 3x3 (8 filtrów) -> [1, 8, 26, 26]
            _conv1 = new ConvLayer(inChannels: 1, outChannels: 8, h: 28, w: 28, kSize: 3);

            // Po MaxPool 2x2 wynik to [1, 8, 13, 13], czyli 1352 cechy
            _bn1 = new BatchNorm1D(1352);
            _fc1 = new LinearLayer(1352, 10);

            // 2. Kontener do ładowania wag
            _weightsContainer = new Sequential(_conv1, _bn1, _fc1);

            // 3. Ładowanie wag z pliku
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"Brak pliku modelu: {modelPath}");

            using (var fs = new FileStream(modelPath, FileMode.Open))
            using (var br = new BinaryReader(fs))
            {
                _weightsContainer.Load(br);
            }

            // 4. Tryb inferencji (Eval) - wyłącza Dropout i blokuje aktualizację średnich w BatchNorm
            _weightsContainer.Eval();

            // 5. Zapewnienie aktywnej Taśmy dla wątku UI
            if (ComputationGraph.Active == null)
            {
                ComputationGraph.Active = new ComputationGraph();
            }
        }

        public int Predict(float[] pixelData)
        {
            if (pixelData == null || pixelData.Length != 784)
                throw new ArgumentException("Niepoprawne dane wejściowe. Oczekiwano 784 pikseli.");

            // Wyłączamy nagrywanie operacji (Inference nie wymaga grafu)
            ComputationGraph.Active.Reset();
            ComputationGraph.Active.IsRecording = false;

            try
            {
                // Tworzymy wejście 4D: [Batch: 1, Channel: 1, H: 28, W: 28]
                using var inputMat = new FastTensor<float>(1, 1, 28, 28);
                pixelData.CopyTo(inputMat.AsSpan());

                using var input = new AutogradNode(inputMat, requiresGrad: false);

                // --- PRZEJŚCIE W PRZÓD (FORWARD PASS) ---

                // 1. Warstwa konwolucyjna
                using var h1 = _conv1.Forward(input);
                using var a1 = TensorMath.ReLU(h1);

                // 2. Pooling
                using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);

                // 3. KRYTYCZNY KROK: Spłaszczenie do 2D dla BatchNorm i FC
                // Wykorzystujemy operację Reshape (Zero-Copy)
                using var p1Flat = new AutogradNode(p1.Data.Reshape(1, 1352), false);

                // 4. Normalizacja i Klasyfikacja
                using var bnOut = _bn1.Forward(p1Flat);
                using var output = _fc1.Forward(bnOut);

                // 5. Zwracamy wynik
                return GetArgMax(output.Data.AsSpan());
            }
            finally
            {
                ComputationGraph.Active.IsRecording = true;
            }
        }

        // Pomocniczy helper zastępujący ArgMax z FastMatrix
        private int GetArgMax(ReadOnlySpan<float> span)
        {
            int maxIdx = 0;
            for (int i = 1; i < span.Length; i++)
            {
                if (span[i] > span[maxIdx]) maxIdx = i;
            }
            return maxIdx;
        }

        public void Dispose()
        {
            _weightsContainer?.Dispose();
        }
    }
}