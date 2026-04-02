using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit
{
    public sealed class MnistPredictor : IDisposable
    {
        private readonly ConvLayer _conv1;
        private readonly BatchNorm1D _bn1;
        private readonly ResidualBlock _res1;
        private readonly LinearLayer _fcOut;

        public MnistPredictor(string modelPath)
        {
            // 1. Definicja architektury (Wymiary zgodne z nowym TensorMath)
            _conv1 = new ConvLayer(1, 8, 28, 28, 3);
            _bn1 = new BatchNorm1D(8 * 26 * 26); // 8 kanałów * 26x26
            _res1 = new ResidualBlock(8 * 13 * 13); // Po MaxPool: 8 * 13x13 = 1352
            _fcOut = new LinearLayer(8, 10); // Wynik z GAP to 8 wartości

            // 2. Wczytywanie wag (Używa zaktualizowanego Load opartego na Shape)
            using (var fs = new FileStream(modelPath, FileMode.Open))
            using (var br = new BinaryReader(fs))
            {
                _conv1.Load(br);
                _bn1.Load(br);
                _res1.Load(br);
                _fcOut.Load(br);
            }

            // 3. Ustawienie trybu inferencji
            _conv1.Eval();
            _bn1.Eval();
            _res1.Eval();
            _fcOut.Eval();
        }

        public int Predict(float[] pixelData)
        {
            if (pixelData.Length != 784)
                throw new ArgumentException("Obrazek musi mieć 784 piksele.");

            // Resetujemy taśmę i wyłączamy nagrywanie (Inferencja nie wymaga Autogradu)
            ComputationGraph.Active.Reset();
            ComputationGraph.Active.IsRecording = false;

            try
            {
                // Tworzymy wejście 4D: [Batch: 1, Channel: 1, Height: 28, Width: 28]
                using var inputMat = new FastTensor<float>(1, 1, 28, 28);
                pixelData.CopyTo(inputMat.AsSpan());
                using var input = new AutogradNode(inputMat, requiresGrad: false);

                // --- FORWARD PASS (Magia Reshape Zero-Copy) ---

                // 1. Splot (Operacja 4D)
                using var c1 = _conv1.Forward(input); // Wynik: [1, 8, 26, 26]

                // 2. BatchNorm1D (Wymaga 2D: [Batch, Features])
                using var c1Flat = new AutogradNode(c1.Data.Reshape(1, 8 * 26 * 26), false);
                using var bc1 = _bn1.Forward(c1Flat);

                // 3. ReLU i MaxPool2D (Wymaga powrotu do 4D)
                using var a1Flat = TensorMath.ReLU(bc1);
                using var a1 = new AutogradNode(a1Flat.Data.Reshape(1, 8, 26, 26), false);
                using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2); // Wynik: [1, 8, 13, 13]

                // 4. ResidualBlock (Opisany w LinearLayer - wymaga 2D)
                using var p1Flat = new AutogradNode(p1.Data.Reshape(1, 8 * 13 * 13), false);
                using var r1 = _res1.Forward(p1Flat); // Wynik: [1, 1352]

                // 5. Global Average Pooling (Wymaga powrotu do 4D)
                using var r1Spatial = new AutogradNode(r1.Data.Reshape(1, 8, 13, 13), false);
                using var gap = TensorMath.GlobalAveragePool2D(r1Spatial, 8, 13, 13); // Wynik: [1, 8]

                // 6. Finalna klasyfikacja
                using var output = _fcOut.Forward(gap); // Wynik: [1, 10]

                return GetArgMax(output.Data.AsSpan());
            }
            finally
            {
                ComputationGraph.Active.IsRecording = true;
            }
        }

        // Pomocnicza metoda do wyciągania predykcji ze Spana
        private int GetArgMax(ReadOnlySpan<float> span)
        {
            var maxIdx = 0;
            for (var i = 1; i < span.Length; i++)
            {
                if (span[i] > span[maxIdx]) maxIdx = i;
            }
            return maxIdx;
        }

        public void Dispose()
        {
            _conv1?.Dispose();
            _bn1?.Dispose();
            _res1?.Dispose();
            _fcOut?.Dispose();
        }
    }
}