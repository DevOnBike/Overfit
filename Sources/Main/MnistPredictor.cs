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
            // 1. Definicja architektury (Wymiary zgodne z nowym TensorMath i Twoim treningiem)
            _conv1 = new ConvLayer(1, 8, 28, 28, 3);          // Output: 8x26x26
            _bn1 = new BatchNorm1D(8 * 26 * 26);             // Rozmiar 5408 - musi być po Conv
            _res1 = new ResidualBlock(8 * 13 * 13);          // Rozmiar 1352 - po MaxPool
            _fcOut = new LinearLayer(8, 10);                 // 8 wejść z GAP na 10 wyjść

            // 2. Wczytywanie wag
            if (File.Exists(modelPath))
            {
                using var fs = new FileStream(modelPath, FileMode.Open);
                using var br = new BinaryReader(fs);
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
            if (pixelData == null || pixelData.Length != 784)
                throw new ArgumentException("Niepoprawne dane wejściowe.");

            // Wyłączamy nagrywanie operacji na grafie dla samej predakcji
            ComputationGraph.Active.IsRecording = false;
            try
            {
                ComputationGraph.Active.Reset();

                // Konwersja na Tensor 4D (1 batch, 1 kanał, 28x28)
                using var inputMat = new FastTensor<float>(1, 1, 28, 28);
                pixelData.CopyTo(inputMat.AsSpan());
                using var input = new AutogradNode(inputMat, requiresGrad: false);

                // --- FORWARD PASS ---
                // 1. Conv -> BN -> ReLU
                using var h1 = _conv1.Forward(input);
                using var bn1Out = _bn1.Forward(h1);
                using var a1 = TensorMath.ReLU(bn1Out);

                // 2. MaxPool (8x26x26 -> 8x13x13)
                using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);

                // 3. Reshape -> ResidualBlock (1352 features)
                using var p1Flat = TensorMath.Reshape(p1, 1, 1352);
                using var resOut = _res1.Forward(p1Flat);

                // 4. Global Average Pool (Redukcja 8x13x13 -> 8 kanałów)
                // Musimy przywrócić kształt 4D dla operacji GAP
                using var res4D = TensorMath.Reshape(resOut, 1, 8, 13, 13);
                using var gapOut = TensorMath.GlobalAveragePool2D(res4D, 8, 13, 13);

                // 5. Linear Output (8 -> 10)
                using var output = _fcOut.Forward(gapOut); // Poprawione z _fc1 na _fcOut

                return GetArgMax(output.Data.AsSpan());
            }
            finally
            {
                ComputationGraph.Active.IsRecording = true;
            }
        }

        private int GetArgMax(ReadOnlySpan<float> span)
        {
            int maxIdx = 0;
            for (int j = 1; j < span.Length; j++)
                if (span[j] > span[maxIdx]) maxIdx = j;
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