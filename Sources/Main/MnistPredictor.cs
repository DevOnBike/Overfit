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
            // 1. Definicja identycznej architektury jak w treningu
            _conv1 = new ConvLayer(1, 8, 28, 28, 3);
            _bn1 = new BatchNorm1D(8 * 26 * 26);
            _res1 = new ResidualBlock(8 * 13 * 13);
            _fcOut = new LinearLayer(8, 10);

            // 2. Wczytywanie wag z jednego spójnego pliku .bin
            using (var fs = new FileStream(modelPath, FileMode.Open))
            using (var br = new BinaryReader(fs))
            {
                _conv1.Load(br);
                _bn1.Load(br);
                _res1.Load(br);
                _fcOut.Load(br);
            }

            // 3. Ustawienie trybu inferencji (wyłącza Dropout i statystyki BatchNorm)
            _conv1.Eval();
            _bn1.Eval();
            _res1.Eval();
            _fcOut.Eval();
        }

        public int Predict(float[] pixelData)
        {
            if (pixelData.Length != 784)
                throw new ArgumentException("Obrazek musi mieć 784 piksele.");

            // Resetujemy taśmę i wyłączamy nagrywanie operacji
            ComputationGraph.Active.Reset();
            ComputationGraph.Active.IsRecording = false;

            try
            {
                using var inputMat = new FastMatrix<float>(1, 784);
                pixelData.CopyTo(inputMat.AsSpan());
                using var input = new AutogradNode(inputMat, requiresGrad: false);

                // --- FORWARD PASS ---
                using var c1 = _conv1.Forward(input);
                using var bc1 = _bn1.Forward(c1);
                using var a1 = TensorMath.ReLU(bc1);
                using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);

                using var r1 = _res1.Forward(p1);

                // Używamy GAP, aby zredukować mapy cech do 8 wartości
                using var gap = TensorMath.GlobalAveragePool2D(r1, 8, 13, 13);
                using var output = _fcOut.Forward(gap);

                return output.Data.ArgMax();
            }
            finally
            {
                // Przywracamy nagrywanie na wypadek, gdyby ten wątek robił też trening
                ComputationGraph.Active.IsRecording = true;
            }
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