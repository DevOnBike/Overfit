using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit
{
    public sealed class MnistPredictor : IDisposable
    {
        // Komponenty architektury "GigaBestia Lightweight"
        private readonly ConvLayer _conv1;
        private readonly BatchNorm1D _bn1;
        private readonly ResidualBlock _res1;
        private readonly LinearLayer _fcOut;

        public MnistPredictor(string modelPrefix)
        {
            // 1. Definicja architektury
            _conv1 = new ConvLayer(1, 8, 28, 28, 3);
            _bn1 = new BatchNorm1D(8 * 26 * 26);
            _res1 = new ResidualBlock(8 * 13 * 13);
            _fcOut = new LinearLayer(8, 10);

            // 2. Wczytywanie wag (korzystając z nowej, bezpiecznej metody pomocniczej)
            LoadModule(_conv1, $"{modelPrefix}_conv1.bin");
            LoadModule(_bn1, $"{modelPrefix}_bn1.bin");
            LoadModule(_res1, $"{modelPrefix}_res1.bin"); // ResidualBlock ładuje się teraz z jednego, spójnego pliku!
            LoadModule(_fcOut, $"{modelPrefix}_fc.bin");

            // 3. MAGIA ARCHITEKTURY: Ustawiamy tryb inferencji RAZ na całe życie obiektu
            _conv1.Eval();
            _bn1.Eval();
            _res1.Eval();
            _fcOut.Eval();
        }

        // Metoda pomocnicza wymuszająca korzystanie z kontraktu IModule
        private void LoadModule(IModule module, string path)
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"Brak pliku wag: {path}");

            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);
            module.Load(br);
        }

        public int Predict(double[] pixelData)
        {
            if (pixelData.Length != 784)
                throw new ArgumentException("Obrazek musi mieć 784 piksele.");

            using var inputMat = new FastMatrix<double>(1, 784);
            pixelData.CopyTo(inputMat.AsSpan());
            using var input = new AutogradNode(inputMat, requiresGrad: false);

            // --- FORWARD PASS (Inference Mode) ---
            // Brak isTraining: false! Warstwy już wiedzą, że są w trybie Eval().
            using var c1 = _conv1.Forward(input);
            using var bc1 = _bn1.Forward(c1);
            using var a1 = TensorMath.ReLU(bc1);
            using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);

            using var r1 = _res1.Forward(p1);

            using var gap = TensorMath.GlobalAveragePool2D(r1, 8, 13, 13);
            using var output = _fcOut.Forward(gap);

            return output.Data.ArgMax();
        }

        public void Dispose()
        {
            // IModule implementuje IDisposable, więc zwalnianie jest teraz eleganckie i bezpieczne
            _conv1?.Dispose();
            _bn1?.Dispose();
            _res1?.Dispose();
            _fcOut?.Dispose();
        }
    }
}