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

        // Używamy Sequential jako "kontenera" do załadowania całego pliku naraz
        private readonly Sequential _weightsContainer;

        public MnistPredictor(string modelPath)
        {
            // 1. Definicja architektury (MUSI BYĆ IDENTYCZNA JAK W TEŚCIE!)
            _conv1 = new ConvLayer(inChannels: 1, outChannels: 8, h: 28, w: 28, kSize: 3);
            _bn1 = new BatchNorm1D(1352);
            _fc1 = new LinearLayer(1352, 10);

            // 2. Sklejamy je w kontener w DOKŁADNIE takiej samej kolejności jak w teście
            _weightsContainer = new Sequential(_conv1, _bn1, _fc1);

            // 3. Ładowanie jednym strzałem
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"Brak pliku modelu: {modelPath}");

            using var fs = new FileStream(modelPath, FileMode.Open);
            using var br = new BinaryReader(fs);
            _weightsContainer.Load(br);

            // 4. Przestawiamy całą sieć w tryb WNIOSKOWANIA (Eval)
            _weightsContainer.Eval();
        }

        public int Predict(double[] pixelData)
        {
            if (pixelData.Length != 784)
                throw new ArgumentException("Obrazek musi mieć 784 piksele.");

            using var inputMat = new FastMatrix<double>(1, 784);
            pixelData.CopyTo(inputMat.AsSpan());
            using var input = new AutogradNode(inputMat, requiresGrad: false);

            // --- FORWARD PASS (Odwzorowanie pętli z testu treningowego) ---
            using var h1 = _conv1.Forward(input);
            using var a1 = TensorMath.ReLU(h1);
            using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);

            using var bnOut = _bn1.Forward(p1);

            // Dropout w trybie wnioskowania jest pomijany, więc w ogóle go tu nie wywołujemy
            using var output = _fc1.Forward(bnOut);

            // Zwracamy indeks (0-9) z najwyższym prawdopodobieństwem (ArgMax via SIMD)
            return output.Data.ArgMax();
        }

        public void Dispose()
        {
            _weightsContainer?.Dispose();
        }
    }
}