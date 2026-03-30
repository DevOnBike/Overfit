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
            // 1. Definicja architektury (identyczna jak w teście!)
            _conv1 = new ConvLayer(1, 8, 28, 28, 3);
            _bn1 = new BatchNorm1D(8 * 26 * 26);
            _res1 = new ResidualBlock(8 * 13 * 13); 
            _fcOut = new LinearLayer(8, 10); 

            // 2. Wczytywanie wag z plików .bin
            _conv1.Load($"{modelPrefix}_conv1.bin");
            _bn1.Load($"{modelPrefix}_bn1.bin");
            _res1.Load($"{modelPrefix}_res1"); // ResidualBlock sam dba o sufiksy
            _fcOut.Load($"{modelPrefix}_fc.bin");
        }

        public int Predict(double[] pixelData)
        {
            if (pixelData.Length != 784) 
                throw new ArgumentException("Obrazek musi mieć 784 piksele.");

            // Przygotowanie wejścia (Batch Size = 1)
            using var inputMat = new FastMatrix<double>(1, 784);
            pixelData.CopyTo(inputMat.AsSpan());
            using var input = new AutogradNode(inputMat, requiresGrad: false);

            // --- FORWARD PASS (Inference Mode) ---
            // Ustawiamy isTraining: false dla BatchNorm i ResNet!
            using var c1 = _conv1.Forward(input);
            using var bc1 = _bn1.Forward(c1, isTraining: false);
            using var a1 = TensorMath.ReLU(bc1);
            using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);
            
            using var r1 = _res1.Forward(p1, isTraining: false);
            
            // Global Average Pooling redukuje wymiary do 8 wartości
            using var gap = TensorMath.GlobalAveragePool2D(r1, 8, 13, 13);
            using var output = _fcOut.Forward(gap);

            // Zwracamy indeks najsilniejszej odpowiedzi
            return output.Data.ArgMax();
        }

        public void Dispose()
        {
            // Pamiętaj o zwolnieniu zasobów wszystkich warstw
            _conv1.Kernels.Dispose();
            _bn1.Parameters().ToList().ForEach(p => p.Dispose());
            _res1.Parameters().ToList().ForEach(p => p.Dispose());
            _fcOut.Weights.Dispose();
            _fcOut.Biases.Dispose();
        }
    }
}