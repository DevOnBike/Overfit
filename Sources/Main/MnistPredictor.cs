using DevOnBike.Overfit.Layers;

namespace DevOnBike.Overfit
{
    public class MnistPredictor : IDisposable
    {
        private readonly LinearLayer _l1;
        private readonly LinearLayer _l2;

        public MnistPredictor(string modelPrefix)
        {
            // 1. Definiujemy architekturę (musi być identyczna jak przy treningu!)
            _l1 = new LinearLayer(784, 128);
            _l2 = new LinearLayer(128, 10);

            // 2. Wstrzykujemy wytrenowaną wiedzę
            _l1.Load($"{modelPrefix}_l1");
            _l2.Load($"{modelPrefix}_l2");
        }

        public int Predict(double[] pixelData)
        {
            if (pixelData.Length != 784) throw new ArgumentException("Obrazek musi mieć 784 piksele.");

            // Tworzymy tensor wejściowy (batch size = 1)
            using var inputMat = new FastMatrix<double>(1, 784);
            pixelData.CopyTo(inputMat.AsSpan());
            using var input = new Tensor(inputMat, requiresGrad: false);

            // FORWARD PASS (Tylko w przód, bez śledzenia gradientów!)
            using var h1 = _l1.Forward(input);
            using var a1 = TensorMath.ReLU(h1);
            using var output = _l2.Forward(a1);

            // Zwracamy indeks o najwyższym prawdopodobieństwie
            return output.Data.ArgMax();
        }

        public void Dispose()
        {
            _l1.Weights.Dispose();
            _l1.Biases.Dispose();
            _l2.Weights.Dispose();
            _l2.Biases.Dispose();
        }
    }
}