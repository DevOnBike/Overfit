using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.Tests
{
    public class GradientChecksTests
    {
        [Fact]
        public void BatchNorm1D_GradientCheck_ShouldPass()
        {
            // 1. Ustawienie środowiska
            int batchSize = 4;
            int features = 5;

            // Inicjujemy warstwę z losowymi wagami
            var bn = new BatchNorm1D(features);
            bn.Gamma.DataView.AsSpan().Fill(1.2f);
            bn.Beta.DataView.AsSpan().Fill(0.3f);
            bn.Train(); // Musimy być w trybie treningu!

            // Inicjujemy losowe wejście (X) i cel (Y)
            var inputData = new FastTensor<float>(batchSize, features).Randomize(1.0f);
            var targetData = new FastTensor<float>(batchSize, features).Randomize(1.0f);

            var inputNode = new AutogradNode(inputData, requiresGrad: true);
            var targetNode = new AutogradNode(targetData, requiresGrad: false);

            // 2. Definiujemy funkcję wykonującą Forward Pass + Obliczenie Loss
            // Ważne: Ta funkcja będzie wywoływana setki razy przez GradientChecker.
            AutogradNode ForwardAndLoss(ComputationGraph g)
            {
                // Wykonujemy propagację w przód
                var bnOutput = bn.Forward(g, inputNode);

                // Liczymy MSE Loss między wyjściem BatchNormu a celem
                var loss = TensorMath.MSELoss(g, bnOutput, targetNode);
                return loss;
            }

            // 3. Uruchamiamy weryfikację
            // Przekazujemy moduł (żeby narzędzie wiedziało, jakie wagi sprawdzać)
            GradientChecker.Verify(bn, ForwardAndLoss, epsilon: 1e-3f, tolerance: 1e-2f);

            // Jeśli wykonanie dojdzie tutaj bez wyjątku - Twój BatchNorm1D jest matematycznie perfekcyjny!
        }
    }
}