using DevOnBike.Overfit.Layers;
using DevOnBike.Overfit.Optimizers;

namespace DevOnBike.Overfit.Tests
{
    public class EndToEndTrainingTests
    {
        [Fact]
        public void NeuralNetwork_TrainsOnXORProblem_AndConvergesToCorrectPredictions()
        {
            // ==========================================
            // ARRANGE
            // ==========================================
            
            // 1. Zbiór danych XOR
            using var xData = new FastMatrix<double>(4, 2);
            xData[0,0] = 0; xData[0,1] = 0;
            xData[1,0] = 0; xData[1,1] = 1;
            xData[2,0] = 1; xData[2,1] = 0;
            xData[3,0] = 1; xData[3,1] = 1;

            using var yData = new FastMatrix<double>(4, 1);
            yData[0,0] = 0;
            yData[1,0] = 1;
            yData[2,0] = 1;
            yData[3,0] = 0;

            using var X = new Tensor(xData, requiresGrad: false);
            using var Y = new Tensor(yData, requiresGrad: false);

            // 2. Architektura Sieci Neuronowej
            var layer1 = new LinearLayer(inputSize: 2, outputSize: 16);
            var layer2 = new LinearLayer(inputSize: 16, outputSize: 1);

            var allParameters = layer1.Parameters().Concat(layer2.Parameters());
            
            // Używamy nieco wyższego Learning Rate w teście, aby szybciej zbiegł
            var sgd = new SGD(allParameters, learningRate: 0.1); 

            int epochs = 2000;
            double finalLoss = double.MaxValue;

            // ==========================================
            // ACT (Trening)
            // ==========================================
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                sgd.ZeroGrad();

                // Forward pass (Pamiętamy o zwalnianiu zasobów tymczasowych!)
                using var hidden = layer1.Forward(X);
                using var activated = TensorMath.ReLU(hidden);
                using var prediction = layer2.Forward(activated);

                using var loss = TensorMath.MSE(prediction, Y);
                finalLoss = loss.Data[0, 0];

                // Backward pass
                loss.Grad[0, 0] = 1.0; 
                loss.Backward();

                // Optymalizacja
                sgd.Step();
            }

            // ==========================================
            // ASSERT (Weryfikacja wiedzy sieci)
            // ==========================================
            
            // 1. Sprawdzamy, czy błąd (Loss) skutecznie spadł
            Assert.True(finalLoss < 0.05, $"Trening nie powiódł się. Końcowy błąd MSE: {finalLoss:F5}");

            // 2. Odpytujemy wytrenowany model
            using var finalHidden = layer1.Forward(X);
            using var finalActivated = TensorMath.ReLU(finalHidden);
            using var finalPrediction = layer2.Forward(finalActivated);

            // Oczekiwane wyniki to: [0, 1, 1, 0]
            // Ustalamy margines błędu. Wartości > 0.85 traktujemy jako 1, a < 0.15 jako 0.
            Assert.True(finalPrediction.Data[0, 0] < 0.15, $"Dla wejścia [0,0] oczekiwano blisko 0, otrzymano: {finalPrediction.Data[0, 0]:F4}");
            Assert.True(finalPrediction.Data[1, 0] > 0.85, $"Dla wejścia [0,1] oczekiwano blisko 1, otrzymano: {finalPrediction.Data[1, 0]:F4}");
            Assert.True(finalPrediction.Data[2, 0] > 0.85, $"Dla wejścia [1,0] oczekiwano blisko 1, otrzymano: {finalPrediction.Data[2, 0]:F4}");
            Assert.True(finalPrediction.Data[3, 0] < 0.15, $"Dla wejścia [1,1] oczekiwano blisko 0, otrzymano: {finalPrediction.Data[3, 0]:F4}");
        }
    }
}