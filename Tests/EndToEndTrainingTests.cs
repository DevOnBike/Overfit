using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;

namespace DevOnBike.Overfit.Tests
{
    public class EndToEndTrainingTests : IDisposable
    {
        public EndToEndTrainingTests()
        {
            // Inicjalizacja taśmy dla bieżącego wątku testowego
            ComputationGraph.Active = new ComputationGraph();
        }

        public void Dispose()
        {
            // Sprzątanie po teście
            ComputationGraph.Active = null;
        }

        [Fact]
        public void NeuralNetwork_TrainsOnXORProblem_AndConvergesToCorrectPredictions()
        {
            // ==========================================
            // ARRANGE
            // ==========================================
            using var xData = new FastMatrix<float>(4, 2);
            xData.CopyFrom([0, 0, 0, 1, 1, 0, 1, 1]);

            using var yData = new FastMatrix<float>(4, 1);
            yData.CopyFrom([0, 1, 1, 0]);

            using var X = new AutogradNode(xData, requiresGrad: false);
            using var Y = new AutogradNode(yData, requiresGrad: false);

            using var layer1 = new LinearLayer(inputSize: 2, outputSize: 16);
            using var layer2 = new LinearLayer(inputSize: 16, outputSize: 1);

            var model = new Sequential(layer1, new ReluActivation(), layer2);
            var sgd = new SGD(model.Parameters(), learningRate: 0.1f);

            var epochs = 2000;
            var finalLoss = double.MaxValue;

            // ==========================================
            // ACT (Trening)
            // ==========================================
            for (var epoch = 0; epoch < epochs; epoch++)
            {
                // Reset taśmy przed każdym forwardem - zero alokacji!
                ComputationGraph.Active.Reset();
                sgd.ZeroGrad();

                using var prediction = model.Forward(X);
                using var loss = TensorMath.MSELoss(prediction, Y);
                finalLoss = loss.Data[0, 0];

                // Backward pass przez graf
                ComputationGraph.Active.Backward(loss);

                sgd.Step();
            }

            // ==========================================
            // ASSERT (Weryfikacja w trybie No-Grad)
            // ==========================================
            Assert.True(finalLoss < 0.05, $"Trening nie powiódł się. Loss: {finalLoss:F5}");

            // Wyłączamy nagrywanie operacji dla fazy testowej
            ComputationGraph.Active.IsRecording = false;
            try
            {
                using var finalPrediction = model.Forward(X);

                Assert.True(finalPrediction.Data[0, 0] < 0.15);
                Assert.True(finalPrediction.Data[1, 0] > 0.85);
                Assert.True(finalPrediction.Data[2, 0] > 0.85);
                Assert.True(finalPrediction.Data[3, 0] < 0.15);
            }
            finally
            {
                ComputationGraph.Active.IsRecording = true;
            }
        }

        [Fact]
        public void NeuralNetwork_TrainsOnConcentricCircles_MakesCPUWorkHarder()
        {
            // ==========================================
            // ARRANGE: Generowanie 300 punktów danych
            // ==========================================
            var numSamples = 300;
            using var xData = new FastMatrix<float>(numSamples, 2);
            using var yData = new FastMatrix<float>(numSamples, 1);
            var rnd = new Random(42);

            for (var i = 0; i < numSamples; i++)
            {
                var isOuter = i % 2 == 0;
                var radius = isOuter ? rnd.NextSingle() * 0.5f + 0.5f : rnd.NextSingle() * 0.4f;
                var angle = rnd.NextSingle() * 2 * MathF.PI;

                xData[i, 0] = radius * MathF.Cos(angle);
                xData[i, 1] = radius * MathF.Sin(angle);
                yData[i, 0] = isOuter ? 1.0f : 0.0f;
            }

            using var X = new AutogradNode(xData, requiresGrad: false);
            using var Y = new AutogradNode(yData, requiresGrad: false);

            using var layer1 = new LinearLayer(2, 32);
            using var layer2 = new LinearLayer(32, 16);
            using var layer3 = new LinearLayer(16, 1);

            var model = new Sequential(
                layer1, new ReluActivation(),
                layer2, new ReluActivation(),
                layer3);

            var sgd = new SGD(model.Parameters(), learningRate: 0.05f);
            var epochs = 3000;
            var finalLoss = double.MaxValue;

            // ==========================================
            // ACT: Pętla Treningowa
            // ==========================================
            for (var epoch = 0; epoch < epochs; epoch++)
            {
                ComputationGraph.Active.Reset();
                sgd.ZeroGrad();

                using var prediction = model.Forward(X);
                using var loss = TensorMath.MSELoss(prediction, Y);
                finalLoss = loss.Data[0, 0];

                ComputationGraph.Active.Backward(loss);
                sgd.Step();
            }

            // ==========================================
            // ASSERT
            // ==========================================
            Assert.True(finalLoss < 0.1, $"Sieć nie podołała okręgom. Końcowy błąd: {finalLoss:F5}");
        }
    }
}