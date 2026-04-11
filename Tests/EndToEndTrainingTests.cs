// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;

namespace DevOnBike.Overfit.Tests
{
    public class EndToEndTrainingTests
    {
        [Fact]
        public void NeuralNetwork_TrainsOnXORProblem_AndConvergesToCorrectPredictions()
        {
            // ARRANGE
            using var xData = new FastTensor<float>(4, 2);
            ((Span<float>)[0, 0, 0, 1, 1, 0, 1, 1]).CopyTo(xData.AsSpan());

            using var yData = new FastTensor<float>(4, 1);
            ((Span<float>)[0, 1, 1, 0]).CopyTo(yData.AsSpan());

            using var X = new AutogradNode(xData, false);
            using var Y = new AutogradNode(yData, false);

            using var layer1 = new LinearLayer(2, 16);
            using var layer2 = new LinearLayer(16, 1);

            var model = new Sequential(layer1, new ReluActivation(), layer2);
            var sgd = new SGD(model.Parameters(), 0.1f);

            var epochs = 2000;
            var finalLoss = 0f;

            // ZMIANA: Tworzymy własny graf treningowy
            var graph = new ComputationGraph();

            // ACT (Trening)
            for (var epoch = 0; epoch < epochs; epoch++)
            {
                graph.Reset();
                sgd.ZeroGrad();

                // Forward z nagrywaniem
                using var prediction = model.Forward(graph, X);
                using var loss = TensorMath.MSELoss(graph, prediction, Y);
                finalLoss = loss.Data[0, 0];

                // Backward pass przez graf
                graph.Backward(loss);

                sgd.Step();
            }

            // ASSERT
            Assert.True(finalLoss < 0.05f, $"Trening nie powiódł się. Loss: {finalLoss:F5}");

            // ZMIANA: Zamiast IsRecording=false, podajemy null jako graph
            using var finalPrediction = model.Forward(null, X);

            Assert.True(finalPrediction.Data[0, 0] < 0.15f);
            Assert.True(finalPrediction.Data[1, 0] > 0.85f);
            Assert.True(finalPrediction.Data[2, 0] > 0.85f);
            Assert.True(finalPrediction.Data[3, 0] < 0.15f);

            model.Dispose();
        }

        [Fact]
        public void NeuralNetwork_TrainsOnConcentricCircles_MakesCPUWorkHarder()
        {
            // ARRANGE
            var numSamples = 300;
            using var xData = new FastTensor<float>(numSamples, 2);
            using var yData = new FastTensor<float>(numSamples, 1);
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

            using var X = new AutogradNode(xData, false);
            using var Y = new AutogradNode(yData, false);

            using var layer1 = new LinearLayer(2, 32);
            using var layer2 = new LinearLayer(32, 16);
            using var layer3 = new LinearLayer(16, 1);

            var model = new Sequential(
            layer1, new ReluActivation(),
            layer2, new ReluActivation(),
            layer3);

            var sgd = new SGD(model.Parameters(), 0.05f);
            var epochs = 3000;
            var finalLoss = 0f;
            var graph = new ComputationGraph();

            // ACT
            for (var epoch = 0; epoch < epochs; epoch++)
            {
                graph.Reset();
                sgd.ZeroGrad();

                using var prediction = model.Forward(graph, X);
                using var loss = TensorMath.MSELoss(graph, prediction, Y);
                finalLoss = loss.Data[0, 0];

                graph.Backward(loss);
                sgd.Step();
            }

            // ASSERT
            Assert.True(finalLoss < 0.1f, $"Sieć nie podołała okręgom. Końcowy błąd: {finalLoss:F5}");
            model.Dispose();
        }
    }
}