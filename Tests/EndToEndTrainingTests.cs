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
            using var xData = new FastTensor<float>(4, 2, clearMemory: true);
            ((Span<float>)[0, 0, 0, 1, 1, 0, 1, 1]).CopyTo(xData.GetView().AsSpan());

            using var yData = new FastTensor<float>(4, 1, clearMemory: true);
            ((Span<float>)[0, 1, 1, 0]).CopyTo(yData.GetView().AsSpan());

            using var X = new AutogradNode(xData, false);
            using var Y = new AutogradNode(yData, false);

            using var layer1 = new LinearLayer(2, 16);
            using var layer2 = new LinearLayer(16, 1);

            var model = new Sequential(layer1, new ReluActivation(), layer2);
            var sgd = new SGD(model.Parameters(), 0.1f);
            var epochs = 500;
            var finalLoss = 0f;
            var graph = new ComputationGraph();

            for (var epoch = 0; epoch < epochs; epoch++)
            {
                graph.Reset();
                sgd.ZeroGrad();

                using var prediction = model.Forward(graph, X);
                using var loss = TensorMath.MSELoss(graph, prediction, Y);
                finalLoss = loss.DataView.AsReadOnlySpan()[0];

                graph.Backward(loss);
                sgd.Step();
            }

            Assert.True(finalLoss < 0.1f, $"Loss did not converge well enough. Final Loss: {finalLoss}");
        }

        [Fact]
        public void NeuralNetwork_TrainsOnCircleProblem_WithAdamW_AndConverges()
        {
            var samples = 200;
            using var xData = new FastTensor<float>(samples, 2, clearMemory: true);
            using var yData = new FastTensor<float>(samples, 1, clearMemory: true);

            var rnd = new Random(42);
            var xSpan = xData.GetView().AsSpan();
            var ySpan = yData.GetView().AsSpan();

            for (var i = 0; i < samples; i++)
            {
                var isOuter = i % 2 == 0;
                var radius = isOuter ? 0.8f + rnd.NextSingle() * 0.2f : rnd.NextSingle() * 0.4f;
                var angle = rnd.NextSingle() * 2 * MathF.PI;

                xSpan[i * 2 + 0] = radius * MathF.Cos(angle);
                xSpan[i * 2 + 1] = radius * MathF.Sin(angle);
                ySpan[i] = isOuter ? 1.0f : 0.0f;
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

            var adam = new Adam(model.Parameters(), 0.01f) { UseAdamW = true };
            var epochs = 300;
            var finalLoss = 0f;
            var graph = new ComputationGraph();

            for (var epoch = 0; epoch < epochs; epoch++)
            {
                graph.Reset();
                adam.ZeroGrad();

                using var prediction = model.Forward(graph, X);
                using var loss = TensorMath.MSELoss(graph, prediction, Y);
                finalLoss = loss.DataView.AsReadOnlySpan()[0];

                graph.Backward(loss);
                adam.Step();
            }

            Assert.True(finalLoss < 0.15f, $"Loss did not converge well enough. Final Loss: {finalLoss}");
        }
    }
}