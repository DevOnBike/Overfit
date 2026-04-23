// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Tests
{
    public class EndToEndTrainingTests
    {
        [Fact]
        public void NeuralNetwork_TrainsOnXORProblem_AndConvergesToCorrectPredictions()
        {
            using var xData = new TensorStorage<float>(8, clearMemory: true);
            ((Span<float>)[0, 0, 0, 1, 1, 0, 1, 1]).CopyTo(xData.AsSpan());

            using var yData = new TensorStorage<float>(4, clearMemory: true);
            ((Span<float>)[0, 1, 1, 0]).CopyTo(yData.AsSpan());

            using var X = new AutogradNode(xData, new TensorShape(4, 2), false);
            using var Y = new AutogradNode(yData, new TensorShape(4, 1), false);

            using var layer1 = new LinearLayer(2, 16);
            using var layer2 = new LinearLayer(16, 1);

            var model = new Sequential(
                layer1, new ReluActivation(),
                layer2, new ReluActivation());

            var adam = new Adam(model.Parameters(), 0.05f) { UseAdamW = true };

            var epochs = 200;
            var finalLoss = 0f;
            var graph = new ComputationGraph();

            for (var epoch = 0; epoch < epochs; epoch++)
            {
                graph.Reset();
                adam.ZeroGrad();

                using var prediction = model.Forward(graph, X);
                using var loss = TensorMath.MSELoss(graph, prediction, Y);

                graph.Backward(loss);
                adam.Step();

                finalLoss = loss.DataView.AsReadOnlySpan()[0];
            }

            Assert.True(finalLoss < 0.1f, $"Loss should be close to 0, but was {finalLoss}");
        }

        [Fact]
        public void NeuralNetwork_TrainsOnCircleDataset_AndConverges()
        {
            var samples = 200;
            var rnd = new Random(42);

            var xData = new TensorStorage<float>(samples * 2, clearMemory: true);
            var yData = new TensorStorage<float>(samples, clearMemory: true);
            var xSpan = xData.AsSpan();
            var ySpan = yData.AsSpan();

            for (var i = 0; i < samples; i++)
            {
                var isOuter = i % 2 == 0;
                var radius = isOuter ? 0.8f + rnd.NextSingle() * 0.2f : rnd.NextSingle() * 0.4f;
                var angle = rnd.NextSingle() * 2 * MathF.PI;

                xSpan[i * 2 + 0] = radius * MathF.Cos(angle);
                xSpan[i * 2 + 1] = radius * MathF.Sin(angle);
                ySpan[i] = isOuter ? 1.0f : 0.0f;
            }

            using var X = new AutogradNode(xData, new TensorShape(samples, 2), false);
            using var Y = new AutogradNode(yData, new TensorShape(samples, 1), false);

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

                graph.Backward(loss);
                adam.Step();

                finalLoss = loss.DataView.AsReadOnlySpan()[0];
            }

            Assert.True(finalLoss < 0.15f, $"Loss should converge below 0.15, but was {finalLoss}");
        }
    }
}