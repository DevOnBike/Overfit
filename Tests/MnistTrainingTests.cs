// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests
{
    public class MnistTrainingTests
    {
        private readonly ITestOutputHelper _output;
        public MnistTrainingTests(ITestOutputHelper output) => _output = output;

        [Fact(Skip = "a")]
        public void Mnist_FullTrain60k_CnnBeastMode_Benchmark()
        {
            var trainSize = 60000; var batchSize = 64; var epochs = 5; var lr = 0.001f;
            _output.WriteLine("=== START: Trening ResNet na Taśmie (NativeBuffer) ===");
            var (trainX, trainY) = MnistLoader.Load("d:/ml/train-images.idx3-ubyte", "d:/ml/train-labels.idx1-ubyte", trainSize);

            var conv1 = new ConvLayer(1, 8, 28, 28, 3);
            var bn1 = new BatchNorm1D(1352);
            var res1 = new ResidualBlock(1352);
            var fcOut = new LinearLayer(8, 10);

            var parameters = conv1.Parameters().Concat(bn1.Parameters()).Concat(res1.Parameters()).Concat(fcOut.Parameters()).ToArray();
            using var optimizer = new Adam(parameters, lr) { UseAdamW = true };

            var graph = new ComputationGraph();
            var sw = Stopwatch.StartNew();

            for (var epoch = 0; epoch < epochs; epoch++)
            {
                var epochLoss = 0f;
                var batches = trainSize / batchSize;

                for (var b = 0; b < batches; b++)
                {
                    graph.Reset();
                    optimizer.ZeroGrad();

                    using var xBData = new FastTensor<float>(batchSize, 1, 28, 28, clearMemory: false);
                    using var yBData = new FastTensor<float>(batchSize, 10, clearMemory: false);
                    using var xBNode = new AutogradNode(xBData, false);
                    using var yBNode = new AutogradNode(yBData, false);

                    trainX.GetView().AsReadOnlySpan().Slice(b * batchSize * 784, batchSize * 784).CopyTo(xBData.GetView().AsSpan());
                    trainY.GetView().AsReadOnlySpan().Slice(b * batchSize * 10, batchSize * 10).CopyTo(yBData.GetView().AsSpan());

                    using var h1 = conv1.Forward(graph, xBNode);
                    using var a1 = TensorMath.ReLU(graph, h1);
                    using var p1 = TensorMath.MaxPool2D(graph, a1, 8, 26, 26, 2);

                    // Płaszczymy do 2D dla BatchNorm i Residual
                    using var p1F = TensorMath.Reshape(graph, p1, batchSize, 1352);
                    using var bn1O = bn1.Forward(graph, p1F);
                    using var resO = res1.Forward(graph, bn1O);

                    // GlobalAveragePool2D czyta prosto ze spana (8 kanałów, 13x13)
                    using var gapO = TensorMath.GlobalAveragePool2D(graph, resO, 8, 13, 13);
                    using var logits = fcOut.Forward(graph, gapO);

                    using var loss = TensorMath.SoftmaxCrossEntropy(graph, logits, yBNode);
                    epochLoss += loss.DataView.AsReadOnlySpan()[0];

                    graph.Backward(loss);
                    optimizer.Step();
                }
                _output.WriteLine($"Epoch {epoch + 1} | Loss: {epochLoss / batches:F4} | Time: {sw.ElapsedMilliseconds}ms");
            }
            _output.WriteLine("=== KONIEC ===");
        }
    }
}