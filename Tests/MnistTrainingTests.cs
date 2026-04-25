// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using DevOnBike.Overfit.Tests.Helpers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests
{
    public sealed class MnistTrainingTests
    {
        private readonly ITestOutputHelper _output;

        public MnistTrainingTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact(Skip = "a")]
        public void Mnist_FullTrain60k_CnnBeastMode_Benchmark()
        {
            const int trainSize = 60_000;
            const int batchSize = 64;
            const int epochs = 5;
            const float lr = 0.001f;

            var trainImagesPath = "d:/ml/train-images.idx3-ubyte";
            var trainLabelsPath = "d:/ml/train-labels.idx1-ubyte";

            if (!File.Exists(trainImagesPath) || !File.Exists(trainLabelsPath))
            {
                _output.WriteLine("MNIST files not found.");
                return;
            }

            var (trainX, trainY) = MnistLoader.Load(trainImagesPath, trainLabelsPath, trainSize);

            using var conv1 = new ConvLayer(1, 8, 28, 28, 3);
            using var bn1 = new BatchNorm1D(1352);
            using var res1 = new ResidualBlock(1352);
            using var fcOut = new LinearLayer(8, 10);

            var parameters = conv1.Parameters()
                .Concat(bn1.Parameters())
                .Concat(res1.Parameters())
                .Concat(fcOut.Parameters())
                .ToArray();

            using var optimizer = new Adam(parameters, lr)
            {
                UseAdamW = true
            };

            using var graph = new ComputationGraph();

            // 1. Podpinamy nasłuchiwacz metryk - koniec z ręcznym mierzeniem czasu i pamięci!
            using var telemetry = new TelemetryListener(_output);
            telemetry.Subscribe(OverfitTelemetry.MeterName);

            var totalSw = ValueStopwatch.StartNew();
            _output.WriteLine("=== START: Trening ResNet na Taśmie (DOD + Zero-Alloc) ===");

            // 2. PRE-ALOKACJA BUFORÓW (Zero-alloc wewnątrz pętli)
            using var xBData = new TensorStorage<float>(batchSize * 1 * 28 * 28, clearMemory: false);
            using var yBData = new TensorStorage<float>(batchSize * 10, clearMemory: false);

            for (var epoch = 0; epoch < epochs; epoch++)
            {
                conv1.Train();
                bn1.Train();
                res1.Train();
                fcOut.Train();

                var epochLoss = 0f;
                var batches = trainSize / batchSize;

                for (var b = 0; b < batches; b++)
                {
                    graph.Reset();
                    optimizer.ZeroGrad();

                    // Kopiowanie do istniejących buforów
                    trainX.AsReadOnlySpan()
                        .Slice(b * batchSize * 784, batchSize * 784)
                        .CopyTo(xBData.AsSpan());

                    trainY.AsReadOnlySpan()
                        .Slice(b * batchSize * 10, batchSize * 10)
                        .CopyTo(yBData.AsSpan());

                    // Węzły wejściowe bez ownStorage (nie niszczą bufora)
                    var xBNode = new AutogradNode(xBData, new TensorShape(batchSize, 1, 28, 28), requiresGrad: false);
                    var yBNode = new AutogradNode(yBData, new TensorShape(batchSize, 10), requiresGrad: false);

                    // --- CZYSTY FORWARD PASS (Pomiar wewnątrz warstw przez ModuleScope) ---
                    var h1 = conv1.Forward(graph, xBNode);
                    var a1 = TensorMath.ReLU(graph, h1);
                    var p1 = TensorMath.MaxPool2D(graph, a1, 8, 26, 26, 2);
                    var p1F = TensorMath.Reshape(graph, p1, batchSize, 1352);

                    var bn1O = bn1.Forward(graph, p1F);
                    var resO = res1.Forward(graph, bn1O);
                    var gapO = TensorMath.GlobalAveragePool2D(graph, resO, 8, 13, 13);
                    var logits = fcOut.Forward(graph, gapO);

                    var loss = TensorMath.SoftmaxCrossEntropy(graph, logits, yBNode);

                    epochLoss += loss.DataView.AsReadOnlySpan()[0];

                    graph.Backward(loss);
                    optimizer.Step();
                }

                var elapsed = totalSw.GetElapsedTime();

                _output.WriteLine($"Epoch {epoch + 1} | Loss: {epochLoss / batches:F4} | Time so far: {elapsed.TotalMilliseconds}ms");
            }

            // Metryki wypiszą się automatycznie dzięki TelemetryListenerowi!
            _output.WriteLine("=== KONIEC ===");
        }
    }
}