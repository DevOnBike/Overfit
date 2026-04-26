// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core; // Wpinamy Core

namespace DevOnBike.Overfit.Tests
{
    public class MnistCnnTrainingTests
    {
        [Fact(Skip = "Long running test")]
        public void Mnist_CnnTraining_ShouldConvergeFast()
        {
            var trainSize = 1000;
            var batchSize = 32;
            var epochs = 3;
            var learningRate = 0.001f;

            // MnistLoader musi teraz zwracać krotkę (TensorStorage<float> trainX, TensorStorage<float> trainY)
            var (trainX, trainY) = MnistLoader.Load("d:/ml/train-images.idx3-ubyte", "d:/ml/train-labels.idx1-ubyte", trainSize);

            using var X = new AutogradNode(trainX, new TensorShape(trainSize, 1, 28, 28), false);
            using var Y = new AutogradNode(trainY, new TensorShape(trainSize, 10), false);

            var conv1 = new ConvLayer(1, 8, 28, 28, 3);
            var fc1 = new LinearLayer(1352, 10);

            var allParameters = conv1.Parameters().Concat(fc1.Parameters());
            var adam = new Adam(allParameters, learningRate) { UseAdamW = true };

            var graph = new ComputationGraph();
            var sw = ValueStopwatch.StartNew();

            for (var epoch = 0; epoch < epochs; epoch++)
            {
                var lossValue = 0f;
                var batches = trainSize / batchSize;

                for (var b = 0; b < batches; b++)
                {
                    graph.Reset();
                    adam.ZeroGrad();

                    // Używamy surowych buforów i kształtów
                    using var batchX = new TensorStorage<float>(batchSize * 1 * 28 * 28, clearMemory: false);
                    using var batchY = new TensorStorage<float>(batchSize * 10, clearMemory: false);

                    trainX.AsReadOnlySpan().Slice(b * batchSize * 784, batchSize * 784).CopyTo(batchX.AsSpan());
                    trainY.AsReadOnlySpan().Slice(b * batchSize * 10, batchSize * 10).CopyTo(batchY.AsSpan());

                    using var bXNode = new AutogradNode(batchX, new TensorShape(batchSize, 1, 28, 28), false);
                    using var bYNode = new AutogradNode(batchY, new TensorShape(batchSize, 10), false);

                    using var h1 = conv1.Forward(graph, bXNode);
                    using var a1 = TensorMath.ReLU(graph, h1);
                    using var p1 = TensorMath.MaxPool2D(graph, a1, 8, 26, 26, 2);

                    using var p1Flat = TensorMath.Reshape(graph, p1, batchSize, 1352);
                    using var output = fc1.Forward(graph, p1Flat);

                    using var loss = TensorMath.SoftmaxCrossEntropy(graph, output, bYNode);

                    graph.Backward(loss);
                    adam.Step();

                    lossValue = loss.DataView.AsReadOnlySpan()[0];
                }

                Console.WriteLine($"Epoch {epoch + 1}/{epochs} | Loss: {lossValue:F4}");
            }

            var elapsed = sw.GetElapsedTime();
            Console.WriteLine($"Training took: {elapsed.TotalMilliseconds} ms");

            PrintConfusionMatrix(conv1, fc1, trainX, trainY);
        }

        private void PrintConfusionMatrix(ConvLayer conv, LinearLayer fc, TensorStorage<float> testX, TensorStorage<float> testY)
        {
            conv.Eval();
            fc.Eval();

            var matrix = new int[10, 10];
            var samples = Math.Min(500, testX.Length / 784);

            for (var i = 0; i < samples; i++)
            {
                using var rowData = new TensorStorage<float>(784, clearMemory: false);
                testX.AsReadOnlySpan().Slice(i * 784, 784).CopyTo(rowData.AsSpan());

                using var input = new AutogradNode(rowData, new TensorShape(1, 1, 28, 28), false);

                var h1 = conv.Forward(null, input);
                using var a1 = TensorMath.ReLU(null, h1);
                using var p1 = TensorMath.MaxPool2D(null, a1, 8, 26, 26, 2);

                using var p1Flat = TensorMath.Reshape(null, p1, 1, 1352);
                var output = fc.Forward(null, p1Flat);

                var predicted = GetArgMax(output.DataView.AsReadOnlySpan());
                var actual = GetArgMax(testY.AsReadOnlySpan().Slice(i * 10, 10));
                matrix[actual, predicted]++;
            }
        }

        private int GetArgMax(ReadOnlySpan<float> span)
        {
            var maxIdx = 0;
            for (var j = 1; j < span.Length; j++)
            {
                if (span[j] > span[maxIdx])
                {
                    maxIdx = j;
                }
            }
            return maxIdx;
        }
    }
}