// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;

namespace DevOnBike.Overfit.Tests
{
    public class MnistCnnTrainingTests
    {
        [Fact(Skip = "Odkomentuj by odpalić bestię!")]
        public void Mnist_CnnTraining_ShouldConvergeFast()
        {
            var trainSize = 1000;
            var batchSize = 32;
            var epochs = 3;
            var learningRate = 0.001f;

            var (trainX, trainY) = MnistLoader.Load("d:/ml/train-images.idx3-ubyte", "d:/ml/train-labels.idx1-ubyte", trainSize);

            using var X = new AutogradNode(trainX, requiresGrad: false);
            using var Y = new AutogradNode(trainY, requiresGrad: false);

            var conv1 = new ConvLayer(inChannels: 1, outChannels: 8, h: 28, w: 28, kSize: 3);
            var fc1 = new LinearLayer(1352, 10);

            var allParameters = conv1.Parameters().Concat(fc1.Parameters());
            using var optimizer = new Adam(allParameters, learningRate);

            var numBatches = trainSize / batchSize;
            var sw = Stopwatch.StartNew();
            float finalLoss = 0;

            // JAWNY GRAF OBLICZENIOWY
            var graph = new ComputationGraph();

            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                float epochLoss = 0;
                conv1.Train();
                fc1.Train();

                for (var b = 0; b < numBatches; b++)
                {
                    graph.Reset();
                    optimizer.ZeroGrad();

                    var xBatchData = new FastTensor<float>(batchSize, 1, 28, 28);
                    X.Data.AsSpan().Slice(b * batchSize * 784, batchSize * 784).CopyTo(xBatchData.AsSpan());
                    using var xBatch = new AutogradNode(xBatchData, false);

                    var yBatchData = new FastTensor<float>(batchSize, 10);
                    Y.Data.AsSpan().Slice(b * batchSize * 10, batchSize * 10).CopyTo(yBatchData.AsSpan());
                    using var yBatch = new AutogradNode(yBatchData, false);

                    // FORWARD PASS Z GRAFEM
                    using var h1 = conv1.Forward(graph, xBatch);
                    using var a1 = TensorMath.ReLU(graph, h1);
                    using var p1 = TensorMath.MaxPool2D(graph, a1, 8, 26, 26, 2);

                    using var p1Flat = TensorMath.Reshape(graph, p1, batchSize, 1352);
                    using var prediction = fc1.Forward(graph, p1Flat);

                    using var loss = TensorMath.MSELoss(graph, prediction, yBatch);
                    epochLoss += loss.Data[0, 0];

                    graph.Backward(loss);
                    optimizer.Step();
                }

                finalLoss = epochLoss / numBatches;
                Debug.WriteLine($"Epoch {epoch} | Loss: {finalLoss:F6}");
            }

            sw.Stop();

            Assert.True(sw.ElapsedMilliseconds < 10000, "Trening CNN jest zbyt wolny!");
            Assert.True(finalLoss < 0.1f, $"Model nie zbiega poprawnie. Loss: {finalLoss}");

            // Ewaluacja
            PrintConfusionMatrix(conv1, fc1, trainX, trainY);
        }

        private void PrintConfusionMatrix(ConvLayer conv, LinearLayer fc, FastTensor<float> testX, FastTensor<float> testY)
        {
            conv.Eval();
            fc.Eval();

            var matrix = new int[10, 10];
            var samples = Math.Min(500, testX.Shape[0]);

            for (var i = 0; i < samples; i++)
            {
                var rowData = new FastTensor<float>(1, 1, 28, 28);
                testX.AsSpan().Slice(i * 784, 784).CopyTo(rowData.AsSpan());
                using var input = new AutogradNode(rowData, false);

                // INFERENCJA -> NULL GRAF
                using var h1 = conv.Forward(null, input);
                using var a1 = TensorMath.ReLU(null, h1);
                using var p1 = TensorMath.MaxPool2D(null, a1, 8, 26, 26, 2);

                using var p1Flat = new AutogradNode(p1.Data.Reshape(1, 1352), false);
                using var output = fc.Forward(null, p1Flat);

                var predicted = GetArgMax(output.Data.AsSpan());
                var actual = GetArgMax(testY.AsSpan().Slice(i * 10, 10));
                matrix[actual, predicted]++;
            }
        }

        private int GetArgMax(ReadOnlySpan<float> span)
        {
            var maxIdx = 0;
            for (var j = 1; j < span.Length; j++)
            {
                if (span[j] > span[maxIdx]) maxIdx = j;
            }
            return maxIdx;
        }
    }
}