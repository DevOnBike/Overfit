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
            _output.WriteLine("=== START: Trening ResNet na Taśmie (CLEAN BENCHMARK) ===");
            var (trainX, trainY) = MnistLoader.Load("d:/ml/train-images.idx3-ubyte", "d:/ml/train-labels.idx1-ubyte", trainSize);

            var conv1 = new ConvLayer(1, 8, 28, 28, 3); var bn1 = new BatchNorm1D(1352); var res1 = new ResidualBlock(1352); var fcOut = new LinearLayer(8, 10);
            var parameters = conv1.Parameters().Concat(bn1.Parameters()).Concat(res1.Parameters()).Concat(fcOut.Parameters()).ToArray();
            using var optimizer = new Adam(parameters, lr) { UseAdamW = true };

            var numBatches = trainSize / batchSize; var totalSw = Stopwatch.StartNew(); var graph = new ComputationGraph();

            // --- BUFFER HOISTING: Alokujemy batche raz przed pętlą ---
            using var xBData = new FastTensor<float>(batchSize, 1, 28, 28);
            using var yBData = new FastTensor<float>(batchSize, 10);
            using var xBNode = new AutogradNode(xBData, false);
            using var yBNode = new AutogradNode(yBData, false);

            _output.WriteLine($"Start treningu: {epochs} epok, {numBatches} batchy na epokę.");

            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                var epochSw = Stopwatch.StartNew(); var epochLoss = 0f;
                conv1.Train(); bn1.Train(); res1.Train(); fcOut.Train();

                for (var b = 0; b < numBatches; b++)
                {
                    graph.Reset(); optimizer.ZeroGrad();

                    // Kopiowanie danych do gotowych buforów (brak alokacji FastTensor)
                    trainX.AsReadOnlySpan().Slice(b * batchSize * 784, batchSize * 784).CopyTo(xBData.AsSpan());
                    trainY.AsReadOnlySpan().Slice(b * batchSize * 10, batchSize * 10).CopyTo(yBData.AsSpan());

                    using var h1 = conv1.Forward(graph, xBNode);
                    using var a1 = TensorMath.ReLU(graph, h1);
                    using var p1 = TensorMath.MaxPool2D(graph, a1, 8, 26, 26, 2);
                    using var p1F = TensorMath.Reshape(graph, p1, batchSize, 1352);
                    using var bn1O = bn1.Forward(graph, p1F);
                    using var resO = res1.Forward(graph, bn1O);
                    using var res4D = TensorMath.Reshape(graph, resO, batchSize, 8, 13, 13);
                    using var gapO = TensorMath.GlobalAveragePool2D(graph, res4D, 8, 13, 13);
                    using var logits = fcOut.Forward(graph, gapO);

                    using var loss = TensorMath.SoftmaxCrossEntropy(graph, logits, yBNode);
                    epochLoss += loss.Data[0, 0];
                    graph.Backward(loss); optimizer.Step();
                }
                _output.WriteLine($"> EPOCH {epoch} GOTOWA | Loss: {epochLoss / numBatches:F6} | Czas: {epochSw.ElapsedMilliseconds}ms");
            }
            _output.WriteLine($"TRENING ZAKOŃCZONY! Całkowity czas: {totalSw.Elapsed.TotalSeconds:F2}s");
        }
    }
}