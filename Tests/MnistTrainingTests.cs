using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;
using System.Diagnostics;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests
{
    public class MnistTrainingTests
    {
        private readonly ITestOutputHelper _output;

        public MnistTrainingTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact(Skip = "a")]
        public void Mnist_FullTrain60k_CnnBeastMode_Augmented()
        {
            // --- ARRANGE (SMOKE TEST) ---
            var trainSize = 60000; // Trenujemy na całym zbiorze danych!
            var batchSize = 64;
            var epochs = 5;        // 5 epok x ~24 sekundy = ~120 sekund (2 minuty)
            var learningRate = 0.001f;

            _output.WriteLine("=== START: Trening ResNet na Taśmie ===");
            _output.WriteLine("Ładowanie danych MNIST...");

            var (trainX, trainY) = MnistLoader.Load("d:/ml/train-images.idx3-ubyte", "d:/ml/train-labels.idx1-ubyte", trainSize);

            using var X = new AutogradNode(trainX, requiresGrad: false);
            using var Y = new AutogradNode(trainY, requiresGrad: false);

            // --- POPRAWIONA ARCHITEKTURA RESNET ---
            var conv1 = new ConvLayer(1, 8, 28, 28, 3);
            var bn1 = new BatchNorm1D(1352);
            var res1 = new ResidualBlock(1352);
            var fcOut = new LinearLayer(8, 10);

            var parameters = conv1.Parameters()
                .Concat(bn1.Parameters())
                .Concat(res1.Parameters())
                .Concat(fcOut.Parameters())
                .ToArray();

            using var optimizer = new Adam(parameters, learningRate);
            using var scheduler = new LRScheduler(optimizer, parameters, _output.WriteLine, factor: 0.5f, patience: 3);

            var numBatches = trainSize / batchSize;
            var totalSw = Stopwatch.StartNew();

            // ZMIANA: Jawna inicjalizacja grafu
            var graph = new ComputationGraph();

            _output.WriteLine($"Start treningu: {epochs} epok, {numBatches} batchy na epokę.");
            _output.WriteLine("----------------------------------------------------------");

            // --- ACT ---
            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                var epochSw = Stopwatch.StartNew();
                var epochLoss = 0f;

                conv1.Train(); bn1.Train(); res1.Train(); fcOut.Train();

                for (var b = 0; b < numBatches; b++)
                {
                    // ZMIANA: Operujemy na naszej jawnej instancji
                    graph.Reset();
                    optimizer.ZeroGrad();

                    using var xBatchData = new FastTensor<float>(batchSize, 1, 28, 28);
                    X.Data.AsSpan().Slice(b * batchSize * 784, batchSize * 784).CopyTo(xBatchData.AsSpan());

                    using var mutatedData = DataAugmenter.AugmentBatch(xBatchData, 28, 28);
                    using var xBatch = new AutogradNode(mutatedData, false);

                    using var yBatchData = new FastTensor<float>(batchSize, 10);
                    Y.Data.AsSpan().Slice(b * batchSize * 10, batchSize * 10).CopyTo(yBatchData.AsSpan());
                    using var yBatch = new AutogradNode(yBatchData, false);

                    // --- FORWARD PASS (Przekazujemy 'graph' jako pierwszy argument) ---
                    using var h1 = conv1.Forward(graph, xBatch);
                    using var a1 = TensorMath.ReLU(graph, h1);
                    using var p1 = TensorMath.MaxPool2D(graph, a1, 8, 26, 26, 2);

                    using var p1Flat = TensorMath.Reshape(graph, p1, batchSize, 1352);
                    using var bn1Out = bn1.Forward(graph, p1Flat);
                    using var resOut = res1.Forward(graph, bn1Out);

                    using var res4D = TensorMath.Reshape(graph, resOut, batchSize, 8, 13, 13);
                    using var gapOut = TensorMath.GlobalAveragePool2D(graph, res4D, 8, 13, 13);
                    using var predictionLogits = fcOut.Forward(graph, gapOut);

                    using var loss = TensorMath.SoftmaxCrossEntropy(graph, predictionLogits, yBatch);
                    epochLoss += loss.Data[0, 0];

                    // ZMIANA: Backward na jawnym grafie
                    graph.Backward(loss);
                    optimizer.Step();
                }

                epochSw.Stop();
                var avgLoss = epochLoss / numBatches;
                _output.WriteLine($"> EPOCH {epoch} GOTOWA | Loss: {avgLoss:F6} | LR: {optimizer.LearningRate:F6} | Czas: {epochSw.ElapsedMilliseconds}ms");

                scheduler.Step(avgLoss);
            }

            totalSw.Stop();

            var (testX, testY) = MnistLoader.Load("d:/ml/t10k-images.idx3-ubyte", "d:/ml/t10k-labels.idx1-ubyte", 1000);
            PrintConfusionMatrix(conv1, bn1, res1, fcOut, testX, testY);

            _output.WriteLine("----------------------------------------------------------");
            _output.WriteLine($"TRENING ZAKOŃCZONY! Całkowity czas: {totalSw.Elapsed.TotalSeconds:F2}s");

            using var fs = new FileStream("d:/ml/resnet_beast.bin", FileMode.Create);
            using var bw = new BinaryWriter(fs);
            conv1.Save(bw); bn1.Save(bw); res1.Save(bw); fcOut.Save(bw);

            _output.WriteLine("Zapisano wagi do pliku: d:/ml/resnet_beast.bin");
        }

        private void PrintConfusionMatrix(ConvLayer conv1, BatchNorm1D bn1, ResidualBlock res1, LinearLayer fcOut, FastTensor<float> testX, FastTensor<float> testY)
        {
            var matrix = new int[10, 10];
            var samples = Math.Min(1000, testX.Shape[0]);

            conv1.Eval(); bn1.Eval(); res1.Eval(); fcOut.Eval();

            _output.WriteLine("\nGenerowanie Macierzy Pomyłek dla 1000 próbek testowych...");

            for (var i = 0; i < samples; i++)
            {
                // ZMIANA: Usunięto graph.Reset() i blok try-finally

                var rowData = new FastTensor<float>(1, 1, 28, 28);
                testX.AsSpan().Slice(i * 784, 784).CopyTo(rowData.AsSpan());
                using var input = new AutogradNode(rowData, false);

                // ZMIANA: Przekazujemy 'null' zamiast grafu – wymusza to tryb Inference
                using var h1 = conv1.Forward(null, input);
                using var a1 = TensorMath.ReLU(null, h1);
                using var p1 = TensorMath.MaxPool2D(null, a1, 8, 26, 26, 2);

                using var p1Flat = TensorMath.Reshape(null, p1, 1, 1352);
                using var bn1Out = bn1.Forward(null, p1Flat);
                using var resOut = res1.Forward(null, bn1Out);

                using var res4D = TensorMath.Reshape(null, resOut, 1, 8, 13, 13);
                using var gapOut = TensorMath.GlobalAveragePool2D(null, res4D, 8, 13, 13);
                using var output = fcOut.Forward(null, gapOut);

                var predicted = GetArgMax(output.Data.AsSpan());
                var actual = GetArgMax(testY.AsSpan().Slice(i * 10, 10));
                matrix[actual, predicted]++;
            }

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("\n=== MACIERZ POMYŁEK (CONFUSION MATRIX) ===");
            for (var r = 0; r < 10; r++)
            {
                sb.Append($"{r,3} |");
                for (var c = 0; c < 10; c++) sb.Append($"{matrix[r, c],4}|");
                sb.AppendLine();
            }
            _output.WriteLine(sb.ToString());
        }

        private int GetArgMax(ReadOnlySpan<float> span)
        {
            var maxIdx = 0;
            for (var j = 1; j < span.Length; j++) if (span[j] > span[maxIdx]) maxIdx = j;
            return maxIdx;
        }
    }
}