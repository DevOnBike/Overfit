using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;
using System.Diagnostics;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests
{
    public class MnistTrainingTests : IDisposable
    {
        private readonly ITestOutputHelper _output;

        public MnistTrainingTests(ITestOutputHelper output)
        {
            _output = output;
            // Inicjalizacja globalnej Taśmy dla każdego testu
            ComputationGraph.Active = new ComputationGraph();
        }

        [Fact(Skip = "a")]
        public void Mnist_FullTrain60k_CnnBeastMode_Augmented()
        {
            // --- ARRANGE ---
            var trainSize = 60000;
            var batchSize = 64;
            var epochs = 20;
            var learningRate = 0.001f;

            _output.WriteLine("=== START: Trening na Taśmie (Computation Graph) z Augmentacją ===");
            _output.WriteLine("Ładowanie danych MNIST...");

            var (trainX, trainY) = MnistLoader.Load("d:/ml/train-images.idx3-ubyte", "d:/ml/train-labels.idx1-ubyte", trainSize);

            using var X = new AutogradNode(trainX, requiresGrad: false);
            using var Y = new AutogradNode(trainY, requiresGrad: false);

            var conv1 = new ConvLayer(1, 8, 28, 28, 3);
            var bn1 = new BatchNorm1D(13 * 13 * 8); // 8 kanałów * 13 * 13
            var fc1 = new LinearLayer(13 * 13 * 8, 10);

            var model = new Sequential(conv1, bn1, fc1);
            using var optimizer = new Adam(model.Parameters(), learningRate);
            var scheduler = new LRScheduler(optimizer, _output.WriteLine, factor: 0.5f, patience: 1);

            var numBatches = trainSize / batchSize;
            var totalSw = Stopwatch.StartNew();

            _output.WriteLine($"Start treningu: {epochs} epok, {numBatches} batchy na epokę.");
            _output.WriteLine("----------------------------------------------------------");

            // --- ACT ---
            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                var epochSw = Stopwatch.StartNew();
                var epochLoss = 0f;

                model.Train();

                for (var b = 0; b < numBatches; b++)
                {
                    ComputationGraph.Active.Reset();
                    optimizer.ZeroGrad();

                    // Używamy 'using' dla wszystkiego, co bierze pamięć z puli
                    using var xBatchData = new FastTensor<float>(batchSize, 1, 28, 28);
                    X.Data.AsSpan().Slice(b * batchSize * 784, batchSize * 784).CopyTo(xBatchData.AsSpan());

                    using var mutatedData = DataAugmenter.AugmentBatch(xBatchData, 28, 28);
                    using var xBatch = new AutogradNode(mutatedData, false);

                    using var yBatchData = new FastTensor<float>(batchSize, 10);
                    Y.Data.AsSpan().Slice(b * batchSize * 10, batchSize * 10).CopyTo(yBatchData.AsSpan());
                    using var yBatch = new AutogradNode(yBatchData, false);

                    // --- FORWARD ---
                    using var h1 = conv1.Forward(xBatch);
                    using var a1 = TensorMath.ReLU(h1);
                    using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);

                    // POPRAWKA: Używamy TensorMath.Reshape zamiast przerywania grafu!
                    using var p1Flat = TensorMath.Reshape(p1, batchSize, 1352);
                    using var bnOut = bn1.Forward(p1Flat);
                    using var predictionLogits = fc1.Forward(bnOut);

                    using var loss = TensorMath.SoftmaxCrossEntropy(predictionLogits, yBatch);
                    epochLoss += loss.Data[0, 0];

                    ComputationGraph.Active.Backward(loss);
                    optimizer.Step();
                }

                epochSw.Stop();
                var avgLoss = epochLoss / numBatches;
                _output.WriteLine($"> EPOCH {epoch} GOTOWA | Loss: {avgLoss:F6} | LR: {optimizer.LearningRate:F6} | Czas: {epochSw.ElapsedMilliseconds}ms");

                scheduler.Step(avgLoss);
            }

            totalSw.Stop();

            // Ewaluacja końcowa
            var (testX, testY) = MnistLoader.Load("d:/ml/t10k-images.idx3-ubyte", "d:/ml/t10k-labels.idx1-ubyte", 1000);
            PrintConfusionMatrix(model, testX, testY);

            _output.WriteLine("----------------------------------------------------------");
            _output.WriteLine($"TRENING ZAKOŃCZONY! Całkowity czas: {totalSw.Elapsed.TotalSeconds:F2}s");

            model.Dispose();
        }

        private void PrintConfusionMatrix(Sequential model, FastTensor<float> testX, FastTensor<float> testY)
        {
            var matrix = new int[10, 10];
            var samples = Math.Min(1000, testX.Shape[0]);

            model.Eval();

            // Pobranie modułów do manualnego przejścia (z uwagi na wymaganą zmianę kształtu)
            var modulesField = typeof(Sequential).GetField("_modules", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var modules = (List<IModule>)modulesField.GetValue(model);
            var conv1 = (ConvLayer)modules[0];
            var bn1 = (BatchNorm1D)modules[1];
            var fc1 = (LinearLayer)modules[2];

            _output.WriteLine("\nGenerowanie Macierzy Pomyłek dla 1000 próbek testowych...");

            for (var i = 0; i < samples; i++)
            {
                // Resetujemy Taśmę, aby uniknąć narzutu pamięci przy inferencji
                ComputationGraph.Active.Reset();
                ComputationGraph.Active.IsRecording = false;

                try
                {
                    var rowData = new FastTensor<float>(1, 1, 28, 28);
                    testX.AsSpan().Slice(i * 784, 784).CopyTo(rowData.AsSpan());
                    using var input = new AutogradNode(rowData, false);

                    // Forward Pass (z manualnym Reshape)
                    using var h1 = conv1.Forward(input);
                    using var a1 = TensorMath.ReLU(h1);
                    using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);

                    using var p1Flat = TensorMath.Reshape(p1, 1, 1352);
                    using var bnOut = bn1.Forward(p1Flat);
                    using var output = fc1.Forward(bnOut);

                    var predicted = GetArgMax(output.Data.AsSpan());
                    var actual = GetArgMax(testY.AsSpan().Slice(i * 10, 10));
                    matrix[actual, predicted]++;
                }
                finally
                {
                    ComputationGraph.Active.IsRecording = true;
                }
            }

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("\n=== MACIERZ POMYŁEK (CONFUSION MATRIX) ===");
            sb.AppendLine("Oś pionowa: Prawda | Oś pozioma: Predykcja\n");

            sb.Append("A\\P |");
            for (var i = 0; i < 10; i++) sb.Append($"{i,4}|");
            sb.AppendLine("\n" + new string('-', 55));

            for (var r = 0; r < 10; r++)
            {
                sb.Append($"{r,3} |");
                for (var c = 0; c < 10; c++)
                {
                    var val = matrix[r, c];
                    if (r != c && val > 0) sb.Append($"[{val,2}]|");
                    else sb.Append($"{val,4}|");
                }
                sb.AppendLine();
            }

            _output.WriteLine(sb.ToString());
        }

        public void Dispose() => ComputationGraph.Active = null;

        private int GetArgMax(ReadOnlySpan<float> span)
        {
            var maxIdx = 0;
            for (var j = 1; j < span.Length; j++) if (span[j] > span[maxIdx]) maxIdx = j;
            return maxIdx;
        }


    }
}