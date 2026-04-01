using System.Diagnostics;
using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;

namespace DevOnBike.Overfit.Tests
{
    public class MnistCnnTrainingTests : IDisposable
    {
        public MnistCnnTrainingTests()
        {
            // Inicjalizacja grafu dla bieżącego wątku
            ComputationGraph.Active = new ComputationGraph();
        }

        public void Dispose()
        {
            // Sprzątanie referencji
            ComputationGraph.Active = null;
        }

        [Fact]
        public void Mnist_CnnTraining_ShouldConvergeFast()
        {
            // --- ARRANGE ---
            var trainSize = 1000;
            var batchSize = 32;
            var epochs = 3;
            var learningRate = 0.001;

            var (trainX, trainY) = MnistLoader.Load("d:/ml/train-images.idx3-ubyte", "d:/ml/train-labels.idx1-ubyte", trainSize);

            using var X = new AutogradNode(trainX, requiresGrad: false);
            using var Y = new AutogradNode(trainY, requiresGrad: false);

            var conv1 = new ConvLayer(inChannels: 1, outChannels: 8, h: 28, w: 28, kSize: 3);
            var fc1 = new LinearLayer(1352, 10);

            // Pobieranie parametrów do Adama
            var allParameters = conv1.Parameters().Concat(fc1.Parameters());
            using var optimizer = new Adam(allParameters, learningRate);

            var numBatches = trainSize / batchSize;
            var sw = Stopwatch.StartNew();
            double finalLoss = 0;

            // --- ACT ---
            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                double epochLoss = 0;
                conv1.Train();
                fc1.Train();

                for (var b = 0; b < numBatches; b++)
                {
                    // Reset taśmy i zerowanie gradientów wag - zero alokacji!
                    ComputationGraph.Active.Reset();
                    optimizer.ZeroGrad();

                    var xView = X.Data.AsView().Slice(b * batchSize, 0, batchSize, 784);
                    var yView = Y.Data.AsView().Slice(b * batchSize, 0, batchSize, 10);

                    using var xBatch = new AutogradNode(xView.ToContiguousFastMatrix(), false);
                    using var yBatch = new AutogradNode(yView.ToContiguousFastMatrix(), false);

                    // --- FORWARD PASS ---
                    using var h1 = conv1.Forward(xBatch);
                    using var a1 = TensorMath.ReLU(h1);
                    using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);
                    using var prediction = fc1.Forward(p1);

                    // --- LOSS ---
                    using var loss = TensorMath.MSELoss(prediction, yBatch);
                    epochLoss += loss.Data[0, 0];

                    // --- BACKWARD PASS (Przez Taśmę) ---
                    ComputationGraph.Active.Backward(loss); // Automatycznie inicjuje 1.0

                    // Update wag
                    optimizer.Step();
                }

                finalLoss = epochLoss / numBatches;
                Debug.WriteLine($"Epoch {epoch} | Loss: {finalLoss:F6}");
            }

            sw.Stop();

            // --- ASSERT ---
            Assert.True(sw.ElapsedMilliseconds < 10000, "Trening CNN jest zbyt wolny!");
            Assert.True(finalLoss < 0.1, $"Model nie zbiega poprawnie. Loss: {finalLoss}");

            // Ewaluacja w trybie No-Grad
            ComputationGraph.Active.IsRecording = false;
            try
            {
                PrintConfusionMatrix(conv1, fc1, trainX, trainY);
            }
            finally
            {
                ComputationGraph.Active.IsRecording = true;
            }
        }

        public void PrintConfusionMatrix(ConvLayer conv, LinearLayer fc, FastMatrix<double> testX, FastMatrix<double> testY)
        {
            conv.Eval();
            fc.Eval();

            var matrix = new int[10, 10];
            var samples = Math.Min(500, testX.Rows);

            for (var i = 0; i < samples; i++)
            {
                // Resetujemy tylko licznik taśmy przy wyłączonym nagrywaniu
                ComputationGraph.Active.Reset();

                var rowView = testX.AsView().Slice(i, 0, 1, 784);
                using var input = new AutogradNode(rowView.ToContiguousFastMatrix(), false);

                using var h1 = conv.Forward(input);
                using var a1 = TensorMath.ReLU(h1);
                using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);
                using var output = fc.Forward(p1);

                matrix[testY.ArgMax(i), output.Data.ArgMax()]++;
            }

            // ... (logika rysowania tabeli bez zmian) ...
        }
    }
}