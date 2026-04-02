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
        
        [Fact(Skip = "Odkomentuj by odpalić bestię!")]
        public void Mnist_CnnTraining_ShouldConvergeFast()
        {
            // --- ARRANGE ---
            var trainSize = 1000;
            var batchSize = 32;
            var epochs = 3;
            var learningRate = 0.001f;

            // MnistLoader zwraca teraz natywne FastTensor<float>
            var (trainX, trainY) = MnistLoader.Load("d:/ml/train-images.idx3-ubyte", "d:/ml/train-labels.idx1-ubyte", trainSize);

            using var X = new AutogradNode(trainX, requiresGrad: false);
            using var Y = new AutogradNode(trainY, requiresGrad: false);

            var conv1 = new ConvLayer(inChannels: 1, outChannels: 8, h: 28, w: 28, kSize: 3);
            var fc1 = new LinearLayer(1352, 10); // Wynik 28x28 -> 26x26 (Conv) -> 13x13 (MaxPool) * 8 ch = 1352

            // Pobieranie parametrów do Adama
            var allParameters = conv1.Parameters().Concat(fc1.Parameters());
            using var optimizer = new Adam(allParameters, learningRate);

            var numBatches = trainSize / batchSize;
            var sw = Stopwatch.StartNew();
            float finalLoss = 0;

            // --- ACT ---
            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                float epochLoss = 0;
                conv1.Train();
                fc1.Train();

                for (var b = 0; b < numBatches; b++)
                {
                    // Reset taśmy przed każdym forwardem
                    ComputationGraph.Active.Reset();
                    optimizer.ZeroGrad();

                    // SLICING NCHW - Kopiowanie bloków pamięci RAM do FastTensor
                    var xBatchData = new FastTensor<float>(batchSize, 1, 28, 28);
                    X.Data.AsSpan().Slice(b * batchSize * 784, batchSize * 784).CopyTo(xBatchData.AsSpan());
                    using var xBatch = new AutogradNode(xBatchData, false);

                    var yBatchData = new FastTensor<float>(batchSize, 10);
                    Y.Data.AsSpan().Slice(b * batchSize * 10, batchSize * 10).CopyTo(yBatchData.AsSpan());
                    using var yBatch = new AutogradNode(yBatchData, false);

                    // --- FORWARD PASS (Pełna moc NCHW) ---
                    using var h1 = conv1.Forward(xBatch); // Wyjście: [batchSize, 8, 26, 26]
                    using var a1 = TensorMath.ReLU(h1);
                    using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2); // Wyjście: [batchSize, 8, 13, 13]

                    // ZERO-COPY RESHAPE: Spłaszczamy mapy cech do 2D dla warstwy liniowej
                    using var p1Flat = new AutogradNode(p1.Data.Reshape(batchSize, 1352), false);
                    using var prediction = fc1.Forward(p1Flat);

                    // --- LOSS ---
                    using var loss = TensorMath.MSELoss(prediction, yBatch);
                    epochLoss += loss.Data[0, 0];

                    // --- BACKWARD PASS ---
                    ComputationGraph.Active.Backward(loss);
                    optimizer.Step();
                }

                finalLoss = epochLoss / numBatches;
                Debug.WriteLine($"Epoch {epoch} | Loss: {finalLoss:F6}");
            }

            sw.Stop();

            // --- ASSERT ---
            Assert.True(sw.ElapsedMilliseconds < 10000, "Trening CNN jest zbyt wolny!");
            Assert.True(finalLoss < 0.1f, $"Model nie zbiega poprawnie. Loss: {finalLoss}");

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

        public void PrintConfusionMatrix(ConvLayer conv, LinearLayer fc, FastTensor<float> testX, FastTensor<float> testY)
        {
            conv.Eval();
            fc.Eval();

            var matrix = new int[10, 10];
            var samples = Math.Min(500, testX.Shape[0]);

            for (var i = 0; i < samples; i++)
            {
                ComputationGraph.Active.Reset();

                // Wycinanie pojedynczego obrazka z pamięci
                var rowData = new FastTensor<float>(1, 1, 28, 28);
                testX.AsSpan().Slice(i * 784, 784).CopyTo(rowData.AsSpan());
                using var input = new AutogradNode(rowData, false);

                using var h1 = conv.Forward(input);
                using var a1 = TensorMath.ReLU(h1);
                using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);

                // Reshape 4D -> 2D dla warstwy końcowej
                using var p1Flat = new AutogradNode(p1.Data.Reshape(1, 1352), false);
                using var output = fc.Forward(p1Flat);

                var predicted = GetArgMax(output.Data.AsSpan());
                var actual = GetArgMax(testY.AsSpan().Slice(i * 10, 10));
                matrix[actual, predicted]++;
            }
        }

        public void Dispose()
        {
            // Sprzątanie referencji
            ComputationGraph.Active = null;
        }

        // Helper do wyciągania klasy o najwyższym prawdopodobieństwie ze Spana
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