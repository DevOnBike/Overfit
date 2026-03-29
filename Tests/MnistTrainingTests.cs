using DevOnBike.Overfit.Layers;
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

        [Fact(Skip = "aaa")]
        public void Mnist_FullTrain60k_CnnBeastMode()
        {
            // --- ARRANGE ---
            // PEŁNY ZBIÓR DANYCH
            var trainSize = 60000; 
            var batchSize = 64; // Zwiększamy batch dla lepszego wykorzystania SIMD
            var epochs = 5;
            var learningRate = 0.001;

            Debug.WriteLine("Ładowanie pełnego zbioru 60,000 obrazów...");
            var (trainX, trainY) = MnistLoader.Load(
                "d:/ml/train-images.idx3-ubyte", 
                "d:/ml/train-labels.idx1-ubyte", 
                trainSize);

            using var X = new Tensor(trainX, requiresGrad: false);
            using var Y = new Tensor(trainY, requiresGrad: false);

            // ARCHITEKTURA: Conv(8x3x3) -> ReLU -> MaxPool(2x2) -> Dropout(0.25) -> Linear(1352 -> 10)
            var conv1 = new ConvLayer(inChannels: 1, outChannels: 8, h: 28, w: 28, kSize: 3);
            var fc1 = new LinearLayer(1352, 10);

            var allParams = new[] { conv1.Kernels }.Concat(fc1.Parameters());
            var optimizer = new Adam(allParams, learningRate);
            
            var numBatches = trainSize / batchSize;
            var totalSw = Stopwatch.StartNew();

            Debug.WriteLine($"Start treningu: {epochs} epok, {numBatches} batchy na epokę.");
            Debug.WriteLine("----------------------------------------------------------");

            // --- ACT ---
            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                var epochSw = Stopwatch.StartNew();
                double epochLoss = 0;

                for (var b = 0; b < numBatches; b++)
                {
                    optimizer.ZeroGrad();

                    // Slicing i materializacja batcha
                    var xView = X.Data.AsView().Slice(b * batchSize, 0, batchSize, 784);
                    var yView = Y.Data.AsView().Slice(b * batchSize, 0, batchSize, 10);

                    using var xBatch = new Tensor(xView.ToContiguousFastMatrix(), false);
                    using var yBatch = new Tensor(yView.ToContiguousFastMatrix(), false);

                    // --- FORWARD ---
                    using var h1 = conv1.Forward(xBatch);
                    using var a1 = TensorMath.ReLU(h1);
                    using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);
                    
                    // DROPOUT: 25% neuronów "odpoczywa" w tej iteracji
                    using var d1 = TensorMath.Dropout(p1, 0.25, isTraining: true);
                    
                    using var prediction = fc1.Forward(d1);

                    // LOSS
                    using var loss = TensorMath.MSE(prediction, yBatch);
                    epochLoss += loss.Data[0, 0];

                    // --- BACKWARD ---
                    loss.Grad[0, 0] = 1.0;
                    loss.Backward();

                    // UPDATE
                    optimizer.Step();

                    // Opcjonalnie: Log co 200 batchy, żeby widzieć progres
                    if (b % 200 == 0 && b > 0)
                    {
                        Debug.WriteLine($"  Batch {b}/{numBatches} | Current Loss: {loss.Data[0,0]:F6}");
                    }
                }

                epochSw.Stop();
                var avgLoss = epochLoss / numBatches;
                Debug.WriteLine($"> EPOCH {epoch} GOTOWA | Loss: {avgLoss:F6} | Czas: {epochSw.ElapsedMilliseconds}ms");
            }

            totalSw.Stop();

            // 1. Ładujemy dane testowe (te, których sieć NIE WIDZIAŁA podczas treningu)
            var (testX, testY) = MnistLoader.Load(
                "d:/ml/t10k-images.idx3-ubyte",
                "d:/ml/t10k-labels.idx1-ubyte",
                1000);

            // 2. Odpalamy diagnostykę
            PrintConfusionMatrix(conv1, fc1, testX, testY);

            // --- SUMMARY ---
            Debug.WriteLine("----------------------------------------------------------");
            Debug.WriteLine($"TRENING ZAKOŃCZONY!");
            Debug.WriteLine($"Całkowity czas: {totalSw.Elapsed.TotalSeconds:F2}s");
            Debug.WriteLine($"Średni czas na epokę: {totalSw.ElapsedMilliseconds / epochs}ms");

            // --- ASSERT ---
            Assert.True(totalSw.Elapsed.TotalSeconds < 120, "Bestia zbyt wolna! Miało być poniżej 2 minut.");
        }
        
        [Fact(Skip = "aaa")]
        public void Mnist_FullTrainingLoop_ShouldConvergeAndPersist()
        {
            // --- ARRANGE ---
            var trainSize = 5000;
            var batchSize = 64;
            var epochs = 5;
            var learningRate = 0.1;
            var modelPathPrefix = "mnist_model_v1";

            // Ścieżki do plików MNIST
            var (trainX, trainY) = MnistLoader.Load("d:/ml/train-images.idx3-ubyte", "d:/ml/train-labels.idx1-ubyte", trainSize);

            using var X = new Tensor(trainX, requiresGrad: false);
            using var Y = new Tensor(trainY, requiresGrad: false);

            var layer1 = new LinearLayer(784, 128);
            var layer2 = new LinearLayer(128, 10);

            // var optimizer = new SGD(layer1.Parameters().Concat(layer2.Parameters()), learningRate);
            var optimizer = new Adam(layer1.Parameters().Concat(layer2.Parameters()), learningRate: 0.001);

            var numBatches = trainSize / batchSize;
            double initialLoss = 0;
            double finalLoss = 0;

            var sw = Stopwatch.StartNew();

            // --- ACT: Training ---
            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                double epochLoss = 0;

                for (var b = 0; b < numBatches; b++)
                {
                    optimizer.ZeroGrad();

                    // Slicing i materializacja[cite: 5]
                    var xView = X.Data.AsView().Slice(b * batchSize, 0, batchSize, 784);
                    var yView = Y.Data.AsView().Slice(b * batchSize, 0, batchSize, 10);

                    using var xBatch = new Tensor(xView.ToContiguousFastMatrix(), requiresGrad: false);
                    using var yBatch = new Tensor(yView.ToContiguousFastMatrix(), requiresGrad: false);

                    // Forward Pass[cite: 5]
                    using var h1 = layer1.Forward(xBatch);
                    using var a1 = TensorMath.ReLU(h1);
                    using var prediction = layer2.Forward(a1);

                    // Loss & Backward[cite: 5]
                    using var loss = TensorMath.MSE(prediction, yBatch);
                    epochLoss += loss.Data[0, 0];

                    loss.Grad[0, 0] = 1.0;
                    loss.Backward();

                    optimizer.Step();
                }

                epochLoss /= numBatches;
                if (epoch == 1) initialLoss = epochLoss;
                finalLoss = epochLoss;

                Debug.WriteLine($"Epoch {epoch} | Loss: {epochLoss:F6}");
            }

            // --- ACT: Save and Load (Weryfikacja Serializacji) ---
            layer1.Save($"{modelPathPrefix}_l1");
            layer2.Save($"{modelPathPrefix}_l2");

            // Tworzymy nowe warstwy ("czyste") i ładujemy do nich wagi
            var loadedL1 = new LinearLayer(784, 128);
            var loadedL2 = new LinearLayer(128, 10);
            loadedL1.Load($"{modelPathPrefix}_l1");
            loadedL2.Load($"{modelPathPrefix}_l2");

            sw.Stop();

            // --- ASSERT ---

            // 1. Sprawdzamy czy błąd spadł[cite: 5]
            Assert.True(finalLoss < initialLoss, $"Loss did not decrease. Initial: {initialLoss}, Final: {finalLoss}");

            // 2. Weryfikacja bitowa modelu po wczytaniu
            // Pobieramy jeden testowy batch
            var testXView = X.Data.AsView().Slice(0, 0, 1, 784);
            using var testX = new Tensor(testXView.ToContiguousFastMatrix(), false);

            // Predykcja starym modelem
            using var oldH = layer1.Forward(testX);
            using var oldA = TensorMath.ReLU(oldH);
            using var oldPred = layer2.Forward(oldA);

            // Predykcja wczytanym modelem
            using var newH = loadedL1.Forward(testX);
            using var newA = TensorMath.ReLU(newH);
            using var newPred = loadedL2.Forward(newA);

            // Sprawdzamy, czy wyniki są identyczne (Precision = 10 miejsc po przecinku)
            for (var i = 0; i < 10; i++)
            {
                Assert.Equal(oldPred.Data[0, i], newPred.Data[0, i], 10);
            }

            Debug.WriteLine($"Training & Persistence check finished in {sw.ElapsedMilliseconds}ms");
        }

        private void PrintConfusionMatrix(ConvLayer conv, LinearLayer fc, FastMatrix<double> testX, FastMatrix<double> testY)
        {
            var matrix = new int[10, 10];
            var samples = 1000;

            // Zbieranie danych
            for (var i = 0; i < samples; i++)
            {
                var rowView = testX.AsView().Slice(i, 0, 1, 784);
                using var input = new Tensor(rowView.ToContiguousFastMatrix(), false);

                using var h1 = conv.Forward(input);
                using var a1 = TensorMath.ReLU(h1);
                using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);
                using var output = fc.Forward(p1);

                var predicted = output.Data.ArgMax();
                var actual = testY.ArgMax(i);
                matrix[actual, predicted]++;
            }

            // Budowanie tabeli w pamięci (xUnit woli jeden duży string lub linia po linii)
            var sb = new System.Text.StringBuilder();
            sb.AppendLine("\n=== MACIERZ POMYŁEK (CONFUSION MATRIX) ===");
            sb.AppendLine("Oś pionowa: Prawda | Oś pozioma: Predykcja\n");

            // Nagłówek kolumn
            sb.Append("A\\P |");
            for (var i = 0; i < 10; i++) sb.Append($"{i,4}|");
            sb.AppendLine("\n" + new string('-', 55));

            // Wiersze
            for (var r = 0; r < 10; r++)
            {
                sb.Append($"{r,3} |");
                for (var c = 0; c < 10; c++)
                {
                    var val = matrix[r, c];
                    if (r != c && val > 0)
                    {
                        // Błędy oznaczamy klamrami, skoro nie mamy kolorów
                        sb.Append($"[{val,2}]|");
                    }
                    else
                    {
                        sb.Append($"{val,4}|");
                    }
                }
                sb.AppendLine();
            }

            // FINALNY WYDRUK DO XUNIT
            _output.WriteLine(sb.ToString());
        }
    }
}