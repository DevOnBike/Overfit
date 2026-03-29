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
            var trainSize = 60000;
            var batchSize = 64;
            var epochs = 5;
            var learningRate = 0.001;

            Debug.WriteLine("Ładowanie pełnego zbioru 60,000 obrazów...");
            var (trainX, trainY) = MnistLoader.Load(
                "d:/ml/train-images.idx3-ubyte",
                "d:/ml/train-labels.idx1-ubyte",
                trainSize);

            using var X = new Tensor(trainX, requiresGrad: false);
            using var Y = new Tensor(trainY, requiresGrad: false);

            var conv1 = new ConvLayer(inChannels: 1, outChannels: 8, h: 28, w: 28, kSize: 3);

            // Wpinamy BatchNorm po Poolingu (rozmiar spłaszczony to 8 * 13 * 13 = 1352)
            var bn1 = new BatchNorm1D(1352);

            var fc1 = new LinearLayer(1352, 10);

            // Rejestrujemy parametry ze wszystkich warstw do optymalizatora!
            var allParams = new[] { conv1.Kernels }.Concat(bn1.Parameters()).Concat(fc1.Parameters());
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

                    var xView = X.Data.AsView().Slice(b * batchSize, 0, batchSize, 784);
                    var yView = Y.Data.AsView().Slice(b * batchSize, 0, batchSize, 10);

                    using var xBatch = new Tensor(xView.ToContiguousFastMatrix(), false);
                    using var yBatch = new Tensor(yView.ToContiguousFastMatrix(), false);

                    // --- FORWARD ---
                    using var h1 = conv1.Forward(xBatch);
                    using var a1 = TensorMath.ReLU(h1);
                    using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);

                    // Przepuszczamy przez BatchNorm z flagą isTraining: true
                    using var bnOut = bn1.Forward(p1, isTraining: true);

                    using var d1 = TensorMath.Dropout(bnOut, 0.25, isTraining: true);
                    using var predictionLogits = fc1.Forward(d1); // Uwaga: To są surowe logity!

                    // --- LOSS: Wdrożenie SoftmaxCrossEntropy ---
                    using var loss = TensorMath.SoftmaxCrossEntropy(predictionLogits, yBatch);
                    epochLoss += loss.Data[0, 0];

                    // --- BACKWARD ---
                    loss.Grad[0, 0] = 1.0;
                    loss.Backward();

                    // --- UPDATE ---
                    optimizer.Step();

                    if (b % 200 == 0 && b > 0)
                    {
                        Debug.WriteLine($"  Batch {b}/{numBatches} | Current Loss: {loss.Data[0, 0]:F6}");
                    }
                }

                epochSw.Stop();
                var avgLoss = epochLoss / numBatches;
                Debug.WriteLine($"> EPOCH {epoch} GOTOWA | Loss: {avgLoss:F6} | Czas: {epochSw.ElapsedMilliseconds}ms");
            }

            totalSw.Stop();

            var (testX, testY) = MnistLoader.Load(
                "d:/ml/t10k-images.idx3-ubyte",
                "d:/ml/t10k-labels.idx1-ubyte",
                1000);

            // Przekazujemy warstwę BatchNorm do ewaluacji!
            PrintConfusionMatrix(conv1, bn1, fc1, testX, testY);

            Debug.WriteLine("----------------------------------------------------------");
            Debug.WriteLine($"TRENING ZAKOŃCZONY!");
            Debug.WriteLine($"Całkowity czas: {totalSw.Elapsed.TotalSeconds:F2}s");
            Debug.WriteLine($"Średni czas na epokę: {totalSw.ElapsedMilliseconds / epochs}ms");

            Assert.True(totalSw.Elapsed.TotalSeconds < 120, "Bestia zbyt wolna! Miało być poniżej 2 minut.");
        }

        [Fact(Skip = "aaa")]
        public void Mnist_FullTrain60k_CnnBeastModeWithJanitor()
        {
            // --- ARRANGE ---
            var trainSize = 10000;
            var batchSize = 32; // Mniejszy batch = mniejszy nacisk na RAM
            var epochs = 2;
            var learningRate = 0.001;

            Debug.WriteLine("Inicjalizacja GigaBestii ResNet (60k)...");
            var (trainX, trainY) = MnistLoader.Load(
                "d:/ml/train-images.idx3-ubyte",
                "d:/ml/train-labels.idx1-ubyte",
                trainSize);

            using var X = new Tensor(trainX, requiresGrad: false);
            using var Y = new Tensor(trainY, requiresGrad: false);

            // Architektura: Conv -> BN -> ReLU -> MaxPool -> ResNet x2 -> FC
            var conv1 = new ConvLayer(inChannels: 1, outChannels: 16, h: 28, w: 28, kSize: 3);
            var bn_conv = new BatchNorm1D(16 * 26 * 26);

            // 16 filtrów * 13x13 (po poolingu) = 2704
            var resBlock1 = new ResidualBlock(2704);
            var resBlock2 = new ResidualBlock(2704);
            var fcOut = new LinearLayer(2704, 10);

            // Zbieramy parametry i tworzymy HashSet dla błyskawicznego sprawdzania w pętli sprzątającej
            var allParams = conv1.Parameters()
                .Concat(bn_conv.Parameters())
                .Concat(resBlock1.Parameters())
                .Concat(resBlock2.Parameters())
                .Concat(fcOut.Parameters())
                .ToList();

            var allParamsSet = new HashSet<Tensor>(allParams);
            var optimizer = new Adam(allParams, learningRate);

            var numBatches = trainSize / batchSize;
            var totalSw = Stopwatch.StartNew();

            // --- ACT ---
            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                var epochSw = Stopwatch.StartNew();
                double epochLoss = 0;

                for (var b = 0; b < numBatches; b++)
                {
                    optimizer.ZeroGrad();

                    // 1. Przygotowanie Batcha (używamy using, by zwolnić macierz widoku)
                    var xView = X.Data.AsView().Slice(b * batchSize, 0, batchSize, 784);
                    var yView = Y.Data.AsView().Slice(b * batchSize, 0, batchSize, 10);

                    var xBatch = new Tensor(xView.ToContiguousFastMatrix(), false);
                    var yBatch = new Tensor(yView.ToContiguousFastMatrix(), false);

                    // 2. Forward Pass
                    var c1 = conv1.Forward(xBatch);
                    var bc1 = bn_conv.Forward(c1, isTraining: true);
                    var a1 = TensorMath.ReLU(bc1);
                    var p1 = TensorMath.MaxPool2D(a1, 16, 26, 26, 2);

                    var r1 = resBlock1.Forward(p1, isTraining: true);
                    var r2 = resBlock2.Forward(r1, isTraining: true);

                    var d1 = TensorMath.Dropout(r2, 0.2, isTraining: true);
                    var logits = fcOut.Forward(d1);

                    // 3. Loss & Backward
                    var loss = TensorMath.SoftmaxCrossEntropy(logits, yBatch);
                    epochLoss += loss.Data[0, 0];

                    // POBIERAMY LISTĘ WSZYSTKICH WĘZŁÓW GRAFU (wymaga poprawki w Tensor.cs!)
                    var graphNodes = loss.Backward();

                    // 4. Update Wag
                    optimizer.Step();

                    // 5. JANITOR MODE: Atomowe sprzątanie RAM
                    // Czyścimy wszystko poza wagami zapisanymi w optimizerze
                    foreach (var node in graphNodes)
                    {
                        if (!allParamsSet.Contains(node))
                        {
                            node.Dispose();
                        }
                    }

                    // Ręcznie czyścimy wejścia batcha, bo nie są w grafie Backward od loss
                    xBatch.Dispose();
                    yBatch.Dispose();
                    graphNodes.Clear();

                    if (b % 200 == 0 && b > 0)
                        Debug.WriteLine($"  Batch {b}/{numBatches} | Avg Loss: {epochLoss / (b + 1):F6}");
                }

                epochSw.Stop();
                Debug.WriteLine($"> EPOCH {epoch} | Loss: {epochLoss / numBatches:F6} | Time: {epochSw.ElapsedMilliseconds}ms");

                // Wymuś na GC zebranie drobnych obiektów C#
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }

            totalSw.Stop();

            // --- ZAPIS MODELU ---
            conv1.Save("d:/ml/beast_resnet_conv1.bin");
            bn_conv.Save("d:/ml/beast_resnet_bn.bin");
            resBlock1.Save("d:/ml/beast_resnet_r1");
            resBlock2.Save("d:/ml/beast_resnet_r2");
            fcOut.Save("d:/ml/beast_resnet_fc.bin");

            // --- EVALUATION ---
            var (testX, testY) = MnistLoader.Load("d:/ml/t10k-images.idx3-ubyte", "d:/ml/t10k-labels.idx1-ubyte", 1000);
            PrintResNetConfusionMatrix(conv1, bn_conv, resBlock1, resBlock2, fcOut, testX, testY);

            Debug.WriteLine($"TRENING ZAKOŃCZONY! Czas: {totalSw.Elapsed.TotalSeconds:F2}s");
        }

        [Fact]
        public void Mnist_FullTrainingLoop_ShouldConvergeAndPersist()
        {
            // --- ARRANGE ---
            var trainSize = 5000;
            var batchSize = 64;
            var epochs = 5;
            var learningRate = 0.001;
            var modelPathPrefix = "mnist_model_resnet";

            var (trainX, trainY) = MnistLoader.Load("d:/ml/train-images.idx3-ubyte", "d:/ml/train-labels.idx1-ubyte", trainSize);

            using var X = new Tensor(trainX, requiresGrad: false);
            using var Y = new Tensor(trainY, requiresGrad: false);

            // Architektura Głębokiego ResNetu (1D)
            var layer1 = new LinearLayer(784, 128);
            var bn1 = new BatchNorm1D(128);

            // Wpinamy nasze bloki resztkowe (autostrady dla gradientu)
            var resBlock1 = new ResidualBlock(128);
            var resBlock2 = new ResidualBlock(128);

            var layerOut = new LinearLayer(128, 10);

            // Rejestrujemy parametry ze wszystkich warstw i bloków
            var allParams = layer1.Parameters()
                .Concat(bn1.Parameters())
                .Concat(resBlock1.Parameters())
                .Concat(resBlock2.Parameters())
                .Concat(layerOut.Parameters());

            var optimizer = new Adam(allParams, learningRate);

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

                    var xView = X.Data.AsView().Slice(b * batchSize, 0, batchSize, 784);
                    var yView = Y.Data.AsView().Slice(b * batchSize, 0, batchSize, 10);

                    using var xBatch = new Tensor(xView.ToContiguousFastMatrix(), requiresGrad: false);
                    using var yBatch = new Tensor(yView.ToContiguousFastMatrix(), requiresGrad: false);

                    // --- FORWARD ---
                    using var h1 = layer1.Forward(xBatch);
                    using var bnOut = bn1.Forward(h1, isTraining: true);
                    using var a1 = TensorMath.ReLU(bnOut);

                    // Przepuszczamy przez głębokie bloki ResNet
                    using var res1Out = resBlock1.Forward(a1, isTraining: true);
                    using var res2Out = resBlock2.Forward(res1Out, isTraining: true);

                    // Wyjście do klasyfikacji
                    using var predictionLogits = layerOut.Forward(res2Out);

                    // --- LOSS: Wdrożenie SoftmaxCrossEntropy ---
                    using var loss = TensorMath.SoftmaxCrossEntropy(predictionLogits, yBatch);
                    epochLoss += loss.Data[0, 0];

                    // --- BACKWARD ---
                    loss.Grad[0, 0] = 1.0;
                    loss.Backward();

                    optimizer.Step();
                }

                epochLoss /= numBatches;
                if (epoch == 1) initialLoss = epochLoss;
                finalLoss = epochLoss;

                Debug.WriteLine($"Epoch {epoch} | Loss: {epochLoss:F6}");
            }

            // --- ZAPIS MODELU (Z nowymi blokami) ---
            layer1.Save($"{modelPathPrefix}_l1");
            bn1.Save($"{modelPathPrefix}_bn1");
            resBlock1.Save($"{modelPathPrefix}_res1");
            resBlock2.Save($"{modelPathPrefix}_res2");
            layerOut.Save($"{modelPathPrefix}_out");

            // --- ODTWORZENIE MODELU Z PLIKÓW ---
            var loadedL1 = new LinearLayer(784, 128);
            var loadedBn1 = new BatchNorm1D(128);
            var loadedRes1 = new ResidualBlock(128);
            var loadedRes2 = new ResidualBlock(128);
            var loadedOut = new LinearLayer(128, 10);

            loadedL1.Load($"{modelPathPrefix}_l1");
            loadedBn1.Load($"{modelPathPrefix}_bn1");
            loadedRes1.Load($"{modelPathPrefix}_res1");
            loadedRes2.Load($"{modelPathPrefix}_res2");
            loadedOut.Load($"{modelPathPrefix}_out");

            sw.Stop();

            // --- ASSERT ---
            Assert.True(finalLoss < initialLoss, $"Loss did not decrease. Initial: {initialLoss}, Final: {finalLoss}");

            var testXView = X.Data.AsView().Slice(0, 0, 1, 784);
            using var testX = new Tensor(testXView.ToContiguousFastMatrix(), false);

            // Forward starego modelu (Inference)
            using var oldH = layer1.Forward(testX);
            using var oldBn = bn1.Forward(oldH, isTraining: false);
            using var oldA = TensorMath.ReLU(oldBn);
            using var oldRes1 = resBlock1.Forward(oldA, isTraining: false);
            using var oldRes2 = resBlock2.Forward(oldRes1, isTraining: false);
            using var oldPred = layerOut.Forward(oldRes2);

            // Forward nowego (wczytanego z dysku) modelu (Inference)
            using var newH = loadedL1.Forward(testX);
            using var newBn = loadedBn1.Forward(newH, isTraining: false);
            using var newA = TensorMath.ReLU(newBn);
            using var newRes1 = loadedRes1.Forward(newA, isTraining: false);
            using var newRes2 = loadedRes2.Forward(newRes1, isTraining: false);
            using var newPred = loadedOut.Forward(newRes2);

            // Sprawdzamy czy predykcje się zgadzają z dokładnością do 10 miejsc po przecinku
            for (var i = 0; i < 10; i++)
            {
                Assert.Equal(oldPred.Data[0, i], newPred.Data[0, i], 10);
            }

            Debug.WriteLine($"Training & Persistence check finished in {sw.ElapsedMilliseconds}ms");
        }

        private void PrintResNetConfusionMatrix(ConvLayer conv, BatchNorm1D bn, ResidualBlock r1, ResidualBlock r2, LinearLayer fc, FastMatrix<double> testX, FastMatrix<double> testY)
        {
            var matrix = new int[10, 10];
            for (var i = 0; i < 1000; i++)
            {
                var rowView = testX.AsView().Slice(i, 0, 1, 784);
                using var input = new Tensor(rowView.ToContiguousFastMatrix(), false);

                using var c1 = conv.Forward(input);
                using var bc = bn.Forward(c1, isTraining: false);
                using var a1 = TensorMath.ReLU(bc);
                using var p1 = TensorMath.MaxPool2D(a1, 16, 26, 26, 2);
                using var res1 = r1.Forward(p1, isTraining: false);
                using var res2 = r2.Forward(res1, isTraining: false);
                using var output = fc.Forward(res2);

                matrix[testY.ArgMax(i), output.Data.ArgMax()]++;
            }

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("\n=== MACIERZ POMYŁEK (GIGABESTIA) ===");
            for (var r = 0; r < 10; r++)
            {
                sb.Append($"{r} |");
                for (var c = 0; c < 10; c++) sb.Append($"{matrix[r, c],4}|");
                sb.AppendLine();
            }
            _output.WriteLine(sb.ToString());
        }

        private void PrintConfusionMatrix(ConvLayer conv, BatchNorm1D bn, LinearLayer fc, FastMatrix<double> testX, FastMatrix<double> testY)
        {
            var matrix = new int[10, 10];
            var samples = 1000;

            for (var i = 0; i < samples; i++)
            {
                var rowView = testX.AsView().Slice(i, 0, 1, 784);
                using var input = new Tensor(rowView.ToContiguousFastMatrix(), false);

                using var h1 = conv.Forward(input);
                using var a1 = TensorMath.ReLU(h1);
                using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);
                
                // UWAGA: Podczas wnioskowania isTraining JEST FALSE!
                using var bnOut = bn.Forward(p1, isTraining: false);
                
                using var output = fc.Forward(bnOut);

                // Z uwagi na monotoniczność funkcji Softmax, używamy ArgMax bezpośrednio na logitach
                var predicted = output.Data.ArgMax();
                var actual = testY.ArgMax(i);
                matrix[actual, predicted]++;
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
                    if (r != c && val > 0)
                    {
                        sb.Append($"[{val,2}]|");
                    }
                    else
                    {
                        sb.Append($"{val,4}|");
                    }
                }
                sb.AppendLine();
            }

            _output.WriteLine(sb.ToString());
        }
    }
}