using System.Diagnostics;
using DevOnBike.Overfit.Layers;
using DevOnBike.Overfit.Optimizers;

namespace DevOnBike.Overfit.Tests
{
    public class MnistTrainingTests
    {
        [Fact(Skip = "This test requires a training model")]
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
            for (int i = 0; i < 10; i++)
            {
                Assert.Equal(oldPred.Data[0, i], newPred.Data[0, i], 10);
            }

            Debug.WriteLine($"Training & Persistence check finished in {sw.ElapsedMilliseconds}ms");
        }
    }
}