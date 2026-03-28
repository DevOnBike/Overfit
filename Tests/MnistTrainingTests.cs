using System.Diagnostics;
using DevOnBike.Overfit.Layers;
using DevOnBike.Overfit.Optimizers;

namespace DevOnBike.Overfit.Tests
{
    public class MnistTrainingTests
    {
        [Fact(Skip = "integration test")]
        public void Mnist_FullTrainingLoop_ShouldConverge()
        {
            // --- ARRANGE ---
            var trainSize = 5000; // Używamy 5k próbek, żeby test nie trwał wieków
            var batchSize = 64;
            var epochs = 5;
            var learningRate = 0.1;

            // Ładowanie plików (muszą być w folderze bin/Debug/net8.0/...)
            var (trainX, trainY) = MnistLoader.Load("d:/ml/train-images.idx3-ubyte", "d:/ml/train-labels.idx1-ubyte", trainSize);

            using var X = new Tensor(trainX, requiresGrad: false);
            using var Y = new Tensor(trainY, requiresGrad: false);

            // Sieć: 784 (wejście) -> 128 (ReLU) -> 10 (Softmax/Linear output)
            var layer1 = new LinearLayer(784, 128);
            var layer2 = new LinearLayer(128, 10);

            var optimizer = new SGD(layer1.Parameters().Concat(layer2.Parameters()), learningRate);
            
            var numBatches = trainSize / batchSize;
            double initialLoss = 0;
            double finalLoss = 0;

            var sw = Stopwatch.StartNew();

            // --- ACT ---
            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                double epochLoss = 0;

                for (var b = 0; b < numBatches; b++)
                {
                    optimizer.ZeroGrad();

                    // Slicing batcha (O(1)) i materializacja do Tensora (wymagane dla GEMM)
                    var xView = X.Data.AsView().Slice(b * batchSize, 0, batchSize, 784);
                    var yView = Y.Data.AsView().Slice(b * batchSize, 0, batchSize, 10);

                    using var xBatch = new Tensor(xView.ToContiguousFastMatrix(), requiresGrad: false);
                    using var yBatch = new Tensor(yView.ToContiguousFastMatrix(), requiresGrad: false);

                    // Forward
                    using var h1 = layer1.Forward(xBatch);
                    using var a1 = TensorMath.ReLU(h1);
                    using var prediction = layer2.Forward(a1);

                    // Loss
                    using var loss = TensorMath.MSE(prediction, yBatch);
                    epochLoss += loss.Data[0, 0];

                    // Backward
                    loss.Grad[0, 0] = 1.0;
                    loss.Backward();

                    // Step
                    optimizer.Step();
                }

                epochLoss /= numBatches;
                if (epoch == 1) initialLoss = epochLoss;
                finalLoss = epochLoss;

                Debug.WriteLine($"Epoch {epoch} | Loss: {epochLoss:F6}");
            }

            sw.Stop();

            // --- ASSERT ---
            // 1. Sprawdzamy czy błąd spadł (sieć się uczy)
            Assert.True(finalLoss < initialLoss, $"Loss did not decrease. Initial: {initialLoss}, Final: {finalLoss}");
            
            // 2. Wydajność: MNIST na 5k próbek/5 epok powinien zająć kilka sekund na nowoczesnym CPU
            Assert.True(sw.ElapsedMilliseconds < 15000, $"Training too slow: {sw.ElapsedMilliseconds}ms");
            
            Debug.WriteLine($"Training finished in {sw.ElapsedMilliseconds}ms");
        }
    }
}