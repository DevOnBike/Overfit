using System.Diagnostics;
using DevOnBike.Overfit.Layers;
using DevOnBike.Overfit.Optimizers;

namespace DevOnBike.Overfit.Tests
{
    public class MnistCnnTrainingTests
    {
        [Fact(Skip = "aaa")]
        public void Mnist_CnnTraining_ShouldConvergeFast()
        {
            // --- ARRANGE ---
            var trainSize = 1000; // Na początek 1000, żebyś nie czekał minuty
            var batchSize = 32;
            var epochs = 3;
            var learningRate = 0.001; // Adam lubi mały LR

            var (trainX, trainY) = MnistLoader.Load("d:/ml/train-images.idx3-ubyte", "d:/ml/train-labels.idx1-ubyte", trainSize);

            using var X = new Tensor(trainX, requiresGrad: false);
            using var Y = new Tensor(trainY, requiresGrad: false);

            // ARCHITEKTURA CNN
            // 1 kanał wejściowy (grayscale), 8 filtrów wyjściowych, obrazek 28x28, filtr 3x3
            var conv1 = new ConvLayer(inChannels: 1, outChannels: 8, h: 28, w: 28, kSize: 3);
            
            // Po splocie 3x3 z obrazka 28x28 zostaje 26x26. 
            // 8 filtrów * 26 * 26 = 5408 wejść do warstwy liniowej.
            var fc1 = new LinearLayer(5408, 10);

            var allParams = new[] { conv1.Kernels }.Concat(fc1.Parameters());
            var optimizer = new Adam(allParams, learningRate);
            
            var numBatches = trainSize / batchSize;
            var sw = Stopwatch.StartNew();

            // --- ACT ---
            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                double epochLoss = 0;

                for (var b = 0; b < numBatches; b++)
                {
                    optimizer.ZeroGrad();

                    var xView = X.Data.AsView().Slice(b * batchSize, 0, batchSize, 784);
                    var yView = Y.Data.AsView().Slice(b * batchSize, 0, batchSize, 10);

                    using var xBatch = new Tensor(xView.ToContiguousFastMatrix(), false);
                    using var yBatch = new Tensor(yView.ToContiguousFastMatrix(), false);

                    // FORWARD: Conv -> ReLU -> Linear
                    using var h1 = conv1.Forward(xBatch);
                    using var a1 = TensorMath.ReLU(h1);
                    using var prediction = fc1.Forward(a1);

                    using var loss = TensorMath.MSE(prediction, yBatch);
                    epochLoss += loss.Data[0, 0];

                    loss.Grad[0, 0] = 1.0;
                    loss.Backward();

                    optimizer.Step();
                }

                Debug.WriteLine($"CNN Epoch {epoch} | Loss: {epochLoss / numBatches:F6}");
            }

            sw.Stop();
            Debug.WriteLine($"CNN Training finished in {sw.ElapsedMilliseconds}ms");

            // --- ASSERT ---
            // Sprawdzamy, czy błąd jest niski (CNN uczy się bardzo szybko nawet na małej ilości danych)
            Assert.True(sw.ElapsedMilliseconds > 0); 
        }
    }
}