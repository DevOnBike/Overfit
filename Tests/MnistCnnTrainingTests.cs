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
            var trainSize = 1000; 
            var batchSize = 32;
            var epochs = 3;
            var learningRate = 0.001;

            // Ścieżki do plików MNIST - upewnij się, że są poprawne!
            var (trainX, trainY) = MnistLoader.Load("d:/ml/train-images.idx3-ubyte", "d:/ml/train-labels.idx1-ubyte", trainSize);

            using var X = new Tensor(trainX, requiresGrad: false);
            using var Y = new Tensor(trainY, requiresGrad: false);

            // 1. Warstwa splotowa: 1 kanał wej, 8 filtrów, filtr 3x3
            var conv1 = new ConvLayer(inChannels: 1, outChannels: 8, h: 28, w: 28, kSize: 3);
            
            // 2. Warstwa liniowa: 
            // Po splocie: 28-3+1 = 26. Po poolingu 2x2: 26/2 = 13.
            // Łącznie cech: 8 filtrów * 13 * 13 = 1352.
            var fc1 = new LinearLayer(1352, 10);

            // Łączymy parametry dla Adama (rozwiązanie problemu Yield)
            var allParameters = new[] { conv1.Kernels }.Concat(fc1.Parameters());
            var optimizer = new Adam(allParameters, learningRate);
            
            var numBatches = trainSize / batchSize;
            var sw = Stopwatch.StartNew();
            double finalLoss = 0;

            // --- ACT ---
            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                double epochLoss = 0;

                for (var b = 0; b < numBatches; b++)
                {
                    optimizer.ZeroGrad();

                    // Slicing batcha
                    var xView = X.Data.AsView().Slice(b * batchSize, 0, batchSize, 784);
                    var yView = Y.Data.AsView().Slice(b * batchSize, 0, batchSize, 10);

                    using var xBatch = new Tensor(xView.ToContiguousFastMatrix(), false);
                    using var yBatch = new Tensor(yView.ToContiguousFastMatrix(), false);

                    // --- FORWARD PASS ---
                    // Conv: 28x28 -> 26x26 (8 kanałów)
                    using var h1 = conv1.Forward(xBatch);
                    using var a1 = TensorMath.ReLU(h1);
                    
                    // Pool: 26x26 -> 13x13 (8 kanałów)
                    using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);
                    
                    // Linear: 1352 -> 10
                    using var prediction = fc1.Forward(p1);

                    // Loss
                    using var loss = TensorMath.MSE(prediction, yBatch);
                    epochLoss += loss.Data[0, 0];

                    // --- BACKWARD PASS ---
                    loss.Grad[0, 0] = 1.0;
                    loss.Backward();

                    // Update wag
                    optimizer.Step();
                }

                finalLoss = epochLoss / numBatches;
                Debug.WriteLine($"Epoch {epoch} | Loss: {finalLoss:F6}");
            }

            sw.Stop();
            Debug.WriteLine($"CNN + Pooling Training finished in {sw.ElapsedMilliseconds}ms");

            // --- ASSERT ---
            // 1. Sprawdzamy czy czas jest sensowny (powinien być zbliżony do 2,5 - 3s)
            Assert.True(sw.ElapsedMilliseconds < 10000, "Trening CNN jest zbyt wolny!");
            
            // 2. Sprawdzamy czy sieć faktycznie zbiła błąd
            Assert.True(finalLoss < 0.1, $"Model nie zbiega poprawnie. Loss: {finalLoss}");
        }
    }
}