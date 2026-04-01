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
            // Inicjalizacja globalnej Taśmy dla każdego testu w tej klasie
            ComputationGraph.Active = new ComputationGraph();
        }

        public void Dispose()
        {
            // Bezpieczne zwolnienie referencji po teście
            ComputationGraph.Active = null;
        }

        [Fact(Skip = "aa")]
        public void Mnist_FullTrain60k_CnnBeastMode_Augmented()
        {
            // --- ARRANGE ---
            var trainSize = 60000;
            var batchSize = 64;
            var epochs = 10;
            var learningRate = 0.001;

            _output.WriteLine("=== START: Trening na Taśmie (Computation Graph) z Augmentacją ===");
            _output.WriteLine("Ładowanie pełnego zbioru 60,000 obrazów...");
            var (trainX, trainY) = MnistLoader.Load(
                "d:/ml/train-images.idx3-ubyte",
                "d:/ml/train-labels.idx1-ubyte",
                trainSize);

            using var X = new AutogradNode(trainX, requiresGrad: false);
            using var Y = new AutogradNode(trainY, requiresGrad: false);

            var conv1 = new ConvLayer(inChannels: 1, outChannels: 8, h: 28, w: 28, kSize: 3);
            var bn1 = new BatchNorm1D(1352);
            var fc1 = new LinearLayer(1352, 10);

            var model = new Sequential(conv1, bn1, fc1);

            var optimizer = new Adam(model.Parameters(), learningRate);
            var scheduler = new LRScheduler(optimizer, _output.WriteLine, factor: 0.5, patience: 1);

            var numBatches = trainSize / batchSize;
            var totalSw = Stopwatch.StartNew();

            _output.WriteLine($"Start treningu: {epochs} epok, {numBatches} batchy na epokę.");
            _output.WriteLine("----------------------------------------------------------");

            // --- ACT ---
            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                var epochSw = Stopwatch.StartNew();
                double epochLoss = 0;

                model.Train();

                for (var b = 0; b < numBatches; b++)
                {
                    // KLUCZOWE: Reset Taśmy na starcie batcha
                    ComputationGraph.Active.Reset();
                    optimizer.ZeroGrad();

                    var xView = X.Data.AsView().Slice(b * batchSize, 0, batchSize, 784);
                    using var tempMatrix = xView.ToContiguousFastMatrix();
                    var mutatedMatrix = DataAugmenter.AugmentBatch(tempMatrix, width: 28, height: 28);
                    using var xBatch = new AutogradNode(mutatedMatrix, requiresGrad: false);

                    var yView = Y.Data.AsView().Slice(b * batchSize, 0, batchSize, 10);
                    using var yBatch = new AutogradNode(yView.ToContiguousFastMatrix(), requiresGrad: false);

                    // --- FORWARD ---
                    using var h1 = conv1.Forward(xBatch);
                    using var a1 = TensorMath.ReLU(h1);
                    using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);
                    using var bnOut = bn1.Forward(p1);
                    using var d1 = TensorMath.Dropout(bnOut, 0.25, isTraining: model.IsTraining);
                    using var predictionLogits = fc1.Forward(d1);

                    // --- LOSS ---
                    using var loss = TensorMath.SoftmaxCrossEntropy(predictionLogits, yBatch);
                    epochLoss += loss.Data[0, 0];

                    // --- BACKWARD ---
                    ComputationGraph.Active.Backward(loss);

                    // --- UPDATE ---
                    optimizer.Step();
                }

                epochSw.Stop();
                var avgLoss = epochLoss / numBatches;
                _output.WriteLine($"> EPOCH {epoch} GOTOWA | Loss: {avgLoss:F6} | LR: {optimizer.LearningRate:F6} | Czas: {epochSw.ElapsedMilliseconds}ms");

                scheduler.Step(avgLoss);
            }

            totalSw.Stop();

            var (testX, testY) = MnistLoader.Load(
                "d:/ml/t10k-images.idx3-ubyte",
                "d:/ml/t10k-labels.idx1-ubyte",
                1000);

            PrintConfusionMatrix(model, testX, testY);

            _output.WriteLine("----------------------------------------------------------");
            _output.WriteLine($"TRENING AUGMENTOWANY ZAKOŃCZONY!");
            _output.WriteLine($"Całkowity czas: {totalSw.Elapsed.TotalSeconds:F2}s");

            var exportPath = @"d:\ml\bestia.bin";
            using (var fs = new FileStream(exportPath, FileMode.Create))
            using (var bw = new BinaryWriter(fs))
            {
                model.Save(bw);
            }
            _output.WriteLine($"Model wyeksportowany do: {exportPath}");

            model.Dispose();
        }

        [Fact]
        public void Mnist_FullTrainingLoop_ShouldConvergeAndPersist()
        {
            // --- ARRANGE ---
            var trainSize = 5000;
            var batchSize = 64;
            var epochs = 5;
            var learningRate = 0.001;
            var modelPath = "mnist_model_resnet.bin";

            var (trainX, trainY) = MnistLoader.Load("d:/ml/train-images.idx3-ubyte", "d:/ml/train-labels.idx1-ubyte", trainSize);

            using var X = new AutogradNode(trainX, requiresGrad: false);
            using var Y = new AutogradNode(trainY, requiresGrad: false);

            var layer1 = new LinearLayer(784, 128);
            var bn1 = new BatchNorm1D(128);
            var resBlock1 = new ResidualBlock(128);
            var resBlock2 = new ResidualBlock(128);
            var layerOut = new LinearLayer(128, 10);

            var model = new Sequential(layer1, bn1, resBlock1, resBlock2, layerOut);
            var optimizer = new Adam(model.Parameters(), learningRate);

            var numBatches = trainSize / batchSize;
            double initialLoss = 0;
            double finalLoss = 0;

            var sw = Stopwatch.StartNew();

            // --- ACT: Training ---
            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                double epochLoss = 0;
                model.Train();

                for (var b = 0; b < numBatches; b++)
                {
                    // Reset Taśmy!
                    ComputationGraph.Active.Reset();
                    optimizer.ZeroGrad();

                    var xView = X.Data.AsView().Slice(b * batchSize, 0, batchSize, 784);
                    var yView = Y.Data.AsView().Slice(b * batchSize, 0, batchSize, 10);

                    using var xBatch = new AutogradNode(xView.ToContiguousFastMatrix(), requiresGrad: false);
                    using var yBatch = new AutogradNode(yView.ToContiguousFastMatrix(), requiresGrad: false);

                    // --- FORWARD ---
                    using var h1 = layer1.Forward(xBatch);
                    using var bnOut = bn1.Forward(h1);
                    using var a1 = TensorMath.ReLU(bnOut);

                    using var res1Out = resBlock1.Forward(a1);
                    using var res2Out = resBlock2.Forward(res1Out);

                    using var predictionLogits = layerOut.Forward(res2Out);

                    // --- LOSS ---
                    using var loss = TensorMath.SoftmaxCrossEntropy(predictionLogits, yBatch);
                    epochLoss += loss.Data[0, 0];

                    // --- BACKWARD ---
                    ComputationGraph.Active.Backward(loss);
                    optimizer.Step();
                }

                epochLoss /= numBatches;
                if (epoch == 1) initialLoss = epochLoss;
                finalLoss = epochLoss;

                Debug.WriteLine($"Epoch {epoch} | Loss: {epochLoss:F6}");
            }

            // --- ZAPIS MODELU ---
            using (var fs = new FileStream(modelPath, FileMode.Create))
            using (var bw = new BinaryWriter(fs))
            {
                model.Save(bw);
            }

            // --- ODTWORZENIE MODELU ---
            var loadedModel = new Sequential(
                new LinearLayer(784, 128),
                new BatchNorm1D(128),
                new ResidualBlock(128),
                new ResidualBlock(128),
                new LinearLayer(128, 10)
            );

            using (var fs = new FileStream(modelPath, FileMode.Open))
            using (var br = new BinaryReader(fs))
            {
                loadedModel.Load(br);
            }

            sw.Stop();

            // --- ASSERT ---
            Assert.True(finalLoss < initialLoss, $"Loss did not decrease. Initial: {initialLoss}, Final: {finalLoss}");

            var testXView = X.Data.AsView().Slice(0, 0, 1, 784);
            using var testX = new AutogradNode(testXView.ToContiguousFastMatrix(), false);

            model.Eval();
            loadedModel.Eval();

            // Czysty forward ewaluacyjny wymaga resetu taśmy
            ComputationGraph.Active.Reset();
            using var oldPred = ManualForwardForTest(model, testX);

            ComputationGraph.Active.Reset();
            using var newPred = ManualForwardForTest(loadedModel, testX);

            for (var i = 0; i < 10; i++)
            {
                Assert.Equal(oldPred.Data[0, i], newPred.Data[0, i], 10);
            }

            model.Dispose();
            loadedModel.Dispose();

            Debug.WriteLine($"Training & Persistence check finished in {sw.ElapsedMilliseconds}ms");
        }

        private AutogradNode ManualForwardForTest(Sequential model, AutogradNode input)
        {
            var fieldInfo = typeof(Sequential).GetField("_modules", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var modules = (System.Collections.Generic.List<IModule>)fieldInfo.GetValue(model);

            var l1 = (LinearLayer)modules[0];
            var bn1 = (BatchNorm1D)modules[1];
            var r1 = (ResidualBlock)modules[2];
            var r2 = (ResidualBlock)modules[3];
            var outL = (LinearLayer)modules[4];

            var h = l1.Forward(input);
            var bn = bn1.Forward(h);
            var a = TensorMath.ReLU(bn);
            var res1 = r1.Forward(a);
            var res2 = r2.Forward(res1);

            return outL.Forward(res2);
        }

        private void PrintConfusionMatrix(Sequential model, FastMatrix<double> testX, FastMatrix<double> testY)
        {
            var matrix = new int[10, 10];
            var samples = 1000;

            model.Eval();

            var modules = (System.Collections.Generic.List<IModule>)model.GetType().GetField("_modules", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(model);
            var conv1 = (ConvLayer)modules[0];
            var bn1 = (BatchNorm1D)modules[1];
            var fc1 = (LinearLayer)modules[2];

            for (var i = 0; i < samples; i++)
            {
                // Nawet przy Eval, graf nagrywa wagi. Należy zresetować licznik na Taśmie.
                ComputationGraph.Active.Reset();

                var rowView = testX.AsView().Slice(i, 0, 1, 784);
                using var input = new AutogradNode(rowView.ToContiguousFastMatrix(), false);

                using var h1 = conv1.Forward(input);
                using var a1 = TensorMath.ReLU(h1);
                using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);
                using var bnOut = bn1.Forward(p1);
                using var output = fc1.Forward(bnOut);

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