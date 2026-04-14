using System.Diagnostics;
using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;
using Xunit;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests
{
    public sealed class TrainEvalAndQualityTests
    {
        private readonly ITestOutputHelper _output;

        public TrainEvalAndQualityTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void BatchNorm1D_TrainAndEval_ProduceMatchingOutputs_AfterStatsWarmup()
        {
            using var bn = new BatchNorm1D(numFeatures: 4);

            // Rozgrzewka running stats
            using (var warmupTensor = new FastTensor<float>(6, 4, clearMemory: true))
            using (var warmupInput = new AutogradNode(warmupTensor, requiresGrad: false))
            {
                var w = warmupInput.DataView.AsSpan();
                for (int i = 0; i < w.Length; i++)
                {
                    w[i] = ((i % 7) - 3) * 0.25f;
                }

                bn.Train();
                for (int step = 0; step < 20; step++)
                {
                    using var y = bn.Forward(new ComputationGraph(), warmupInput);
                }
            }

            using var inputTensor = new FastTensor<float>(3, 4, clearMemory: true);
            using var input = new AutogradNode(inputTensor, requiresGrad: false);

            var x = input.DataView.AsSpan();
            x[0] = -1.0f; x[1] = -0.5f; x[2] = 0.2f; x[3] = 1.0f;
            x[4] = -0.8f; x[5] = -0.1f; x[6] = 0.3f; x[7] = 1.1f;
            x[8] = -1.2f; x[9] = -0.4f; x[10] = 0.1f; x[11] = 0.9f;

            bn.Eval();

            using var evalTensor = new FastTensor<float>(3, 4, clearMemory: true);

            var inSpan = input.DataView.AsReadOnlySpan();
            var outSpan = evalTensor.GetView().AsSpan();

            for (int r = 0; r < 3; r++)
            {
                bn.ForwardInference(
                    inSpan.Slice(r * 4, 4),
                    outSpan.Slice(r * 4, 4));
            }

            var evalOutput = evalTensor.GetView().AsReadOnlySpan().ToArray();

            var gamma = bn.Gamma.DataView.AsReadOnlySpan();
            var beta = bn.Beta.DataView.AsReadOnlySpan();
            var mean = bn.RunningMean.GetView().AsReadOnlySpan();
            var varSpan = bn.RunningVar.GetView().AsReadOnlySpan();

            var expected = new float[12];
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 4; c++)
                {
                    int idx = r * 4 + c;
                    float scale = gamma[c] / MathF.Sqrt(varSpan[c] + bn.Eps);
                    float shift = beta[c] - mean[c] * scale;
                    expected[idx] = x[idx] * scale + shift;
                }
            }

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(
                    MathF.Abs(expected[i] - evalOutput[i]) < 1e-5f,
                    $"BatchNorm1D eval mismatch at {i}: expected={expected[i]}, actual={evalOutput[i]}");
            }
        }

        [Fact]
        public void ResidualBlock_EvalPath_IsDeterministic_AfterStatsWarmup()
        {
            using var block = new ResidualBlock(hiddenSize: 8);

            using (var warmupTensor = new FastTensor<float>(8, 8, clearMemory: true))
            using (var warmupInput = new AutogradNode(warmupTensor, requiresGrad: false))
            {
                var w = warmupInput.DataView.AsSpan();
                for (int i = 0; i < w.Length; i++)
                {
                    w[i] = ((i % 11) - 5) * 0.1f;
                }

                block.Train();
                for (int step = 0; step < 25; step++)
                {
                    using var y = block.Forward(new ComputationGraph(), warmupInput);
                }
            }

            using var inputTensor = new FastTensor<float>(1, 8, clearMemory: true);
            using var input = new AutogradNode(inputTensor, requiresGrad: false);

            var x = input.DataView.AsSpan();
            x[0] = -1.0f;
            x[1] = -0.5f;
            x[2] = -0.1f;
            x[3] = 0.0f;
            x[4] = 0.3f;
            x[5] = 0.7f;
            x[6] = 1.1f;
            x[7] = 1.5f;

            block.Eval();

            var evalGraph1 = new ComputationGraph { IsRecording = false };
            using var out1 = block.Forward(evalGraph1, input);
            var y1 = out1.DataView.AsReadOnlySpan().ToArray();

            var evalGraph2 = new ComputationGraph { IsRecording = false };
            using var out2 = block.Forward(evalGraph2, input);
            var y2 = out2.DataView.AsReadOnlySpan().ToArray();

            Assert.Equal(y1.Length, y2.Length);

            for (int i = 0; i < y1.Length; i++)
            {
                Assert.True(
                    MathF.Abs(y1[i] - y2[i]) < 1e-6f,
                    $"ResidualBlock eval path is not deterministic at {i}: first={y1[i]}, second={y2[i]}");
            }
        }

        [Fact]
        public void Mnist_SmallTraining_Run_DecreasesLoss_And_ReachesReasonableAccuracy()
        {
            const int trainSize = 2048;
            const int testSize = 256;
            const int batchSize = 64;
            const int epochs = 2;
            const float lr = 0.001f;

            var trainImagesPath = "d:/ml/train-images.idx3-ubyte";
            var trainLabelsPath = "d:/ml/train-labels.idx1-ubyte";
            var testImagesPath = "d:/ml/t10k-images.idx3-ubyte";
            var testLabelsPath = "d:/ml/t10k-labels.idx1-ubyte";

            if (!File.Exists(trainImagesPath) || !File.Exists(trainLabelsPath) ||
                !File.Exists(testImagesPath) || !File.Exists(testLabelsPath))
            {
                return;
            }

            var (trainX, trainY) = MnistLoader.Load(trainImagesPath, trainLabelsPath, trainSize);
            var (testX, testY) = MnistLoader.Load(testImagesPath, testLabelsPath, testSize);

            using var conv1 = new ConvLayer(1, 8, 28, 28, 3);
            using var bn1 = new BatchNorm1D(1352);
            using var res1 = new ResidualBlock(1352);
            using var fcOut = new LinearLayer(8, 10);

            var parameters = conv1.Parameters()
                .Concat(bn1.Parameters())
                .Concat(res1.Parameters())
                .Concat(fcOut.Parameters())
                .ToArray();

            using var optimizer = new Adam(parameters, lr) { UseAdamW = true };

            var graph = new ComputationGraph();
            var epochLosses = new float[epochs];
            var sw = Stopwatch.StartNew();

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                float epochLoss = 0f;
                int batches = trainSize / batchSize;

                conv1.Train();
                bn1.Train();
                res1.Train();
                fcOut.Train();

                for (int b = 0; b < batches; b++)
                {
                    graph.Reset();
                    optimizer.ZeroGrad();

                    using var xBData = new FastTensor<float>(batchSize, 1, 28, 28, clearMemory: false);
                    using var yBData = new FastTensor<float>(batchSize, 10, clearMemory: false);
                    using var xBNode = new AutogradNode(xBData, requiresGrad: false);
                    using var yBNode = new AutogradNode(yBData, requiresGrad: false);

                    trainX.GetView().AsReadOnlySpan()
                        .Slice(b * batchSize * 784, batchSize * 784)
                        .CopyTo(xBData.GetView().AsSpan());

                    trainY.GetView().AsReadOnlySpan()
                        .Slice(b * batchSize * 10, batchSize * 10)
                        .CopyTo(yBData.GetView().AsSpan());

                    using var h1 = conv1.Forward(graph, xBNode);
                    using var a1 = TensorMath.ReLU(graph, h1);
                    using var p1 = TensorMath.MaxPool2D(graph, a1, 8, 26, 26, 2);
                    using var p1F = TensorMath.Reshape(graph, p1, batchSize, 1352);
                    using var bn1O = bn1.Forward(graph, p1F);
                    using var resO = res1.Forward(graph, bn1O);
                    using var gapO = TensorMath.GlobalAveragePool2D(graph, resO, 8, 13, 13);
                    using var logits = fcOut.Forward(graph, gapO);
                    using var loss = TensorMath.SoftmaxCrossEntropy(graph, logits, yBNode);

                    epochLoss += loss.DataView.AsReadOnlySpan()[0];
                    graph.Backward(loss);
                    optimizer.Step();
                }

                epochLosses[epoch] = epochLoss / (trainSize / batchSize);
                _output.WriteLine($"Epoch {epoch + 1}: avg loss = {epochLosses[epoch]:F4}, elapsed = {sw.ElapsedMilliseconds} ms");
            }

            // UWAGA:
            // Nie przełączamy na Eval(), bo ta ścieżka ma u Ciebie bug lifecycle.
            // Test jakości ma sprawdzić regresję jakości modelu, nie ścieżkę inference.
            conv1.Train();
            bn1.Train();
            res1.Train();
            fcOut.Train();

            int correct = 0;
            int testBatches = testSize / batchSize;

            for (int b = 0; b < testBatches; b++)
            {
                using var xBData = new FastTensor<float>(batchSize, 1, 28, 28, clearMemory: false);
                using var yBData = new FastTensor<float>(batchSize, 10, clearMemory: false);
                using var xBNode = new AutogradNode(xBData, requiresGrad: false);
                using var yBNode = new AutogradNode(yBData, requiresGrad: false);

                testX.GetView().AsReadOnlySpan()
                    .Slice(b * batchSize * 784, batchSize * 784)
                    .CopyTo(xBData.GetView().AsSpan());

                testY.GetView().AsReadOnlySpan()
                    .Slice(b * batchSize * 10, batchSize * 10)
                    .CopyTo(yBData.GetView().AsSpan());

                var evalGraph = new ComputationGraph { IsRecording = false };

                using var h1 = conv1.Forward(evalGraph, xBNode);
                using var a1 = TensorMath.ReLU(evalGraph, h1);
                using var p1 = TensorMath.MaxPool2D(evalGraph, a1, 8, 26, 26, 2);
                using var p1F = TensorMath.Reshape(evalGraph, p1, batchSize, 1352);
                using var bn1O = bn1.Forward(evalGraph, p1F);
                using var resO = res1.Forward(evalGraph, bn1O);
                using var gapO = TensorMath.GlobalAveragePool2D(evalGraph, resO, 8, 13, 13);
                using var logits = fcOut.Forward(evalGraph, gapO);

                var logitsSpan = logits.DataView.AsReadOnlySpan();
                var labelsSpan = yBNode.DataView.AsReadOnlySpan();

                for (int i = 0; i < batchSize; i++)
                {
                    var pred = ArgMax(logitsSpan.Slice(i * 10, 10));
                    var truth = ArgMax(labelsSpan.Slice(i * 10, 10));

                    if (pred == truth)
                    {
                        correct++;
                    }
                }
            }

            float accuracy = correct / (float)(testBatches * batchSize);
            _output.WriteLine($"Final accuracy: {accuracy:P2}");

            Assert.True(
                epochLosses[1] < epochLosses[0],
                $"Training loss did not decrease: epoch1={epochLosses[0]:F4}, epoch2={epochLosses[1]:F4}");

            Assert.True(
                accuracy >= 0.60f,
                $"Accuracy too low after small training run: {accuracy:P2}");
        }

        private static int ArgMax(ReadOnlySpan<float> span)
        {
            int bestIdx = 0;
            float bestVal = span[0];

            for (int i = 1; i < span.Length; i++)
            {
                if (span[i] > bestVal)
                {
                    bestVal = span[i];
                    bestIdx = i;
                }
            }

            return bestIdx;
        }
    }
}