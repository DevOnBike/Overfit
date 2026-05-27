// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using DevOnBike.Overfit.Tests.TestSupport;
using DevOnBike.Overfit.Training;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Examples
{
    /// <summary>
    /// A real <b>MNIST CNN classifier</b> trained end-to-end in pure C#, exercising the SAME-padding /
    /// bias conv that the training path now supports:
    /// <c>Conv(1→8,SAME)→ReLU→MaxPool → Conv(8→16,SAME)→ReLU→MaxPool → Flatten → FC(→64)→ReLU → FC(→10)</c>,
    /// softmax cross-entropy, Adam + cosine LR. Loads the standard IDX files and asserts test-set
    /// accuracy clears a real bar. [LongFact] — needs the MNIST files (default <c>d:\ml</c> or
    /// <c>OVERFIT_MNIST_DIR</c>).
    /// </summary>
    public sealed class MnistCnnDemoTests
    {
        private const int Side = 28;
        private const int Pixels = Side * Side;   // 784
        private const int Classes = 10;

        private readonly ITestOutputHelper _out;
        public MnistCnnDemoTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        [Trait("Category", "Demo")]
        public void TrainsCnn_OnRealMnist_ToHighAccuracy()
        {
            if (!File.Exists(TestModelPaths.Mnist.TrainImagesPath)) { _out.WriteLine($"missing MNIST at {TestModelPaths.Mnist.Dir}"); return; }

            var (trainX, trainN) = LoadImages(TestModelPaths.Mnist.TrainImagesPath);
            var trainY = LoadLabels(TestModelPaths.Mnist.TrainLabelsPath);
            var (testX, testN) = LoadImages(TestModelPaths.Mnist.TestImagesPath);
            var testY = LoadLabels(TestModelPaths.Mnist.TestLabelsPath);
            _out.WriteLine($"MNIST: {trainN} train / {testN} test");

            const int batch = 64;
            const int steps = 1200;
            const float lrMax = 1e-3f, lrMin = 1e-4f;

            // Conv stack uses SAME padding (preserve dims) + bias; MaxPool halves spatial each stage.
            using var conv1 = new ConvLayer(1, 8, Side, Side, kSize: 3, padding: 1, stride: 1, useBias: true);
            using var conv2 = new ConvLayer(8, 16, 14, 14, kSize: 3, padding: 1, stride: 1, useBias: true);
            using var fc1 = new LinearLayer(16 * 7 * 7, 64);
            using var fc2 = new LinearLayer(64, Classes);
            foreach (var m in new IModule[] { conv1, conv2, fc1, fc2 }) { m.Train(); }

            var parameters = new List<AutogradNode>();
            foreach (var m in new IModule[] { conv1, conv2, fc1, fc2 })
            {
                foreach (var p in m.Parameters()) { parameters.Add(p); }
            }

            using var optimizer = new Adam(parameters, lrMax);
            using var graph = new ComputationGraph(48_000_000);
            var rng = new Random(20260527);

            var inputData = new float[batch * Pixels];
            var targetData = new float[batch * Classes];

            var firstLoss = 0f;
            var lastLoss = 0f;

            for (var step = 0; step < steps; step++)
            {
                // Sample a minibatch (with replacement) into the input + one-hot target buffers.
                Array.Clear(targetData);
                for (var b = 0; b < batch; b++)
                {
                    var idx = rng.Next(trainN);
                    trainX.AsSpan(idx * Pixels, Pixels).CopyTo(inputData.AsSpan(b * Pixels, Pixels));
                    targetData[b * Classes + trainY[idx]] = 1f;
                }

                optimizer.LearningRate = LearningRateSchedule.Cosine(step, steps, lrMax, lrMin);
                graph.Reset();
                conv1.InvalidateParameterCaches(); conv2.InvalidateParameterCaches();
                fc1.InvalidateParameterCaches(); fc2.InvalidateParameterCaches();

                using var input = NewNode(inputData, new TensorShape(batch, 1, Side, Side));
                using var target = NewNode(targetData, new TensorShape(batch, Classes));
                var logits = ForwardLogits(graph, conv1, conv2, fc1, fc2, input, batch);
                var loss = graph.SoftmaxCrossEntropy(logits, target);

                optimizer.ZeroGrad();
                graph.Backward(loss);
                optimizer.Step();

                var l = loss.DataView.AsReadOnlySpan()[0];
                if (step == 0) { firstLoss = l; }
                lastLoss = l;
                if (step == 0 || (step + 1) % 100 == 0) { _out.WriteLine($"step {step + 1,4}/{steps}  loss={l:F4}"); }
            }

            // ── Test-set accuracy ──
            conv1.Eval(); conv2.Eval(); fc1.Eval(); fc2.Eval();
            var evalCount = Math.Min(testN, 2000);
            var correct = 0;
            var evalInput = new float[batch * Pixels];
            for (var start = 0; start < evalCount; start += batch)
            {
                var b = Math.Min(batch, evalCount - start);
                for (var i = 0; i < b; i++)
                {
                    testX.AsSpan((start + i) * Pixels, Pixels).CopyTo(evalInput.AsSpan(i * Pixels, Pixels));
                }

                graph.Reset();
                using var input = NewNode(evalInput, new TensorShape(batch, 1, Side, Side));
                var logits = ForwardLogits(graph, conv1, conv2, fc1, fc2, input, batch);
                var data = logits.DataView.AsReadOnlySpan();
                for (var i = 0; i < b; i++)
                {
                    if (ArgMax(data.Slice(i * Classes, Classes)) == testY[start + i]) { correct++; }
                }
            }

            var accuracy = (float)correct / evalCount;
            _out.WriteLine($"loss {firstLoss:F4} -> {lastLoss:F4}   test accuracy {accuracy:P2} ({correct}/{evalCount})");
            Assert.True(accuracy > 0.95f, $"MNIST CNN test accuracy {accuracy:P2} below 95%.");
        }

        private static AutogradNode ForwardLogits(
            ComputationGraph graph, ConvLayer conv1, ConvLayer conv2, LinearLayer fc1, LinearLayer fc2,
            AutogradNode input, int batch)
        {
            var x = graph.Relu(conv1.Forward(graph, input));        // [B,8,28,28]
            x = graph.MaxPool2D(x, 8, 28, 28, 2);                   // [B,8,14,14]
            x = graph.Relu(conv2.Forward(graph, x));                // [B,16,14,14]
            x = graph.MaxPool2D(x, 16, 14, 14, 2);                  // [B,16,7,7]
            x = graph.Reshape(x, batch, 16 * 7 * 7);                // flatten
            x = graph.Relu(fc1.Forward(graph, x));                  // [B,64]
            return fc2.Forward(graph, x);                           // [B,10]
        }

        private static AutogradNode NewNode(float[] data, TensorShape shape)
        {
            var storage = new TensorStorage<float>(data.Length, clearMemory: false);
            data.AsSpan().CopyTo(storage.AsSpan());
            return new AutogradNode(storage, shape, requiresGrad: false);
        }

        private static int ArgMax(ReadOnlySpan<float> v)
        {
            var best = 0;
            for (var i = 1; i < v.Length; i++) { if (v[i] > v[best]) { best = i; } }
            return best;
        }

        private static (float[] images, int count) LoadImages(string path)
        {
            using var br = new BinaryReader(File.OpenRead(path));
            _ = ReadBigInt32(br);          // magic 2051
            var count = ReadBigInt32(br);
            var rows = ReadBigInt32(br);
            var cols = ReadBigInt32(br);
            var n = count * rows * cols;
            var bytes = br.ReadBytes(n);
            var images = new float[n];
            for (var i = 0; i < n; i++) { images[i] = bytes[i] / 255f; }
            return (images, count);
        }

        private static int[] LoadLabels(string path)
        {
            using var br = new BinaryReader(File.OpenRead(path));
            _ = ReadBigInt32(br);          // magic 2049
            var count = ReadBigInt32(br);
            var bytes = br.ReadBytes(count);
            var labels = new int[count];
            for (var i = 0; i < count; i++) { labels[i] = bytes[i]; }
            return labels;
        }

        private static int ReadBigInt32(BinaryReader br)
        {
            var b = br.ReadBytes(4);
            return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
        }
    }
}
