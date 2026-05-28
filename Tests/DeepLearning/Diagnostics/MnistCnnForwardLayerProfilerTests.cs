// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.DeepLearning.Diagnostics
{
    /// <summary>
    /// Forward-side per-layer profiler for the MNIST CNN. Backward is handled by the
    /// reflection-based <c>MnistCnnBackwardOpProfilerTests</c>; forward isn't replayable from the tape
    /// (the tape is the *output* of forward), so we time each composition step in the demo's forward
    /// pass directly. Drives a steady-state median over a small warm-up + repeat budget — first call
    /// includes JIT and cache misses we want to exclude.
    /// </summary>
    public sealed class MnistCnnForwardLayerProfilerTests
    {
        private const int ImageSide = 28;
        private const int Classes = 10;
        private const int ArenaSize = 64_000_000;
        private const int Warmups = 2;
        private const int Repeats = 5;

        private readonly ITestOutputHelper _output;

        public MnistCnnForwardLayerProfilerTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        [Trait("Category", "Diagnostics")]
        public void Profile_MnistCnnForward_PerLayer()
        {
            var batchSize = GetIntEnvVar("OVERFIT_MNIST_PROFILE_BATCH", 32);
            var rng = new Random(11);

            using var conv1 = new ConvLayer(1, 8, ImageSide, ImageSide, kSize: 3, padding: 1, stride: 1, useBias: true);
            using var bn1 = new BatchNorm2D(8);
            using var conv2 = new ConvLayer(8, 16, 14, 14, kSize: 3, padding: 1, stride: 1, useBias: true);
            using var bn2 = new BatchNorm2D(16);
            using var fc1 = new LinearLayer(16 * 7 * 7, 64);
            using var fc2 = new LinearLayer(64, Classes);
            conv1.Train(); bn1.Train(); conv2.Train(); bn2.Train(); fc1.Train(); fc2.Train();

            var inputData = new float[batchSize * 1 * ImageSide * ImageSide];
            for (var i = 0; i < inputData.Length; i++) { inputData[i] = (float)rng.NextDouble(); }

            // Per-layer accumulators across repeats.
            string[] labels = { "Conv1", "BN1", "ReLU1", "MaxPool1", "Conv2", "BN2", "ReLU2", "MaxPool2", "Reshape", "FC1", "ReLU3", "FC2" };
            var ticks = new long[labels.Length];

            using var graph = new ComputationGraph(ArenaSize);

            for (var rep = 0; rep < Warmups + Repeats; rep++)
            {
                graph.Reset();
                // Fresh storage per iteration — AutogradNode takes ownership and disposing the node
                // would invalidate a shared TensorStorage for the next pass.
                var inStore = new TensorStorage<float>(inputData.Length, clearMemory: false);
                inputData.AsSpan().CopyTo(inStore.AsSpan());
                using var input = new AutogradNode(inStore, new TensorShape(batchSize, 1, ImageSide, ImageSide), requiresGrad: false);

                var sw = Stopwatch.StartNew();
                var c1 = conv1.Forward(graph, input);                         var t1 = sw.ElapsedTicks; sw.Restart();
                var b1 = bn1.Forward(graph, c1);                              var t2 = sw.ElapsedTicks; sw.Restart();
                var r1 = graph.Relu(b1);                                      var t3 = sw.ElapsedTicks; sw.Restart();
                var p1 = graph.MaxPool2D(r1, 8, 28, 28, 2);                   var t4 = sw.ElapsedTicks; sw.Restart();
                var c2 = conv2.Forward(graph, p1);                            var t5 = sw.ElapsedTicks; sw.Restart();
                var b2 = bn2.Forward(graph, c2);                              var t6 = sw.ElapsedTicks; sw.Restart();
                var r2 = graph.Relu(b2);                                      var t7 = sw.ElapsedTicks; sw.Restart();
                var p2 = graph.MaxPool2D(r2, 16, 14, 14, 2);                  var t8 = sw.ElapsedTicks; sw.Restart();
                var fl = graph.Reshape(p2, batchSize, 16 * 7 * 7);            var t9 = sw.ElapsedTicks; sw.Restart();
                var d1 = fc1.Forward(graph, fl);                              var t10 = sw.ElapsedTicks; sw.Restart();
                var r3 = graph.Relu(d1);                                      var t11 = sw.ElapsedTicks; sw.Restart();
                _ = fc2.Forward(graph, r3);                                   var t12 = sw.ElapsedTicks;

                if (rep < Warmups) { continue; }
                ticks[0] += t1; ticks[1] += t2; ticks[2] += t3; ticks[3] += t4;
                ticks[4] += t5; ticks[5] += t6; ticks[6] += t7; ticks[7] += t8;
                ticks[8] += t9; ticks[9] += t10; ticks[10] += t11; ticks[11] += t12;
            }

            var total = 0L;
            for (var i = 0; i < ticks.Length; i++) { total += ticks[i]; }

            _output.WriteLine("=== MNIST CNN Forward Per-Layer Profiler ===");
            _output.WriteLine($"Shape: batch={batchSize}, H×W={ImageSide}×{ImageSide}, classes={Classes}");
            _output.WriteLine($"Warmups={Warmups}, Repeats={Repeats}, total forward time (averaged): {(total * 1000.0 / Stopwatch.Frequency / Repeats):F3} ms");
            _output.WriteLine(string.Empty);
            _output.WriteLine("| Layer | Avg ms | Share |");
            _output.WriteLine("|---|---:|---:|");

            // Sort by ticks descending for hot-spot visibility.
            var indices = new int[labels.Length];
            for (var i = 0; i < indices.Length; i++) { indices[i] = i; }
            Array.Sort(indices, (a, b) => ticks[b].CompareTo(ticks[a]));

            foreach (var idx in indices)
            {
                var avgMs = ticks[idx] * 1000.0 / Stopwatch.Frequency / Repeats;
                var share = total == 0 ? 0.0 : ticks[idx] * 100.0 / total;
                _output.WriteLine($"| {labels[idx]} | {avgMs:F3} | {share:F1}% |");
            }

            Assert.True(total > 0);
        }

        private static int GetIntEnvVar(string name, int defaultValue)
        {
            var raw = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(raw)) { return defaultValue; }
            return int.TryParse(raw, out var v) && v > 0 ? v : defaultValue;
        }
    }
}
