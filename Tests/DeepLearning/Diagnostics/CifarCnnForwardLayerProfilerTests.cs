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
    /// Forward per-layer profiler for the CIFAR-class CNN (5 conv layers, C up to 128). Same warm-up +
    /// repeat methodology as <c>MnistCnnForwardLayerProfilerTests</c>.
    /// </summary>
    public sealed class CifarCnnForwardLayerProfilerTests
    {
        private const int ImageSide = 32;
        private const int InChannels = 3;
        private const int Classes = 10;
        private const int ArenaSize = 128_000_000;
        private const int Warmups = 2;
        private const int Repeats = 3;

        private readonly ITestOutputHelper _output;

        public CifarCnnForwardLayerProfilerTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        [Trait("Category", "Diagnostics")]
        public void Profile_CifarCnnForward_PerLayer()
        {
            var batchSize = GetIntEnvVar("OVERFIT_CIFAR_PROFILE_BATCH", 32);
            var rng = new Random(13);

            using var conv1 = new ConvLayer(InChannels, 32, ImageSide, ImageSide, kSize: 3, padding: 1, stride: 1, useBias: true);
            using var bn1 = new BatchNorm2D(32);
            using var conv2 = new ConvLayer(32, 32, 32, 32, kSize: 3, padding: 1, stride: 1, useBias: true);
            using var bn2 = new BatchNorm2D(32);
            using var conv3 = new ConvLayer(32, 64, 16, 16, kSize: 3, padding: 1, stride: 1, useBias: true);
            using var bn3 = new BatchNorm2D(64);
            using var conv4 = new ConvLayer(64, 64, 16, 16, kSize: 3, padding: 1, stride: 1, useBias: true);
            using var bn4 = new BatchNorm2D(64);
            using var conv5 = new ConvLayer(64, 128, 8, 8, kSize: 3, padding: 1, stride: 1, useBias: true);
            using var bn5 = new BatchNorm2D(128);
            using var fc1 = new LinearLayer(128 * 4 * 4, 256);
            using var fc2 = new LinearLayer(256, Classes);

            conv1.Train();
            bn1.Train();
            conv2.Train();
            bn2.Train();
            conv3.Train();
            bn3.Train();
            conv4.Train();
            bn4.Train();
            conv5.Train();
            bn5.Train();
            fc1.Train();
            fc2.Train();

            var inputData = new float[batchSize * InChannels * ImageSide * ImageSide];
            for (var i = 0; i < inputData.Length; i++)
            {
                inputData[i] = (float)rng.NextDouble();
            }

            string[] labels = {
                "Conv1","BN1","ReLU1","Conv2","BN2","ReLU2","Pool1",
                "Conv3","BN3","ReLU3","Conv4","BN4","ReLU4","Pool2",
                "Conv5","BN5","ReLU5","Pool3","Reshape","FC1","ReLU6","FC2",
            };
            var ticks = new long[labels.Length];

            using var graph = new ComputationGraph(ArenaSize);

            for (var rep = 0; rep < Warmups + Repeats; rep++)
            {
                graph.Reset();
                var inStore = new TensorStorage<float>(inputData.Length, clearMemory: false);
                inputData.AsSpan().CopyTo(inStore.AsSpan());
                using var input = new AutogradNode(inStore, new TensorShape(batchSize, InChannels, ImageSide, ImageSide), requiresGrad: false);

                var sw = Stopwatch.StartNew();
                var c1 = conv1.Forward(graph, input);
                var t01 = sw.ElapsedTicks;
                sw.Restart();
                var b1 = bn1.Forward(graph, c1);
                var t02 = sw.ElapsedTicks;
                sw.Restart();
                var r1 = graph.Relu(b1);
                var t03 = sw.ElapsedTicks;
                sw.Restart();
                var c2 = conv2.Forward(graph, r1);
                var t04 = sw.ElapsedTicks;
                sw.Restart();
                var b2 = bn2.Forward(graph, c2);
                var t05 = sw.ElapsedTicks;
                sw.Restart();
                var r2 = graph.Relu(b2);
                var t06 = sw.ElapsedTicks;
                sw.Restart();
                var p1 = graph.MaxPool2D(r2, 32, 32, 32, 2);
                var t07 = sw.ElapsedTicks;
                sw.Restart();
                var c3 = conv3.Forward(graph, p1);
                var t08 = sw.ElapsedTicks;
                sw.Restart();
                var b3 = bn3.Forward(graph, c3);
                var t09 = sw.ElapsedTicks;
                sw.Restart();
                var r3 = graph.Relu(b3);
                var t10 = sw.ElapsedTicks;
                sw.Restart();
                var c4 = conv4.Forward(graph, r3);
                var t11 = sw.ElapsedTicks;
                sw.Restart();
                var b4 = bn4.Forward(graph, c4);
                var t12 = sw.ElapsedTicks;
                sw.Restart();
                var r4 = graph.Relu(b4);
                var t13 = sw.ElapsedTicks;
                sw.Restart();
                var p2 = graph.MaxPool2D(r4, 64, 16, 16, 2);
                var t14 = sw.ElapsedTicks;
                sw.Restart();
                var c5 = conv5.Forward(graph, p2);
                var t15 = sw.ElapsedTicks;
                sw.Restart();
                var b5 = bn5.Forward(graph, c5);
                var t16 = sw.ElapsedTicks;
                sw.Restart();
                var r5 = graph.Relu(b5);
                var t17 = sw.ElapsedTicks;
                sw.Restart();
                var p3 = graph.MaxPool2D(r5, 128, 8, 8, 2);
                var t18 = sw.ElapsedTicks;
                sw.Restart();
                var fl = graph.Reshape(p3, batchSize, 128 * 4 * 4);
                var t19 = sw.ElapsedTicks;
                sw.Restart();
                var d1 = fc1.Forward(graph, fl);
                var t20 = sw.ElapsedTicks;
                sw.Restart();
                var rd = graph.Relu(d1);
                var t21 = sw.ElapsedTicks;
                sw.Restart();
                _ = fc2.Forward(graph, rd);
                var t22 = sw.ElapsedTicks;

                if (rep < Warmups)
                {
                    continue;
                }
                ticks[0] += t01;
                ticks[1] += t02;
                ticks[2] += t03;
                ticks[3] += t04;
                ticks[4] += t05;
                ticks[5] += t06;
                ticks[6] += t07;
                ticks[7] += t08;
                ticks[8] += t09;
                ticks[9] += t10;
                ticks[10] += t11;
                ticks[11] += t12;
                ticks[12] += t13;
                ticks[13] += t14;
                ticks[14] += t15;
                ticks[15] += t16;
                ticks[16] += t17;
                ticks[17] += t18;
                ticks[18] += t19;
                ticks[19] += t20;
                ticks[20] += t21;
                ticks[21] += t22;
            }

            var total = 0L;
            for (var i = 0; i < ticks.Length; i++)
            {
                total += ticks[i];
            }

            _output.WriteLine("=== CIFAR CNN Forward Per-Layer Profiler ===");
            _output.WriteLine($"Shape: batch={batchSize}, H×W={ImageSide}×{ImageSide}, C={InChannels}→128, classes={Classes}");
            _output.WriteLine($"Warmups={Warmups}, Repeats={Repeats}, total forward (avg): {(total * 1000.0 / Stopwatch.Frequency / Repeats):F3} ms");
            _output.WriteLine(string.Empty);
            _output.WriteLine("| Layer | Avg ms | Share |");
            _output.WriteLine("|---|---:|---:|");

            var idx = new int[labels.Length];
            for (var i = 0; i < idx.Length; i++)
            {
                idx[i] = i;
            }
            Array.Sort(idx, (a, b) => ticks[b].CompareTo(ticks[a]));

            foreach (var i in idx)
            {
                var avgMs = ticks[i] * 1000.0 / Stopwatch.Frequency / Repeats;
                var share = total == 0 ? 0.0 : ticks[i] * 100.0 / total;
                _output.WriteLine($"| {labels[i]} | {avgMs:F3} | {share:F1}% |");
            }

            Assert.True(total > 0);
        }

        private static int GetIntEnvVar(string name, int defaultValue)
        {
            var raw = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(raw))
            {
                return defaultValue;
            }
            return int.TryParse(raw, out var v) && v > 0 ? v : defaultValue;
        }
    }
}
