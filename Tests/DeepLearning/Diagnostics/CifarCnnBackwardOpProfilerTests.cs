// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Reflection;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.DeepLearning.Diagnostics
{
    /// <summary>
    /// Same reflection-driven per-op backward profiler as <c>MnistCnnBackwardOpProfilerTests</c>, but on
    /// a representative CIFAR-class CNN: 5 conv layers (C=32/64/128) over 32×32×3 inputs. Purpose: A/B
    /// whether the MNIST-scale "SIMD/Parallel overhead dominates" regime flips at this scale — i.e.
    /// whether Conv/BN/ReLU per-op times scale enough for fused or parallel-channel optimisations to
    /// actually pay off here.
    /// </summary>
    public sealed class CifarCnnBackwardOpProfilerTests
    {
        private const int ImageSide = 32;
        private const int InChannels = 3;
        private const int Classes = 10;
        private const int ArenaSize = 128_000_000;   // ~512 MB

        private readonly ITestOutputHelper _output;

        public CifarCnnBackwardOpProfilerTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        [Trait("Category", "Diagnostics")]
        public void Profile_CifarCnnTrainingStep_BackwardOps()
        {
            var batchSize = GetIntEnvVar("OVERFIT_CIFAR_PROFILE_BATCH", 32);
            var rng = new Random(7);

            // Block 1: 32×32 → 32×32 (2× conv) → 16×16 (MaxPool)
            using var conv1 = new ConvLayer(InChannels, 32, ImageSide, ImageSide, kSize: 3, padding: 1, stride: 1, useBias: true);
            using var bn1 = new BatchNorm2D(32);
            using var conv2 = new ConvLayer(32, 32, 32, 32, kSize: 3, padding: 1, stride: 1, useBias: true);
            using var bn2 = new BatchNorm2D(32);

            // Block 2: 16×16 → 16×16 (2× conv) → 8×8
            using var conv3 = new ConvLayer(32, 64, 16, 16, kSize: 3, padding: 1, stride: 1, useBias: true);
            using var bn3 = new BatchNorm2D(64);
            using var conv4 = new ConvLayer(64, 64, 16, 16, kSize: 3, padding: 1, stride: 1, useBias: true);
            using var bn4 = new BatchNorm2D(64);

            // Block 3: 8×8 → 8×8 → 4×4
            using var conv5 = new ConvLayer(64, 128, 8, 8, kSize: 3, padding: 1, stride: 1, useBias: true);
            using var bn5 = new BatchNorm2D(128);

            using var fc1 = new LinearLayer(128 * 4 * 4, 256);
            using var fc2 = new LinearLayer(256, Classes);

            conv1.Train(); bn1.Train(); conv2.Train(); bn2.Train();
            conv3.Train(); bn3.Train(); conv4.Train(); bn4.Train();
            conv5.Train(); bn5.Train(); fc1.Train(); fc2.Train();

            var inputData = new float[batchSize * InChannels * ImageSide * ImageSide];
            for (var i = 0; i < inputData.Length; i++) { inputData[i] = (float)rng.NextDouble(); }
            var labels = new int[batchSize];
            for (var i = 0; i < labels.Length; i++) { labels[i] = rng.Next(0, Classes); }

            using var graph = new ComputationGraph(ArenaSize);
            using var inStore = new TensorStorage<float>(inputData.Length, clearMemory: false);
            inputData.AsSpan().CopyTo(inStore.AsSpan());
            using var input = new AutogradNode(inStore, new TensorShape(batchSize, InChannels, ImageSide, ImageSide), requiresGrad: false);

            var forwardWatch = Stopwatch.StartNew();
            var x = graph.Relu(bn1.Forward(graph, conv1.Forward(graph, input)));
            x = graph.Relu(bn2.Forward(graph, conv2.Forward(graph, x)));
            x = graph.MaxPool2D(x, 32, 32, 32, 2);
            x = graph.Relu(bn3.Forward(graph, conv3.Forward(graph, x)));
            x = graph.Relu(bn4.Forward(graph, conv4.Forward(graph, x)));
            x = graph.MaxPool2D(x, 64, 16, 16, 2);
            x = graph.Relu(bn5.Forward(graph, conv5.Forward(graph, x)));
            x = graph.MaxPool2D(x, 128, 8, 8, 2);
            x = graph.Reshape(x, batchSize, 128 * 4 * 4);
            x = graph.Relu(fc1.Forward(graph, x));
            var logits = fc2.Forward(graph, x);
            forwardWatch.Stop();

            var lossWatch = Stopwatch.StartNew();
            var loss = ComputeLossAndSeedGrad(logits, labels, batchSize, Classes);
            lossWatch.Stop();

            var profile = ExecuteBackwardWithReflectionProfiler(graph);

            _output.WriteLine("=== CIFAR CNN Backward Op Profiler ===");
            _output.WriteLine($"Shape: batch={batchSize}, H×W={ImageSide}×{ImageSide}, C={InChannels}→128, classes={Classes}");
            _output.WriteLine($"Recorded ops: {profile.RecordedOpCount}");
            _output.WriteLine($"Loss: {loss:F6}");
            _output.WriteLine($"Forward time: {forwardWatch.Elapsed.TotalMilliseconds:F3} ms");
            _output.WriteLine($"Loss seed time: {lossWatch.Elapsed.TotalMilliseconds:F3} ms");
            _output.WriteLine($"Backward timed total: {profile.TotalMilliseconds:F3} ms");
            _output.WriteLine(string.Empty);
            _output.WriteLine("| OpCode | Count | Time ms | Share | Avg us/op |");
            _output.WriteLine("|---|---:|---:|---:|---:|");

            var ordered = new List<BackwardProfileItem>(profile.Items);
            ordered.Sort((a, b) => b.ElapsedTicks.CompareTo(a.ElapsedTicks));
            foreach (var item in ordered)
            {
                var ms = item.ElapsedTicks * 1000.0 / Stopwatch.Frequency;
                var share = profile.TotalTicks == 0 ? 0.0 : item.ElapsedTicks * 100.0 / profile.TotalTicks;
                var avgUs = item.Count == 0 ? 0.0 : item.ElapsedTicks * 1_000_000.0 / Stopwatch.Frequency / item.Count;
                _output.WriteLine($"| {item.OpCode} | {item.Count} | {ms:F3} | {share:F1}% | {avgUs:F3} |");
            }

            Assert.True(profile.RecordedOpCount > 0);
            Assert.True(profile.TotalTicks > 0);
        }

        private static float ComputeLossAndSeedGrad(AutogradNode logits, int[] labels, int batch, int classes)
        {
            var logitArr = logits.DataView.AsReadOnlySpan().ToArray();
            var gradArr = new float[batch * classes];
            var losses = new float[batch];

            Parallel.For(0, batch, b =>
            {
                var off = b * classes;
                var maxVal = logitArr[off];
                for (var c = 1; c < classes; c++)
                {
                    if (logitArr[off + c] > maxVal) { maxVal = logitArr[off + c]; }
                }

                var sumExp = 0f;
                for (var c = 0; c < classes; c++) { sumExp += MathF.Exp(logitArr[off + c] - maxVal); }
                losses[b] = maxVal + MathF.Log(sumExp) - logitArr[off + labels[b]];

                var scale = 1f / batch;
                for (var c = 0; c < classes; c++)
                {
                    var softmax = MathF.Exp(logitArr[off + c] - maxVal) / sumExp;
                    gradArr[off + c] = (softmax - (c == labels[b] ? 1f : 0f)) * scale;
                }
            });

            gradArr.AsSpan().CopyTo(logits.GradView.AsSpan());
            var total = 0f;
            for (var b = 0; b < batch; b++) { total += losses[b]; }
            return total / batch;
        }

        private static BackwardProfile ExecuteBackwardWithReflectionProfiler(ComputationGraph graph)
        {
            var graphType = typeof(ComputationGraph);
            var tapeField = graphType.GetField("_tape", BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new MissingFieldException(graphType.FullName, "_tape");
            var opCountField = graphType.GetField("_opCount", BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new MissingFieldException(graphType.FullName, "_opCount");
            var executeBackwardMethod = graphType.GetMethod("ExecuteBackward", BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new MissingMethodException(graphType.FullName, "ExecuteBackward");

            var tape = tapeField.GetValue(graph) as Array
                ?? throw new InvalidOperationException("Could not read ComputationGraph._tape.");

            var opCount = (int)opCountField.GetValue(graph)!;
            var aggregates = new Dictionary<string, BackwardProfileItem>(StringComparer.Ordinal);
            var totalTicks = 0L;

            for (var i = opCount - 1; i >= 0; i--)
            {
                var op = tape.GetValue(i);
                if (op is null) { continue; }
                var opCode = GetOpCodeName(op);

                var start = Stopwatch.GetTimestamp();
                executeBackwardMethod.Invoke(graph, [op]);
                var elapsed = Stopwatch.GetTimestamp() - start;

                totalTicks += elapsed;
                if (!aggregates.TryGetValue(opCode, out var item))
                {
                    item = new BackwardProfileItem(opCode);
                    aggregates.Add(opCode, item);
                }

                item.Count++;
                item.ElapsedTicks += elapsed;
            }

            var items = new BackwardProfileItem[aggregates.Count];
            var idx = 0;
            foreach (var v in aggregates.Values) { items[idx++] = v; }
            return new BackwardProfile(opCount, totalTicks, items);
        }

        private static string GetOpCodeName(object tapeOp)
        {
            var type = tapeOp.GetType();
            var codeField = type.GetField("Code", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (codeField is not null) { return codeField.GetValue(tapeOp)?.ToString() ?? "<null>"; }
            var codeProperty = type.GetProperty("Code", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (codeProperty is not null) { return codeProperty.GetValue(tapeOp)?.ToString() ?? "<null>"; }
            throw new MissingMemberException(type.FullName, "Code");
        }

        private static int GetIntEnvVar(string name, int defaultValue)
        {
            var raw = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(raw)) { return defaultValue; }
            return int.TryParse(raw, out var v) && v > 0 ? v : defaultValue;
        }

        private sealed class BackwardProfile
        {
            public BackwardProfile(int recordedOpCount, long totalTicks, IReadOnlyList<BackwardProfileItem> items)
            {
                RecordedOpCount = recordedOpCount;
                TotalTicks = totalTicks;
                Items = items;
            }

            public int RecordedOpCount { get; }
            public long TotalTicks { get; }
            public double TotalMilliseconds => TotalTicks * 1000.0 / Stopwatch.Frequency;
            public IReadOnlyList<BackwardProfileItem> Items { get; }
        }

        private sealed class BackwardProfileItem
        {
            public BackwardProfileItem(string opCode) { OpCode = opCode; }
            public string OpCode { get; }
            public int Count { get; set; }
            public long ElapsedTicks { get; set; }
        }
    }
}
