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
    /// Diagnostic profiler for the MNIST CNN backward path — measures which op dominates so the next
    /// CNN/MLP optimization is empirically guided. Architecture mirrors <c>MnistCnnDemoTests</c>:
    /// <c>Conv(1→8,SAME)→BN2D→ReLU→MaxPool → Conv(8→16,SAME)→BN2D→ReLU→MaxPool → flatten → FC(784→64)
    /// →ReLU → FC(→10)</c>. Same reflection-driven per-op timing as
    /// <c>GPT1BackwardOpProfilerTests</c>; numbers are diagnostic, not production-grade
    /// (cheap ops look noisier than they are; expensive ops dominate clearly).
    /// </summary>
    public sealed class MnistCnnBackwardOpProfilerTests
    {
        private const int ImageSide = 28;
        private const int Classes = 10;
        private const int ArenaSize = 64_000_000;   // ~256 MB, plenty for batch=32 28×28

        private readonly ITestOutputHelper _output;

        public MnistCnnBackwardOpProfilerTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        [Trait("Category", "Diagnostics")]
        public void Profile_MnistCnnTrainingStep_BackwardOps()
        {
            var batchSize = GetIntEnvVar("OVERFIT_MNIST_PROFILE_BATCH", 32);
            var rng = new Random(7);

            using var conv1 = new ConvLayer(1, 8, ImageSide, ImageSide, kSize: 3, padding: 1, stride: 1, useBias: true);
            using var bn1 = new BatchNorm2D(8);
            using var conv2 = new ConvLayer(8, 16, 14, 14, kSize: 3, padding: 1, stride: 1, useBias: true);
            using var bn2 = new BatchNorm2D(16);
            using var fc1 = new LinearLayer(16 * 7 * 7, 64);
            using var fc2 = new LinearLayer(64, Classes);

            conv1.Train(); bn1.Train(); conv2.Train(); bn2.Train(); fc1.Train(); fc2.Train();

            // Synthetic [B, 1, 28, 28] input and labels.
            var inputData = new float[batchSize * 1 * ImageSide * ImageSide];
            for (var i = 0; i < inputData.Length; i++) { inputData[i] = (float)rng.NextDouble(); }
            var labels = new int[batchSize];
            for (var i = 0; i < labels.Length; i++) { labels[i] = rng.Next(0, Classes); }

            using var graph = new ComputationGraph(ArenaSize);

            using var inStore = new TensorStorage<float>(inputData.Length, clearMemory: false);
            inputData.AsSpan().CopyTo(inStore.AsSpan());
            using var input = new AutogradNode(inStore, new TensorShape(batchSize, 1, ImageSide, ImageSide), requiresGrad: false);

            var forwardWatch = Stopwatch.StartNew();
            var logits = Forward(graph, conv1, bn1, conv2, bn2, fc1, fc2, input, batchSize);
            forwardWatch.Stop();

            var lossWatch = Stopwatch.StartNew();
            var loss = ComputeLossAndSeedGrad(logits, labels, batchSize, Classes);
            lossWatch.Stop();

            var profile = ExecuteBackwardWithReflectionProfiler(graph);

            _output.WriteLine("=== MNIST CNN Backward Op Profiler ===");
            _output.WriteLine($"Shape: batch={batchSize}, H×W={ImageSide}×{ImageSide}, classes={Classes}");
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

        private static AutogradNode Forward(
            ComputationGraph graph,
            ConvLayer conv1, BatchNorm2D bn1, ConvLayer conv2, BatchNorm2D bn2,
            LinearLayer fc1, LinearLayer fc2,
            AutogradNode input, int batch)
        {
            var x = graph.Relu(bn1.Forward(graph, conv1.Forward(graph, input)));
            x = graph.MaxPool2D(x, 8, 28, 28, 2);
            x = graph.Relu(bn2.Forward(graph, conv2.Forward(graph, x)));
            x = graph.MaxPool2D(x, 16, 14, 14, 2);
            x = graph.Reshape(x, batch, 16 * 7 * 7);
            x = graph.Relu(fc1.Forward(graph, x));
            return fc2.Forward(graph, x);
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

        // ── reflection-driven per-op backward execution (mirrors GPT1BackwardOpProfilerTests) ──

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
