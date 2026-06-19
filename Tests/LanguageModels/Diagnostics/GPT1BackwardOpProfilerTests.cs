// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Reflection;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Diagnostic profiler for GPT-1 backward tape operations.
    ///
    /// This test intentionally does not modify ComputationGraph.
    /// It uses reflection to inspect the recorded tape and invokes the private
    /// ExecuteBackward method one operation at a time, measuring elapsed ticks
    /// grouped by OpCode.
    ///
    /// Use this only as a diagnostic tool:
    ///
    /// - it is not a production profiler,
    /// - reflection adds overhead,
    /// - expensive ops are still visible clearly,
    /// - cheap ops may look noisier than they are.
    ///
    /// Goal:
    /// identify whether the next multicore optimization should target:
    ///
    /// - LinearBackward,
    /// - ScaledDotProductAttentionBackward,
    /// - LayerNormBackward,
    /// - GeluBackward,
    /// - EmbeddingBackward,
    /// - or something else.
    /// </summary>
    public class GPT1BackwardOpProfilerTests
    {
        private const int VocabSize = 68;
        private const int ContextLength = 128;
        private const int DModel = 128;
        private const int HeadCount = 4;
        private const int LayerCount = 4;
        private const int DFF = 512;
        private const int ArenaSize = 180_000_000;

        private readonly ITestOutputHelper _output;

        public GPT1BackwardOpProfilerTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        [Trait("Category", "Diagnostics")]
        public void Profile_GPT1TrainingStep_BackwardOps()
        {
            var batchSize = GetIntEnvironmentVariable(
                "OVERFIT_GPT1_BACKWARD_PROFILE_BATCH",
                8);

            var seqLen = GetIntEnvironmentVariable(
                "OVERFIT_GPT1_BACKWARD_PROFILE_SEQ",
                128);

            var config = new GPT1Config
            {
                VocabSize = VocabSize,
                ContextLength = ContextLength,
                DModel = DModel,
                NHeads = HeadCount,
                NLayers = LayerCount,
                DFF = DFF,
                TieWeights = false,
                PreLayerNorm = true
            };

            using var model = new GPT1Model(config);
            using var graph = new ComputationGraph(ArenaSize);

            model.Train();

            var inputIds = CreateSyntheticTokens(
                batchSize * seqLen,
                VocabSize,
                seed: 123);

            var targetIds = CreateSyntheticTokens(
                batchSize * seqLen,
                VocabSize,
                seed: 456);

            model.InvalidateAllCaches();

            var forwardWatch = Stopwatch.StartNew();

            var logits = model.Forward(
                graph,
                inputIds,
                batchSize,
                seqLen);

            forwardWatch.Stop();

            var lossWatch = Stopwatch.StartNew();

            var loss = ComputeLossAndSeedGradParallel(
                logits,
                targetIds,
                seqLen,
                batchSize,
                VocabSize);

            lossWatch.Stop();

            var profile = ExecuteBackwardWithReflectionProfiler(graph);

            logits.Dispose();

            _output.WriteLine("=== GPT-1 Backward Op Profiler ===");
            _output.WriteLine($"Shape: batch={batchSize}, seq={seqLen}, d={DModel}, heads={HeadCount}, layers={LayerCount}, vocab={VocabSize}");
            _output.WriteLine($"Recorded ops: {profile.RecordedOpCount}");
            _output.WriteLine($"Loss: {loss:F6}");
            _output.WriteLine($"Forward time: {forwardWatch.Elapsed.TotalMilliseconds:F3} ms");
            _output.WriteLine($"Loss seed time: {lossWatch.Elapsed.TotalMilliseconds:F3} ms");
            _output.WriteLine($"Backward timed total: {profile.TotalMilliseconds:F3} ms");
            _output.WriteLine("");
            _output.WriteLine("| OpCode | Count | Time ms | Share | Avg us/op |");
            _output.WriteLine("|---|---:|---:|---:|---:|");

            foreach (var item in profile.Items.OrderByDescending(x => x.ElapsedTicks))
            {
                var ms = item.ElapsedTicks * 1000.0 / Stopwatch.Frequency;
                var share = profile.TotalTicks == 0
                    ? 0.0
                    : item.ElapsedTicks * 100.0 / profile.TotalTicks;

                var avgUs = item.Count == 0
                    ? 0.0
                    : item.ElapsedTicks * 1_000_000.0 / Stopwatch.Frequency / item.Count;

                _output.WriteLine(
                    $"| {item.OpCode} | {item.Count} | {ms:F3} | {share:F1}% | {avgUs:F3} |");
            }

            Assert.True(profile.RecordedOpCount > 0);
            Assert.True(profile.TotalTicks > 0);
        }

        private static BackwardProfile ExecuteBackwardWithReflectionProfiler(
            ComputationGraph graph)
        {
            var graphType = typeof(ComputationGraph);

            var tapeField = graphType.GetField(
                "_tape",
                BindingFlags.Instance | BindingFlags.NonPublic);

            var opCountField = graphType.GetField(
                "_opCount",
                BindingFlags.Instance | BindingFlags.NonPublic);

            var executeBackwardMethod = graphType.GetMethod(
                "ExecuteBackward",
                BindingFlags.Instance | BindingFlags.NonPublic);

            if (tapeField is null)
            {
                throw new MissingFieldException(
                    graphType.FullName,
                    "_tape");
            }

            if (opCountField is null)
            {
                throw new MissingFieldException(
                    graphType.FullName,
                    "_opCount");
            }

            if (executeBackwardMethod is null)
            {
                throw new MissingMethodException(
                    graphType.FullName,
                    "ExecuteBackward");
            }

            var tape = tapeField.GetValue(graph) as Array;

            if (tape is null)
            {
                throw new InvalidOperationException("Could not read ComputationGraph._tape.");
            }

            var opCount = (int)opCountField.GetValue(graph)!;
            var aggregates = new Dictionary<string, BackwardProfileItem>(StringComparer.Ordinal);
            var totalTicks = 0L;

            for (var i = opCount - 1; i >= 0; i--)
            {
                var op = tape.GetValue(i);

                if (op is null)
                {
                    continue;
                }

                var opCode = GetOpCodeName(op);

                var start = Stopwatch.GetTimestamp();

                executeBackwardMethod.Invoke(
                    graph,
                    [op]);

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

            return new BackwardProfile(
                opCount,
                totalTicks,
                aggregates.Values.ToArray());
        }

        private static string GetOpCodeName(object tapeOp)
        {
            var type = tapeOp.GetType();

            var codeField = type.GetField(
                "Code",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);

            if (codeField is not null)
            {
                return codeField.GetValue(tapeOp)?.ToString() ?? "<null>";
            }

            var codeProperty = type.GetProperty(
                "Code",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);

            if (codeProperty is not null)
            {
                return codeProperty.GetValue(tapeOp)?.ToString() ?? "<null>";
            }

            throw new MissingMemberException(
                type.FullName,
                "Code");
        }

        private static int[] CreateSyntheticTokens(
            int length,
            int vocabSize,
            int seed)
        {
            var result = new int[length];
            var rng = new Random(seed);

            for (var i = 0; i < result.Length; i++)
            {
                result[i] = rng.Next(0, vocabSize);
            }

            return result;
        }

        private static float ComputeLossAndSeedGradParallel(
            AutogradNode logits,
            int[] targetIds,
            int seqLen,
            int batchSize,
            int vocabSize)
        {
            var totalTokens = batchSize * seqLen;
            var logitArr = logits.DataView.AsReadOnlySpan().ToArray();
            var gradArr = new float[totalTokens * vocabSize];
            var losses = new float[totalTokens];

            Parallel.For(0, totalTokens, tokenIndex =>
            {
                var offset = tokenIndex * vocabSize;
                var targetId = targetIds[tokenIndex];

                var maxVal = logitArr[offset];

                for (var v = 1; v < vocabSize; v++)
                {
                    if (logitArr[offset + v] > maxVal)
                    {
                        maxVal = logitArr[offset + v];
                    }
                }

                var sumExp = 0f;

                for (var v = 0; v < vocabSize; v++)
                {
                    sumExp += MathF.Exp(logitArr[offset + v] - maxVal);
                }

                losses[tokenIndex] =
                    maxVal +
                    MathF.Log(sumExp) -
                    logitArr[offset + targetId];

                var scale = 1f / seqLen;

                for (var v = 0; v < vocabSize; v++)
                {
                    var softmax =
                        MathF.Exp(logitArr[offset + v] - maxVal) /
                        sumExp;

                    gradArr[offset + v] =
                        (softmax - (v == targetId ? 1f : 0f)) *
                        scale;
                }
            });

            gradArr
                .AsSpan()
                .CopyTo(logits.GradView.AsSpan());

            var total = 0f;

            for (var t = 0; t < totalTokens; t++)
            {
                total += losses[t];
            }

            return total / totalTokens;
        }

        private static int GetIntEnvironmentVariable(
            string name,
            int defaultValue)
        {
            var raw = Environment.GetEnvironmentVariable(name);

            if (string.IsNullOrWhiteSpace(raw))
            {
                return defaultValue;
            }

            return int.TryParse(raw, out var value) && value > 0
                ? value
                : defaultValue;
        }

        private sealed class BackwardProfile
        {
            public BackwardProfile(
                int recordedOpCount,
                long totalTicks,
                IReadOnlyList<BackwardProfileItem> items)
            {
                RecordedOpCount = recordedOpCount;
                TotalTicks = totalTicks;
                Items = items;
            }

            public int RecordedOpCount
            {
                get;
            }

            public long TotalTicks
            {
                get;
            }

            public double TotalMilliseconds => TotalTicks * 1000.0 / Stopwatch.Frequency;

            public IReadOnlyList<BackwardProfileItem> Items
            {
                get;
            }
        }

        private sealed class BackwardProfileItem
        {
            public BackwardProfileItem(string opCode)
            {
                OpCode = opCode;
            }

            public string OpCode
            {
                get;
            }

            public int Count
            {
                get; set;
            }

            public long ElapsedTicks
            {
                get; set;
            }
        }
    }
}
