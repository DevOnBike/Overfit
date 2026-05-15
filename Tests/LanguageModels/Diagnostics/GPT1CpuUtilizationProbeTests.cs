// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Parameters;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Diagnostic CPU-utilization probe for GPT-1 training step.
    /// Mirrors <c>GPT1TrainingStepBenchmark</c> but adds
    /// <see cref="Process.TotalProcessorTime"/> sampling so we can read
    /// "cores effective" — the metric that tells whether the multi-core box
    /// is actually being saturated or whether sequential slices dominate.
    ///
    /// MNIST baseline (per ROADMAP) was 6.81 / 32 cores (21.3%). MNIST is
    /// Amdahl-limited at this scale. GPT-1 has bigger per-op body work
    /// (FFN d=128→512→128, MHA 4 heads × d=32) — this test answers whether
    /// dispatcher migration converts to more cores effective on a workload
    /// that's NOT Amdahl-bound by tiny per-call overhead.
    /// </summary>
    public sealed class GPT1CpuUtilizationProbeTests
    {
        private const int VocabSize = 68;
        private const int ContextLength = 128;
        private const int DModel = 128;
        private const int HeadCount = 4;
        private const int LayerCount = 4;
        private const int DFF = 512;
        private const int ArenaSize = 180_000_000;

        private const float LearningRate = 3e-4f;
        private const float WeightDecay = 0.1f;
        private const float MaxGradNorm = 1.0f;

        private readonly ITestOutputHelper _output;

        public GPT1CpuUtilizationProbeTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        [Trait("Category", "Diagnostics")]
        public void GPT1_TrainingStep_CpuUtilization_BatchSweep()
        {
            _output.WriteLine($"Environment.ProcessorCount: {Environment.ProcessorCount}");

            foreach (var batchSize in new[] { 8, 16, 32 })
            {
                RunForBatchSize(batchSize);
            }
        }

        private void RunForBatchSize(int batchSize)
        {
            const int seqLen = 128;
            const int stepCount = 20;
            const int warmupSteps = 3;

            var config = new GPT1Config
            {
                VocabSize = VocabSize,
                ContextLength = ContextLength,
                DModel = DModel,
                NHeads = HeadCount,
                NLayers = LayerCount,
                DFF = DFF,
                TieWeights = false,
                PreLayerNorm = true,
            };

            using var model = new GPT1Model(config);
            model.Train();

            var parameters = model.TrainableParameters().ToList();

            using var optimizer = new Adam(parameters, LearningRate)
            {
                UseAdamW = true,
                WeightDecay = WeightDecay,
            };

            using var graph = new ComputationGraph(ArenaSize);
            graph.BackwardProfileEnabled = true;

            var rng = new Random(42);
            var corpus = CreateSyntheticCorpus(1_115_394, VocabSize);
            var inputIds = new int[batchSize * seqLen];
            var targetIds = new int[batchSize * seqLen];

            // Warmup — JIT, allocate scratch buffers, settle thread pool.
            for (var w = 0; w < warmupSteps; w++)
            {
                SampleBatch(corpus, seqLen, batchSize, rng, inputIds, targetIds);
                RunOneStep(model, graph, optimizer, parameters, inputIds, targetIds,
                           batchSize, seqLen, VocabSize);
            }

            // Reset profile counters after warmup so they reflect measured steps only.
            graph.ResetBackwardProfile();

            using var process = Process.GetCurrentProcess();
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            process.Refresh();
            var cpuBefore = process.TotalProcessorTime;
            var allocBefore = GC.GetTotalAllocatedBytes(precise: false);
            var watch = ValueStopwatch.StartNew();

            for (var step = 0; step < stepCount; step++)
            {
                SampleBatch(corpus, seqLen, batchSize, rng, inputIds, targetIds);
                RunOneStep(model, graph, optimizer, parameters, inputIds, targetIds,
                           batchSize, seqLen, VocabSize);
            }

            var wallMs = watch.GetElapsedTime().TotalMilliseconds;
            process.Refresh();
            var cpuMs = (process.TotalProcessorTime - cpuBefore).TotalMilliseconds;
            var allocBytes = GC.GetTotalAllocatedBytes(precise: false) - allocBefore;

            var coresUsed = cpuMs / wallMs;
            var utilization = coresUsed / Environment.ProcessorCount;
            var allocPerStep = allocBytes / (double)stepCount;
            var wallPerStep = wallMs / stepCount;

            _output.WriteLine($"--- BatchSize={batchSize}, SeqLen={seqLen}, Steps={stepCount} ---");
            _output.WriteLine($"  wall time total:    {wallMs:F1} ms ({wallPerStep:F2} ms/step)");
            _output.WriteLine($"  total CPU time:     {cpuMs:F1} ms");
            _output.WriteLine($"  cores effective:    {coresUsed:F2} of {Environment.ProcessorCount}");
            _output.WriteLine($"  utilization:        {utilization:P1}");
            _output.WriteLine($"  alloc per step:     {allocPerStep / 1024.0:F1} KB");
            _output.WriteLine(string.Empty);
            _output.WriteLine("  per-OpCode backward (post-warmup):");

            var profile = graph.GetBackwardOpProfile()
                .OrderByDescending(p => p.ElapsedMs)
                .ToArray();

            foreach (var p in profile)
            {
                _output.WriteLine($"    {p.Code,-32} {p.ElapsedMs,9:F1} ms | calls {p.Count,6}");
            }

            _output.WriteLine(string.Empty);
        }

        private static void RunOneStep(
            GPT1Model model,
            ComputationGraph graph,
            Adam optimizer,
            IReadOnlyList<Parameter> parameters,
            int[] inputIds,
            int[] targetIds,
            int batchSize,
            int seqLen,
            int vocabSize)
        {
            optimizer.ZeroGrad();
            graph.Reset();
            model.InvalidateAllCaches();

            var logits = model.Forward(graph, inputIds, batchSize, seqLen);
            ComputeLossAndSeedGrad(logits, targetIds, seqLen, batchSize, vocabSize);
            graph.BackwardFromGrad(logits);
            logits.Dispose();

            ClipGradNorm(parameters, MaxGradNorm);
            optimizer.Step();
        }

        private static int[] CreateSyntheticCorpus(int length, int vocabSize)
        {
            var rng = new Random(123);
            var result = new int[length];
            for (var i = 0; i < length; i++) { result[i] = rng.Next(0, vocabSize); }
            return result;
        }

        private static void SampleBatch(
            int[] corpus, int seqLen, int batchSize, Random rng,
            int[] inputIds, int[] targetIds)
        {
            for (var b = 0; b < batchSize; b++)
            {
                var start = rng.Next(0, corpus.Length - seqLen - 1);
                corpus.AsSpan(start, seqLen).CopyTo(inputIds.AsSpan(b * seqLen, seqLen));
                corpus.AsSpan(start + 1, seqLen).CopyTo(targetIds.AsSpan(b * seqLen, seqLen));
            }
        }

        private static void ComputeLossAndSeedGrad(
            AutogradNode logits, int[] targetIds, int seqLen, int batchSize, int vocabSize)
        {
            var totalTokens = batchSize * seqLen;
            var logitsSpan = logits.DataView.AsReadOnlySpan();
            var gradSpan = logits.GradView.AsSpan();
            var scale = 1f / seqLen;

            for (var tokenIndex = 0; tokenIndex < totalTokens; tokenIndex++)
            {
                var offset = tokenIndex * vocabSize;
                var targetId = targetIds[tokenIndex];

                var maxVal = logitsSpan[offset];
                for (var v = 1; v < vocabSize; v++)
                {
                    if (logitsSpan[offset + v] > maxVal) { maxVal = logitsSpan[offset + v]; }
                }

                var sumExp = 0f;
                for (var v = 0; v < vocabSize; v++)
                {
                    sumExp += MathF.Exp(logitsSpan[offset + v] - maxVal);
                }

                for (var v = 0; v < vocabSize; v++)
                {
                    var softmax = MathF.Exp(logitsSpan[offset + v] - maxVal) / sumExp;
                    gradSpan[offset + v] = (softmax - (v == targetId ? 1f : 0f)) * scale;
                }
            }
        }

        private static void ClipGradNorm(IReadOnlyList<Parameter> parameters, float maxNorm)
        {
            var totalNormSq = 0f;
            foreach (var parameter in parameters)
            {
                var grad = parameter.GradSpan;
                for (var i = 0; i < grad.Length; i++) { totalNormSq += grad[i] * grad[i]; }
            }
            var norm = MathF.Sqrt(totalNormSq);
            if (norm <= maxNorm) { return; }
            var scale = maxNorm / (norm + 1e-6f);
            foreach (var parameter in parameters)
            {
                var grad = parameter.GradSpan;
                for (var i = 0; i < grad.Length; i++) { grad[i] *= scale; }
            }
        }

        private readonly struct ValueStopwatch
        {
            private readonly long _startTimestamp;

            private ValueStopwatch(long startTimestamp)
            {
                _startTimestamp = startTimestamp;
            }

            public static ValueStopwatch StartNew() => new(Stopwatch.GetTimestamp());

            public TimeSpan GetElapsedTime() =>
                Stopwatch.GetElapsedTime(_startTimestamp);
        }
    }
}
