// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tokenization;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Demo.TinyShakespeare
{
    /// <summary>
    /// Manual single-model / sequential TinyShakespeare quality checkpoint demo.
    ///
    /// Intent:
    /// - no data-parallel trainer,
    /// - no experimental parallel attention flag,
    /// - no Parallel.For in loss seeding,
    /// - optionally restrict the process to one logical CPU while this test runs,
    /// - train a better checkpoint.bin for the cached runtime demo.
    ///
    /// This is a manual long-running demo test. It is not a regular unit test.
    ///
    /// Recommended presets:
    ///
    /// Quality:
    ///   steps=20000, batch=4, seq=128, d=128, heads=4, layers=4
    ///
    /// Better but longer:
    ///   OVERFIT_TINY_SHAKESPEARE_SEQ_QUALITY_STEPS=30000
    ///
    /// Paths intentionally match the existing demo:
    /// - corpus:     test_fixtures/tiny_shakespeare.txt
    /// - checkpoint: test_fixtures/checkpoint.bin
    /// </summary>
    public class TinyShakespeareSequentialQualityCheckpointTests
    {
        private const string FixturePath = "test_fixtures/tiny_shakespeare.txt";
        private const string DemoCheckpointPath = "test_fixtures/checkpoint.bin";
        private const string DemoPrompt = "ROMEO:\n";

        private const int DemoSeqLen = 128;
        private const int DemoBatchSize = 4;
        private const int DemoDefaultTrainingSteps = 20_000;
        private const int DemoArenaSize = 180_000_000;

        private const int DemoReportEvery = 100;
        private const int DemoEvalEvery = 1_000;
        private const int DemoValidationBatches = 20;

        private const int DemoRequestedNewTokens = 120;

        private const float DemoLearningRateMax = 3e-4f;
        private const float DemoLearningRateMin = 3e-5f;
        private const float DemoWeightDecay = 0.05f;
        private const float DemoMaxGradNorm = 1.0f;

        private const int DemoTopK = 16;
        private const float DemoTemperature = 0.85f;
        private const float DemoRepetitionPenalty = 1.15f;
        private const int DemoRepetitionWindow = 64;
        private const int DemoSamplingSeed = 42;

        private readonly ITestOutputHelper _output;

        public TinyShakespeareSequentialQualityCheckpointTests(
            ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact(Skip = "Manual long-running single-model GPT quality demo. Remove Skip locally, run once, then restore Skip.")]
        [Trait("Category", "Demo")]
        [Trait("Category", "LongRunning")]
        public void Demo_Train_TinyShakespeare_SequentialQuality_AndWriteCheckpointBin()
        {
            SkipIfMissing(FixturePath);

            var steps = GetIntEnvironmentVariable(
                "OVERFIT_TINY_SHAKESPEARE_SEQ_QUALITY_STEPS",
                DemoDefaultTrainingSteps);

            var forceSingleLogicalProcessor = GetBoolEnvironmentVariable(
                "OVERFIT_TINY_SHAKESPEARE_FORCE_ONE_CORE",
                defaultValue: false);

            using var affinityScope = forceSingleLogicalProcessor ? SingleLogicalProcessorScope.TryEnter(_output) : null;

            var text = File.ReadAllText(FixturePath);
            var tokenizer = CharacterTokenizer.FromCorpus(text);
            var allIds = tokenizer.Encode(text);

            var trainSize = (int)(allIds.Length * 0.9);
            var trainIds = allIds.AsSpan(0, trainSize).ToArray();
            var valIds = allIds.AsSpan(trainSize).ToArray();

            var config = CreateDemoConfig(tokenizer.VocabSize);

            _output.WriteLine("=== Overfit TinyShakespeare Sequential Quality Checkpoint Demo ===");
            _output.WriteLine($"Corpus: {text.Length:N0} chars");
            _output.WriteLine($"Train tokens: {trainIds.Length:N0}");
            _output.WriteLine($"Validation tokens: {valIds.Length:N0}");
            _output.WriteLine($"Vocab: {tokenizer.VocabSize}");
            _output.WriteLine($"Model: {config}");
            _output.WriteLine($"Parameters: {config.ParameterCount:N0}");
            _output.WriteLine($"Steps: {steps:N0}");
            _output.WriteLine($"BatchSize: {DemoBatchSize}");
            _output.WriteLine($"SeqLen / ContextLength: {DemoSeqLen}");
            _output.WriteLine($"LR schedule: {DemoLearningRateMax} -> {DemoLearningRateMin}");
            _output.WriteLine($"WeightDecay: {DemoWeightDecay}");
            _output.WriteLine($"MaxGradNorm: {DemoMaxGradNorm}");
            _output.WriteLine($"Single logical processor requested: {forceSingleLogicalProcessor}");
            _output.WriteLine($"Single logical processor active: {affinityScope?.IsActive ?? false}");
            _output.WriteLine($"Display sampling: TopK(k={DemoTopK}), temperature={DemoTemperature}, repetitionPenalty={DemoRepetitionPenalty}, window={DemoRepetitionWindow}, seed={DemoSamplingSeed}");
            _output.WriteLine($"Checkpoint: {Path.GetFullPath(DemoCheckpointPath)}");
            _output.WriteLine(string.Empty);

            using var model = new GPT1Model(config);
            var parameters = model
                .TrainableParameters()
                .ToList();

            using var optimizer = new Adam(parameters, DemoLearningRateMax)
            {
                UseAdamW = true,
                WeightDecay = DemoWeightDecay
            };

            model.Train();

            var rng = new Random(42);
            var firstLoss = 0f;
            var lastLoss = 0f;
            var bestValLoss = float.PositiveInfinity;
            var finalValLoss = float.PositiveInfinity;
            var windowLoss = 0f;

            var started = Stopwatch.StartNew();

            using var graph = new ComputationGraph(DemoArenaSize);

            for (var step = 0; step < steps; step++)
            {
                var (inputIds, targetIds) = SampleBatch(
                    trainIds,
                    DemoSeqLen,
                    DemoBatchSize,
                    rng);

                optimizer.ZeroGrad();

                graph.Reset();
                model.InvalidateAllCaches();

                var logits = model.Forward(
                    graph,
                    inputIds,
                    DemoBatchSize,
                    DemoSeqLen);

                var loss = ComputeLossAndSeedGradSequential(
                    logits,
                    targetIds,
                    DemoSeqLen,
                    DemoBatchSize,
                    config.VocabSize);

                Assert.False(float.IsNaN(loss), $"NaN loss at step {step}.");
                Assert.False(float.IsInfinity(loss), $"Infinite loss at step {step}.");

                graph.BackwardFromGrad(logits);
                logits.Dispose();

                ClipGradNorm(
                    parameters,
                    DemoMaxGradNorm);

                optimizer.LearningRate = CosineDecay(
                    DemoLearningRateMax,
                    DemoLearningRateMin,
                    step,
                    steps);

                optimizer.Step();

                if (step == 0)
                {
                    firstLoss = loss;
                }

                lastLoss = loss;
                windowLoss += loss;

                if (step == 0 ||
                    (step + 1) % DemoReportEvery == 0 ||
                    step + 1 == steps)
                {
                    var denominator = step == 0
                        ? 1
                        : Math.Min(DemoReportEvery, step + 1);

                    var avg = windowLoss / denominator;
                    windowLoss = 0f;

                    _output.WriteLine(
                        $"Step {step + 1,6}/{steps} | loss={loss:F4} | avg={avg:F4} | lr={optimizer.LearningRate:E2} | elapsed={started.Elapsed:hh\\:mm\\:ss}");
                }

                if ((step + 1) % DemoEvalEvery == 0 || step + 1 == steps)
                {
                    var valLoss = EvaluateLossSequential(
                        model,
                        valIds,
                        config,
                        DemoSeqLen,
                        rng,
                        DemoValidationBatches);

                    finalValLoss = valLoss;
                    bestValLoss = MathF.Min(bestValLoss, valLoss);

                    _output.WriteLine(
                        $"Validation @ step {step + 1,6}: val={valLoss:F4} | best={bestValLoss:F4}");

                    var safeSampleTokens = GetSafeGeneratedTokenCount(
                        config.ContextLength,
                        tokenizer.Encode(DemoPrompt).Length,
                        DemoRequestedNewTokens);

                    var sample = GenerateDisplaySampleWithRepetitionPenalty(
                        model,
                        tokenizer,
                        DemoPrompt,
                        safeSampleTokens);

                    _output.WriteLine("Sample:");
                    _output.WriteLine(sample);
                    _output.WriteLine(string.Empty);

                    model.Train();
                }
            }

            model.Eval();

            Directory.CreateDirectory(Path.GetDirectoryName(DemoCheckpointPath)!);

            using (var stream = File.Create(DemoCheckpointPath))
            using (var writer = new BinaryWriter(stream))
            {
                model.Save(writer);
            }

            var finalSafeSampleTokens = GetSafeGeneratedTokenCount(
                config.ContextLength,
                tokenizer.Encode(DemoPrompt).Length,
                DemoRequestedNewTokens);

            var finalSample = GenerateDisplaySampleWithRepetitionPenalty(
                model,
                tokenizer,
                DemoPrompt,
                finalSafeSampleTokens);

            _output.WriteLine("");
            _output.WriteLine($"Loss: {firstLoss:F4} -> {lastLoss:F4}");
            _output.WriteLine($"Final validation loss: {finalValLoss:F4}");
            _output.WriteLine($"Best validation loss: {bestValLoss:F4}");
            _output.WriteLine($"Total elapsed: {started.Elapsed:hh\\:mm\\:ss}");
            _output.WriteLine("");
            _output.WriteLine("Final display sample:");
            _output.WriteLine(finalSample);
            _output.WriteLine("");
            _output.WriteLine($"Checkpoint written: {Path.GetFullPath(DemoCheckpointPath)}");

            Assert.True(File.Exists(DemoCheckpointPath));
            Assert.True(new FileInfo(DemoCheckpointPath).Length > 0);
            Assert.True(lastLoss < firstLoss, $"Loss did not go down: {firstLoss:F4} -> {lastLoss:F4}.");
            Assert.False(float.IsNaN(finalValLoss) || float.IsInfinity(finalValLoss), "Final validation loss is NaN/Infinity.");
        }

        private static GPT1Config CreateDemoConfig(
            int vocabSize)
        {
            return new GPT1Config
            {
                VocabSize = vocabSize,
                ContextLength = DemoSeqLen,
                DModel = 128,
                NHeads = 4,
                NLayers = 4,
                DFF = 512,
                TieWeights = false,
                PreLayerNorm = true
            };
        }

        private static (int[] inputIds, int[] targetIds) SampleBatch(
            int[] corpus,
            int seqLen,
            int batchSize,
            Random rng)
        {
            var inputIds = new int[batchSize * seqLen];
            var targetIds = new int[batchSize * seqLen];

            for (var b = 0; b < batchSize; b++)
            {
                var start = rng.Next(0, corpus.Length - seqLen - 1);

                corpus
                    .AsSpan(start, seqLen)
                    .CopyTo(inputIds.AsSpan(b * seqLen, seqLen));

                corpus
                    .AsSpan(start + 1, seqLen)
                    .CopyTo(targetIds.AsSpan(b * seqLen, seqLen));
            }

            return (inputIds, targetIds);
        }

        private static float ComputeLossAndSeedGradSequential(
            AutogradNode logits,
            int[] targetIds,
            int seqLen,
            int batchSize,
            int vocabSize)
        {
            var totalTokens = batchSize * seqLen;
            var logitsSpan = logits.DataView.AsReadOnlySpan();
            var gradSpan = logits.GradView.AsSpan();

            var total = 0f;

            for (var tokenIndex = 0; tokenIndex < totalTokens; tokenIndex++)
            {
                var offset = tokenIndex * vocabSize;
                var targetId = targetIds[tokenIndex];

                var maxVal = logitsSpan[offset];

                for (var v = 1; v < vocabSize; v++)
                {
                    var value = logitsSpan[offset + v];

                    if (value > maxVal)
                    {
                        maxVal = value;
                    }
                }

                var sumExp = 0f;

                for (var v = 0; v < vocabSize; v++)
                {
                    sumExp += MathF.Exp(logitsSpan[offset + v] - maxVal);
                }

                total +=
                    maxVal +
                    MathF.Log(sumExp) -
                    logitsSpan[offset + targetId];

                var scale = 1f / seqLen;

                for (var v = 0; v < vocabSize; v++)
                {
                    var softmax =
                        MathF.Exp(logitsSpan[offset + v] - maxVal) /
                        sumExp;

                    gradSpan[offset + v] =
                        (softmax - (v == targetId ? 1f : 0f)) *
                        scale;
                }
            }

            return total / totalTokens;
        }

        private static float EvaluateLossSequential(
            GPT1Model model,
            int[] valCorpus,
            GPT1Config config,
            int seqLen,
            Random rng,
            int validationBatches)
        {
            model.Eval();

            var total = 0f;

            using var graph = new ComputationGraph(100_000_000);

            for (var i = 0; i < validationBatches; i++)
            {
                var (inputIds, targetIds) = SampleBatch(
                    valCorpus,
                    seqLen,
                    batchSize: 1,
                    rng);

                graph.Reset();
                model.InvalidateAllCaches();

                var logits = model.Forward(
                    graph,
                    inputIds,
                    batchSize: 1,
                    seqLen);

                total += ComputeLossAndSeedGradSequential(
                    logits,
                    targetIds,
                    seqLen,
                    batchSize: 1,
                    config.VocabSize);

                logits.Dispose();
            }

            return total / validationBatches;
        }

        private static void ClipGradNorm(
            IReadOnlyList<Parameter> parameters,
            float maxNorm)
        {
            var totalNormSq = 0f;

            foreach (var parameter in parameters)
            {
                var grad = parameter.GradSpan;

                for (var i = 0; i < grad.Length; i++)
                {
                    totalNormSq += grad[i] * grad[i];
                }
            }

            var norm = MathF.Sqrt(totalNormSq);

            if (norm <= maxNorm)
            {
                return;
            }

            var scale = maxNorm / (norm + 1e-6f);

            foreach (var parameter in parameters)
            {
                var grad = parameter.GradSpan;

                for (var i = 0; i < grad.Length; i++)
                {
                    grad[i] *= scale;
                }
            }
        }

        private static float CosineDecay(
            float maxLearningRate,
            float minLearningRate,
            int step,
            int totalSteps)
        {
            if (totalSteps <= 1)
            {
                return minLearningRate;
            }

            var progress = Math.Clamp(
                step / (float)(totalSteps - 1),
                0f,
                1f);

            var cosine = 0.5f * (1f + MathF.Cos(MathF.PI * progress));

            return minLearningRate + (maxLearningRate - minLearningRate) * cosine;
        }

        private static string GenerateDisplaySampleWithRepetitionPenalty(
            GPT1Model model,
            CharacterTokenizer tokenizer,
            string prompt,
            int maxTokens)
        {
            model.Eval();

            var promptTokens = tokenizer.Encode(prompt);
            var generatedTokens = new List<int>(promptTokens.Length + maxTokens);

            generatedTokens.AddRange(promptTokens);

            using var adapter = new CachedGpt1ModelAdapter(model);

            var logits = new float[model.Config.VocabSize];
            var adjustedLogits = new float[model.Config.VocabSize];
            var indexScratch = new int[model.Config.VocabSize];
            var scoreScratch = new float[model.Config.VocabSize];
            var random = new Random(DemoSamplingSeed);

            for (var i = 0; i < promptTokens.Length; i++)
            {
                adapter.DecodeNextToken(
                    promptTokens[i],
                    logits);
            }

            var sampling = new SamplingOptions(
                SamplingStrategy.TopK,
                temperature: DemoTemperature,
                topK: DemoTopK,
                topP: 1.0f,
                seed: 0);

            for (var i = 0; i < maxTokens; i++)
            {
                if (adapter.IsFull)
                {
                    break;
                }

                logits.AsSpan().CopyTo(adjustedLogits);
                ApplyRepetitionPenalty(
                    adjustedLogits,
                    generatedTokens,
                    DemoRepetitionWindow,
                    DemoRepetitionPenalty);

                var nextToken = TokenSampler.Sample(
                    adjustedLogits,
                    in sampling,
                    random,
                    indexScratch,
                    scoreScratch);

                generatedTokens.Add(nextToken);

                adapter.DecodeNextToken(
                    nextToken,
                    logits);
            }

            return tokenizer.Decode(generatedTokens.ToArray());
        }

        private static void ApplyRepetitionPenalty(
            Span<float> logits,
            IReadOnlyList<int> generatedTokens,
            int window,
            float penalty)
        {
            if (penalty <= 1f || generatedTokens.Count == 0)
            {
                return;
            }

            var start = Math.Max(
                0,
                generatedTokens.Count - window);

            for (var i = start; i < generatedTokens.Count; i++)
            {
                var token = generatedTokens[i];

                if ((uint)token >= (uint)logits.Length)
                {
                    continue;
                }

                if (logits[token] >= 0f)
                {
                    logits[token] /= penalty;
                }
                else
                {
                    logits[token] *= penalty;
                }
            }
        }

        private static int GetSafeGeneratedTokenCount(
            int contextLength,
            int promptTokenCount,
            int requestedGeneratedTokenCount)
        {
            var available = contextLength - promptTokenCount;

            if (available <= 0)
            {
                throw new InvalidOperationException(
                    $"Prompt length {promptTokenCount} does not fit context length {contextLength}.");
            }

            return Math.Min(
                requestedGeneratedTokenCount,
                available);
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

        private static bool GetBoolEnvironmentVariable(
            string name,
            bool defaultValue)
        {
            var raw = Environment.GetEnvironmentVariable(name);

            if (string.IsNullOrWhiteSpace(raw))
            {
                return defaultValue;
            }

            return
                string.Equals(raw, "1", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(raw, "true", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(raw, "yes", StringComparison.OrdinalIgnoreCase);
        }

        private static void SkipIfMissing(
            string path)
        {
            if (!File.Exists(path))
            {
                throw new InvalidOperationException(
                    $"Required fixture is missing: {Path.GetFullPath(path)}");
            }
        }

        private sealed class SingleLogicalProcessorScope : IDisposable
        {
            private readonly ITestOutputHelper _output;
            private readonly Process _process;
            private readonly IntPtr _originalAffinity;
            private bool _disposed;

            private SingleLogicalProcessorScope(
                ITestOutputHelper output,
                Process process,
                IntPtr originalAffinity,
                bool isActive)
            {
                _output = output;
                _process = process;
                _originalAffinity = originalAffinity;
                IsActive = isActive;
            }

            public bool IsActive { get; }

            public static SingleLogicalProcessorScope TryEnter(
                ITestOutputHelper output)
            {
                var process = Process.GetCurrentProcess();

                try
                {
                    var original = process.ProcessorAffinity;

                    if (original == IntPtr.Zero)
                    {
                        output.WriteLine("Processor affinity is zero/unknown. Single-core restriction was not applied.");

                        return new SingleLogicalProcessorScope(
                            output,
                            process,
                            original,
                            isActive: false);
                    }

                    process.ProcessorAffinity = (IntPtr)1;

                    return new SingleLogicalProcessorScope(
                        output,
                        process,
                        original,
                        isActive: true);
                }
                catch (Exception ex) when (
                    ex is PlatformNotSupportedException ||
                    ex is NotSupportedException ||
                    ex is InvalidOperationException ||
                    ex is System.ComponentModel.Win32Exception)
                {
                    output.WriteLine($"Could not restrict process affinity to one logical CPU: {ex.Message}");

                    return new SingleLogicalProcessorScope(
                        output,
                        process,
                        IntPtr.Zero,
                        isActive: false);
                }
            }

            public void Dispose()
            {
                if (_disposed)
                {
                    return;
                }

                _disposed = true;

                if (!IsActive)
                {
                    return;
                }

                try
                {
                    _process.ProcessorAffinity = _originalAffinity;
                }
                catch (Exception ex) when (
                    ex is PlatformNotSupportedException ||
                    ex is NotSupportedException ||
                    ex is InvalidOperationException ||
                    ex is System.ComponentModel.Win32Exception)
                {
                    _output.WriteLine($"Could not restore process affinity: {ex.Message}");
                }
            }
        }
    }
}
