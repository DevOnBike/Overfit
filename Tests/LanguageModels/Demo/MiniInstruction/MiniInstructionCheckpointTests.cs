// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Text;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tokenization;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Demo.MiniInstruction
{
    /// <summary>
    /// Mini instruction-tuning demo.
    ///
    /// This is intentionally small and synthetic.
    ///
    /// Goal:
    /// - prove that the GPT training/checkpoint/runtime pipeline can learn a
    ///   User/Assistant style format,
    /// - write checkpoint_instruction.bin,
    /// - load it back,
    /// - show cached KV generation from instruction prompts,
    /// - validate cached/legacy parity and 0 B cached continuation allocation.
    ///
    /// This is not a general-purpose assistant and not a real instruction model.
    /// It is an overfit demo over a controlled synthetic corpus.
    /// </summary>
    public class MiniInstructionCheckpointTests
    {
        private const string InstructionCheckpointPath = "test_fixtures/checkpoint_instruction.bin";

        private const int ContextLength = 128;
        private const int BatchSize = 8;
        private const int DefaultTrainingSteps = 5_000;
        private const int ArenaSize = 180_000_000;

        private const int ReportEvery = 100;
        private const int EvalEvery = 1_000;
        private const int ValidationBatches = 20;

        private const int DModel = 128;
        private const int HeadCount = 4;
        private const int LayerCount = 4;
        private const int DFF = 512;

        private const float LearningRateMax = 3e-4f;
        private const float LearningRateMin = 3e-5f;
        private const float WeightDecay = 0.02f;
        private const float MaxGradNorm = 1.0f;

        private const int DisplayTopK = 8;
        private const float DisplayTemperature = 0.70f;
        private const float DisplayRepetitionPenalty = 1.12f;
        private const int DisplayRepetitionWindow = 48;
        private const int DisplaySeed = 42;

        private const int RequestedNewTokens = 80;
        private const int ParityRequestedNewTokens = 32;
        private const int AllocationMeasuredTokenCount = 8;

        private readonly ITestOutputHelper _output;

        public MiniInstructionCheckpointTests(
            ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact(Skip = "Manual long-running mini instruction demo. Remove Skip locally, run once, then restore Skip.")]
        [Trait("Category", "Demo")]
        [Trait("Category", "LongRunning")]
        public void Demo_Train_MiniInstruction_AndWriteCheckpointBin()
        {
            var steps = GetIntEnvironmentVariable(
                "OVERFIT_MINI_INSTRUCTION_STEPS",
                DefaultTrainingSteps);

            var corpus = BuildInstructionCorpus(repetitions: 120);
            var tokenizer = CharacterTokenizer.FromCorpus(corpus);
            var allIds = tokenizer.Encode(corpus);

            var trainSize = (int)(allIds.Length * 0.9);
            var trainIds = allIds.AsSpan(0, trainSize).ToArray();
            var valIds = allIds.AsSpan(trainSize).ToArray();

            var config = CreateConfig(tokenizer.VocabSize);

            _output.WriteLine("=== Overfit MiniInstruction Training Demo ===");
            _output.WriteLine($"Corpus chars: {corpus.Length:N0}");
            _output.WriteLine($"Train tokens: {trainIds.Length:N0}");
            _output.WriteLine($"Validation tokens: {valIds.Length:N0}");
            _output.WriteLine($"Vocab: {tokenizer.VocabSize}");
            _output.WriteLine($"Model: {config}");
            _output.WriteLine($"Parameters: {config.ParameterCount:N0}");
            _output.WriteLine($"Steps: {steps:N0}");
            _output.WriteLine($"BatchSize: {BatchSize}");
            _output.WriteLine($"ContextLength: {ContextLength}");
            _output.WriteLine($"LR schedule: {LearningRateMax} -> {LearningRateMin}");
            _output.WriteLine($"WeightDecay: {WeightDecay}");
            _output.WriteLine($"Checkpoint: {Path.GetFullPath(InstructionCheckpointPath)}");
            _output.WriteLine(string.Empty);

            using var model = new GPT1Model(config);
            using var optimizer = new Adam(model.TrainableParameters(), LearningRateMax)
            {
                UseAdamW = true,
                WeightDecay = WeightDecay
            };

            model.Train();

            var parameters = model
                .TrainableParameters()
                .ToList();

            var rng = new Random(42);
            var firstLoss = 0f;
            var lastLoss = 0f;
            var windowLoss = 0f;
            var bestValLoss = float.PositiveInfinity;
            var finalValLoss = float.PositiveInfinity;
            var stopwatch = Stopwatch.StartNew();

            using var graph = new ComputationGraph(ArenaSize);

            for (var step = 0; step < steps; step++)
            {
                var (inputIds, targetIds) = SampleBatch(
                    trainIds,
                    ContextLength,
                    BatchSize,
                    rng);

                optimizer.ZeroGrad();

                graph.Reset();
                model.InvalidateAllCaches();

                var logits = model.Forward(
                    graph,
                    inputIds,
                    BatchSize,
                    ContextLength);

                var loss = ComputeLossAndSeedGradSequential(
                    logits,
                    targetIds,
                    ContextLength,
                    BatchSize,
                    config.VocabSize);

                Assert.False(float.IsNaN(loss), $"NaN loss at step {step}.");
                Assert.False(float.IsInfinity(loss), $"Infinite loss at step {step}.");

                graph.BackwardFromGrad(logits);
                logits.Dispose();

                ClipGradNorm(parameters, MaxGradNorm);

                optimizer.LearningRate = CosineDecay(
                    LearningRateMax,
                    LearningRateMin,
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
                    (step + 1) % ReportEvery == 0 ||
                    step + 1 == steps)
                {
                    var denominator = step == 0
                        ? 1
                        : Math.Min(ReportEvery, step + 1);

                    var avg = windowLoss / denominator;
                    windowLoss = 0f;

                    _output.WriteLine(
                        $"Step {step + 1,6}/{steps} | loss={loss:F4} | avg={avg:F4} | lr={optimizer.LearningRate:E2} | elapsed={stopwatch.Elapsed:hh\\:mm\\:ss}");
                }

                if ((step + 1) % EvalEvery == 0 || step + 1 == steps)
                {
                    var valLoss = EvaluateLossSequential(
                        model,
                        valIds,
                        config,
                        ContextLength,
                        rng,
                        ValidationBatches);

                    finalValLoss = valLoss;
                    bestValLoss = MathF.Min(bestValLoss, valLoss);

                    _output.WriteLine(
                        $"Validation @ step {step + 1,6}: val={valLoss:F4} | best={bestValLoss:F4}");

                    var sample = GenerateInstructionSample(
                        model,
                        tokenizer,
                        "User: What is 2 plus 2?\nAssistant:",
                        RequestedNewTokens);

                    var answer = ExtractFirstAssistantAnswer(
                        sample,
                        "User: What is 2 plus 2?\nAssistant:");

                    _output.WriteLine("Sample answer:");
                    _output.WriteLine(answer);
                    _output.WriteLine(string.Empty);

                    model.Train();
                }
            }

            model.Eval();

            Directory.CreateDirectory(Path.GetDirectoryName(InstructionCheckpointPath)!);

            using (var stream = File.Create(InstructionCheckpointPath))
            using (var writer = new BinaryWriter(stream))
            {
                model.Save(writer);
            }

            _output.WriteLine("");
            _output.WriteLine($"Loss: {firstLoss:F4} -> {lastLoss:F4}");
            _output.WriteLine($"Final validation loss: {finalValLoss:F4}");
            _output.WriteLine($"Best validation loss: {bestValLoss:F4}");
            _output.WriteLine($"Total elapsed: {stopwatch.Elapsed:hh\\:mm\\:ss}");
            _output.WriteLine($"Checkpoint written: {Path.GetFullPath(InstructionCheckpointPath)}");

            Assert.True(File.Exists(InstructionCheckpointPath));
            Assert.True(new FileInfo(InstructionCheckpointPath).Length > 0);
            Assert.True(lastLoss < firstLoss, $"Loss did not go down: {firstLoss:F4} -> {lastLoss:F4}.");
            Assert.False(float.IsNaN(finalValLoss) || float.IsInfinity(finalValLoss), "Final validation loss is NaN/Infinity.");
        }

        [Fact]
        [Trait("Category", "Demo")]
        public void Demo_LoadCheckpoint_AndShowMiniInstructionGeneration()
        {
            if (!File.Exists(InstructionCheckpointPath))
            {
                throw new InvalidOperationException(
                    $"Checkpoint '{InstructionCheckpointPath}' not found. Run Demo_Train_MiniInstruction_AndWriteCheckpointBin first.");
            }

            var corpus = BuildInstructionCorpus(repetitions: 120);
            var tokenizer = CharacterTokenizer.FromCorpus(corpus);
            var config = CreateConfig(tokenizer.VocabSize);

            using var model = new GPT1Model(config);

            using (var stream = File.OpenRead(InstructionCheckpointPath))
            using (var reader = new BinaryReader(stream))
            {
                model.Load(reader);
            }

            model.Eval();

            var prompts = new[]
            {
                "User: What is 2 plus 2?\nAssistant:",
                "User: What is the capital of France?\nAssistant:",
                "User: What is Overfit?\nAssistant:",
                "User: Say hello.\nAssistant:"
            };

            _output.WriteLine("=== Overfit MiniInstruction Cached Runtime Demo ===");
            _output.WriteLine("");
            _output.WriteLine("This is a synthetic overfit instruction-format demo.");
            _output.WriteLine("It is not a general-purpose assistant.");
            _output.WriteLine("");

            foreach (var prompt in prompts)
            {
                var generated = GenerateInstructionSample(
                    model,
                    tokenizer,
                    prompt,
                    RequestedNewTokens);

                var answer = ExtractFirstAssistantAnswer(
                    generated,
                    prompt);

                _output.WriteLine("Prompt:");
                _output.WriteLine(prompt);
                _output.WriteLine("");
                _output.WriteLine("Answer:");
                _output.WriteLine(answer);
                _output.WriteLine("");

                AssertInstructionTextLooksValid(generated);
                AssertInstructionAnswerLooksValid(answer);
                AssertExpectedDemoAnswer(prompt, answer);
            }

            AssertDemoCachedMatchesLegacyGreedy(
                model,
                tokenizer,
                prompts[0],
                GetSafeGeneratedTokenCount(
                    config.ContextLength,
                    tokenizer.Encode(prompts[0]).Length,
                    ParityRequestedNewTokens));

            AssertDemoCachedContinuationDoesNotAllocate(
                model,
                tokenizer,
                prompts[0],
                AllocationMeasuredTokenCount);

            _output.WriteLine("Validation:");
            _output.WriteLine("Legacy parity: OK");
            _output.WriteLine($"Continuation allocation check: 0 B for {AllocationMeasuredTokenCount} tokens");
        }

        private static GPT1Config CreateConfig(
            int vocabSize)
        {
            return new GPT1Config
            {
                VocabSize = vocabSize,
                ContextLength = ContextLength,
                DModel = DModel,
                NHeads = HeadCount,
                NLayers = LayerCount,
                DFF = DFF,
                TieWeights = false,
                PreLayerNorm = true
            };
        }

        private static string BuildInstructionCorpus(
            int repetitions)
        {
            var examples = new (string User, string Assistant)[]
            {
                ("What is 2 plus 2?", "4."),
                ("What is two plus two?", "4."),
                ("Add 2 and 2.", "4."),
                ("What is 3 plus 5?", "8."),
                ("What is 10 minus 4?", "6."),
                ("What is 6 times 7?", "42."),
                ("What is the capital of France?", "Paris."),
                ("What is the capital of Poland?", "Warsaw."),
                ("What is the capital of Germany?", "Berlin."),
                ("What color is the sky on a clear day?", "Blue."),
                ("What color is grass?", "Green."),
                ("Say hello.", "Hello."),
                ("Say goodbye.", "Goodbye."),
                ("Answer yes.", "Yes."),
                ("Answer no.", "No."),
                ("What is Overfit?", "Overfit is a C# deep-learning engine."),
                ("What language is Overfit written in?", "C#."),
                ("What does checkpoint.bin store?", "Model weights."),
                ("What does cached KV runtime do?", "It reuses key and value tensors during generation."),
                ("What does zero allocation mean?", "No managed allocations on the measured hot path."),
                ("Is this a real assistant?", "No. This is a small demo model."),
                ("What is a neural network?", "A neural network is a model made of trainable layers."),
                ("What is training?", "Training changes model weights to reduce loss."),
                ("What is inference?", "Inference runs a trained model to produce output."),
                ("What is a token?", "A token is a small piece of text used by a language model."),
                ("What is GPT?", "GPT is a transformer language model."),
                ("Complete: User asks and Assistant", "answers."),
                ("Who are you?", "I am a small Overfit demo model."),
                ("What should you do when unsure?", "Say that you are unsure."),
                ("Give a short answer.", "OK."),
                ("Keep the answer short.", "Short answer."),
                ("What is 1 plus 1?", "2."),
                ("What is 5 plus 5?", "10."),
                ("What is 9 minus 3?", "6."),
                ("What is 4 times 3?", "12.")
            };

            var builder = new StringBuilder();

            for (var repetition = 0; repetition < repetitions; repetition++)
            {
                for (var i = 0; i < examples.Length; i++)
                {
                    var example = examples[(i + repetition) % examples.Length];

                    builder
                        .Append("User: ")
                        .Append(example.User)
                        .Append('\n')
                        .Append("Assistant: ")
                        .Append(example.Assistant)
                        .Append("\n\n");
                }
            }

            return builder.ToString();
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

        private static string GenerateInstructionSample(
            GPT1Model model,
            CharacterTokenizer tokenizer,
            string prompt,
            int requestedNewTokens)
        {
            var promptTokens = tokenizer.Encode(prompt);
            var safeTokens = GetSafeGeneratedTokenCount(
                model.Config.ContextLength,
                promptTokens.Length,
                requestedNewTokens);

            return GenerateDisplaySampleWithRepetitionPenalty(
                model,
                tokenizer,
                prompt,
                safeTokens);
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
            var random = new Random(DisplaySeed);

            for (var i = 0; i < promptTokens.Length; i++)
            {
                adapter.DecodeNextToken(
                    promptTokens[i],
                    logits);
            }

            var sampling = new SamplingOptions(
                SamplingStrategy.TopK,
                temperature: DisplayTemperature,
                topK: DisplayTopK,
                topP: 1.0f,
                seed: DisplaySeed);

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
                    DisplayRepetitionWindow,
                    DisplayRepetitionPenalty);

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

        private static string ExtractFirstAssistantAnswer(
            string generatedText,
            string prompt)
        {
            var candidate = generatedText.StartsWith(prompt, StringComparison.Ordinal)
                ? generatedText[prompt.Length..]
                : ExtractAfterFirstAssistantMarker(generatedText);

            var stopIndex = FindFirstStopMarker(candidate);

            if (stopIndex >= 0)
            {
                candidate = candidate[..stopIndex];
            }

            return candidate.Trim();
        }

        private static string ExtractAfterFirstAssistantMarker(
            string generatedText)
        {
            const string marker = "Assistant:";

            var index = generatedText.IndexOf(
                marker,
                StringComparison.OrdinalIgnoreCase);

            return index < 0
                ? generatedText
                : generatedText[(index + marker.Length)..];
        }

        private static int FindFirstStopMarker(
            string text)
        {
            var markers = new[]
            {
                "\r\n\r\nUser:",
                "\n\nUser:",
                "\r\nUser:",
                "\nUser:"
            };

            var best = -1;

            foreach (var marker in markers)
            {
                var index = text.IndexOf(
                    marker,
                    StringComparison.OrdinalIgnoreCase);

                if (index >= 0 && (best < 0 || index < best))
                {
                    best = index;
                }
            }

            return best;
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

        private static void AssertDemoCachedMatchesLegacyGreedy(
            GPT1Model model,
            CharacterTokenizer tokenizer,
            string prompt,
            int maxNewTokens)
        {
            var promptTokens = tokenizer.Encode(prompt);

            var legacyTokens = new int[maxNewTokens];
            var cachedTokens = new int[maxNewTokens];

            using var legacy = SlmRuntimeFactory.CreateGpt1(
                model,
                SlmRuntimeMode.Legacy);

            using var cached = SlmRuntimeFactory.CreateGpt1(
                model,
                SlmRuntimeMode.Cached);

            var legacyGenerated = legacy.GenerateGreedy(
                promptTokens,
                legacyTokens,
                maxNewTokens);

            var cachedGenerated = cached.GenerateGreedy(
                promptTokens,
                cachedTokens,
                maxNewTokens);

            Assert.Equal(legacyGenerated, cachedGenerated);
            Assert.Equal(
                legacyTokens.AsSpan(0, legacyGenerated).ToArray(),
                cachedTokens.AsSpan(0, cachedGenerated).ToArray());
        }

        private static void AssertDemoCachedContinuationDoesNotAllocate(
            GPT1Model model,
            CharacterTokenizer tokenizer,
            string prompt,
            int measuredTokenCount)
        {
            var promptTokens = tokenizer.Encode(prompt);
            var sampling = SamplingOptions.Greedy;

            using var runtime = SlmRuntimeFactory.CreateGpt1(
                model,
                SlmRuntimeMode.Cached);

            runtime.Session.Reset(promptTokens);

            // Warm up the continuation path before allocation measurement.
            _ = runtime.Session.GenerateNextToken(in sampling);

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < measuredTokenCount; i++)
            {
                _ = runtime.Session.GenerateNextToken(in sampling);
            }

            var after = GC.GetAllocatedBytesForCurrentThread();

            Assert.Equal(0, after - before);
        }

        private static void AssertInstructionTextLooksValid(
            string text)
        {
            Assert.False(string.IsNullOrWhiteSpace(text));
            AssertNoNullCharacters(text);
            Assert.Contains("User:", text, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("Assistant:", text, StringComparison.OrdinalIgnoreCase);
        }

        private static void AssertInstructionAnswerLooksValid(
            string answer)
        {
            Assert.False(string.IsNullOrWhiteSpace(answer));
            AssertNoNullCharacters(answer);
            Assert.DoesNotContain("User:", answer, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("Assistant:", answer, StringComparison.OrdinalIgnoreCase);
        }

        private static void AssertExpectedDemoAnswer(
            string prompt,
            string answer)
        {
            if (prompt.Contains("2 plus 2", StringComparison.OrdinalIgnoreCase))
            {
                Assert.Contains("4", answer, StringComparison.OrdinalIgnoreCase);
                return;
            }

            if (prompt.Contains("capital of France", StringComparison.OrdinalIgnoreCase))
            {
                Assert.Contains("Paris", answer, StringComparison.OrdinalIgnoreCase);
                return;
            }

            if (prompt.Contains("Overfit", StringComparison.OrdinalIgnoreCase))
            {
                Assert.Contains("Overfit", answer, StringComparison.OrdinalIgnoreCase);
                Assert.Contains("C#", answer, StringComparison.OrdinalIgnoreCase);
                return;
            }

            if (prompt.Contains("Say hello", StringComparison.OrdinalIgnoreCase))
            {
                Assert.Contains("Hello", answer, StringComparison.OrdinalIgnoreCase);
            }
        }

        private static void AssertNoNullCharacters(
            string text)
        {
            for (var i = 0; i < text.Length; i++)
            {
                if (text[i] == '\0')
                {
                    throw new InvalidOperationException(
                        $"Generated text contains a null character at index {i}.");
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
    }
}
