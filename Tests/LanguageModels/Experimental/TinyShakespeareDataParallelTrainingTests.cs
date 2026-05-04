// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Experimental;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Experimental
{
    /// <summary>
    /// Manual experimental data-parallel TinyShakespeare trainer.
    ///
    /// This is not a normal correctness test and not a stable public training API.
    ///
    /// Master:
    /// - owns optimizer state,
    /// - receives averaged gradients,
    /// - writes checkpoint.bin.
    ///
    /// Workers:
    /// - each owns a GPT1Model,
    /// - each owns a ComputationGraph,
    /// - each computes gradients on a local mini-batch.
    /// </summary>
    public class TinyShakespeareDataParallelTrainingTests
    {
        private const string CorpusPath = "test_fixtures/tiny_shakespeare.txt";
        private const string CheckpointPath = "test_fixtures/checkpoint.bin";

        private const string Prompt = "ROMEO:";

        private const int SeqLen = 128;
        private const int DefaultLocalBatchSize = 8;
        private const int DefaultWorkerCount = 12;
        private const int DefaultDataParallelSteps = 5000;

        private const int ArenaSizePerWorker = 180_000_000;

        private const float LearningRateMax = 3e-4f;
        private const float LearningRateMin = 5e-5f;
        private const float WeightDecay = 0.1f;
        private const float MaxGradNorm = 1.0f;

        private readonly ITestOutputHelper _output;

        public TinyShakespeareDataParallelTrainingTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact(Skip = "Manual experimental long-running GPT data-parallel training demo. Remove Skip locally, run once, then restore Skip.")]
        [Trait("Category", "Demo")]
        [Trait("Category", "LongRunning")]
        [Trait("Category", "Experimental")]
        public void Demo_Train_TinyShakespeare_DataParallel_AndWriteCheckpointBin()
        {
            var previousParallelAttention =
                ExperimentalLanguageModelOptions.EnableParallelAttentionBackward;

            ExperimentalLanguageModelOptions.EnableParallelAttentionBackward = true;

            try
            {
                RunDataParallelTrainingDemo();
            }
            finally
            {
                ExperimentalLanguageModelOptions.EnableParallelAttentionBackward =
                    previousParallelAttention;
            }
        }

        private void RunDataParallelTrainingDemo()
        {
            if (!File.Exists(CorpusPath))
            {
                _output.WriteLine("TinyShakespeare corpus is missing.");
                _output.WriteLine($"Expected: {Path.GetFullPath(CorpusPath)}");
                return;
            }

            var steps = GetIntEnvironmentVariable(
                "OVERFIT_TINY_SHAKESPEARE_DP_STEPS",
                DefaultDataParallelSteps);

            var workerCount = GetIntEnvironmentVariable(
                "OVERFIT_TINY_SHAKESPEARE_DP_WORKERS",
                DefaultWorkerCount);

            var localBatchSize = GetIntEnvironmentVariable(
                "OVERFIT_TINY_SHAKESPEARE_DP_LOCAL_BATCH",
                DefaultLocalBatchSize);

            workerCount = Math.Max(1, workerCount);
            localBatchSize = Math.Max(1, localBatchSize);

            var globalBatchSize = workerCount * localBatchSize;

            var text = File.ReadAllText(CorpusPath);
            var tokenizer = CharacterTokenizer.FromCorpus(text);
            var allIds = tokenizer.Encode(text);

            if (allIds.Length <= SeqLen + 1)
            {
                throw new InvalidOperationException(
                    $"Corpus is too small. Need more than {SeqLen + 1} tokens.");
            }

            var config = CreateDemoConfig(tokenizer.VocabSize);

            using var master = new GPT1Model(config);
            master.Train();

            var masterParameters = master
                .TrainableParameters()
                .ToList();

            using var optimizer = new Adam(masterParameters, LearningRateMax)
            {
                UseAdamW = true,
                WeightDecay = WeightDecay
            };

            var workers = new WorkerState[workerCount];

            try
            {
                for (var i = 0; i < workers.Length; i++)
                {
                    workers[i] = new WorkerState(
                        workerIndex: i,
                        config: config,
                        localBatchSize: localBatchSize,
                        seqLen: SeqLen,
                        arenaSize: ArenaSizePerWorker,
                        rngSeed: 42 + i * 997);
                }

                CopyParametersToWorkers(masterParameters, workers);

                _output.WriteLine("=== Overfit TinyShakespeare Experimental Data-Parallel Training Demo ===");
                _output.WriteLine($"Corpus: {text.Length:N0} chars");
                _output.WriteLine($"Vocab: {tokenizer.VocabSize}");
                _output.WriteLine($"Model: {config}");
                _output.WriteLine($"Parameters: {config.ParameterCount:N0}");
                _output.WriteLine($"Steps: {steps}");
                _output.WriteLine($"WorkerCount: {workerCount}");
                _output.WriteLine($"LocalBatchSize: {localBatchSize}");
                _output.WriteLine($"GlobalBatchSize: {globalBatchSize}");
                _output.WriteLine($"SeqLen / ContextLength: {SeqLen}");
                _output.WriteLine($"LR schedule: {LearningRateMax} -> {LearningRateMin}");
                _output.WriteLine($"Experimental parallel attention backward: {ExperimentalLanguageModelOptions.EnableParallelAttentionBackward}");
                _output.WriteLine($"Checkpoint: {Path.GetFullPath(CheckpointPath)}");
                _output.WriteLine("");

                var started = Stopwatch.StartNew();
                var firstLoss = 0f;
                var lastLoss = 0f;
                var windowLoss = 0f;
                var lastLogElapsed = TimeSpan.Zero;
                var processedSequences = 0L;

                for (var step = 0; step < steps; step++)
                {
                    var lr = CosineDecay(
                        LearningRateMax,
                        LearningRateMin,
                        step,
                        steps);

                    optimizer.LearningRate = lr;

                    ClearGradients(masterParameters);

                    Parallel.For(
                        0,
                        workerCount,
                        new ParallelOptions
                        {
                            MaxDegreeOfParallelism = workerCount
                        },
                        workerIndex =>
                        {
                            var worker = workers[workerIndex];

                            worker.Loss = TrainWorkerStep(
                                worker,
                                allIds);
                        });

                    ReduceWorkerGradientsIntoMaster(
                        masterParameters,
                        workers,
                        scale: 1f / workerCount);

                    ClipGradNorm(
                        masterParameters,
                        MaxGradNorm);

                    optimizer.Step();

                    CopyParametersToWorkers(masterParameters, workers);

                    var avgLoss = 0f;

                    for (var i = 0; i < workers.Length; i++)
                    {
                        avgLoss += workers[i].Loss;
                    }

                    avgLoss /= workerCount;

                    if (step == 0)
                    {
                        firstLoss = avgLoss;
                    }

                    lastLoss = avgLoss;
                    windowLoss += avgLoss;
                    processedSequences += globalBatchSize;

                    if (step == 0 || (step + 1) % 25 == 0 || step + 1 == steps)
                    {
                        var elapsed = started.Elapsed;
                        var sinceLastLog = elapsed - lastLogElapsed;
                        lastLogElapsed = elapsed;

                        var denominator = step == 0 ? 1 : Math.Min(25, step + 1);
                        var rollingLoss = windowLoss / denominator;
                        windowLoss = 0f;

                        var sequencesPerSecond =
                            elapsed.TotalSeconds <= 0
                                ? 0.0
                                : processedSequences / elapsed.TotalSeconds;

                        _output.WriteLine(
                            $"Step {step + 1,5}/{steps} | loss={avgLoss:F4} | avg={rollingLoss:F4} | lr={lr:E2} | seq/s={sequencesPerSecond:F1} | elapsed={elapsed:mm\\:ss} | +{sinceLastLog.TotalSeconds:F1}s");
                    }
                }

                master.Eval();

                Directory.CreateDirectory(Path.GetDirectoryName(CheckpointPath)!);

                using (var stream = File.Create(CheckpointPath))
                using (var writer = new BinaryWriter(stream))
                {
                    master.Save(writer);
                }

                var sample = GenerateSample(
                    master,
                    tokenizer,
                    Prompt,
                    maxTokens: GetSafeGeneratedTokenCount(
                        config.ContextLength,
                        tokenizer.Encode(Prompt).Length,
                        requestedGeneratedTokenCount: 120));

                _output.WriteLine("");
                _output.WriteLine($"Loss: {firstLoss:F4} -> {lastLoss:F4}");
                _output.WriteLine($"Total elapsed: {started.Elapsed:mm\\:ss}");
                _output.WriteLine($"Processed sequences: {processedSequences:N0}");
                _output.WriteLine($"Average sequences/sec: {processedSequences / Math.Max(started.Elapsed.TotalSeconds, 0.001):F1}");
                _output.WriteLine("");
                _output.WriteLine("Training sample:");
                _output.WriteLine(sample);
                _output.WriteLine("");
                _output.WriteLine($"Checkpoint written: {Path.GetFullPath(CheckpointPath)}");

                Assert.True(File.Exists(CheckpointPath));
                Assert.True(new FileInfo(CheckpointPath).Length > 0);
                Assert.True(lastLoss < firstLoss, $"Loss did not go down: {firstLoss:F4} -> {lastLoss:F4}.");
            }
            finally
            {
                foreach (var worker in workers)
                {
                    worker?.Dispose();
                }
            }
        }

        private static GPT1Config CreateDemoConfig(int vocabSize)
        {
            return new GPT1Config
            {
                VocabSize = vocabSize,
                ContextLength = SeqLen,
                DModel = 128,
                NHeads = 4,
                NLayers = 4,
                DFF = 512,
                TieWeights = false,
                PreLayerNorm = true
            };
        }

        private static float TrainWorkerStep(
            WorkerState worker,
            int[] corpus)
        {
            SampleBatch(
                corpus,
                worker.SeqLen,
                worker.LocalBatchSize,
                worker.Rng,
                worker.InputIds,
                worker.TargetIds);

            ClearGradients(worker.Parameters);

            worker.Graph.Reset();
            worker.Model.InvalidateAllCaches();

            var logits = worker.Model.Forward(
                worker.Graph,
                worker.InputIds,
                worker.LocalBatchSize,
                worker.SeqLen);

            var loss = ComputeLossAndSeedGradSequential(
                logits,
                worker.TargetIds,
                worker.SeqLen,
                worker.LocalBatchSize,
                worker.Model.Config.VocabSize);

            worker.Graph.BackwardFromGrad(logits);
            logits.Dispose();

            return loss;
        }

        private static void CopyParametersToWorkers(
            IReadOnlyList<Parameter> masterParameters,
            IReadOnlyList<WorkerState> workers)
        {
            foreach (var worker in workers)
            {
                var workerParameters = worker.Parameters;

                if (workerParameters.Count != masterParameters.Count)
                {
                    throw new InvalidOperationException(
                        $"Worker parameter count mismatch. Master={masterParameters.Count}, Worker={workerParameters.Count}.");
                }

                for (var i = 0; i < masterParameters.Count; i++)
                {
                    masterParameters[i]
                        .DataSpan
                        .CopyTo(workerParameters[i].DataSpan);
                }
            }
        }

        private static void ClearGradients(
            IReadOnlyList<Parameter> parameters)
        {
            for (var i = 0; i < parameters.Count; i++)
            {
                parameters[i].GradSpan.Clear();
            }
        }

        private static void ReduceWorkerGradientsIntoMaster(
            IReadOnlyList<Parameter> masterParameters,
            IReadOnlyList<WorkerState> workers,
            float scale)
        {
            for (var p = 0; p < masterParameters.Count; p++)
            {
                var masterGrad = masterParameters[p].GradSpan;

                for (var w = 0; w < workers.Count; w++)
                {
                    var workerGrad = workers[w].Parameters[p].GradSpan;

                    if (workerGrad.Length != masterGrad.Length)
                    {
                        throw new InvalidOperationException(
                            $"Gradient length mismatch at parameter {p}. Master={masterGrad.Length}, Worker={workerGrad.Length}.");
                    }

                    for (var i = 0; i < masterGrad.Length; i++)
                    {
                        masterGrad[i] += workerGrad[i] * scale;
                    }
                }
            }
        }

        private static void SampleBatch(
            int[] corpus,
            int seqLen,
            int batchSize,
            Random rng,
            int[] inputIds,
            int[] targetIds)
        {
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

        private static string GenerateSample(
            GPT1Model model,
            CharacterTokenizer tokenizer,
            string prompt,
            int maxTokens)
        {
            model.Eval();

            var ids = tokenizer.Encode(prompt);
            var generated = model.Generate(
                ids,
                maxTokens);

            return prompt + tokenizer.Decode(generated);
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

        private sealed class WorkerState : IDisposable
        {
            public WorkerState(
                int workerIndex,
                GPT1Config config,
                int localBatchSize,
                int seqLen,
                int arenaSize,
                int rngSeed)
            {
                WorkerIndex = workerIndex;
                LocalBatchSize = localBatchSize;
                SeqLen = seqLen;

                Model = new GPT1Model(config);
                Model.Train();

                Graph = new ComputationGraph(arenaSize);

                Parameters = Model
                    .TrainableParameters()
                    .ToList();

                InputIds = new int[localBatchSize * seqLen];
                TargetIds = new int[localBatchSize * seqLen];
                Rng = new Random(rngSeed);
            }

            public int WorkerIndex { get; }

            public int LocalBatchSize { get; }

            public int SeqLen { get; }

            public GPT1Model Model { get; }

            public ComputationGraph Graph { get; }

            public List<Parameter> Parameters { get; }

            public int[] InputIds { get; }

            public int[] TargetIds { get; }

            public Random Rng { get; }

            public float Loss { get; set; }

            public void Dispose()
            {
                Graph.Dispose();
                Model.Dispose();
            }
        }
    }
}
