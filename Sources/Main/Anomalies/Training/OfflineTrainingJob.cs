// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Experimental;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Parameters;

namespace DevOnBike.Overfit.Anomalies.Training
{
    /// <summary>
    /// Trains a GPT model on historical metric CSV data and saves a checkpoint.
    ///
    /// Flow:
    ///   1. Load CSV via HistoricalCsvLoader
    ///   2. Tokenize MetricSnapshots → int[] via MetricTokenizer
    ///   3. Train GPT with next-token prediction loss
    ///   4. Save checkpoint.bin compatible with SlmRuntimeFactory.CreateGpt1()
    ///
    /// Usage:
    ///   var job    = new OfflineTrainingJob(OfflineTrainingConfig.Quick);
    ///   var result = await job.RunAsync("metrics.csv", "checkpoint.bin", progress, ct);
    ///   Console.WriteLine(result);
    /// </summary>
    public sealed class OfflineTrainingJob
    {
        private readonly GptTrainingConfig _cfg;

        public OfflineTrainingJob(GptTrainingConfig config)
        {
            _cfg = config ?? throw new ArgumentNullException(nameof(config));
        }

        public async Task<OfflineTrainingResult> RunAsync(
            string csvPath,
            string checkpointPath,
            IProgress<TrainingProgress>? progress = null,
            CancellationToken ct = default)
        {
            // 1. Load
            progress?.Report(new TrainingProgress { Phase = "Loading CSV", Step = 0, TotalSteps = _cfg.Steps });
            var snapshots = HistoricalCsvLoader.Load(csvPath, out var skipped);
            if (snapshots.Count == 0)
            {
                throw new InvalidDataException($"No snapshots loaded from '{csvPath}'. Skipped rows: {skipped}");
            }

            progress?.Report(new TrainingProgress
            {
                Phase      = $"Loaded {snapshots.Count:N0} snapshots, {skipped} skipped",
                Step       = 0,
                TotalSteps = _cfg.Steps,
            });

            // 2. Tokenize
            var tokenizer = new Gpt.MetricTokenizer();
            var allTokens = tokenizer.EncodeSequence(snapshots);

            // 3. Model
            var gptConfig = new GPT1Config
            {
                VocabSize     = Gpt.MetricTokenizer.VocabSize,
                ContextLength = _cfg.ContextLength,
                DModel        = _cfg.DModel,
                NHeads        = _cfg.NHeads,
                NLayers       = _cfg.NLayers,
                DFF           = _cfg.DModel * 4,
                TieWeights    = false,
                PreLayerNorm  = true,
            };

            using var model = new GPT1Model(gptConfig);
            using var optimizer = new Adam(model.TrainableParameters(), _cfg.LearningRateMax)
            {
                UseAdamW    = true,
                WeightDecay = _cfg.WeightDecay,
            };
            model.Train();

            var trainSize = (int)(allTokens.Length * 0.9);
            var trainIds  = allTokens.AsSpan(0, trainSize).ToArray();
            var valIds    = allTokens.AsSpan(trainSize).ToArray();
            var rng       = new Random(_cfg.Seed);
            var sw        = System.Diagnostics.Stopwatch.StartNew();

            float initialLoss  = 0f;
            float finalValLoss = 0f;
            float windowLoss   = 0f;

            // Parallel SDPA backward — -27% czasu backward przy dużych modelach.
            ExperimentalLanguageModelOptions.EnableParallelAttentionBackward = true;

            // Data parallel: N workers share gradient per step.
            // Linear Scaling Rule: lr × sqrt(WorkerCount).
            var workerCount = _cfg.WorkerCount;
            var lrScale     = MathF.Sqrt(workerCount);
            var lrMax       = _cfg.LearningRateMax * lrScale;
            var lrMin       = _cfg.LearningRateMin * lrScale;

            // Worker models share weights with master — each has its own graph and gradients.
            var workers = Enumerable.Range(0, workerCount)
                .Select(_ => new GPT1Model(gptConfig))
                .ToList();

            using var graph = new ComputationGraph(_cfg.ArenaSize);

            // 4. Train
            for (var step = 0; step < _cfg.Steps && !ct.IsCancellationRequested; step++)
            {
                // Copy master weights to workers
                CopyParametersToWorkers(model.TrainableParameters(), workers);

                // Parallel forward+backward across workers
                var losses   = new float[workerCount];
                var workerGraphs = workers.Select(_ => new ComputationGraph(_cfg.ArenaSize / workerCount)).ToList();

                Parallel.For(0, workerCount, w =>
                {
                    var (inp, tgt) = Sample(trainIds, _cfg.ContextLength, rng);
                    workerGraphs[w].Reset();
                    workers[w].InvalidateAllCaches();
                    var logits = workers[w].Forward(workerGraphs[w], inp, batchSize: 1, _cfg.ContextLength);
                    losses[w]  = LossAndGrad(logits, tgt, _cfg.ContextLength, gptConfig.VocabSize);
                    workerGraphs[w].BackwardFromGrad(logits);
                    logits.Dispose();
                });

                // Aggregate gradients from workers → master
                AggregateGradients(model.TrainableParameters(), workers, scale: 1f / workerCount);
                foreach (var wg in workerGraphs)
                {
                    wg.Dispose();
                }

                var loss = losses.Average();
                windowLoss += loss;

                optimizer.ZeroGrad();
                ClipGradNorm(model.TrainableParameters(), _cfg.MaxGradNorm);
                var cosine = 0.5f * (1f + MathF.Cos(MathF.PI * (float)step / _cfg.Steps));
                optimizer.LearningRate = lrMin + (lrMax - lrMin) * cosine;
                optimizer.Step();

                if (step == 0)
                {
                    initialLoss = loss;
                }

                if ((step + 1) % _cfg.ReportEvery == 0 || step == _cfg.Steps - 1)
                {
                    var avg     = windowLoss / _cfg.ReportEvery;
                    windowLoss  = 0f;
                    var valLoss = Evaluate(model, graph, gptConfig, valIds, rng);
                    finalValLoss = valLoss;

                    progress?.Report(new TrainingProgress
                    {
                        Phase      = "Training",
                        Step       = step + 1,
                        TotalSteps = _cfg.Steps,
                        TrainLoss  = avg,
                        ValLoss    = valLoss,
                        Elapsed    = sw.Elapsed,
                    });
                }
            }

            foreach (var w in workers)
            {
                w.Dispose();
            }
            ct.ThrowIfCancellationRequested();

            // 5. Save checkpoint
            Directory.CreateDirectory(Path.GetDirectoryName(checkpointPath) ?? ".");
            await using var fs = File.Create(checkpointPath);
            using var bw = new BinaryWriter(fs);
            model.Save(bw);

            return new OfflineTrainingResult
            {
                SnapshotsLoaded = snapshots.Count,
                SkippedCsvRows  = skipped,
                InitialLoss     = initialLoss,
                FinalValLoss    = finalValLoss,
                CheckpointPath  = checkpointPath,
                TrainingTime    = sw.Elapsed,
            };
        }

        // ── Helpers ──────────────────────────────────────────────────────────

        private static (int[] input, int[] target) Sample(int[] corpus, int seqLen, Random rng)
        {
            var start = rng.Next(0, corpus.Length - seqLen - 1);
            return (corpus.AsSpan(start, seqLen).ToArray(),
                    corpus.AsSpan(start + 1, seqLen).ToArray());
        }

        private static float LossAndGrad(AutogradNode logits, int[] targets, int seqLen, int vocab)
        {
            var arr  = logits.DataView.AsReadOnlySpan().ToArray();
            var grad = new float[seqLen * vocab];
            var loss = new float[seqLen];

            System.Threading.Tasks.Parallel.For(0, seqLen, t =>
            {
                var off = t * vocab;
                var tgt = targets[t];
                var max = arr[off];
                for (var v = 1; v < vocab; v++)
                {
                    if (arr[off + v] > max)
                    {
                        max = arr[off + v];
                    }
                }
                var sum = 0f;
                for (var v = 0; v < vocab; v++)
                {
                    sum += MathF.Exp(arr[off + v] - max);
                }
                loss[t] = max + MathF.Log(sum) - arr[off + tgt];
                var sc = 1f / seqLen;
                for (var v = 0; v < vocab; v++)
                {
                    var sm = MathF.Exp(arr[off + v] - max) / sum;
                    grad[off + v] = (sm - (v == tgt ? 1f : 0f)) * sc;
                }
            });

            grad.AsSpan().CopyTo(logits.GradView.AsSpan());
            return loss.Sum() / seqLen;
        }

        private static void CopyParametersToWorkers(
            IEnumerable<Parameter> masterParams,
            List<GPT1Model> workers)
        {
            var master = masterParams.ToList();
            foreach (var worker in workers)
            {
                var wp = worker.TrainableParameters().ToList();
                for (var i = 0; i < master.Count && i < wp.Count; i++)
                {
                    master[i].DataReadOnlySpan.CopyTo(wp[i].DataSpan);
                }
            }
        }

        private static void AggregateGradients(
            IEnumerable<Parameter> masterParams,
            List<GPT1Model> workers,
            float scale)
        {
            var master = masterParams.ToList();
            master.ForEach(p => p.GradSpan.Clear());

            foreach (var worker in workers)
            {
                var wp = worker.TrainableParameters().ToList();
                for (var i = 0; i < master.Count && i < wp.Count; i++)
                {
                    var mg = master[i].GradSpan;
                    var wg = wp[i].GradSpan;
                    for (var j = 0; j < mg.Length; j++)
                    {
                        mg[j] += wg[j] * scale;
                    }
                }
            }
        }

        private static void ClipGradNorm(IEnumerable<Parameter> parameters, float maxNorm)
        {
            var list = parameters.ToList();
            var sq   = 0f;
            foreach (var p in list) { var g = p.GradSpan; for (var i = 0; i < g.Length; i++)
                {
                    sq += g[i] * g[i];
                }
            }
            var n = MathF.Sqrt(sq);
            if (n <= maxNorm)
            {
                return;
            }
            var s = maxNorm / (n + 1e-6f);
            foreach (var p in list) { var g = p.GradSpan; for (var i = 0; i < g.Length; i++)
                {
                    g[i] *= s;
                }
            }
        }

        private float Evaluate(GPT1Model model, ComputationGraph graph, GPT1Config config, int[] val, Random rng)
        {
            model.Eval();
            var total = 0f;
            for (var s = 0; s < _cfg.ValSteps; s++)
            {
                var (inp, tgt) = Sample(val, _cfg.ContextLength, rng);
                graph.Reset(); model.InvalidateAllCaches();
                var l = model.Forward(graph, inp, batchSize: 1, _cfg.ContextLength);
                total += LossAndGrad(l, tgt, _cfg.ContextLength, config.VocabSize);
                l.Dispose();
            }
            model.Train();
            return total / _cfg.ValSteps;
        }
    }

    public sealed class TrainingProgress
    {
        public string  Phase      { get; init; } = string.Empty;
        public int     Step       { get; init; }
        public int     TotalSteps { get; init; }
        public float   TrainLoss  { get; init; }
        public float   ValLoss    { get; init; }
        public TimeSpan Elapsed   { get; init; }

        public override string ToString() =>
            $"[{Phase}] {Step}/{TotalSteps} train={TrainLoss:F4} val={ValLoss:F4} {Elapsed:mm\\:ss}";
    }
}
