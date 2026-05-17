// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Linq;
using DevOnBike.Overfit.Anomalies.Gpt;
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
            var tokenizer = new MetricTokenizer();
            var allTokens = tokenizer.EncodeSequence(snapshots);

            // 3. Model
            var gptConfig = new GPT1Config
            {
                VocabSize     = MetricTokenizer.VocabSize,
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
            var sw        = Stopwatch.StartNew();

            var initialLoss  = 0f;
            var finalValLoss = 0f;
            var windowLoss   = 0f;

            // Parallel SDPA backward — -27% backward time on large models.
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

            // Per-worker scratch — allocated once and reused every step. These (the worker
            // graphs especially, each carrying its own arena) were previously re-created and
            // disposed inside the step loop, churning a whole arena's worth of memory per
            // step; the Reset() call below is the giveaway that reuse was always intended.
            var workerGraphs  = new ComputationGraph[workerCount];
            var sampleInputs  = new int[workerCount][];
            var sampleTargets = new int[workerCount][];
            var lossScratch   = new float[workerCount][];
            var losses        = new float[workerCount];

            for (var w = 0; w < workerCount; w++)
            {
                workerGraphs[w]  = new ComputationGraph(_cfg.ArenaSize / workerCount);
                sampleInputs[w]  = new int[_cfg.ContextLength];
                sampleTargets[w] = new int[_cfg.ContextLength];
                lossScratch[w]   = new float[_cfg.ContextLength];
            }

            // 4. Train
            for (var step = 0; step < _cfg.Steps && !ct.IsCancellationRequested; step++)
            {
                // Copy master weights to workers
                CopyParametersToWorkers(model.TrainableParameters(), workers);

                // Parallel forward+backward across workers — reusing the per-worker graphs.
                Parallel.For(0, workerCount, w =>
                {
                    Sample(trainIds, _cfg.ContextLength, rng, sampleInputs[w], sampleTargets[w]);
                    workerGraphs[w].Reset();
                    workers[w].InvalidateAllCaches();

                    // BackwardFromGrad accumulates into Parameter.Grad, and graph.Reset()
                    // does not touch parameter gradients — so each worker must clear its
                    // own grads every step. Otherwise AggregateGradients would read a
                    // running sum over all previous steps instead of this step's gradient.
                    ZeroGrads(workers[w]);

                    var logits = workers[w].Forward(workerGraphs[w], sampleInputs[w], batchSize: 1, _cfg.ContextLength);
                    losses[w]  = LossAndGrad(logits, sampleTargets[w], _cfg.ContextLength, gptConfig.VocabSize, lossScratch[w]);
                    workerGraphs[w].BackwardFromGrad(logits);
                    logits.Dispose();
                });

                // Aggregate gradients from workers → master
                AggregateGradients(model.TrainableParameters(), workers, scale: 1f / workerCount);

                var lossSum = 0f;
                for (var w = 0; w < workerCount; w++)
                {
                    lossSum += losses[w];
                }

                var loss = lossSum / workerCount;
                windowLoss += loss;

                // Do NOT call optimizer.ZeroGrad() here. Master gradients are produced
                // fresh by AggregateGradients (which clears then accumulates) immediately
                // above. Zeroing them now would wipe the just-aggregated gradient, and
                // Step() would run on zeros — only AdamW weight decay would apply, so the
                // model would never learn (loss would merely decay toward ln(VocabSize)).
                ClipGradNorm(model.TrainableParameters(), _cfg.MaxGradNorm);
                var cosine = 0.5f * (1f + MathF.Cos(MathF.PI * step / _cfg.Steps));
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

            foreach (var wg in workerGraphs)
            {
                wg.Dispose();
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

        private static void Sample(
            int[] corpus,
            int seqLen,
            Random rng,
            Span<int> input,
            Span<int> target)
        {
            var start = rng.Next(0, corpus.Length - seqLen - 1);
            corpus.AsSpan(start, seqLen).CopyTo(input);
            corpus.AsSpan(start + 1, seqLen).CopyTo(target);
        }

        private static float LossAndGrad(
            AutogradNode logits,
            int[] targets,
            int seqLen,
            int vocab,
            float[] lossScratch)
        {
            // Read logits and write gradients straight through the node's own spans —
            // each parallel iteration owns a disjoint [t*vocab, (t+1)*vocab) slice — so
            // no per-call logits copy or gradient buffer is allocated.
            Parallel.For(0, seqLen, t =>
            {
                var data = logits.DataView.AsReadOnlySpan();
                var grad = logits.GradView.AsSpan();
                var off = t * vocab;
                var tgt = targets[t];
                var max = data[off];
                for (var v = 1; v < vocab; v++)
                {
                    if (data[off + v] > max)
                    {
                        max = data[off + v];
                    }
                }
                var sum = 0f;
                for (var v = 0; v < vocab; v++)
                {
                    sum += MathF.Exp(data[off + v] - max);
                }
                lossScratch[t] = max + MathF.Log(sum) - data[off + tgt];
                var sc = 1f / seqLen;
                for (var v = 0; v < vocab; v++)
                {
                    var sm = MathF.Exp(data[off + v] - max) / sum;
                    grad[off + v] = (sm - (v == tgt ? 1f : 0f)) * sc;
                }
            });

            var total = 0f;
            for (var t = 0; t < seqLen; t++)
            {
                total += lossScratch[t];
            }

            return total / seqLen;
        }

        private static void ZeroGrads(GPT1Model model)
        {
            foreach (var p in model.TrainableParameters())
            {
                p.ZeroGrad();
            }
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

            // Reused across all validation steps — no per-step array allocation.
            var input       = new int[_cfg.ContextLength];
            var target      = new int[_cfg.ContextLength];
            var lossScratch = new float[_cfg.ContextLength];

            var total = 0f;
            for (var s = 0; s < _cfg.ValSteps; s++)
            {
                Sample(val, _cfg.ContextLength, rng, input, target);
                graph.Reset(); model.InvalidateAllCaches();
                var l = model.Forward(graph, input, batchSize: 1, _cfg.ContextLength);
                total += LossAndGrad(l, target, _cfg.ContextLength, config.VocabSize, lossScratch);
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
