// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using DevOnBike.Overfit.Training;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Data.Mnist
{
    /// <summary>
    /// MNIST CNN training wall-time A/B: single replica (the beast-benchmark baseline) vs
    /// <see cref="DataParallelTrainer"/> with 4 / 8 replicas — same total samples (1 epoch, 60k, batch 64
    /// per replica), same model, avg loss reported as the quality proxy. Gate: ship the DP wiring only if
    /// ≥2.5× at 8 replicas with comparable loss. [LongFact] — needs MNIST files + minutes of CPU.
    /// </summary>
    public sealed class MnistDataParallelBenchTests
    {
        private const int TrainSize = 60_000;
        private const int BatchSize = 64;
        private readonly ITestOutputHelper _out;
        public MnistDataParallelBenchTests(ITestOutputHelper output) => _out = output;

        private sealed class Replica : IDisposable
        {
            public ConvLayer Conv = new(1, 8, 28, 28, 3);
            public LinearLayer Hidden = new(1352, 64);
            public LinearLayer Out = new(64, 10);
            public ComputationGraph Graph = new();
            public TensorStorage<float> XData = new(BatchSize * 784, clearMemory: false);
            public TensorStorage<float> YData = new(BatchSize * 10, clearMemory: false);
            public AutogradNode X;
            public AutogradNode Y;
            private Parameter[]? _params;

            public Replica()
            {
                X = new AutogradNode(XData, new TensorShape(BatchSize, 1, 28, 28), requiresGrad: false);
                Y = new AutogradNode(YData, new TensorShape(BatchSize, 10), requiresGrad: false);
            }

            public Parameter[] Parameters() => _params ??=
                [.. Conv.TrainableParameters(), .. Hidden.TrainableParameters(), .. Out.TrainableParameters()];

            public float TrainBatch(ReadOnlySpan<float> trainX, ReadOnlySpan<float> trainY, int batch)
            {
                Graph.Reset();
                foreach (var p in Parameters()) { p.ZeroGrad(); }

                trainX.Slice(batch * BatchSize * 784, BatchSize * 784).CopyTo(XData.AsSpan());
                trainY.Slice(batch * BatchSize * 10, BatchSize * 10).CopyTo(YData.AsSpan());

                using var h1 = Conv.Forward(Graph, X);
                using var a1 = Ops.TensorMath.ReLU(Graph, h1);
                using var p1 = Ops.TensorMath.MaxPool2D(Graph, a1, 8, 26, 26, 2);
                using var p1F = Ops.TensorMath.Reshape(Graph, p1, BatchSize, 1352);
                using var hidden = Hidden.Forward(Graph, p1F);
                using var hiddenAct = Ops.TensorMath.ReLU(Graph, hidden);
                using var logits = Out.Forward(Graph, hiddenAct);
                using var loss = Ops.TensorMath.SoftmaxCrossEntropy(Graph, logits, Y);
                Graph.Backward(loss);
                return loss.DataView.AsReadOnlySpan()[0];
            }

            public void Dispose()
            {
                X.Dispose(); Y.Dispose();
                Graph.Dispose(); Conv.Dispose(); Hidden.Dispose(); Out.Dispose();
                XData.Dispose(); YData.Dispose();
            }
        }

        [LongFact]
        public void SingleReplica_vs_DataParallel()
        {
            var imgs = TestSupport.TestModelPaths.Mnist.TrainImagesPath;
            var lbls = TestSupport.TestModelPaths.Mnist.TrainLabelsPath;
            if (!File.Exists(imgs)) { _out.WriteLine("MNIST files not found"); return; }
            var (trainX, trainY) = MnistLoader.Load(imgs, lbls);
            var batches = TrainSize / BatchSize;

            // ── baseline: single replica, intra-op parallelism on ──
            {
                using var r = new Replica();
                using var opt = new Adam(r.Parameters()) { UseAdamW = true };
                var sw = Stopwatch.StartNew(); var lossSum = 0f; var tailSum = 0f;
                var tailStart = batches - batches / 10;
                for (var b = 0; b < batches; b++)
                {
                    opt.ZeroGrad();
                    var l = r.TrainBatch(trainX.AsReadOnlySpan(), trainY.AsReadOnlySpan(), b);
                    lossSum += l;
                    if (b >= tailStart) { tailSum += l; }
                    opt.Step();
                }
                sw.Stop();
                _out.WriteLine($"single   : {sw.Elapsed.TotalSeconds,6:F1}s  avgLoss {lossSum / batches:F4}  finalLoss {tailSum / (batches - tailStart):F4}");
            }

            // ── data-parallel: R replicas, each step consumes R batches. lr follows the linear
            // scaling rule (lr × R) to compensate for the R× fewer optimizer steps per epoch —
            // the unscaled arms quantify the large-batch loss penalty, the scaled arm whether
            // the rule recovers the single-replica loss at data-parallel wall time. ──
            // lr arms: unscaled, linear rule (lr×R — derived for SGD), sqrt rule (lr×√R — the
            // usual Adam recommendation).
            foreach (var (replicas, lr) in new[] { (4, 0.001f), (8, 0.001f), (8, 0.008f), (8, 0.00283f) })
            {
                using var master = new Replica();
                var workers = new Replica[replicas];
                for (var w = 0; w < replicas; w++) { workers[w] = new Replica(); }
                try
                {
                    using var opt = new Adam(master.Parameters(), lr) { UseAdamW = true };
                    var trainer = new DataParallelTrainer(
                        master.Parameters(), [.. workers.Select(w => (IReadOnlyList<Parameter>)w.Parameters())]);
                    trainer.BroadcastParameters();

                    var steps = batches / replicas;
                    var tailStart = steps - Math.Max(1, steps / 10);
                    var sw = Stopwatch.StartNew(); var lossSum = 0f; var tailSum = 0f;
                    for (var s = 0; s < steps; s++)
                    {
                        var baseBatch = s * replicas;
                        var l = trainer.Step(opt, w =>
                            workers[w].TrainBatch(trainX.AsReadOnlySpan(), trainY.AsReadOnlySpan(), baseBatch + w));
                        lossSum += l;
                        if (s >= tailStart) { tailSum += l; }
                    }
                    sw.Stop();
                    _out.WriteLine($"replicas {replicas} lr={lr}: {sw.Elapsed.TotalSeconds,6:F1}s  avgLoss {lossSum / steps:F4}  finalLoss {tailSum / (steps - tailStart):F4}  (steps {steps})");
                }
                finally
                {
                    foreach (var w in workers) { w.Dispose(); }
                }
            }
        }
    }
}
