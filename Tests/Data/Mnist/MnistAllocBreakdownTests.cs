// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Data.Mnist
{
    /// <summary>
    /// Attributes the ~1.31 MB/epoch from <c>MnistTrainingEpochBenchmark</c>'s MemoryDiagnoser:
    /// measures GC.GetTotalAllocatedBytes (ALL threads) for (a) the bare tape path — 58× TrainBatch
    /// on one replica, no trainer/optimizer; (b) the full DP×8 epoch — 58× trainer.Step. The delta
    /// is the trainer + Adam + TPL-dispatch share. MEASURED 2026-06-12: tape 754 KB/epoch
    /// (1 624 B/batch/replica — AutogradNode/TapeOp objects), trainer+Adam+TPL 656 KB/epoch
    /// (~11.3 KB/step). Total ≈ 1.4 MB/epoch = ~2.8 MB/s at 503 ms/epoch → GC cost ≤1 ms/epoch
    /// (~0.2%), BELOW the benchmark's own ±4 ms stddev — zero-alloc here buys nothing measurable.
    /// [LongFact] — diagnostic, needs d:\ml.
    /// </summary>
    public sealed class MnistAllocBreakdownTests
    {
        private readonly ITestOutputHelper _out;
        public MnistAllocBreakdownTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Epoch_Allocation_Breakdown()
        {
            const int batchSize = 128;
            const int replicas = 8;
            const int steps = 60_000 / batchSize / replicas;   // 58

            var imgs = TestModelPaths.Mnist.TrainImagesPath;
            if (!File.Exists(imgs)) { _out.WriteLine("MNIST files not found."); return; }
            var (trainX, trainY) = MnistLoader.Load(imgs, TestModelPaths.Mnist.TrainLabelsPath);

            var convs = new ConvLayer[replicas + 1];
            var hiddens = new LinearLayer[replicas + 1];
            var outs = new LinearLayer[replicas + 1];
            var graphs = new ComputationGraph[replicas + 1];
            var xs = new TensorStorage<float>[replicas + 1];
            var ys = new TensorStorage<float>[replicas + 1];
            var xn = new AutogradNode[replicas + 1];
            var yn = new AutogradNode[replicas + 1];
            var pars = new Parameters.Parameter[replicas + 1][];
            for (var i = 0; i <= replicas; i++)
            {
                convs[i] = new ConvLayer(1, 8, 28, 28, 3);
                hiddens[i] = new LinearLayer(1352, 64);
                outs[i] = new LinearLayer(64, 10);
                graphs[i] = new ComputationGraph();
                xs[i] = new TensorStorage<float>(batchSize * 784, clearMemory: false);
                ys[i] = new TensorStorage<float>(batchSize * 10, clearMemory: false);
                xn[i] = new AutogradNode(xs[i], new TensorShape(batchSize, 1, 28, 28), requiresGrad: false);
                yn[i] = new AutogradNode(ys[i], new TensorShape(batchSize, 10), requiresGrad: false);
                pars[i] = [.. convs[i].TrainableParameters(), .. hiddens[i].TrainableParameters(), .. outs[i].TrainableParameters()];
            }

            try
            {
                using var opt = new Adam(pars[0], 0.008f) { UseAdamW = true };
                var trainer = new DevOnBike.Overfit.Training.DataParallelTrainer(
                    pars[0], [.. Enumerable.Range(1, replicas).Select(i => (IReadOnlyList<Parameters.Parameter>)pars[i])],
                    runWorkerOpsInline: true);
                trainer.BroadcastParameters();

                float TrainBatch(int i, int batch)
                {
                    graphs[i].Reset();
                    foreach (var p in pars[i]) { p.ZeroGrad(); }
                    trainX.AsReadOnlySpan().Slice(batch * batchSize * 784, batchSize * 784).CopyTo(xs[i].AsSpan());
                    trainY.AsReadOnlySpan().Slice(batch * batchSize * 10, batchSize * 10).CopyTo(ys[i].AsSpan());
                    using var h1 = convs[i].Forward(graphs[i], xn[i]);
                    using var a1 = TensorMath.ReLU(graphs[i], h1);
                    using var p1 = TensorMath.MaxPool2D(graphs[i], a1, 8, 26, 26, 2);
                    using var f = TensorMath.Reshape(graphs[i], p1, batchSize, 1352);
                    using var hd = hiddens[i].Forward(graphs[i], f);
                    using var ha = TensorMath.ReLU(graphs[i], hd);
                    using var lg = outs[i].Forward(graphs[i], ha);
                    using var loss = TensorMath.SoftmaxCrossEntropy(graphs[i], lg, yn[i]);
                    graphs[i].Backward(loss);
                    return loss.DataView.AsReadOnlySpan()[0];
                }

                // Warmup (JIT) — both paths.
                for (var s = 0; s < 4; s++) { TrainBatch(1, s); }
                trainer.Step(opt, w => TrainBatch(1 + w, w));

                // (a) bare tape: 58 single-replica batches, no trainer/optimizer.
                var before = GC.GetTotalAllocatedBytes(precise: true);
                for (var s = 0; s < steps; s++) { TrainBatch(1, s); }
                var tapeOnly = GC.GetTotalAllocatedBytes(precise: true) - before;
                _out.WriteLine($"(a) tape-only, 58 batches × 1 replica:  {tapeOnly:N0} B  ({tapeOnly / steps:N0} B/batch)");

                // ×8 replicas equivalent (what one epoch's worth of TrainBatch work allocates).
                _out.WriteLine($"    ×8 replicas equivalent:             {tapeOnly * replicas:N0} B");

                // (b) full DP epoch: 58 trainer steps (8 replicas + averaging + Adam + broadcast).
                before = GC.GetTotalAllocatedBytes(precise: true);
                for (var s = 0; s < steps; s++)
                {
                    var b0 = s * replicas;
                    trainer.Step(opt, w => TrainBatch(1 + w, b0 + w));
                }
                var full = GC.GetTotalAllocatedBytes(precise: true) - before;
                _out.WriteLine($"(b) full DP×8 epoch (58 trainer steps): {full:N0} B  ({full / steps:N0} B/step)");
                _out.WriteLine($"(b)−(a)×8 = trainer+Adam+TPL share:     {full - tapeOnly * replicas:N0} B");
            }
            finally
            {
                for (var i = 0; i <= replicas; i++)
                {
                    xn[i].Dispose(); yn[i].Dispose(); graphs[i].Dispose();
                    convs[i].Dispose(); hiddens[i].Dispose(); outs[i].Dispose();
                    xs[i].Dispose(); ys[i].Dispose();
                }
            }
        }
    }
}
