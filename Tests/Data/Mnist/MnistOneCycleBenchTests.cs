// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using DevOnBike.Overfit.Tests.TestSupport;
using DevOnBike.Overfit.Tests.TestSupport.Helpers;
using DevOnBike.Overfit.Training;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Data.Mnist
{
    /// <summary>
    /// Time-to-quality result (2026-06-12, 3 runs, init noise ±0.006): a one-cycle LR (warmup 10% →
    /// high peak → cosine to 0.0005) reaches the 5-epoch constant-lr-0.008 baseline quality in FEWER
    /// epochs. Measured: baseline 5ep ≈ loss 0.036–0.049 @ ~2.8s; **3ep @ peak 0.048 ≈ 0.0397 @ 1.47s
    /// (−47% time, matches baseline band)**; **4ep @ peak 0.032 ≈ 0.0304 @ 2.0s (−28% time, BEATS the
    /// baseline band)**. Low peaks don't work (3ep @ 0.008 → 0.088); peak 4–6× base lr is the regime.
    /// PyTorch 2.11 CPU reference on the same box/arch/batch (Scripts/bench_mnist_torch.py, thread
    /// sweep → ITS optimum is 8 threads): ~524–570 ms/epoch vs our 503 ± 4 ms (BenchmarkDotNet,
    /// MnistTrainingEpochBenchmark) — on par, consistently ~5–10% faster. (32 torch threads = 13 s/ep.)
    /// Same DP×8 / batch-128 rig as the DataParallel8 benchmark; fresh layers per arm.
    /// </summary>
    public sealed class MnistOneCycleBenchTests
    {
        private readonly ITestOutputHelper _output;
        public MnistOneCycleBenchTests(ITestOutputHelper output) => _output = output;

        [LongFact]
        public void OneCycle_FewerEpochs_VsConstantLrBaseline()
        {
            var imgs = TestModelPaths.Mnist.TrainImagesPath;
            var lbls = TestModelPaths.Mnist.TrainLabelsPath;
            if (!File.Exists(imgs))
            {
                _output.WriteLine("MNIST files not found.");
                return;
            }
            var (trainX, trainY) = MnistLoader.Load(imgs, lbls);

            RunArm("A: 5ep const lr 0.008 (baseline)", trainX, trainY, epochs: 5, lrAt: (_, _) => 0.008f);
            RunArm("H: 3ep one-cycle max 0.048", trainX, trainY, epochs: 3,
                lrAt: (step, total) => LearningRateSchedule.WarmupCosine(step, total, Math.Max(1, total / 10), 0.048f, 0.0005f));
            RunArm("I: 4ep one-cycle max 0.032", trainX, trainY, epochs: 4,
                lrAt: (step, total) => LearningRateSchedule.WarmupCosine(step, total, Math.Max(1, total / 10), 0.032f, 0.0005f));
        }

        private void RunArm(
            string name, TensorStorage<float> trainX, TensorStorage<float> trainY,
            int epochs, Func<int, int, float> lrAt)
        {
            const int trainSize = 60_000;
            const int batchSize = 128;
            const int replicas = 8;

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
                var trainer = new DataParallelTrainer(
                    pars[0], [.. Enumerable.Range(1, replicas).Select(i => (IReadOnlyList<Parameters.Parameter>)pars[i])],
                    runWorkerOpsInline: true);
                trainer.BroadcastParameters();

                float TrainBatch(int i, int batch)
                {
                    graphs[i].Reset();
                    foreach (var p in pars[i])
                    {
                        p.ZeroGrad();
                    }
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

                var steps = trainSize / batchSize / replicas;
                var totalSteps = steps * epochs;
                var globalStep = 0;

                _output.WriteLine($"=== {name} ===");
                var runWatch = ValueStopwatch.StartNew();
                var lastEpochLoss = 0f;
                for (var epoch = 0; epoch < epochs; epoch++)
                {
                    var epochWatch = ValueStopwatch.StartNew();
                    var epochLoss = 0f;
                    for (var s = 0; s < steps; s++)
                    {
                        opt.LearningRate = lrAt(globalStep++, totalSteps);
                        var b0 = s * replicas;
                        epochLoss += trainer.Step(opt, w => TrainBatch(1 + w, b0 + w));
                    }
                    lastEpochLoss = epochLoss / steps;
                    _output.WriteLine($"Epoch {epoch + 1} | Loss: {lastEpochLoss:F4} | Time: {epochWatch.GetElapsedTime().TotalMilliseconds:F1}ms");
                }
                _output.WriteLine($"TOTAL {name}: {runWatch.GetElapsedTime().TotalMilliseconds:F0} ms | final-epoch avg loss {lastEpochLoss:F4}");
                _output.WriteLine("");
            }
            finally
            {
                for (var i = 0; i <= replicas; i++)
                {
                    xn[i].Dispose();
                    yn[i].Dispose();
                    graphs[i].Dispose();
                    convs[i].Dispose();
                    hiddens[i].Dispose();
                    outs[i].Dispose();
                    xs[i].Dispose();
                    ys[i].Dispose();
                }
            }
        }
    }
}
