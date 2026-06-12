// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using DevOnBike.Overfit.Training;

namespace Benchmarks
{
    /// <summary>
    /// Statistically-grounded ms/epoch for the MNIST beast CNN under DataParallel ×8 / batch 128 —
    /// the number behind the "vs PyTorch CPU" comparison (the test-based measurement is a single
    /// run; this is BenchmarkDotNet: warmup + N iterations + stddev). One benchmark op = one FULL
    /// epoch over 60k samples (468 trainer steps). The model keeps training across iterations —
    /// compute per epoch is shape-identical, so throughput is unaffected by the loss value.
    /// PyTorch reference on the same arch/data/box: Scripts/bench_mnist_torch.py (best-of-N, its
    /// optimal 8 threads — sweep first!). Data: d:\ml (override OVERFIT_MNIST_DIR).
    /// </summary>
    [SimpleJob(RunStrategy.Monitoring, warmupCount: 2, iterationCount: 10, invocationCount: 1)]
    [MemoryDiagnoser]
    public class MnistTrainingEpochBenchmark
    {
        private const int TrainSize = 60_000;
        private const int BatchSize = 128;
        private const int Replicas = 8;

        private TensorStorage<float> _trainX = null!;
        private TensorStorage<float> _trainY = null!;
        private ConvLayer[] _convs = null!;
        private LinearLayer[] _hiddens = null!;
        private LinearLayer[] _outs = null!;
        private ComputationGraph[] _graphs = null!;
        private TensorStorage<float>[] _xs = null!;
        private TensorStorage<float>[] _ys = null!;
        private AutogradNode[] _xn = null!;
        private AutogradNode[] _yn = null!;
        private Parameter[][] _pars = null!;
        private Adam _opt = null!;
        private DataParallelTrainer _trainer = null!;

        [GlobalSetup]
        public void Setup()
        {
            var dir = Environment.GetEnvironmentVariable("OVERFIT_MNIST_DIR") ?? @"d:\ml";
            (_trainX, _trainY) = LoadMnist(
                Path.Combine(dir, "train-images.idx3-ubyte"),
                Path.Combine(dir, "train-labels.idx1-ubyte"));

            _convs = new ConvLayer[Replicas + 1];
            _hiddens = new LinearLayer[Replicas + 1];
            _outs = new LinearLayer[Replicas + 1];
            _graphs = new ComputationGraph[Replicas + 1];
            _xs = new TensorStorage<float>[Replicas + 1];
            _ys = new TensorStorage<float>[Replicas + 1];
            _xn = new AutogradNode[Replicas + 1];
            _yn = new AutogradNode[Replicas + 1];
            _pars = new Parameter[Replicas + 1][];
            for (var i = 0; i <= Replicas; i++)
            {
                _convs[i] = new ConvLayer(1, 8, 28, 28, 3);
                _hiddens[i] = new LinearLayer(1352, 64);
                _outs[i] = new LinearLayer(64, 10);
                _graphs[i] = new ComputationGraph();
                _xs[i] = new TensorStorage<float>(BatchSize * 784, clearMemory: false);
                _ys[i] = new TensorStorage<float>(BatchSize * 10, clearMemory: false);
                _xn[i] = new AutogradNode(_xs[i], new TensorShape(BatchSize, 1, 28, 28), requiresGrad: false);
                _yn[i] = new AutogradNode(_ys[i], new TensorShape(BatchSize, 10), requiresGrad: false);
                _pars[i] = [.. _convs[i].TrainableParameters(), .. _hiddens[i].TrainableParameters(), .. _outs[i].TrainableParameters()];
            }

            _opt = new Adam(_pars[0], 0.008f) { UseAdamW = true };
            _trainer = new DataParallelTrainer(
                _pars[0], [.. Enumerable.Range(1, Replicas).Select(i => (IReadOnlyList<Parameter>)_pars[i])],
                runWorkerOpsInline: true);
            _trainer.BroadcastParameters();
        }

        [Benchmark]
        public float TrainOneEpoch_DataParallel8_Batch128()
        {
            var steps = TrainSize / BatchSize / Replicas;
            var epochLoss = 0f;
            for (var s = 0; s < steps; s++)
            {
                var b0 = s * Replicas;
                epochLoss += _trainer.Step(_opt, w => TrainBatch(1 + w, b0 + w));
            }
            return epochLoss;
        }

        private float TrainBatch(int i, int batch)
        {
            _graphs[i].Reset();
            foreach (var p in _pars[i]) { p.ZeroGrad(); }
            _trainX.AsReadOnlySpan().Slice(batch * BatchSize * 784, BatchSize * 784).CopyTo(_xs[i].AsSpan());
            _trainY.AsReadOnlySpan().Slice(batch * BatchSize * 10, BatchSize * 10).CopyTo(_ys[i].AsSpan());
            using var h1 = _convs[i].Forward(_graphs[i], _xn[i]);
            using var a1 = TensorMath.ReLU(_graphs[i], h1);
            using var p1 = TensorMath.MaxPool2D(_graphs[i], a1, 8, 26, 26, 2);
            using var f = TensorMath.Reshape(_graphs[i], p1, BatchSize, 1352);
            using var hd = _hiddens[i].Forward(_graphs[i], f);
            using var ha = TensorMath.ReLU(_graphs[i], hd);
            using var lg = _outs[i].Forward(_graphs[i], ha);
            using var loss = TensorMath.SoftmaxCrossEntropy(_graphs[i], lg, _yn[i]);
            _graphs[i].Backward(loss);
            return loss.DataView.AsReadOnlySpan()[0];
        }

        private static (TensorStorage<float> images, TensorStorage<float> labels) LoadMnist(string imagesPath, string labelsPath)
        {
            using var imgReader = new BinaryReader(File.OpenRead(imagesPath));
            using var lblReader = new BinaryReader(File.OpenRead(labelsPath));
            imgReader.ReadBytes(16);
            lblReader.ReadBytes(8);

            var images = new TensorStorage<float>(TrainSize * 784, clearMemory: false);
            var labels = new TensorStorage<float>(TrainSize * 10, clearMemory: true);
            var imgSpan = images.AsSpan();
            var lblSpan = labels.AsSpan();
            for (var i = 0; i < TrainSize; i++)
            {
                var pixels = imgReader.ReadBytes(784);
                for (var j = 0; j < 784; j++) { imgSpan[i * 784 + j] = pixels[j] / 255f; }
                lblSpan[i * 10 + lblReader.ReadByte()] = 1f;
            }
            return (images, labels);
        }
    }
}
