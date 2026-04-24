// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using DevOnBike.Overfit.Tests.Monitoring;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests
{
    public sealed class MnistTrainingTests
    {
        private readonly ITestOutputHelper _output;

        public MnistTrainingTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void Mnist_FullTrain60k_CnnBeastMode_Benchmark()
        {
            const int trainSize = 60_000;
            const int batchSize = 64;
            const int epochs = 5;
            const float lr = 0.001f;

            const bool enableDiagnostics = false;
            const bool enableBackwardProfiling = true;

            var trainImagesPath = "d:/ml/train-images.idx3-ubyte";
            var trainLabelsPath = "d:/ml/train-labels.idx1-ubyte";

            if (!File.Exists(trainImagesPath) || !File.Exists(trainLabelsPath))
            {
                _output.WriteLine("MNIST files not found.");
                return;
            }

            var (trainX, trainY) = MnistLoader.Load(trainImagesPath, trainLabelsPath, trainSize);

            using var conv1 = new ConvLayer(1, 8, 28, 28, 3);
            using var bn1 = new BatchNorm1D(1352);
            using var res1 = new ResidualBlock(1352);
            using var fcOut = new LinearLayer(8, 10);

            var parameters = conv1.Parameters()
                .Concat(bn1.Parameters())
                .Concat(res1.Parameters())
                .Concat(fcOut.Parameters())
                .ToArray();

            using var optimizer = new Adam(parameters, lr)
            {
                UseAdamW = true
            };

            using var graph = new ComputationGraph
            {
                EnableBackwardProfiling = enableBackwardProfiling
            };

            var totalSw = Stopwatch.StartNew();

            var traceDir = Path.Combine(AppContext.BaseDirectory, "diagnostics", "mnist");
            Directory.CreateDirectory(traceDir);

            var textTracePath = Path.Combine(traceDir, "mnist_trace.log");
            var jsonlTracePath = Path.Combine(traceDir, "mnist_trace.jsonl");

            if (File.Exists(textTracePath))
            {
                File.Delete(textTracePath);
            }

            if (File.Exists(jsonlTracePath))
            {
                File.Delete(jsonlTracePath);
            }

            _output.WriteLine("=== START: Trening ResNet na Taśmie (NativeBuffer) ===");
            _output.WriteLine($"Diagnostics enabled: {enableDiagnostics}");
            _output.WriteLine($"Backward profiling enabled: {enableBackwardProfiling}");

            var runStartSnapshot = MemoryLeakProbe.Capture(forceFullGc: true);
            _output.WriteLine(MemoryLeakProbe.Format("RUN START", runStartSnapshot));

            for (var epoch = 0; epoch < epochs; epoch++)
            {
                conv1.Train();
                bn1.Train();
                res1.Train();
                fcOut.Train();

                var epochLoss = 0f;
                var batches = trainSize / batchSize;

                long tConv = 0, tBn = 0, tRes = 0, tHead = 0, tLoss = 0, tBackward = 0, tOptimizer = 0;
                long aConv = 0, aBn = 0, aRes = 0, aHead = 0, aLoss = 0, aBackward = 0, aOptimizer = 0;

                var sectionSw = new Stopwatch();

                var epochAllocBefore = GC.GetTotalAllocatedBytes(true);
                var gc0Before = GC.CollectionCount(0);
                var gc1Before = GC.CollectionCount(1);
                var gc2Before = GC.CollectionCount(2);

                var epochStartSnapshot = MemoryLeakProbe.Capture(forceFullGc: true);

                for (var b = 0; b < batches; b++)
                {
                    graph.Reset();
                    optimizer.ZeroGrad();

                    using var xBData = new TensorStorage<float>(batchSize * 1 * 28 * 28, clearMemory: false);
                    using var yBData = new TensorStorage<float>(batchSize * 10, clearMemory: false);

                    using var xBNode = new AutogradNode(xBData, new TensorShape(batchSize, 1, 28, 28), requiresGrad: false);
                    using var yBNode = new AutogradNode(yBData, new TensorShape(batchSize, 10), requiresGrad: false);

                    trainX.AsReadOnlySpan()
                        .Slice(b * batchSize * 784, batchSize * 784)
                        .CopyTo(xBData.AsSpan());

                    trainY.AsReadOnlySpan()
                        .Slice(b * batchSize * 10, batchSize * 10)
                        .CopyTo(yBData.AsSpan());

                    long allocBefore;

                    allocBefore = GC.GetTotalAllocatedBytes(false);
                    sectionSw.Restart();
                    using var h1 = conv1.Forward(graph, xBNode);
                    using var a1 = TensorMath.ReLU(graph, h1);
                    using var p1 = TensorMath.MaxPool2D(graph, a1, 8, 26, 26, 2);
                    using var p1F = TensorMath.Reshape(graph, p1, batchSize, 1352);
                    sectionSw.Stop();
                    tConv += sectionSw.ElapsedTicks;
                    aConv += GC.GetTotalAllocatedBytes(false) - allocBefore;

                    allocBefore = GC.GetTotalAllocatedBytes(false);
                    sectionSw.Restart();
                    using var bn1O = bn1.Forward(graph, p1F);
                    sectionSw.Stop();
                    tBn += sectionSw.ElapsedTicks;
                    aBn += GC.GetTotalAllocatedBytes(false) - allocBefore;

                    allocBefore = GC.GetTotalAllocatedBytes(false);
                    sectionSw.Restart();
                    using var resO = res1.Forward(graph, bn1O);
                    sectionSw.Stop();
                    tRes += sectionSw.ElapsedTicks;
                    aRes += GC.GetTotalAllocatedBytes(false) - allocBefore;

                    allocBefore = GC.GetTotalAllocatedBytes(false);
                    sectionSw.Restart();
                    using var gapO = TensorMath.GlobalAveragePool2D(graph, resO, 8, 13, 13);
                    using var logits = fcOut.Forward(graph, gapO);
                    sectionSw.Stop();
                    tHead += sectionSw.ElapsedTicks;
                    aHead += GC.GetTotalAllocatedBytes(false) - allocBefore;

                    allocBefore = GC.GetTotalAllocatedBytes(false);
                    sectionSw.Restart();
                    using var loss = TensorMath.SoftmaxCrossEntropy(graph, logits, yBNode);
                    epochLoss += loss.DataView.AsReadOnlySpan()[0];
                    sectionSw.Stop();
                    tLoss += sectionSw.ElapsedTicks;
                    aLoss += GC.GetTotalAllocatedBytes(false) - allocBefore;

                    allocBefore = GC.GetTotalAllocatedBytes(false);
                    sectionSw.Restart();
                    graph.Backward(loss);
                    sectionSw.Stop();
                    tBackward += sectionSw.ElapsedTicks;
                    aBackward += GC.GetTotalAllocatedBytes(false) - allocBefore;

                    if (enableBackwardProfiling)
                    {
                        // lastBackwardProfile = graph.GetBackwardProfileSnapshot();
                    }

                    allocBefore = GC.GetTotalAllocatedBytes(false);
                    sectionSw.Restart();
                    optimizer.Step();
                    sectionSw.Stop();
                    tOptimizer += sectionSw.ElapsedTicks;
                    aOptimizer += GC.GetTotalAllocatedBytes(false) - allocBefore;
                }

                var epochAllocAfter = GC.GetTotalAllocatedBytes(true);

                var epochJson = Path.Combine(traceDir, $"epoch_{epoch + 1:D2}.json");
                var epochCsv = Path.Combine(traceDir, $"epoch_{epoch + 1:D2}.csv");

                var epochEndSnapshot = MemoryLeakProbe.Capture(forceFullGc: true);

                _output.WriteLine($"Epoch {epoch + 1} | Loss: {epochLoss / batches:F4} | Time: {totalSw.ElapsedMilliseconds}ms");
                _output.WriteLine($"  conv+relu+pool+reshape: {TimeSpan.FromTicks(tConv).TotalMilliseconds:F1} ms | alloc {(aConv / 1024.0 / 1024.0):F2} MB");
                _output.WriteLine($"  batchnorm:              {TimeSpan.FromTicks(tBn).TotalMilliseconds:F1} ms | alloc {(aBn / 1024.0 / 1024.0):F2} MB");
                _output.WriteLine($"  residual:               {TimeSpan.FromTicks(tRes).TotalMilliseconds:F1} ms | alloc {(aRes / 1024.0 / 1024.0):F2} MB");
                _output.WriteLine($"  gap+fc:                 {TimeSpan.FromTicks(tHead).TotalMilliseconds:F1} ms | alloc {(aHead / 1024.0 / 1024.0):F2} MB");
                _output.WriteLine($"  loss:                   {TimeSpan.FromTicks(tLoss).TotalMilliseconds:F1} ms | alloc {(aLoss / 1024.0 / 1024.0):F2} MB");
                _output.WriteLine($"  backward:               {TimeSpan.FromTicks(tBackward).TotalMilliseconds:F1} ms | alloc {(aBackward / 1024.0 / 1024.0):F2} MB");
                _output.WriteLine($"  optimizer:              {TimeSpan.FromTicks(tOptimizer).TotalMilliseconds:F1} ms | alloc {(aOptimizer / 1024.0 / 1024.0):F2} MB");
                _output.WriteLine($"  allocated total:        {(epochAllocAfter - epochAllocBefore) / 1024.0 / 1024.0:F2} MB");
                _output.WriteLine($"  GC0/1/2:                {GC.CollectionCount(0) - gc0Before}/{GC.CollectionCount(1) - gc1Before}/{GC.CollectionCount(2) - gc2Before}");

                _output.WriteLine(MemoryLeakProbe.Format($"EPOCH {epoch + 1} START", epochStartSnapshot));
                _output.WriteLine(MemoryLeakProbe.Format($"EPOCH {epoch + 1} END", epochEndSnapshot));
                _output.WriteLine($"live managed delta: {(epochEndSnapshot.LiveManagedBytes - epochStartSnapshot.LiveManagedBytes) / 1024.0 / 1024.0:F2} MB");
                _output.WriteLine($"private bytes delta: {(epochEndSnapshot.PrivateMemoryBytes - epochStartSnapshot.PrivateMemoryBytes) / 1024.0 / 1024.0:F2} MB");
                _output.WriteLine($"working set delta: {(epochEndSnapshot.WorkingSet64 - epochStartSnapshot.WorkingSet64) / 1024.0 / 1024.0:F2} MB");

                if (enableDiagnostics)
                {
                    _output.WriteLine($"  trace.json:             {epochJson}");
                    _output.WriteLine($"  trace.csv:              {epochCsv}");

                    if (epoch == epochs - 1)
                    {
                        var baselinePath = Path.Combine(traceDir, "baseline_epoch_05.json");
                        if (!File.Exists(baselinePath))
                        {
                            File.Copy(epochJson, baselinePath, overwrite: false);
                            _output.WriteLine($"  baseline.json:          {baselinePath} (created)");
                        }
                    }
                }
                else
                {
                    _output.WriteLine("=== RUNTIME DIAGNOSTICS ===");
                    _output.WriteLine("diagnostics disabled");
                }
            }

            var runEndSnapshot = MemoryLeakProbe.Capture(forceFullGc: true);
            _output.WriteLine(MemoryLeakProbe.Format("RUN END", runEndSnapshot));
            _output.WriteLine($"run live managed delta: {(runEndSnapshot.LiveManagedBytes - runStartSnapshot.LiveManagedBytes) / 1024.0 / 1024.0:F2} MB");
            _output.WriteLine($"run private bytes delta: {(runEndSnapshot.PrivateMemoryBytes - runStartSnapshot.PrivateMemoryBytes) / 1024.0 / 1024.0:F2} MB");
            _output.WriteLine($"run working set delta: {(runEndSnapshot.WorkingSet64 - runStartSnapshot.WorkingSet64) / 1024.0 / 1024.0:F2} MB");

            _output.WriteLine("=== KONIEC ===");
        }
    }
}