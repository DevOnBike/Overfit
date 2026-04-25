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
using DevOnBike.Overfit.Tests.Helpers;
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

        [Fact(Skip = "a")]
        public void Mnist_FullTrain60k_CnnBeastMode_Benchmark()
        {
            const int trainSize = 60_000;
            const int batchSize = 64;
            const int epochs = 5;
            const float lr = 0.001f;

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

            using var graph = new ComputationGraph();

            using var telemetryListener = new TelemetryListener2(
    _output,
    includeTags: false,
    maxRows: 64,
    metricNamePrefix: "overfit.tensor_storage.");

            telemetryListener.Subscribe();

            _output.WriteLine("=== START: Trening ResNet na Taśmie (DOD + Zero-Alloc verification) ===");

            ForceFullGc();
            var runStart = MemorySnapshot.Capture();
            var runTotalAllocBefore = GC.GetTotalAllocatedBytes(false);

            _output.WriteLine(FormatMemorySnapshot("RUN START", runStart));

            var runWatch = ValueStopwatch.StartNew();

            for (var epoch = 0; epoch < epochs; epoch++)
            {
                conv1.Train();
                bn1.Train();
                res1.Train();
                fcOut.Train();

                ForceFullGc();
                var epochStart = MemorySnapshot.Capture();

                var epochAllocBefore = GC.GetTotalAllocatedBytes(false);
                var gc0Before = GC.CollectionCount(0);
                var gc1Before = GC.CollectionCount(1);
                var gc2Before = GC.CollectionCount(2);

                var epochWatch = ValueStopwatch.StartNew();

                var epochLoss = 0f;
                var batches = trainSize / batchSize;

                long convTicks = 0;
                long batchNormTicks = 0;
                long residualTicks = 0;
                long headTicks = 0;
                long lossTicks = 0;
                long backwardTicks = 0;
                long optimizerTicks = 0;

                long convAlloc = 0;
                long batchNormAlloc = 0;
                long residualAlloc = 0;
                long headAlloc = 0;
                long lossAlloc = 0;
                long backwardAlloc = 0;
                long optimizerAlloc = 0;

                for (var batch = 0; batch < batches; batch++)
                {
                    graph.Reset();
                    optimizer.ZeroGrad();

                    using var xBData = new TensorStorage<float>(batchSize * 1 * 28 * 28, clearMemory: false);
                    using var yBData = new TensorStorage<float>(batchSize * 10, clearMemory: false);

                    using var xBNode = new AutogradNode(
                        xBData,
                        new TensorShape(batchSize, 1, 28, 28),
                        requiresGrad: false);

                    using var yBNode = new AutogradNode(
                        yBData,
                        new TensorShape(batchSize, 10),
                        requiresGrad: false);

                    trainX.AsReadOnlySpan()
                        .Slice(batch * batchSize * 784, batchSize * 784)
                        .CopyTo(xBData.AsSpan());

                    trainY.AsReadOnlySpan()
                        .Slice(batch * batchSize * 10, batchSize * 10)
                        .CopyTo(yBData.AsSpan());

                    long allocBefore;
                    ValueStopwatch sectionWatch;
                    TimeSpan elapsed;

                    allocBefore = GC.GetTotalAllocatedBytes(false);
                    sectionWatch = ValueStopwatch.StartNew();

                    using var h1 = conv1.Forward(graph, xBNode);
                    using var a1 = TensorMath.ReLU(graph, h1);
                    using var p1 = TensorMath.MaxPool2D(graph, a1, 8, 26, 26, 2);
                    using var p1F = TensorMath.Reshape(graph, p1, batchSize, 1352);

                    elapsed = sectionWatch.GetElapsedTime();
                    convTicks += elapsed.Ticks;
                    convAlloc += GC.GetTotalAllocatedBytes(false) - allocBefore;

                    allocBefore = GC.GetTotalAllocatedBytes(false);
                    sectionWatch = ValueStopwatch.StartNew();

                    using var bn1O = bn1.Forward(graph, p1F);

                    elapsed = sectionWatch.GetElapsedTime();
                    batchNormTicks += elapsed.Ticks;
                    batchNormAlloc += GC.GetTotalAllocatedBytes(false) - allocBefore;

                    allocBefore = GC.GetTotalAllocatedBytes(false);
                    sectionWatch = ValueStopwatch.StartNew();

                    using var resO = res1.Forward(graph, bn1O);

                    elapsed = sectionWatch.GetElapsedTime();
                    residualTicks += elapsed.Ticks;
                    residualAlloc += GC.GetTotalAllocatedBytes(false) - allocBefore;

                    allocBefore = GC.GetTotalAllocatedBytes(false);
                    sectionWatch = ValueStopwatch.StartNew();

                    using var gapO = TensorMath.GlobalAveragePool2D(graph, resO, 8, 13, 13);
                    using var logits = fcOut.Forward(graph, gapO);

                    elapsed = sectionWatch.GetElapsedTime();
                    headTicks += elapsed.Ticks;
                    headAlloc += GC.GetTotalAllocatedBytes(false) - allocBefore;

                    allocBefore = GC.GetTotalAllocatedBytes(false);
                    sectionWatch = ValueStopwatch.StartNew();

                    using var loss = TensorMath.SoftmaxCrossEntropy(graph, logits, yBNode);
                    epochLoss += loss.DataView.AsReadOnlySpan()[0];

                    elapsed = sectionWatch.GetElapsedTime();
                    lossTicks += elapsed.Ticks;
                    lossAlloc += GC.GetTotalAllocatedBytes(false) - allocBefore;

                    allocBefore = GC.GetTotalAllocatedBytes(false);
                    sectionWatch = ValueStopwatch.StartNew();

                    graph.Backward(loss);

                    elapsed = sectionWatch.GetElapsedTime();
                    backwardTicks += elapsed.Ticks;
                    backwardAlloc += GC.GetTotalAllocatedBytes(false) - allocBefore;

                    allocBefore = GC.GetTotalAllocatedBytes(false);
                    sectionWatch = ValueStopwatch.StartNew();

                    optimizer.Step();

                    elapsed = sectionWatch.GetElapsedTime();
                    optimizerTicks += elapsed.Ticks;
                    optimizerAlloc += GC.GetTotalAllocatedBytes(false) - allocBefore;
                }

                var epochElapsed = epochWatch.GetElapsedTime();
                var epochAllocAfter = GC.GetTotalAllocatedBytes(false);

                ForceFullGc();
                var epochEnd = MemorySnapshot.Capture();

                var epochAllocatedBytes = epochAllocAfter - epochAllocBefore;

                _output.WriteLine($"Epoch {epoch + 1} | Loss: {epochLoss / batches:F4} | Time: {epochElapsed.TotalMilliseconds:F1}ms | Time so far: {runWatch.GetElapsedTime().TotalMilliseconds:F0}ms");
                _output.WriteLine($"  conv+relu+pool+reshape: {TicksToMs(convTicks):F1} ms | alloc {BytesToMb(convAlloc):F2} MB");
                _output.WriteLine($"  batchnorm:              {TicksToMs(batchNormTicks):F1} ms | alloc {BytesToMb(batchNormAlloc):F2} MB");
                _output.WriteLine($"  residual:               {TicksToMs(residualTicks):F1} ms | alloc {BytesToMb(residualAlloc):F2} MB");
                _output.WriteLine($"  gap+fc:                 {TicksToMs(headTicks):F1} ms | alloc {BytesToMb(headAlloc):F2} MB");
                _output.WriteLine($"  loss:                   {TicksToMs(lossTicks):F1} ms | alloc {BytesToMb(lossAlloc):F2} MB");
                _output.WriteLine($"  backward:               {TicksToMs(backwardTicks):F1} ms | alloc {BytesToMb(backwardAlloc):F2} MB");
                _output.WriteLine($"  optimizer:              {TicksToMs(optimizerTicks):F1} ms | alloc {BytesToMb(optimizerAlloc):F2} MB");
                _output.WriteLine($"  allocated total:        {BytesToMb(epochAllocatedBytes):F2} MB");
                _output.WriteLine($"  GC0/1/2:                {GC.CollectionCount(0) - gc0Before}/{GC.CollectionCount(1) - gc1Before}/{GC.CollectionCount(2) - gc2Before}");

                _output.WriteLine(FormatMemorySnapshot($"EPOCH {epoch + 1} START", epochStart));
                _output.WriteLine(FormatMemorySnapshot($"EPOCH {epoch + 1} END", epochEnd));
                _output.WriteLine($"  live managed delta:     {BytesToMb(epochEnd.LiveManagedBytes - epochStart.LiveManagedBytes):F2} MB");
                _output.WriteLine($"  private bytes delta:    {BytesToMb(epochEnd.PrivateMemoryBytes - epochStart.PrivateMemoryBytes):F2} MB");
                _output.WriteLine($"  working set delta:      {BytesToMb(epochEnd.WorkingSetBytes - epochStart.WorkingSetBytes):F2} MB");
            }

            ForceFullGc();
            var runEnd = MemorySnapshot.Capture();
            var runTotalAllocAfter = GC.GetTotalAllocatedBytes(false);

            _output.WriteLine(FormatMemorySnapshot("RUN END", runEnd));
            _output.WriteLine($"run elapsed:              {runWatch.GetElapsedTime().TotalMilliseconds:F1} ms");
            _output.WriteLine($"run allocated total:      {BytesToMb(runTotalAllocAfter - runTotalAllocBefore):F2} MB");
            _output.WriteLine($"run live managed delta:   {BytesToMb(runEnd.LiveManagedBytes - runStart.LiveManagedBytes):F2} MB");
            _output.WriteLine($"run private bytes delta:  {BytesToMb(runEnd.PrivateMemoryBytes - runStart.PrivateMemoryBytes):F2} MB");
            _output.WriteLine($"run working set delta:    {BytesToMb(runEnd.WorkingSetBytes - runStart.WorkingSetBytes):F2} MB");

            _output.WriteLine("=== KONIEC ===");
        }

        private static void ForceFullGc()
        {
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
            GC.WaitForPendingFinalizers();
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
        }

        private static double TicksToMs(long ticks)
        {
            return TimeSpan.FromTicks(ticks).TotalMilliseconds;
        }

        private static double BytesToMb(long bytes)
        {
            return bytes / 1024.0 / 1024.0;
        }

        private static string FormatMemorySnapshot(string name, in MemorySnapshot snapshot)
        {
            return
                $"=== MEMORY SNAPSHOT: {name} ==={Environment.NewLine}" +
                $"live managed:   {BytesToMb(snapshot.LiveManagedBytes):F2} MB{Environment.NewLine}" +
                $"total alloc:    {BytesToMb(snapshot.TotalAllocatedBytes):F2} MB{Environment.NewLine}" +
                $"working set:    {BytesToMb(snapshot.WorkingSetBytes):F2} MB{Environment.NewLine}" +
                $"private bytes:  {BytesToMb(snapshot.PrivateMemoryBytes):F2} MB{Environment.NewLine}" +
                $"GC0/1/2 total:  {snapshot.Gen0Collections}/{snapshot.Gen1Collections}/{snapshot.Gen2Collections}";
        }

        private readonly struct MemorySnapshot
        {
            private MemorySnapshot(
                long liveManagedBytes,
                long totalAllocatedBytes,
                long workingSetBytes,
                long privateMemoryBytes,
                int gen0Collections,
                int gen1Collections,
                int gen2Collections)
            {
                LiveManagedBytes = liveManagedBytes;
                TotalAllocatedBytes = totalAllocatedBytes;
                WorkingSetBytes = workingSetBytes;
                PrivateMemoryBytes = privateMemoryBytes;
                Gen0Collections = gen0Collections;
                Gen1Collections = gen1Collections;
                Gen2Collections = gen2Collections;
            }

            public long LiveManagedBytes { get; }
            public long TotalAllocatedBytes { get; }
            public long WorkingSetBytes { get; }
            public long PrivateMemoryBytes { get; }
            public int Gen0Collections { get; }
            public int Gen1Collections { get; }
            public int Gen2Collections { get; }

            public static MemorySnapshot Capture()
            {
                var process = System.Diagnostics.Process.GetCurrentProcess();

                return new MemorySnapshot(
                    liveManagedBytes: GC.GetTotalMemory(forceFullCollection: false),
                    totalAllocatedBytes: GC.GetTotalAllocatedBytes(false),
                    workingSetBytes: process.WorkingSet64,
                    privateMemoryBytes: process.PrivateMemorySize64,
                    gen0Collections: GC.CollectionCount(0),
                    gen1Collections: GC.CollectionCount(1),
                    gen2Collections: GC.CollectionCount(2));
            }
        }
    }
}