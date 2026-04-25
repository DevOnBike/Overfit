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

            // Set to false for a cleaner allocation baseline without Meter/Listener overhead.
            const bool enableTelemetry = false;
            const bool measureSections = false;

            ThreadPool.GetMinThreads(out var minWorkerBefore, out var minIoBefore);
            ThreadPool.GetMaxThreads(out var maxWorker, out var maxIo);

            _output.WriteLine($"Environment.ProcessorCount: {Environment.ProcessorCount}");
            _output.WriteLine($"ThreadPool min worker/io before: {minWorkerBefore}/{minIoBefore}");
            _output.WriteLine($"ThreadPool max worker/io: {maxWorker}/{maxIo}");

            ThreadPool.SetMinThreads(Environment.ProcessorCount, minIoBefore);

            ThreadPool.GetMinThreads(out var minWorkerAfter, out var minIoAfter);
            _output.WriteLine($"ThreadPool min worker/io after: {minWorkerAfter}/{minIoAfter}");



            var trainImagesPath = "d:/ml/train-images.idx3-ubyte";
            var trainLabelsPath = "d:/ml/train-labels.idx1-ubyte";

            if (!File.Exists(trainImagesPath) || !File.Exists(trainLabelsPath))
            {
                _output.WriteLine("MNIST files not found.");
                return;
            }

            var previousTelemetryEnabled = OverfitTelemetry.Enabled;
            OverfitTelemetry.Enabled = enableTelemetry;

            TelemetryListener2? telemetryListener = null;

            try
            {
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

                // Reused batch buffers.
                // These used to be allocated inside the batch loop.
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

                if (enableTelemetry)
                {
                    telemetryListener = new TelemetryListener2(
                        _output,
                        includeTags: false,
                        maxRows: 64,
                        metricNamePrefix: "overfit.tensor_storage.");

                    telemetryListener.Subscribe();
                }

                _output.WriteLine("=== START: Trening ResNet na Taśmie (DOD + Zero-Alloc verification) ===");
                _output.WriteLine($"telemetry enabled: {enableTelemetry}");
                _output.WriteLine($"Environment.ProcessorCount: {Environment.ProcessorCount}");

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

                    var copyInputStats = new OperationStats("copy input/target");
                    var zeroGradStats = new OperationStats("optimizer.ZeroGrad");
                    var resetStats = new OperationStats("graph.Reset");

                    var convStats = new OperationStats("Conv2D");
                    var relu1Stats = new OperationStats("ReLU #1");
                    var maxPoolStats = new OperationStats("MaxPool2D");
                    var reshapeStats = new OperationStats("Reshape");

                    var bn1Stats = new OperationStats("BatchNorm1D #1");
                    var residualStats = new OperationStats("ResidualBlock");

                    var gapStats = new OperationStats("GlobalAveragePool2D");
                    var fcStats = new OperationStats("Linear FC");

                    var lossStats = new OperationStats("SoftmaxCrossEntropy");
                    var backwardStats = new OperationStats("graph.Backward");
                    var optimizerStats = new OperationStats("optimizer.Step");

                    for (var batch = 0; batch < batches; batch++)
                    {
                        long allocBefore;
                        ValueStopwatch sectionWatch;
                        TimeSpan elapsed;

                        allocBefore = GC.GetTotalAllocatedBytes(false);
                        sectionWatch = ValueStopwatch.StartNew();

                        graph.Reset();

                        elapsed = sectionWatch.GetElapsedTime();
                        resetStats.Add(elapsed, GC.GetTotalAllocatedBytes(false) - allocBefore);

                        allocBefore = GC.GetTotalAllocatedBytes(false);
                        sectionWatch = ValueStopwatch.StartNew();

                        optimizer.ZeroGrad();

                        elapsed = sectionWatch.GetElapsedTime();
                        zeroGradStats.Add(elapsed, GC.GetTotalAllocatedBytes(false) - allocBefore);

                        allocBefore = GC.GetTotalAllocatedBytes(false);
                        sectionWatch = ValueStopwatch.StartNew();

                        trainX.AsReadOnlySpan()
                            .Slice(batch * batchSize * 784, batchSize * 784)
                            .CopyTo(xBData.AsSpan());

                        trainY.AsReadOnlySpan()
                            .Slice(batch * batchSize * 10, batchSize * 10)
                            .CopyTo(yBData.AsSpan());

                        elapsed = sectionWatch.GetElapsedTime();
                        copyInputStats.Add(elapsed, GC.GetTotalAllocatedBytes(false) - allocBefore);

                        allocBefore = GC.GetTotalAllocatedBytes(false);
                        sectionWatch = ValueStopwatch.StartNew();

                        using var h1 = conv1.Forward(graph, xBNode);

                        elapsed = sectionWatch.GetElapsedTime();
                        convStats.Add(elapsed, GC.GetTotalAllocatedBytes(false) - allocBefore);

                        allocBefore = GC.GetTotalAllocatedBytes(false);
                        sectionWatch = ValueStopwatch.StartNew();

                        using var a1 = TensorMath.ReLU(graph, h1);

                        elapsed = sectionWatch.GetElapsedTime();
                        relu1Stats.Add(elapsed, GC.GetTotalAllocatedBytes(false) - allocBefore);

                        allocBefore = GC.GetTotalAllocatedBytes(false);
                        sectionWatch = ValueStopwatch.StartNew();

                        using var p1 = TensorMath.MaxPool2D(graph, a1, 8, 26, 26, 2);

                        elapsed = sectionWatch.GetElapsedTime();
                        maxPoolStats.Add(elapsed, GC.GetTotalAllocatedBytes(false) - allocBefore);

                        allocBefore = GC.GetTotalAllocatedBytes(false);
                        sectionWatch = ValueStopwatch.StartNew();

                        using var p1F = TensorMath.Reshape(graph, p1, batchSize, 1352);

                        elapsed = sectionWatch.GetElapsedTime();
                        reshapeStats.Add(elapsed, GC.GetTotalAllocatedBytes(false) - allocBefore);

                        allocBefore = GC.GetTotalAllocatedBytes(false);
                        sectionWatch = ValueStopwatch.StartNew();

                        using var bn1O = bn1.Forward(graph, p1F);

                        elapsed = sectionWatch.GetElapsedTime();
                        bn1Stats.Add(elapsed, GC.GetTotalAllocatedBytes(false) - allocBefore);

                        allocBefore = GC.GetTotalAllocatedBytes(false);
                        sectionWatch = ValueStopwatch.StartNew();

                        using var resO = res1.Forward(graph, bn1O);

                        elapsed = sectionWatch.GetElapsedTime();
                        residualStats.Add(elapsed, GC.GetTotalAllocatedBytes(false) - allocBefore);

                        allocBefore = GC.GetTotalAllocatedBytes(false);
                        sectionWatch = ValueStopwatch.StartNew();

                        using var gapO = TensorMath.GlobalAveragePool2D(graph, resO, 8, 13, 13);

                        elapsed = sectionWatch.GetElapsedTime();
                        gapStats.Add(elapsed, GC.GetTotalAllocatedBytes(false) - allocBefore);

                        allocBefore = GC.GetTotalAllocatedBytes(false);
                        sectionWatch = ValueStopwatch.StartNew();

                        using var logits = fcOut.Forward(graph, gapO);

                        elapsed = sectionWatch.GetElapsedTime();
                        fcStats.Add(elapsed, GC.GetTotalAllocatedBytes(false) - allocBefore);

                        allocBefore = GC.GetTotalAllocatedBytes(false);
                        sectionWatch = ValueStopwatch.StartNew();

                        using var loss = TensorMath.SoftmaxCrossEntropy(graph, logits, yBNode);
                        epochLoss += loss.DataView.AsReadOnlySpan()[0];

                        elapsed = sectionWatch.GetElapsedTime();
                        lossStats.Add(elapsed, GC.GetTotalAllocatedBytes(false) - allocBefore);

                        allocBefore = GC.GetTotalAllocatedBytes(false);
                        sectionWatch = ValueStopwatch.StartNew();

                        graph.Backward(loss);

                        elapsed = sectionWatch.GetElapsedTime();
                        backwardStats.Add(elapsed, GC.GetTotalAllocatedBytes(false) - allocBefore);

                        allocBefore = GC.GetTotalAllocatedBytes(false);
                        sectionWatch = ValueStopwatch.StartNew();

                        optimizer.Step();

                        elapsed = sectionWatch.GetElapsedTime();
                        optimizerStats.Add(elapsed, GC.GetTotalAllocatedBytes(false) - allocBefore);
                    }

                    var epochElapsed = epochWatch.GetElapsedTime();
                    var epochAllocAfter = GC.GetTotalAllocatedBytes(false);

                    ForceFullGc();

                    var epochEnd = MemorySnapshot.Capture();
                    var epochAllocatedBytes = epochAllocAfter - epochAllocBefore;

                    _output.WriteLine($"Epoch {epoch + 1} | Loss: {epochLoss / batches:F4} | Time: {epochElapsed.TotalMilliseconds:F1}ms | Time so far: {runWatch.GetElapsedTime().TotalMilliseconds:F0}ms");
                    _output.WriteLine("  === per-op stats ===");
                    WriteOperationStats(copyInputStats);
                    WriteOperationStats(resetStats);
                    WriteOperationStats(zeroGradStats);
                    WriteOperationStats(convStats);
                    WriteOperationStats(relu1Stats);
                    WriteOperationStats(maxPoolStats);
                    WriteOperationStats(reshapeStats);
                    WriteOperationStats(bn1Stats);
                    WriteOperationStats(residualStats);
                    WriteOperationStats(gapStats);
                    WriteOperationStats(fcStats);
                    WriteOperationStats(lossStats);
                    WriteOperationStats(backwardStats);
                    WriteOperationStats(optimizerStats);

                    _output.WriteLine("  === grouped stats ===");
                    _output.WriteLine($"  conv+relu+pool+reshape: {TicksToMs(convStats.Ticks + relu1Stats.Ticks + maxPoolStats.Ticks + reshapeStats.Ticks):F1} ms | alloc {BytesToMb(convStats.AllocatedBytes + relu1Stats.AllocatedBytes + maxPoolStats.AllocatedBytes + reshapeStats.AllocatedBytes):F2} MB");
                    _output.WriteLine($"  batchnorm:              {TicksToMs(bn1Stats.Ticks):F1} ms | alloc {BytesToMb(bn1Stats.AllocatedBytes):F2} MB");
                    _output.WriteLine($"  residual:               {TicksToMs(residualStats.Ticks):F1} ms | alloc {BytesToMb(residualStats.AllocatedBytes):F2} MB");
                    _output.WriteLine($"  gap+fc:                 {TicksToMs(gapStats.Ticks + fcStats.Ticks):F1} ms | alloc {BytesToMb(gapStats.AllocatedBytes + fcStats.AllocatedBytes):F2} MB");
                    _output.WriteLine($"  loss:                   {TicksToMs(lossStats.Ticks):F1} ms | alloc {BytesToMb(lossStats.AllocatedBytes):F2} MB");
                    _output.WriteLine($"  backward:               {TicksToMs(backwardStats.Ticks):F1} ms | alloc {BytesToMb(backwardStats.AllocatedBytes):F2} MB");
                    _output.WriteLine($"  optimizer:              {TicksToMs(optimizerStats.Ticks):F1} ms | alloc {BytesToMb(optimizerStats.AllocatedBytes):F2} MB");

                    // _output.WriteLine(graph.FormatBackwardProfileSnapshot(reset: true));

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
            finally
            {
                telemetryListener?.Dispose();
                OverfitTelemetry.Enabled = previousTelemetryEnabled;
            }
        }

        private void WriteOperationStats(in OperationStats stats)
        {
            _output.WriteLine(
                $"  {stats.Name,-24} {TicksToMs(stats.Ticks),8:F1} ms | alloc {BytesToMb(stats.AllocatedBytes),8:F2} MB | calls {stats.Calls,5}");
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

        private struct OperationStats
        {
            public OperationStats(string name)
            {
                Name = name;
                Ticks = 0;
                AllocatedBytes = 0;
                Calls = 0;
            }

            public string Name { get; }

            public long Ticks { get; private set; }

            public long AllocatedBytes { get; private set; }

            public int Calls { get; private set; }

            public void Add(TimeSpan elapsed, long allocatedBytes)
            {
                Ticks += elapsed.Ticks;
                AllocatedBytes += allocatedBytes;
                Calls++;
            }
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