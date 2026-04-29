// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.Diagnostics.Contracts;
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

        /// <summary>
        /// MNIST training benchmark with a small, idiomatic CNN classifier.
        ///
        /// Architecture (~14k parameters):
        ///   Conv(1→8, 3×3)              → [B, 8, 26, 26]
        ///   ReLU
        ///   MaxPool(2×2)                → [B, 8, 13, 13]   = [B, 1352] flat
        ///   Linear(1352→64)
        ///   ReLU
        ///   Linear(64→10)               classification head
        ///
        /// Notes:
        ///   - The previous version of this test used BatchNorm1D(1352) and
        ///     ResidualBlock(1352). Each contained two Linear(1352, 1352)
        ///     layers totaling ~3.7 MB of parameters — about 30× too large
        ///     for MNIST. That made each training step ~21 s/epoch, dominated
        ///     by Linear math, optimizer updates and backward graph allocations
        ///     in the giant residual block, not by genuine CNN cost.
        ///   - This version uses ~14k parameters, which is plenty for MNIST
        ///     (LeNet-1 had ~2.6k and reached &gt;98 %). The training cost is
        ///     now dominated by Conv2D and MaxPool2D, which is what we
        ///     actually want to benchmark.
        /// </summary>
        [Fact]
        public void Mnist_FullTrain60k_CnnBeastMode_Benchmark()
        {
            const int trainSize = 60_000;
            const int batchSize = 64;
            const int epochs = 5;
            const float lr = 0.001f;

            // Set to false for a cleaner allocation baseline without Meter/Listener overhead.
            const bool enableTelemetry = false;

            _output.WriteLine($"Environment.ProcessorCount: {Environment.ProcessorCount}");

            var trainImagesPath = "d:/ml/train-images.idx3-ubyte";
            var trainLabelsPath = "d:/ml/train-labels.idx1-ubyte";

            if (!File.Exists(trainImagesPath) || !File.Exists(trainLabelsPath))
            {
                _output.WriteLine("MNIST files not found.");
                return;
            }

            using var process = Process.GetCurrentProcess();

            var previousTelemetryEnabled = OverfitTelemetry.Enabled;
            OverfitTelemetry.Enabled = enableTelemetry;

            TelemetryListener2? telemetryListener = null;

            try
            {
                var (trainX, trainY) = MnistLoader.Load(trainImagesPath, trainLabelsPath, trainSize);

                // Small, idiomatic CNN classifier — not a giant MLP-residual.
                using var conv1 = new ConvLayer(1, 8, 28, 28, 3);   // → [B, 8, 26, 26]
                using var fcHidden = new LinearLayer(1352, 64);      // 1352 = 8*13*13 after MaxPool
                using var fcOut = new LinearLayer(64, 10);

                var parameters = conv1.Parameters()
                    .Concat(fcHidden.Parameters())
                    .Concat(fcOut.Parameters())
                    .ToArray();

                using var optimizer = new Adam(parameters, lr)
                {
                    UseAdamW = true
                };

                using var graph = new ComputationGraph();

                // Reused batch buffers (allocated once, not per batch).
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

                var paramTotal = parameters.Sum(p => p.DataView.Size);
                _output.WriteLine($"=== START: Small CNN MNIST training (params: {paramTotal:N0}) ===");
                _output.WriteLine($"telemetry enabled: {enableTelemetry}");
                _output.WriteLine($"Environment.ProcessorCount: {Environment.ProcessorCount}");

                ForceFullGc();

                var runStart = new DotNetMemorySnapshot(process);
                var runTotalAllocBefore = GC.GetTotalAllocatedBytes(false);

                _output.WriteLine(FormatMemorySnapshot("RUN START", runStart));

                var runWatch = ValueStopwatch.StartNew();

                for (var epoch = 0; epoch < epochs; epoch++)
                {
                    conv1.Train();
                    fcHidden.Train();
                    fcOut.Train();

                    ForceFullGc();

                    var epochStart = new DotNetMemorySnapshot(process);
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

                    var fcHiddenStats = new OperationStats("Linear hidden");
                    var relu2Stats = new OperationStats("ReLU #2");
                    var fcOutStats = new OperationStats("Linear out");

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

                        // ─── Forward ─────────────────────────────────────────────────

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

                        using var hidden = fcHidden.Forward(graph, p1F);

                        elapsed = sectionWatch.GetElapsedTime();
                        fcHiddenStats.Add(elapsed, GC.GetTotalAllocatedBytes(false) - allocBefore);

                        allocBefore = GC.GetTotalAllocatedBytes(false);
                        sectionWatch = ValueStopwatch.StartNew();

                        using var hiddenAct = TensorMath.ReLU(graph, hidden);

                        elapsed = sectionWatch.GetElapsedTime();
                        relu2Stats.Add(elapsed, GC.GetTotalAllocatedBytes(false) - allocBefore);

                        allocBefore = GC.GetTotalAllocatedBytes(false);
                        sectionWatch = ValueStopwatch.StartNew();

                        using var logits = fcOut.Forward(graph, hiddenAct);

                        elapsed = sectionWatch.GetElapsedTime();
                        fcOutStats.Add(elapsed, GC.GetTotalAllocatedBytes(false) - allocBefore);

                        // ─── Loss + Backward + Step ─────────────────────────────────

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

                    var epochEnd = new DotNetMemorySnapshot(process);
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
                    WriteOperationStats(fcHiddenStats);
                    WriteOperationStats(relu2Stats);
                    WriteOperationStats(fcOutStats);
                    WriteOperationStats(lossStats);
                    WriteOperationStats(backwardStats);
                    WriteOperationStats(optimizerStats);

                    _output.WriteLine("  === grouped stats ===");
                    _output.WriteLine($"  forward conv:           {TicksToMs(convStats.Ticks + relu1Stats.Ticks + maxPoolStats.Ticks + reshapeStats.Ticks):F1} ms | alloc {BytesToMb(convStats.AllocatedBytes + relu1Stats.AllocatedBytes + maxPoolStats.AllocatedBytes + reshapeStats.AllocatedBytes):F2} MB");
                    _output.WriteLine($"  forward classifier:     {TicksToMs(fcHiddenStats.Ticks + relu2Stats.Ticks + fcOutStats.Ticks):F1} ms | alloc {BytesToMb(fcHiddenStats.AllocatedBytes + relu2Stats.AllocatedBytes + fcOutStats.AllocatedBytes):F2} MB");
                    _output.WriteLine($"  loss:                   {TicksToMs(lossStats.Ticks):F1} ms | alloc {BytesToMb(lossStats.AllocatedBytes):F2} MB");
                    _output.WriteLine($"  backward:               {TicksToMs(backwardStats.Ticks):F1} ms | alloc {BytesToMb(backwardStats.AllocatedBytes):F2} MB");
                    _output.WriteLine($"  optimizer:              {TicksToMs(optimizerStats.Ticks):F1} ms | alloc {BytesToMb(optimizerStats.AllocatedBytes):F2} MB");

                    _output.WriteLine($"  allocated total:        {BytesToMb(epochAllocatedBytes):F2} MB");
                    _output.WriteLine($"  GC0/1/2:                {GC.CollectionCount(0) - gc0Before}/{GC.CollectionCount(1) - gc1Before}/{GC.CollectionCount(2) - gc2Before}");
                    _output.WriteLine(FormatMemorySnapshot($"EPOCH {epoch + 1} START", epochStart));
                    _output.WriteLine(FormatMemorySnapshot($"EPOCH {epoch + 1} END", epochEnd));
                    _output.WriteLine($"  live managed delta:     {BytesToMb(epochEnd.LiveManagedBytes - epochStart.LiveManagedBytes):F2} MB");
                    _output.WriteLine($"  private bytes delta:    {BytesToMb(epochEnd.PrivateMemoryBytes - epochStart.PrivateMemoryBytes):F2} MB");
                    _output.WriteLine($"  working set delta:      {BytesToMb(epochEnd.WorkingSetBytes - epochStart.WorkingSetBytes):F2} MB");
                }

                ForceFullGc();

                var runEnd = new DotNetMemorySnapshot(process);
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

        private static string FormatMemorySnapshot(string name, in DotNetMemorySnapshot snapshot)
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
    }
}
