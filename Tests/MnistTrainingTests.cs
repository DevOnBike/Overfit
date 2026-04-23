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
using DevOnBike.Overfit.Tensors.Core; // Zmieniono namespace
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

        //[Fact]
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

            var graph = new ComputationGraph();
            var totalSw = Stopwatch.StartNew();

            var traceCollector = new EpochTraceCollector();
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

            using var textSink = TextWriterDiagnosticsSink.CreateFile(textTracePath, append: false);
            using var jsonlSink = JsonLinesDiagnosticsSink.CreateFile(jsonlTracePath, append: false);
            var compositeSink = new CompositeOverfitDiagnosticsSink(traceCollector, textSink, jsonlSink);

            using var session = new DiagnosticsSession(enabled: true, sink: compositeSink);

            traceCollector.Reset();

            _output.WriteLine("=== START: Trening ResNet na Taśmie (NativeBuffer) ===");

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                traceCollector.Reset();

                conv1.Train();
                bn1.Train();
                res1.Train();
                fcOut.Train();

                float epochLoss = 0f;
                int batches = trainSize / batchSize;

                long tConv = 0, tBn = 0, tRes = 0, tHead = 0, tLoss = 0, tBackward = 0, tOptimizer = 0;
                long aConv = 0, aBn = 0, aRes = 0, aHead = 0, aLoss = 0, aBackward = 0, aOptimizer = 0;

                var sectionSw = new Stopwatch();

                long epochAllocBefore = GC.GetTotalAllocatedBytes(true);
                int gc0Before = GC.CollectionCount(0);
                int gc1Before = GC.CollectionCount(1);
                int gc2Before = GC.CollectionCount(2);

                for (int b = 0; b < batches; b++)
                {
                    graph.Reset();
                    optimizer.ZeroGrad();

                    // POPRAWKA: Przejście na TensorStorage i DOD
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

                    allocBefore = GC.GetTotalAllocatedBytes(false);
                    sectionSw.Restart();
                    optimizer.Step();
                    sectionSw.Stop();
                    tOptimizer += sectionSw.ElapsedTicks;
                    aOptimizer += GC.GetTotalAllocatedBytes(false) - allocBefore;
                }

                long epochAllocAfter = GC.GetTotalAllocatedBytes(true);
                var snapshot = traceCollector.Snapshot();

                var epochJson = Path.Combine(traceDir, $"epoch_{epoch + 1:D2}.json");
                var epochCsv = Path.Combine(traceDir, $"epoch_{epoch + 1:D2}.csv");
                EpochTraceExporter.WriteJson(epochJson, epoch + 1, snapshot);
                EpochTraceExporter.WriteCsv(epochCsv, snapshot);

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
                _output.WriteLine(BenchmarkTraceFormatter.Format(snapshot));
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

            _output.WriteLine("=== KONIEC ===");
        }
    }
}