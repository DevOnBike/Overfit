using System.Diagnostics;
using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;
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

            _output.WriteLine("=== START: Trening ResNet na Taśmie (NativeBuffer) ===");

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                conv1.Train();
                bn1.Train();
                res1.Train();
                fcOut.Train();

                ResidualBlock.DiagnosticsEnabled = true;
                ResidualBlock.ResetDiagnostics();

                float epochLoss = 0f;
                int batches = trainSize / batchSize;

                long tConv = 0;
                long tBn = 0;
                long tRes = 0;
                long tHead = 0;
                long tLoss = 0;
                long tBackward = 0;
                long tOptimizer = 0;

                long aConv = 0;
                long aBn = 0;
                long aRes = 0;
                long aHead = 0;
                long aLoss = 0;
                long aBackward = 0;
                long aOptimizer = 0;

                var sectionSw = new Stopwatch();

                long epochAllocBefore = GC.GetTotalAllocatedBytes(true);
                int gc0Before = GC.CollectionCount(0);
                int gc1Before = GC.CollectionCount(1);
                int gc2Before = GC.CollectionCount(2);

                for (int b = 0; b < batches; b++)
                {
                    graph.Reset();
                    optimizer.ZeroGrad();

                    using var xBData = new FastTensor<float>(batchSize, 1, 28, 28, clearMemory: false);
                    using var yBData = new FastTensor<float>(batchSize, 10, clearMemory: false);
                    using var xBNode = new AutogradNode(xBData, requiresGrad: false);
                    using var yBNode = new AutogradNode(yBData, requiresGrad: false);

                    trainX.GetView().AsReadOnlySpan()
                        .Slice(b * batchSize * 784, batchSize * 784)
                        .CopyTo(xBData.GetView().AsSpan());

                    trainY.GetView().AsReadOnlySpan()
                        .Slice(b * batchSize * 10, batchSize * 10)
                        .CopyTo(yBData.GetView().AsSpan());

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
                var rd = ResidualBlock.GetDiagnosticsSnapshot();

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

                _output.WriteLine($"  residual.calls:         {rd.Calls}");
                _output.WriteLine($"  residual.total:         {rd.TotalMs:F1} ms | alloc {(rd.TotalAllocBytes / 1024.0 / 1024.0):F2} MB");
                _output.WriteLine($"    linear1:              {rd.Linear1Ms:F1} ms | alloc {(rd.Linear1AllocBytes / 1024.0 / 1024.0):F2} MB");
                _output.WriteLine($"    bn1:                  {rd.BatchNorm1Ms:F1} ms | alloc {(rd.BatchNorm1AllocBytes / 1024.0 / 1024.0):F2} MB");
                _output.WriteLine($"    relu1:                {rd.ReLU1Ms:F1} ms | alloc {(rd.ReLU1AllocBytes / 1024.0 / 1024.0):F2} MB");
                _output.WriteLine($"    linear2:              {rd.Linear2Ms:F1} ms | alloc {(rd.Linear2AllocBytes / 1024.0 / 1024.0):F2} MB");
                _output.WriteLine($"    bn2:                  {rd.BatchNorm2Ms:F1} ms | alloc {(rd.BatchNorm2AllocBytes / 1024.0 / 1024.0):F2} MB");
                _output.WriteLine($"    add:                  {rd.AddMs:F1} ms | alloc {(rd.AddAllocBytes / 1024.0 / 1024.0):F2} MB");
                _output.WriteLine($"    relu2:                {rd.ReLU2Ms:F1} ms | alloc {(rd.ReLU2AllocBytes / 1024.0 / 1024.0):F2} MB");
            }

            ResidualBlock.DiagnosticsEnabled = false;
            _output.WriteLine("=== KONIEC ===");
        }
    }
}