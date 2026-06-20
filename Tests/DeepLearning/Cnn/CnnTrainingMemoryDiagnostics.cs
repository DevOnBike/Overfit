// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.DeepLearning.Cnn
{
    /// <summary>
    /// Measures WHERE training memory goes in a CNN step (no data files — random input; memory footprint
    /// is data-independent). Reports the tape arena's actual high-water-mark (<c>TapeBuffer.CurrentOffset</c>
    /// = the live-activation footprint forward→backward), the arena capacity, and the im2col workspace
    /// (computed) — for an MNIST-scale conv and a wide conv — so the biggest reducible item is confirmed
    /// before any optimisation. Diagnostic, not a correctness check.
    /// </summary>
    public sealed class CnnTrainingMemoryDiagnostics
    {
        private readonly ITestOutputHelper _out;
        public CnnTrainingMemoryDiagnostics(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Report_TrainingStepMemory_MnistVsWideConv()
        {
            Measure("MNIST  ", batch: 32, inC: 1, hw: 28, outC: 8, k: 3);
            Measure("wide   ", batch: 32, inC: 3, hw: 32, outC: 64, k: 3);
            Measure("wide×2 ", batch: 64, inC: 3, hw: 48, outC: 64, k: 3);
        }

        private void Measure(string tag, int batch, int inC, int hw, int outC, int k)
        {
            var outHw = hw - k + 1;
            var poolHw = outHw / 2;
            var flat = outC * poolHw * poolHw;

            var conv = new ConvLayer(inC, outC, hw, hw, k);
            var fc = new LinearLayer(flat, 10);

            var rng = new Random(1);
            using var xBuf = new TensorStorage<float>(batch * inC * hw * hw, clearMemory: false);
            using var yBuf = new TensorStorage<float>(batch * 10, clearMemory: false);
            var xs = xBuf.AsSpan();
            for (var i = 0; i < xs.Length; i++)
            {
                xs[i] = (float)(rng.NextDouble() - 0.5);
            }
            yBuf.AsSpan().Clear();
            for (var b = 0; b < batch; b++)
            {
                yBuf.AsSpan()[b * 10 + (b % 10)] = 1f;
            }

            using var graph = new ComputationGraph();
            using var xNode = new AutogradNode(xBuf, new TensorShape(batch, inC, hw, hw));
            using var yNode = new AutogradNode(yBuf, new TensorShape(batch, 10));

            using var h1 = conv.Forward(graph, xNode);
            using var a1 = TensorMath.ReLU(graph, h1);
            using var p1 = TensorMath.MaxPool2D(graph, a1, outC, outHw, outHw, 2);
            using var pFlat = TensorMath.Reshape(graph, p1, batch, flat);
            using var output = fc.Forward(graph, pFlat);
            using var loss = TensorMath.SoftmaxCrossEntropy(graph, output, yNode);
            graph.Backward(loss);

            // Arena high-water-mark AFTER forward+backward, BEFORE Reset = the live-activation footprint.
            var arenaUsedMb = graph.TapeBuffer.CurrentOffset * 4.0 / (1024 * 1024);
            var arenaCapMb = graph.TapeBuffer.Size * 4.0 / (1024 * 1024);

            // im2col workspace (separate from arena): workerCount × k²·inC·outHW × 2 (col + dcol).
            var workers = Environment.ProcessorCount;
            var colLen = (long)k * k * inC * outHw * outHw;
            var im2colMb = workers * colLen * 2 * 4.0 / (1024 * 1024);

            _out.WriteLine(
                $"[{tag}] batch={batch} inC={inC} {hw}x{hw} outC={outC} k={k}  " +
                $"arena-used={arenaUsedMb:F1} MB (cap {arenaCapMb:F0})  " +
                $"im2col≈{im2colMb:F1} MB ({workers}w × {colLen} × 2)");
        }
    }
}
