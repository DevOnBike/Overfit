// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Kernels;
using DevOnBike.Overfit.Tensors.Core;

namespace Benchmarks
{
    /// <summary>
    /// Per-image cost of one convolution as the batch grows — the measurement that decides whether batched
    /// graph inference is worth building.
    ///
    /// <para><b>Why this is measured before anything is written.</b> The ONNX graph path is strictly
    /// single-image: <c>OnnxGraphModel.RunInference</c> throws unless <c>input.Length == _inputSize</c>, and
    /// every intermediate buffer is sized for one item. Supporting a batch is a capability change — buffer
    /// sizing, per-node batch propagation, and roughly 12.8 MB per extra image for VGG-16's largest
    /// activation — not a tuning knob.</para>
    ///
    /// <para><b>And the usual justification does not survive arithmetic here.</b> "Batching raises arithmetic
    /// intensity because the weights are reused" is false for this loop order: A is read once per N-panel and
    /// there are <c>N / 32</c> panels, so a batch of B gives B times the panels and B times the A traffic,
    /// exactly proportional. **Nothing is amortised.** What batching can still buy is occupancy — VGG-16's
    /// conv11-13 have <c>N = 196</c>, which is seven panels against thirty-two workers — and the fixed cost
    /// of each call. <b>Those are real but bounded, and this benchmark says by how much.</b></para>
    ///
    /// <para>The shape is VGG-16's conv10: 512 in, 512 out, 28x28, 3x3 pad 1, which is the well-conditioned
    /// case. <see cref="Shape"/> also covers conv13's 14x14, where only seven panels exist and occupancy is
    /// the thing under test.</para>
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark -- --filter "*ConvBatchScaling*"
    /// </summary>
    // The shared config warms up five times with one invocation per iteration, which is not enough for
    // calls this long: the first measurements land bimodally at roughly 6.5 ms and 21 ms, a 3.2x split
    // that is the signature of tier-0 code still being measured. More warmup, and enough iterations for
    // the median to mean something.
    [Config(typeof(BenchmarkConfig))]
    [WarmupCount(25)]
    [IterationCount(30)]
    public class ConvBatchScalingBenchmark : IDisposable
    {
        private const int Channels = 512;
        private const int KernelSize = 3;

        /// <summary>28 is conv10 (N = 784, 25 panels); 14 is conv13 (N = 196, seven panels).</summary>
        [Params(28, 14)]
        public int Hw { get; set; } = 28;

        [Params(1, 2, 4, 8)]
        public int Batch { get; set; } = 1;

        private TensorStorage<float> _input = null!;
        private TensorStorage<float> _kernels = null!;
        private TensorStorage<float> _packed = null!;
        private TensorStorage<float> _output = null!;

        private int K => Channels * KernelSize * KernelSize;

        [GlobalSetup]
        public void Setup()
        {
            _input = new TensorStorage<float>(Batch * Channels * Hw * Hw, clearMemory: false);
            _kernels = new TensorStorage<float>(Channels * K, clearMemory: false);
            _packed = new TensorStorage<float>(
                Conv2DGemmKernels.PackedKernelLength(Channels, K), clearMemory: false);
            _output = new TensorStorage<float>(Batch * Channels * Hw * Hw, clearMemory: true);

            Fill(_input.AsSpan(), seed: 5);
            Fill(_kernels.AsSpan(), seed: 19);

            Conv2DGemmKernels.PackKernels(_kernels.AsReadOnlySpan(), _packed.AsSpan(), Channels, K);
        }

        /// <summary>
        /// One convolution over the whole batch. Divide the reported mean by <see cref="Batch"/> to compare
        /// per-image cost; the benchmark deliberately does not do that arithmetic, so the raw number cannot
        /// be quoted as if it were a single-image figure.
        /// </summary>
        [Benchmark]
        public void Convolution()
        {
            Conv2DKernels.ForwardNchw(
                _input.AsReadOnlySpan(),
                _kernels.AsReadOnlySpan(),
                _output.AsSpan(),
                Batch,
                Channels,
                Channels,
                Hw,
                Hw,
                KernelSize,
                padding: 1,
                stride: 1,
                packedKernels: _packed.AsReadOnlySpan());
        }

        [GlobalCleanup]
        public void Dispose()
        {
            _input?.Dispose();
            _kernels?.Dispose();
            _packed?.Dispose();
            _output?.Dispose();
        }

        private static void Fill(Span<float> values, int seed)
        {
            var state = (uint)(0x9E3779B9 + seed);

            for (var i = 0; i < values.Length; i++)
            {
                state = (state * 1664525u) + 1013904223u;
                values[i] = (((state & 0x00FFFFFF) / 16777216f) * 2f) - 1f;
            }
        }
    }
}
