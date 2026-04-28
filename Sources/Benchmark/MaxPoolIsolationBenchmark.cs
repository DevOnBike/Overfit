// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Kernels;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace Benchmarks
{
    /// <summary>
    /// Diagnostic benchmark to isolate MaxPool2D performance.
    ///
    /// Splits cost between:
    /// 1. Existing generic pooling kernel.
    /// 2. Specialized pool=2/stride=2 fast path.
    /// 3. TensorMath.MaxPool2D without graph.
    /// 4. TensorMath.MaxPool2D with graph.
    /// 5. TensorMath.MaxPool2D with graph + RequiresGrad.
    ///
    /// The RequiresGrad case uses its own storage/node pair created in GlobalSetup.
    /// Do not create a temporary AutogradNode over the same storage inside the benchmark method.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    [MemoryDiagnoser]
    public class MaxPoolIsolationBenchmark
    {
        // MNIST CNN forward: input shape [B, 8, 26, 26] -> output [B, 8, 13, 13]
        private const int BatchSize = 64;
        private const int Channels = 8;
        private const int InputH = 26;
        private const int InputW = 26;
        private const int Pool = 2;

        private const int OutputH = InputH / Pool;
        private const int OutputW = InputW / Pool;

        private const int InputSize = BatchSize * Channels * InputH * InputW;
        private const int OutputSize = BatchSize * Channels * OutputH * OutputW;

        private float[] _input = null!;
        private float[] _output = null!;
        private float[] _fastOutput = null!;

        private TensorStorage<float> _inputStorage = null!;
        private TensorStorage<float> _trainingInputStorage = null!;

        private AutogradNode _inputNode = null!;
        private AutogradNode _trainingInputNode = null!;

        private ComputationGraph _graph = null!;

        [GlobalSetup]
        public void Setup()
        {
            _input = new float[InputSize];
            _output = new float[OutputSize];
            _fastOutput = new float[OutputSize];

            FillDeterministic(
                _input,
                seed: 42);

            _inputStorage = new TensorStorage<float>(
                InputSize,
                clearMemory: false);

            _input.AsSpan().CopyTo(
                _inputStorage.AsSpan());

            _inputNode = new AutogradNode(
                _inputStorage,
                new TensorShape(
                    BatchSize,
                    Channels,
                    InputH,
                    InputW),
                requiresGrad: false);

            _trainingInputStorage = new TensorStorage<float>(
                InputSize,
                clearMemory: false);

            _input.AsSpan().CopyTo(
                _trainingInputStorage.AsSpan());

            _trainingInputNode = new AutogradNode(
                _trainingInputStorage,
                new TensorShape(
                    BatchSize,
                    Channels,
                    InputH,
                    InputW),
                requiresGrad: true);

            _graph = new ComputationGraph();

            // Warmup + correctness guard for the specialized fast path.
            PoolingKernels.MaxPool2DForwardNchw(
                _input,
                _output,
                Channels,
                InputH,
                InputW,
                Pool);

            PoolingFastPathKernels.MaxPool2DForwardNchwPool2Stride2(
                _input,
                _fastOutput,
                Channels,
                InputH,
                InputW);

            AssertClose(
                _output,
                _fastOutput,
                tolerance: 0f);
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _inputNode.Dispose();
            _trainingInputNode.Dispose();

            _inputStorage.Dispose();
            _trainingInputStorage.Dispose();

            _graph.Dispose();
        }

        /// <summary>
        /// Existing generic pooling kernel.
        /// This is the current floor for TensorMath.
        /// </summary>
        [Benchmark(Baseline = true)]
        public void PoolingKernel_Direct()
        {
            PoolingKernels.MaxPool2DForwardNchw(
                _input,
                _output,
                Channels,
                InputH,
                InputW,
                Pool);
        }

        /// <summary>
        /// Specialized pool=2/stride=2/no-padding fast path.
        /// This is the candidate production kernel for the common MNIST CNN case.
        /// </summary>
        [Benchmark]
        public void PoolingKernel_FastPath_Pool2Stride2()
        {
            PoolingFastPathKernels.MaxPool2DForwardNchwPool2Stride2(
                _input,
                _fastOutput,
                Channels,
                InputH,
                InputW);
        }

        /// <summary>
        /// TensorMath.MaxPool2D without graph.
        /// Measures TensorMath wrapper + output/max-index allocation + kernel cost.
        /// </summary>
        [Benchmark]
        public void TensorMath_NoGraph()
        {
            using var output = TensorMath.MaxPool2D(
                null,
                _inputNode,
                Channels,
                InputH,
                InputW,
                Pool);
        }

        /// <summary>
        /// TensorMath.MaxPool2D with graph but no requires-grad input.
        /// </summary>
        [Benchmark]
        public void TensorMath_WithGraph()
        {
            _graph.Reset();

            using var output = TensorMath.MaxPool2D(
                _graph,
                _inputNode,
                Channels,
                InputH,
                InputW,
                Pool);
        }

        /// <summary>
        /// Realistic training scenario: input requires grad.
        /// Uses a dedicated training input node created once in GlobalSetup.
        /// </summary>
        [Benchmark]
        public void TensorMath_WithGraph_RequiresGrad()
        {
            _graph.Reset();

            using var output = TensorMath.MaxPool2D(
                _graph,
                _trainingInputNode,
                Channels,
                InputH,
                InputW,
                Pool);
        }

        private static void FillDeterministic(
            float[] data,
            int seed)
        {
            var rng = new Random(seed);

            for (var i = 0; i < data.Length; i++)
            {
                data[i] = (float)rng.NextDouble();
            }
        }

        private static void AssertClose(
            ReadOnlySpan<float> expected,
            ReadOnlySpan<float> actual,
            float tolerance)
        {
            if (expected.Length != actual.Length)
            {
                throw new InvalidOperationException(
                    $"Length mismatch: expected={expected.Length}, actual={actual.Length}");
            }

            for (var i = 0; i < expected.Length; i++)
            {
                var diff = MathF.Abs(expected[i] - actual[i]);

                if (diff > tolerance)
                {
                    throw new InvalidOperationException(
                        $"Mismatch at {i}: expected={expected[i]}, actual={actual[i]}, diff={diff}, tolerance={tolerance}");
                }
            }
        }
    }
}