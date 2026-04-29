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
    /// Isolates MaxPool2D forward cost at each layer of overhead.
    ///
    /// Three scenarios in ascending overhead order:
    ///   1. PoolingKernel_Direct              — pure span kernel, no autograd (floor)
    ///   2. TensorMath_WithGraph              — inference path, no maxIndices, no grad
    ///   3. TensorMath_WithGraph_RequiresGrad — training path, allocates maxIndices, records tape
    ///
    /// Expected after pool=2 SIMD optimization (was ~815 µs for all):
    ///   PoolingKernel_Direct              ~250 µs
    ///   TensorMath_WithGraph              ~260 µs  (inference: no maxIndices)
    ///   TensorMath_WithGraph_RequiresGrad ~340 µs  (training: maxIndices + tape)
    ///
    /// Input: [64, 8, 26, 26] → Output: [64, 8, 13, 13]  (MNIST CNN after Conv)
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class MaxPoolIsolationBenchmark
    {
        private const int BatchSize = 64;
        private const int Channels = 8;
        private const int InputH = 26;
        private const int InputW = 26;
        private const int Pool = 2;

        private const int InputElements = BatchSize * Channels * InputH * InputW;
        private const int OutputElements = BatchSize * Channels * (InputH / Pool) * (InputW / Pool);

        private float[] _inputArr = null!;
        private float[] _outputArr = null!;

        private TensorStorage<float> _inputStorage = null!;
        private TensorStorage<float> _inputStorageGrad = null!;
        private AutogradNode _inputNode = null!;
        private AutogradNode _inputNodeGrad = null!;
        private ComputationGraph _graph = null!;

        [GlobalSetup]
        public void Setup()
        {
            _inputArr = new float[InputElements];
            _outputArr = new float[OutputElements];

            var rng = new Random(42);
            for (var i = 0; i < InputElements; i++)
                _inputArr[i] = (float)rng.NextDouble() * 2f - 1f;

            _inputStorage = new TensorStorage<float>(InputElements, clearMemory: false);
            _inputStorageGrad = new TensorStorage<float>(InputElements, clearMemory: false);
            _inputArr.AsSpan().CopyTo(_inputStorage.AsSpan());
            _inputArr.AsSpan().CopyTo(_inputStorageGrad.AsSpan());

            var shape = new TensorShape(BatchSize, Channels, InputH, InputW);
            _inputNode = new AutogradNode(_inputStorage, shape, requiresGrad: false);
            _inputNodeGrad = new AutogradNode(_inputStorageGrad, shape, requiresGrad: true);

            _graph = new ComputationGraph();
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _inputNode.Dispose();
            _inputNodeGrad.Dispose();
            _inputStorage.Dispose();
            _inputStorageGrad.Dispose();
            _graph.Dispose();
        }

        [Benchmark(Baseline = true)]
        public void PoolingKernel_Direct()
        {
            PoolingKernels.MaxPool2DForwardNchw(
                _inputArr, _outputArr, Channels, InputH, InputW, Pool);
        }

        [Benchmark]
        public void TensorMath_WithGraph()
        {
            _graph.Reset();
            using var output = TensorMath.MaxPool2D(
                _graph, _inputNode, Channels, InputH, InputW, Pool);
        }

        [Benchmark]
        public void TensorMath_WithGraph_RequiresGrad()
        {
            _graph.Reset();
            using var output = TensorMath.MaxPool2D(
                _graph, _inputNodeGrad, Channels, InputH, InputW, Pool);
        }
    }
}