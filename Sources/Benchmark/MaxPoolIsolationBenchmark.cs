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
    /// Motivation: MnistTrainingTests reports MaxPool2D taking ~1.3 ms/call for
    /// input [64, 8, 26, 26] — about 13× slower than expected based on raw
    /// arithmetic count (~346k comparisons should run in &lt; 100 µs).
    ///
    /// This benchmark splits the cost between three layers:
    ///   1. Pure pooling kernel (PoolingKernels.MaxPool2DForwardNchw)
    ///   2. TensorMath.MaxPool2D without a graph (allocates own storage)
    ///   3. TensorMath.MaxPool2D with a ComputationGraph (arena allocation + tape record)
    ///
    /// The delta between (1) and (3) tells us where the overhead lives.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    [MemoryDiagnoser]
    public class MaxPoolIsolationBenchmark
    {
        // MNIST CNN forward: input shape [B, 8, 26, 26] → output [B, 8, 13, 13]
        private const int BatchSize = 64;
        private const int Channels = 8;
        private const int InputH = 26;
        private const int InputW = 26;
        private const int Pool = 2;

        private const int InputSize = BatchSize * Channels * InputH * InputW;   // 86,528
        private const int OutputSize = BatchSize * Channels * (InputH / Pool) * (InputW / Pool); // 21,632

        private float[] _input = null!;
        private float[] _output = null!;

        private TensorStorage<float> _inputStorage = null!;
        private AutogradNode _inputNode = null!;
        private ComputationGraph _graph = null!;

        [GlobalSetup]
        public void Setup()
        {
            _input = new float[InputSize];
            _output = new float[OutputSize];

            var rng = new Random(42);
            for (var i = 0; i < InputSize; i++)
            {
                _input[i] = (float)rng.NextDouble();
            }

            _inputStorage = new TensorStorage<float>(InputSize, clearMemory: false);
            _input.AsSpan().CopyTo(_inputStorage.AsSpan());

            _inputNode = new AutogradNode(
                _inputStorage,
                new TensorShape(BatchSize, Channels, InputH, InputW),
                requiresGrad: false);

            _graph = new ComputationGraph();
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _inputNode.Dispose();
            _inputStorage.Dispose();
            _graph.Dispose();
        }

        // ────────────────────────────────────────────────────────────────────
        // Layer 1: Pure kernel
        //   This is the floor. Should be ~50-100 µs for 86k input → 21k output.
        //   No allocation, no autograd, no graph.
        // ────────────────────────────────────────────────────────────────────
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

        // ────────────────────────────────────────────────────────────────────
        // Layer 2: TensorMath.MaxPool2D without graph (graph = null)
        //   Allocates output + maxIndices storage as standalone TensorStorage.
        //   No tape, no graph.Record. Measures cost of:
        //     - 2× new TensorStorage<float>(...)
        //     - 2× new AutogradNode(...)
        //     - the pooling math (with the loop in TensorMath.Pooling.cs)
        // ────────────────────────────────────────────────────────────────────
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

        // ────────────────────────────────────────────────────────────────────
        // Layer 3: TensorMath.MaxPool2D with graph
        //   Same as above plus:
        //     - graph.AllocateIntermediate(...) for both nodes
        //     - graph.Record(OpCode.MaxPool2D, ...) — tape append
        //   This is what MnistTrainingTests is measuring.
        // ────────────────────────────────────────────────────────────────────
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

        // ────────────────────────────────────────────────────────────────────
        // Layer 4: TensorMath.MaxPool2D with graph + RequiresGrad
        //   This is the realistic training-loop scenario. After conv1 with
        //   RequiresGrad=true, the input to MaxPool also has RequiresGrad,
        //   so maxIndices is kept and graph.Record runs.
        // ────────────────────────────────────────────────────────────────────
        [Benchmark]
        public void TensorMath_WithGraph_RequiresGrad()
        {
            _graph.Reset();

            // Build a node with requiresGrad to exercise the full training path
            using var trainingInput = new AutogradNode(
                _inputStorage,
                new TensorShape(BatchSize, Channels, InputH, InputW),
                requiresGrad: true);

            using var output = TensorMath.MaxPool2D(
                _graph,
                trainingInput,
                Channels,
                InputH,
                InputW,
                Pool);
        }
    }
}
