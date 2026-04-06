// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;
using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Benchmarks
{
    /// <summary>
    /// Baseline performance comparison for a single inference pass.
    /// Measures the raw overhead of a single Forward call without loop-level optimizations.
    /// </summary>
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    [DisassemblyDiagnoser(maxDepth: 2)]
    public class SingleInferenceBenchmark
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;

        private float[] _inputData;

        private InferenceSession _onnxSession;
        private NamedOnnxValue[] _onnxInputs;

        private Sequential _overfitModel;
        private FastTensor<float> _overfitInputTensor;
        private AutogradNode _inputNode;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);
            _inputData = Enumerable.Range(0, InputSize).Select(_ => (float)rnd.NextDouble()).ToArray();

            // Setup ONNX Runtime session
            _onnxSession = new InferenceSession("benchmark_model.onnx");
            var tensor = new DenseTensor<float>(_inputData, [1, InputSize]);
            _onnxInputs = [NamedOnnxValue.CreateFromTensor("input", tensor)];

            // Setup Overfit model
            _overfitModel = new Sequential(new LinearLayer(InputSize, OutputSize));
            _overfitModel.Load("benchmark_model.bin");
            _overfitModel.Eval(); // Enable inference mode (triggers weight pre-transposition)

            // Prepare Overfit input tensors
            _overfitInputTensor = new FastTensor<float>(false, 1, InputSize);
            _inputData.AsSpan().CopyTo(_overfitInputTensor.AsSpan());
            _inputNode = new AutogradNode(_overfitInputTensor, requiresGrad: false);

            for (var i = 0; i < 100; i++)
            {
                _overfitModel.Forward(null, _inputNode);
            }
        }

        /// <summary>
        /// Benchmarks ONNX Runtime using a pre-allocated input tensor.
        /// Represents the "best-case" scenario for ONNX by minimizing setup overhead.
        /// </summary>
        [Benchmark(Baseline = true)]
        public float OnnxRuntime_PreAllocated()
        {
            using var results = _onnxSession.Run(_onnxInputs);
            return results.First().AsTensor<float>()[0];
        }

        /// <summary>
        /// Benchmarks ONNX Runtime including the cost of tensor creation.
        /// Reflects a typical production scenario where new data arrives per request.
        /// </summary>
        [Benchmark]
        public float OnnxRuntime_FullAllocation()
        {
            var tensor = new DenseTensor<float>(_inputData, [1, InputSize]);
            var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor("input", tensor) };
            using var results = _onnxSession.Run(inputs);
            return results.First().AsTensor<float>()[0];
        }

        /// <summary>
        /// Benchmarks Overfit using the zero-allocation SIMD inference path.
        /// Leverages pre-transposed weights for optimal hardware utilization.
        /// </summary>
        [Benchmark]
        public float Overfit_ZeroAlloc()
        {
            return _overfitModel.Forward(null, _inputNode).Data.AsSpan()[0];
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _onnxSession?.Dispose();
            _overfitInputTensor?.Dispose();
            _inputNode?.Dispose();
            _overfitModel?.Dispose();
        }
    }
}