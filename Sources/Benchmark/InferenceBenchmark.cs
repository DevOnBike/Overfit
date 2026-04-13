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
    ///     Performance comparison between ONNX Runtime and Overfit engine.
    ///     Evaluates execution speed and memory allocation overhead during inference.
    /// </summary>
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    [DisassemblyDiagnoser(maxDepth: 2)]
    public class InferenceBenchmark
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;
        private float[] _inputData;
        private AutogradNode _inputNode;
        private NamedOnnxValue[] _onnxInputs;

        private InferenceSession _onnxSession;
        private Sequential _overfitModel;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);
            _inputData = Enumerable.Range(0, InputSize).Select(_ => (float)rnd.NextDouble()).ToArray();

            _onnxSession = new InferenceSession("benchmark_model.onnx");
            var tensor = new DenseTensor<float>(_inputData, [1, InputSize]);
            _onnxInputs = [NamedOnnxValue.CreateFromTensor("input", tensor)];

            _overfitModel = new Sequential(new LinearLayer(InputSize, OutputSize));
            _overfitModel.Load("benchmark_model.bin");
            _overfitModel.Eval();

            var inputTensor = new FastTensor<float>(1, InputSize, clearMemory: false);
            _inputData.AsSpan().CopyTo(inputTensor.GetView().AsSpan());
            _inputNode = new AutogradNode(inputTensor, false);

            for (var i = 0; i < 100; i++)
            {
                _overfitModel.Forward(null, _inputNode);
            }
        }

        [Benchmark(Baseline = true)]
        public float OnnxRuntime_PreAllocated()
        {
            using var results = _onnxSession.Run(_onnxInputs);
            return results.First().AsTensor<float>()[0];
        }

        /// <summary>
        ///     Benchmarks Overfit with zero-allocation SIMD inference.
        ///     Leverages the full inference path from raw data to prediction.
        /// </summary>
        [Benchmark]
        public float Overfit_ZeroAlloc()
        {
            var outputNode = _overfitModel.Forward(null, _inputNode);

            return outputNode.DataView.AsReadOnlySpan()[0];
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _onnxSession?.Dispose();
            _inputNode?.Dispose();
            _overfitModel?.Dispose();
        }
    }
}