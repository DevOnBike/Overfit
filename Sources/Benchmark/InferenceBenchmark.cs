// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core; // Zmieniono namespace na Core
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Benchmarks
{
    /// <summary>
    ///     Performance comparison between ONNX Runtime and Overfit engine.
    ///     Evaluates execution speed and memory allocation overhead during inference.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    // [HardwareCounters(HardwareCounter.InstructionRetired, HardwareCounter.CacheMisses)]
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

            // POPRAWKA: Używamy TensorStorage i TensorShape
            var inputTensor = new TensorStorage<float>(InputSize, clearMemory: false);
            _inputData.AsSpan().CopyTo(inputTensor.AsSpan());
            _inputNode = new AutogradNode(inputTensor, new TensorShape(1, InputSize), false);

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
            _overfitModel?.Dispose();
            // Dispose logic was moved to _inputNode.Dispose() which holds the storage
            _inputNode?.Dispose();
        }
    }
}