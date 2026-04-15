// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Benchmarks
{
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    public class ThroughputBenchmark
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;
        private const int Iterations = 10_000;

        private float[] _inputData;
        private AutogradNode _inputNode;
        private NamedOnnxValue[] _onnxInputs;
        private InferenceSession _onnxSession;
        private FastTensor<float> _overfitInputTensor;
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

            // POPRAWKA: Poprawny konstruktor (dim0, dim1, clearMemory)
            _overfitInputTensor = new FastTensor<float>(1, InputSize, clearMemory: false);
            // POPRAWKA: GetView().AsSpan()
            _inputData.AsSpan().CopyTo(_overfitInputTensor.GetView().AsSpan());
            _inputNode = new AutogradNode(_overfitInputTensor, false);

            for (var i = 0; i < 100; i++)
            {
                _overfitModel.Forward(null, _inputNode);
            }
        }

        [Benchmark(Baseline = true)]
        public float OnnxRuntime_10k()
        {
            var sum = 0f;
            for (var i = 0; i < Iterations; i++)
            {
                using var results = _onnxSession.Run(_onnxInputs);
                sum += results.First().AsTensor<float>()[0];
            }
            return sum;
        }

        [Benchmark]
        public float Overfit_10k_ZeroAlloc()
        {
            var sum = 0f;
            for (var i = 0; i < Iterations; i++)
            {
                // POPRAWKA: DataView.AsReadOnlySpan() zamiast Data.AsSpan()
                sum += _overfitModel.Forward(null, _inputNode).DataView.AsReadOnlySpan()[0];
            }
            return sum;
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