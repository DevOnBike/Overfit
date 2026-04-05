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
    /// 10 000 inferencji w tight-loop.
    /// ONNX: 912B × 10k = 9.12MB alokacji → GC Gen-0 kicks in.
    /// Overfit: 0B → zero GC, flat throughput.
    /// </summary>
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    public class ThroughputBenchmark
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;
        private const int Iterations = 10_000;

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

            _onnxSession = new InferenceSession("benchmark_model.onnx");
            var tensor = new DenseTensor<float>(_inputData, [1, InputSize]);
            _onnxInputs = [NamedOnnxValue.CreateFromTensor("input", tensor)];

            _overfitModel = new Sequential(new LinearLayer(InputSize, OutputSize));
            _overfitModel.Load("benchmark_model.bin");
            _overfitModel.Eval();

            _overfitInputTensor = new FastTensor<float>(false, 1, InputSize);
            _inputData.AsSpan().CopyTo(_overfitInputTensor.AsSpan());
            _inputNode = new AutogradNode(_overfitInputTensor, requiresGrad: false);

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
                sum += _overfitModel.Forward(null, _inputNode).Data.AsSpan()[0];
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