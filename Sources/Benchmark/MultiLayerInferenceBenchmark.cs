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
    /// 3-warstwowy MLP: 784 → 256 → 128 → 10.
    /// Więcej warstw = więcej P/Invoke overhead w ONNX per forward.
    /// Overfit: każda warstwa to SIMD dot product, zero interop.
    /// </summary>
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    public class MultiLayerInferenceBenchmark
    {
        private const int InputSize = 784;

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

            _onnxSession = new InferenceSession("benchmark_mlp3.onnx");
            var tensor = new DenseTensor<float>(_inputData, [1, InputSize]);
            _onnxInputs = [NamedOnnxValue.CreateFromTensor("input", tensor)];

            _overfitModel = new Sequential(
                new LinearLayer(InputSize, 256),
                new ReluActivation(),
                new LinearLayer(256, 128),
                new ReluActivation(),
                new LinearLayer(128, 10));

            _overfitModel.Load("benchmark_mlp3.bin");
            _overfitModel.Eval();

            _overfitInputTensor = new FastTensor<float>(false, 1, InputSize);
            _inputData.AsSpan().CopyTo(_overfitInputTensor.AsSpan());
            _inputNode = new AutogradNode(_overfitInputTensor, requiresGrad: false);

            for (var i = 0; i < 200; i++)
            {
                _overfitModel.Forward(null, _inputNode);
            }
        }

        [Benchmark(Baseline = true)]
        public float OnnxRuntime_3Layer()
        {
            using var results = _onnxSession.Run(_onnxInputs);
            return results.First().AsTensor<float>()[0];
        }

        [Benchmark]
        public float Overfit_3Layer_ZeroAlloc()
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