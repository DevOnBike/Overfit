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
    ///     Performance benchmark for a 3-layer MLP architecture: 784 → 256 → 128 → 10.
    ///     Evaluates the cumulative impact of layer depth on inference latency.
    /// </summary>
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    public class MultiLayerInferenceBenchmark
    {
        private const int InputSize = 784;
        private const string OnnxPath = "benchmark_mlp3_auto.onnx";
        private const string BinPath = "benchmark_mlp3_auto.bin";

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

            _overfitModel = new Sequential(
            new LinearLayer(InputSize, 256),
            new ReluActivation(),
            new LinearLayer(256, 128),
            new ReluActivation(),
            new LinearLayer(128, 10));

            _overfitModel.Save(BinPath);
            _overfitModel.Eval();

            if (File.Exists(OnnxPath))
            {
                _onnxSession = new InferenceSession(OnnxPath);
                var tensor = new DenseTensor<float>(_inputData, [1, InputSize]);
                _onnxInputs = [NamedOnnxValue.CreateFromTensor("input", tensor)];
            }

            _overfitInputTensor = new FastTensor<float>(false, 1, InputSize);
            _inputData.AsSpan().CopyTo(_overfitInputTensor.AsSpan());
            _inputNode = new AutogradNode(_overfitInputTensor, false);

            for (var i = 0; i < 200; i++)
            {
                _overfitModel.Forward(null, _inputNode);
            }
        }

        /// <summary>
        ///     Benchmarks ONNX Runtime on a 3-layer MLP. Requires an external .onnx file.
        /// </summary>
        [Benchmark(Baseline = true)]
        public float OnnxRuntime_3Layer()
        {
            if (_onnxSession == null)
            {
                throw new InvalidOperationException(
                $"Missing {OnnxPath}. Please export the model from PyTorch using:\n" +
                "python export_mlp3.py --input-size 784 --hidden 256 128 --output 10");
            }

            using var results = _onnxSession.Run(_onnxInputs);
            return results.First().AsTensor<float>()[0];
        }

        /// <summary>
        ///     Benchmarks Overfit on a 3-layer MLP using the optimized zero-allocation path.
        /// </summary>
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