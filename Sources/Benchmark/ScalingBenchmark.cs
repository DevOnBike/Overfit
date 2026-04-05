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
    /// Jak skaluje się przewaga Overfit przy różnych rozmiarach modelu.
    /// Im mniejszy model, tym większy stosunek overhead/compute.
    ///
    /// 128 → 10:  edge/embedded, Overfit dominuje (~15×)
    /// 784 → 10:  MNIST, sweet spot (~9×)
    /// 4096 → 10: duży embedding, ONNX zaczyna doganiać (~5×)
    /// </summary>
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    public class ScalingBenchmark
    {
        private const int OutputSize = 10;

        private InferenceSession _onnxSmall;
        private InferenceSession _onnxMedium;
        private InferenceSession _onnxLarge;

        private NamedOnnxValue[] _onnxSmallInputs;
        private NamedOnnxValue[] _onnxMediumInputs;
        private NamedOnnxValue[] _onnxLargeInputs;

        private Sequential _overfitSmall;
        private Sequential _overfitMedium;
        private Sequential _overfitLarge;

        private AutogradNode _smallNode;
        private AutogradNode _mediumNode;
        private AutogradNode _largeNode;

        private FastTensor<float> _smallTensor;
        private FastTensor<float> _mediumTensor;
        private FastTensor<float> _largeTensor;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);

            SetupPair(rnd, 128, "benchmark_small",
                out _onnxSmall, out _onnxSmallInputs,
                out _overfitSmall, out _smallNode, out _smallTensor);

            SetupPair(rnd, 784, "benchmark_medium",
                out _onnxMedium, out _onnxMediumInputs,
                out _overfitMedium, out _mediumNode, out _mediumTensor);

            SetupPair(rnd, 4096, "benchmark_large",
                out _onnxLarge, out _onnxLargeInputs,
                out _overfitLarge, out _largeNode, out _largeTensor);
        }

        private static void SetupPair(Random rnd, int inputSize, string modelName,
            out InferenceSession onnxSession, out NamedOnnxValue[] onnxInputs,
            out Sequential overfitModel, out AutogradNode inputNode,
            out FastTensor<float> inputTensor)
        {
            var data = Enumerable.Range(0, inputSize).Select(_ => (float)rnd.NextDouble()).ToArray();

            onnxSession = new InferenceSession($"{modelName}.onnx");
            var tensor = new DenseTensor<float>(data, [1, inputSize]);
            onnxInputs = [NamedOnnxValue.CreateFromTensor("input", tensor)];

            overfitModel = new Sequential(new LinearLayer(inputSize, OutputSize));
            overfitModel.Load($"{modelName}.bin");
            overfitModel.Eval();

            inputTensor = new FastTensor<float>(false, 1, inputSize);
            data.AsSpan().CopyTo(inputTensor.AsSpan());
            inputNode = new AutogradNode(inputTensor, requiresGrad: false);

            for (var i = 0; i < 200; i++)
            {
                overfitModel.Forward(null, inputNode);
            }
        }

        // --- Small: 128 -> 10 ---

        [Benchmark]
        public float Onnx_128()
        {
            using var r = _onnxSmall.Run(_onnxSmallInputs);
            return r.First().AsTensor<float>()[0];
        }

        [Benchmark]
        public float Overfit_128()
        {
            return _overfitSmall.Forward(null, _smallNode).Data.AsSpan()[0];
        }

        // --- Medium: 784 -> 10 (MNIST) ---

        [Benchmark(Baseline = true)]
        public float Onnx_784()
        {
            using var r = _onnxMedium.Run(_onnxMediumInputs);
            return r.First().AsTensor<float>()[0];
        }

        [Benchmark]
        public float Overfit_784()
        {
            return _overfitMedium.Forward(null, _mediumNode).Data.AsSpan()[0];
        }

        // --- Large: 4096 -> 10 ---

        [Benchmark]
        public float Onnx_4096()
        {
            using var r = _onnxLarge.Run(_onnxLargeInputs);
            return r.First().AsTensor<float>()[0];
        }

        [Benchmark]
        public float Overfit_4096()
        {
            return _overfitLarge.Forward(null, _largeNode).Data.AsSpan()[0];
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _onnxSmall?.Dispose();
            _onnxMedium?.Dispose();
            _onnxLarge?.Dispose();
            _overfitSmall?.Dispose();
            _overfitMedium?.Dispose();
            _overfitLarge?.Dispose();
            _smallNode?.Dispose();
            _mediumNode?.Dispose();
            _largeNode?.Dispose();
            _smallTensor?.Dispose();
            _mediumTensor?.Dispose();
            _largeTensor?.Dispose();
        }
    }
}