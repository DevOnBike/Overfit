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
    /// <summary>
    ///     Evaluates how the performance advantage of the Overfit engine scales across different model sizes.
    ///     Analyzes the compute-to-overhead ratio compared to ONNX Runtime.
    /// </summary>
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    public class ScalingBenchmark
    {
        private const int OutputSize = 10;
        private AutogradNode _largeNode;
        private FastTensor<float> _largeTensor;
        private AutogradNode _mediumNode;
        private FastTensor<float> _mediumTensor;
        private InferenceSession _onnxLarge;
        private NamedOnnxValue[] _onnxLargeInputs;
        private InferenceSession _onnxMedium;
        private NamedOnnxValue[] _onnxMediumInputs;

        private InferenceSession _onnxSmall;

        private NamedOnnxValue[] _onnxSmallInputs;
        private Sequential _overfitLarge;
        private Sequential _overfitMedium;

        private Sequential _overfitSmall;

        private AutogradNode _smallNode;

        private FastTensor<float> _smallTensor;

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

            // Setup ONNX Runtime session
            onnxSession = new InferenceSession($"{modelName}.onnx");
            var tensor = new DenseTensor<float>(data, [1, inputSize]);
            onnxInputs = [NamedOnnxValue.CreateFromTensor("input", tensor)];

            // Setup Overfit model
            overfitModel = new Sequential(new LinearLayer(inputSize, OutputSize));
            overfitModel.Load($"{modelName}.bin");
            overfitModel.Eval();

            // POPRAWKA: Poprawny konstruktor (dim0, dim1, clearMemory)
            inputTensor = new FastTensor<float>(1, inputSize, clearMemory: false);
            // POPRAWKA: GetView().AsSpan() zamiast AsSpan()
            data.AsSpan().CopyTo(inputTensor.GetView().AsSpan());
            inputNode = new AutogradNode(inputTensor, false);

            for (var i = 0; i < 200; i++)
            {
                overfitModel.Forward(null, inputNode);
            }
        }

        [Benchmark]
        public float Onnx_128()
        {
            using var r = _onnxSmall.Run(_onnxSmallInputs);
            return r.First().AsTensor<float>()[0];
        }

        [Benchmark]
        public float Overfit_128()
        {
            // POPRAWKA: DataView.AsReadOnlySpan() zamiast Data.AsSpan()
            return _overfitSmall.Forward(null, _smallNode).DataView.AsReadOnlySpan()[0];
        }

        [Benchmark(Baseline = true)]
        public float Onnx_784()
        {
            using var r = _onnxMedium.Run(_onnxMediumInputs);
            return r.First().AsTensor<float>()[0];
        }

        [Benchmark]
        public float Overfit_784()
        {
            // POPRAWKA: DataView.AsReadOnlySpan()
            return _overfitMedium.Forward(null, _mediumNode).DataView.AsReadOnlySpan()[0];
        }

        [Benchmark]
        public float Onnx_4096()
        {
            using var r = _onnxLarge.Run(_onnxLargeInputs);
            return r.First().AsTensor<float>()[0];
        }

        [Benchmark]
        public float Overfit_4096()
        {
            // POPRAWKA: DataView.AsReadOnlySpan()
            return _overfitLarge.Forward(null, _largeNode).DataView.AsReadOnlySpan()[0];
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