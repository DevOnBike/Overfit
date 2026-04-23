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
using DevOnBike.Overfit.Tensors.Core; // Zmieniono na Tensors.Core
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
        private TensorStorage<float> _largeTensor;
        private AutogradNode _mediumNode;
        private TensorStorage<float> _mediumTensor;
        private AutogradNode _smallNode;
        private TensorStorage<float> _smallTensor;

        private InferenceSession _onnxLarge;
        private NamedOnnxValue[] _onnxLargeInputs;
        private InferenceSession _onnxMedium;
        private NamedOnnxValue[] _onnxMediumInputs;
        private InferenceSession _onnxSmall;
        private NamedOnnxValue[] _onnxSmallInputs;

        private Sequential _overfitLarge;
        private Sequential _overfitMedium;
        private Sequential _overfitSmall;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);

            var smallData = Enumerable.Range(0, 64).Select(_ => (float)rnd.NextDouble()).ToArray();
            var mediumData = Enumerable.Range(0, 784).Select(_ => (float)rnd.NextDouble()).ToArray();
            var largeData = Enumerable.Range(0, 4096).Select(_ => (float)rnd.NextDouble()).ToArray();

            // ONNX Setup
            _onnxSmall = new InferenceSession("benchmark_small.onnx");
            _onnxMedium = new InferenceSession("benchmark_medium.onnx");
            _onnxLarge = new InferenceSession("benchmark_large.onnx");

            _onnxSmallInputs = [NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(smallData, [1, 64]))];
            _onnxMediumInputs = [NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(mediumData, [1, 784]))];
            _onnxLargeInputs = [NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(largeData, [1, 4096]))];

            // Overfit Setup
            _overfitSmall = new Sequential(new LinearLayer(64, OutputSize));
            _overfitSmall.Load("benchmark_small.bin");
            _overfitSmall.Eval();

            _overfitMedium = new Sequential(new LinearLayer(784, OutputSize));
            _overfitMedium.Load("benchmark_medium.bin");
            _overfitMedium.Eval();

            _overfitLarge = new Sequential(new LinearLayer(4096, OutputSize));
            _overfitLarge.Load("benchmark_large.bin");
            _overfitLarge.Eval();

            // Zmiana na TensorStorage + TensorShape w AutogradNode
            _smallTensor = new TensorStorage<float>(64, clearMemory: false);
            smallData.AsSpan().CopyTo(_smallTensor.AsSpan());
            _smallNode = new AutogradNode(_smallTensor, new TensorShape(1, 64), false);

            _mediumTensor = new TensorStorage<float>(784, clearMemory: false);
            mediumData.AsSpan().CopyTo(_mediumTensor.AsSpan());
            _mediumNode = new AutogradNode(_mediumTensor, new TensorShape(1, 784), false);

            _largeTensor = new TensorStorage<float>(4096, clearMemory: false);
            largeData.AsSpan().CopyTo(_largeTensor.AsSpan());
            _largeNode = new AutogradNode(_largeTensor, new TensorShape(1, 4096), false);
        }

        [Benchmark]
        public float Onnx_64()
        {
            using var r = _onnxSmall.Run(_onnxSmallInputs);
            return r.First().AsTensor<float>()[0];
        }

        [Benchmark]
        public float Overfit_64()
        {
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
            _smallTensor?.Dispose();
            _smallNode?.Dispose();
            _mediumTensor?.Dispose();
            _mediumNode?.Dispose();
            _largeTensor?.Dispose();
            _largeNode?.Dispose();
        }
    }
}