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
    /// Czas od zera do pierwszej predykcji.
    /// ONNX: ładowanie onnxruntime.dll (~30MB) + parsowanie protobuf grafu + alokacja buforów.
    /// Overfit: new Sequential + BinaryReader na float[] + Eval (transpozycja wag).
    /// Krytyczne dla: serverless (Azure Functions), kontenerów z krótkim lifecycle, CLI tools.
    /// </summary>
    [SimpleJob(RuntimeMoniker.Net10_0, iterationCount: 10, warmupCount: 0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    public class ColdStartBenchmark
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;

        private float[] _inputData;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);
            _inputData = Enumerable.Range(0, InputSize).Select(_ => (float)rnd.NextDouble()).ToArray();
        }

        [Benchmark(Baseline = true)]
        public float OnnxRuntime_ColdStart()
        {
            using var session = new InferenceSession("benchmark_model.onnx");
            var tensor = new DenseTensor<float>(_inputData, [1, InputSize]);
            var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor("input", tensor) };
            using var results = session.Run(inputs);
            return results.First().AsTensor<float>()[0];
        }

        [Benchmark]
        public float Overfit_ColdStart()
        {
            var model = new Sequential(new LinearLayer(InputSize, OutputSize));
            model.Load("benchmark_model.bin");
            model.Eval();

            using var inputTensor = new FastTensor<float>(false, 1, InputSize);
            _inputData.AsSpan().CopyTo(inputTensor.AsSpan());
            using var inputNode = new AutogradNode(inputTensor, requiresGrad: false);

            var result = model.Forward(null, inputNode).Data.AsSpan()[0];
            model.Dispose();
            return result;
        }
    }
}