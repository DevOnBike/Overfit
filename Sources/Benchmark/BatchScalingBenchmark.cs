// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnosers;
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
    // [HardwareCounters(HardwareCounter.InstructionRetired, HardwareCounter.CacheMisses)]
    public class BatchScalingBenchmark
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;

        // Testujemy małe paczki (Edge/IoT) i duże paczki (Server/Cloud)
        [Params(1, 16, 64, 256)]
        public int BatchSize { get; set; }

        private InferenceSession _onnxSession;
        private NamedOnnxValue[] _onnxInputs;
        private Sequential _overfitModel;
        private FastTensor<float> _overfitInputTensor;
        private AutogradNode _inputNode;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);
            var inputData = Enumerable.Range(0, BatchSize * InputSize).Select(_ => (float)rnd.NextDouble()).ToArray();

            // ONNX Setup
            _onnxSession = new InferenceSession("benchmark_model.onnx");
            var tensor = new DenseTensor<float>(inputData, [BatchSize, InputSize]);
            _onnxInputs = [NamedOnnxValue.CreateFromTensor("input", tensor)];

            // Overfit Setup
            _overfitModel = new Sequential(new LinearLayer(InputSize, OutputSize));
            _overfitModel.Load("benchmark_model.bin");
            _overfitModel.Eval();

            _overfitInputTensor = new FastTensor<float>(BatchSize, InputSize, clearMemory: false);
            inputData.AsSpan().CopyTo(_overfitInputTensor.GetView().AsSpan());
            _inputNode = new AutogradNode(_overfitInputTensor, false);
        }

        [Benchmark(Baseline = true)]
        public float OnnxRuntime_Batch()
        {
            using var results = _onnxSession.Run(_onnxInputs);
            return results.First().AsTensor<float>()[0];
        }

        [Benchmark]
        public float Overfit_Batch()
        {
            return _overfitModel.Forward(null, _inputNode).DataView.AsReadOnlySpan()[0];
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