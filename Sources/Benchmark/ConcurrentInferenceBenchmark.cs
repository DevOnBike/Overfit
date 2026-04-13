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
    ///     Analyzes multi-threaded inference performance (8 threads × 1000 iterations).
    ///     Compares synchronization overhead and resource contention between ONNX Runtime and Overfit.
    /// </summary>
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    public class ConcurrentInferenceBenchmark
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;
        private const int ThreadCount = 8;
        private const int IterationsPerThread = 1000;

        private float[] _inputData;
        private AutogradNode[] _inputNodes;
        private FastTensor<float>[] _inputTensors;

        private InferenceSession _onnxSession;

        private Sequential[] _overfitModels;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);
            _inputData = Enumerable.Range(0, InputSize).Select(_ => (float)rnd.NextDouble()).ToArray();

            _onnxSession = new InferenceSession("benchmark_model.onnx");

            _overfitModels = new Sequential[ThreadCount];
            _inputTensors = new FastTensor<float>[ThreadCount];
            _inputNodes = new AutogradNode[ThreadCount];

            for (var t = 0; t < ThreadCount; t++)
            {
                _overfitModels[t] = new Sequential(new LinearLayer(InputSize, OutputSize));
                _overfitModels[t].Load("benchmark_model.bin");
                _overfitModels[t].Eval();

                _inputTensors[t] = new FastTensor<float>(1, InputSize, clearMemory: false);
                _inputData.AsSpan().CopyTo(_inputTensors[t].GetView().AsSpan());
                _inputNodes[t] = new AutogradNode(_inputTensors[t], false);
            }
        }

        /// <summary>
        ///     Measures ONNX Runtime performance under concurrent load.
        ///     Expects potential bottlenecks due to internal thread synchronization.
        /// </summary>
        [Benchmark(Baseline = true)]
        public float OnnxRuntime_Concurrent()
        {
            var results = new float[ThreadCount];

            Parallel.For(0, ThreadCount, body: t =>
            {
                var tensor = new DenseTensor<float>(_inputData, [1, InputSize]);
                var inputs = new[]
                {
                    NamedOnnxValue.CreateFromTensor("input", tensor)
                };
                var sum = 0f;

                for (var i = 0; i < IterationsPerThread; i++)
                {
                    using var output = _onnxSession.Run(inputs);
                    sum += output.First().AsTensor<float>()[0];
                }

                results[t] = sum;
            });

            return results.Sum();
        }

        /// <summary>
        ///     Measures Overfit performance with zero-contention concurrent execution.
        ///     Leverages independent memory buffers and pre-transposed weights for peak throughput.
        /// </summary>
        [Benchmark]
        public float Overfit_Concurrent_ZeroContention()
        {
            var results = new float[ThreadCount];

            Parallel.For(0, ThreadCount, body: t =>
            {
                var model = _overfitModels[t];
                var inputNode = _inputNodes[t];
                var sum = 0f;

                for (var i = 0; i < IterationsPerThread; i++)
                {
                    sum += model.Forward(null, inputNode).DataView.AsReadOnlySpan()[0];
                }

                results[t] = sum;
            });
            return results.Sum();
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _onnxSession?.Dispose();

            for (var t = 0; t < ThreadCount; t++)
            {
                _inputTensors[t]?.Dispose();
                _inputNodes[t]?.Dispose();
                _overfitModels[t]?.Dispose();
            }
        }
    }
}