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
    ///     Measures "Time-to-First-Prediction" (Cold Start) latency.
    ///     Compares the initialization and execution overhead of ONNX Runtime vs. the Overfit engine.
    /// </summary>
    /// <remarks>
    ///     <list type="bullet">
    ///         <item>
    ///             <description>
    ///                 <b>ONNX:</b> Significant overhead due to loading <c>onnxruntime.dll</c> (~30MB), parsing the
    ///                 Protobuf graph, and initial workspace buffer allocations.
    ///             </description>
    ///         </item>
    ///         <item>
    ///             <description>
    ///                 <b>Overfit:</b> Lightweight start via direct binary reading into <see cref="FastTensor{}" />
    ///                 and efficient weight pre-transposition during <c>Eval()</c>.
    ///             </description>
    ///         </item>
    ///     </list>
    ///     This benchmark is critical for Serverless environments (Azure Functions), short-lived containers, and CLI tools
    ///     where
    ///     the startup cost often dominates the total execution time.
    /// </remarks>
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

        /// <summary>
        ///     Benchmark for ONNX Runtime cold start, including session creation and the first inference pass.
        /// </summary>
        [Benchmark(Baseline = true)]
        public float OnnxRuntime_ColdStart()
        {
            using var session = new InferenceSession("benchmark_model.onnx");
            var tensor = new DenseTensor<float>(_inputData, [1, InputSize]);
            var inputs = new[]
            {
                NamedOnnxValue.CreateFromTensor("input", tensor)
            };

            using var results = session.Run(inputs);
            return results.First().AsTensor<float>()[0];
        }

        /// <summary>
        ///     Benchmark for Overfit cold start, including model loading, weight transposition, and the first inference pass.
        /// </summary>
        [Benchmark]
        public float Overfit_ColdStart()
        {
            var model = new Sequential(new LinearLayer(InputSize, OutputSize));
            model.Load("benchmark_model.bin");
            model.Eval();

            // POPRAWKA: Konstruktor i GetView()
            using var inputTensor = new FastTensor<float>(1, InputSize, clearMemory: false);
            _inputData.AsSpan().CopyTo(inputTensor.GetView().AsSpan());

            using var inputNode = new AutogradNode(inputTensor, false);
            // POPRAWKA: DataView.AsReadOnlySpan()
            var result = model.Forward(null, inputNode).DataView.AsReadOnlySpan()[0];

            model.Dispose();
            return result;
        }
    }
}