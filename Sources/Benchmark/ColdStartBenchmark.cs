// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core; // Zmieniono namespace na Core
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Benchmarks
{
    /// <summary>
    ///     Measures "Time-to-First-Prediction" (Cold Start) latency.
    ///     Compares the initialization and execution overhead of ONNX Runtime, ML.NET,
    ///     and the Overfit engine — the metric that matters for serverless functions,
    ///     CLI tools, and any short-lived process.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
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
        ///     Benchmark for ML.NET cold start. Includes MLContext creation, the
        ///     OnnxScoringEstimator pipeline build, Fit on an empty IDataView, and
        ///     PredictionEngine creation — i.e. everything a real ML.NET app pays
        ///     once before its first prediction can run.
        /// </summary>
        [Benchmark]
        public float MLNet_ColdStart()
        {
            var ml = new MLContext(seed: 42);
            var emptyData = ml.Data.LoadFromEnumerable(new List<OnnxInput>());

            var pipeline = ml.Transforms.ApplyOnnxModel(
                modelFile: "benchmark_model.onnx",
                outputColumnNames: ["output"],
                inputColumnNames: ["input"]);

            var transformer = pipeline.Fit(emptyData);
            using var engine = ml.Model.CreatePredictionEngine<OnnxInput, OnnxOutput>(transformer);

            var output = engine.Predict(new OnnxInput { Input = _inputData });
            return output.Output[0];
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

            // POPRAWKA: Zmiana na TensorStorage + ucięto GetView()
            using var inputTensor = new TensorStorage<float>(InputSize, clearMemory: false);
            _inputData.AsSpan().CopyTo(inputTensor.AsSpan());

            using var inputNode = new AutogradNode(inputTensor, new TensorShape(1, InputSize), false);

            var result = model.Forward(null, inputNode).DataView.AsReadOnlySpan()[0];

            model.Dispose();
            return result;
        }

        // ML.NET schema classes — same shape as in MLNetSingleInferenceBenchmark
        // but kept here as a copy because BenchmarkDotNet's per-class compilation
        // doesn't share types across benchmark classes cleanly.
        public sealed class OnnxInput
        {
            [VectorType(InputSize)]
            [ColumnName("input")]
            public float[] Input { get; set; }
        }

        public sealed class OnnxOutput
        {
            [VectorType(OutputSize)]
            [ColumnName("output")]
            public float[] Output { get; set; }
        }
    }
}