// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Inference;
using DevOnBike.Overfit.Onnx;

namespace Benchmarks
{
    /// <summary>
    /// Benchmark: DAG inference via <see cref="OnnxGraphImporter"/> vs Sequential.
    ///
    /// Model: TinyResNet — 2 Linear layers with a skip connection.
    ///   fc1 = Linear(8, 8), fc2 = Linear(8, 4)
    ///   forward(x) = fc2(relu(fc1(x)) + x)
    ///
    /// Scenarios:
    ///   1. OnnxGraph_RunInference — OnnxGraphModel.RunInference directly.
    ///   2. OnnxGraph_ViaBackend  — via InferenceEngine.FromBackend.
    ///   3. Sequential_Imported   — same weights via OnnxImporter (linear path only).
    ///      Note: TinyResNet has a skip connection, so this path would fail at import;
    ///      included as reference baseline with a simple linear model instead.
    ///
    /// Purpose: verify that DAG overhead (slot-based buffer routing) is negligible
    /// vs Sequential for the same compute.
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark --filter "*OnnxGraph*"
    ///
    /// Fixtures required:
    ///   Sources/Benchmark/Helpers/tiny_resnet.onnx  (copy from Tests/test_fixtures/)
    ///   Sources/Benchmark/Helpers/benchmark_model.onnx  (existing linear model)
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    [MemoryDiagnoser]
    public class OnnxGraphBenchmark
    {
        private const int InputSize  = 8;
        private const int OutputSize = 4;
        private const string FixtureDir = "Helpers";

        private OnnxGraphModel      _dagModel     = null!;
        private InferenceEngine     _dagEngine    = null!;
        private float[]             _input        = null!;
        private float[]             _output       = null!;

        [GlobalSetup]
        public void Setup()
        {
            _input  = new float[InputSize];
            _output = new float[OutputSize];

            var rng = new Random(42);
            for (var i = 0; i < InputSize; i++)
            {
                _input[i] = (float)rng.NextDouble();
            }

            var onnxPath = Path.Combine(FixtureDir, "tiny_resnet.onnx");

            if (!File.Exists(onnxPath))
            {
                throw new FileNotFoundException(
                    $"Benchmark fixture missing: {onnxPath}. " +
                    "Copy Tests/test_fixtures/tiny_resnet.onnx to Sources/Benchmark/Helpers/.");
            }

            _dagModel = OnnxGraphImporter.Load(onnxPath, InputSize, OutputSize);
            _dagModel.Eval();

            var backend = new OnnxGraphInferenceBackend(_dagModel);
            _dagEngine = InferenceEngine.FromBackend(backend,
                new DevOnBike.Overfit.Inference.Contracts.InferenceEngineOptions
                {
                    WarmupIterations = 256,
                });
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _dagEngine.Dispose();
            // _dagModel is disposed by _dagEngine via backend
        }

        /// <summary>
        /// Direct DAG model inference — no InferenceEngine wrapper overhead.
        /// This is the raw cost of the slot-based buffer routing.
        /// </summary>
        [Benchmark(Baseline = true)]
        public void OnnxGraph_Direct()
        {
            _dagModel.RunInference(_input, _output);
        }

        /// <summary>
        /// DAG model via InferenceEngine.FromBackend.
        /// Adds one virtual dispatch + span bounds check vs Direct.
        /// </summary>
        [Benchmark]
        public void OnnxGraph_ViaEngine()
        {
            _dagEngine.Run(_input, _output);
        }
    }
}
