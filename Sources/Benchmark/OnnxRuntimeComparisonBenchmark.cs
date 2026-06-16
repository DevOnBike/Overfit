// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using DevOnBike.Overfit.Inference;
using DevOnBike.Overfit.Onnx;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Benchmarks
{
    /// <summary>
    /// Apples-to-apples CPU inference on the SAME <c>mnist_cnn.onnx</c>: Overfit's pure-managed,
    /// caller-owned-buffer, zero-allocation engine vs Microsoft's native ONNX Runtime (the C++/MLAS
    /// reference). Answers "did an ORT version bump overtake us, and on which axis" by reporting both
    /// latency (ns/op) and allocations/op — Overfit's identity is competitive-order latency with <b>zero
    /// native dependency and zero per-call allocation</b>, not beating MLAS on raw throughput. ORT is the
    /// native baseline here, exactly as llama.cpp is for the GGUF decode path.
    /// <para>
    /// Both run on the default CPU configuration out of the box (ORT picks its own intra-op threading);
    /// this is the "what you get without tuning" comparison. Run:
    /// <c>dotnet run -c Release --project Sources/Benchmark -- --filter "*OnnxRuntimeComparison*"</c>
    /// </para>
    /// </summary>
    [MemoryDiagnoser]
    public class OnnxRuntimeComparisonBenchmark
    {
        private const int InputSize = 1 * 28 * 28;
        private const int OutputSize = 10;

        private InferenceEngine _overfit = null!;
        private InferenceSession _ort = null!;
        private float[] _input = null!;
        private float[] _output = null!;
        private List<NamedOnnxValue> _ortInputs = null!;

        [GlobalSetup]
        public void Setup()
        {
            var modelPath = Path.Combine(AppContext.BaseDirectory, "mnist_cnn.onnx");

            _input = new float[InputSize];
            _output = new float[OutputSize];
            var rng = new Random(1234);
            for (var i = 0; i < InputSize; i++)
            {
                _input[i] = (float)rng.NextDouble();
            }

            // Overfit: pure-managed Sequential, caller-owned buffers, zero-allocation Run.
            var model = OnnxImporter.Load(modelPath);
            model.Eval();
            _overfit = InferenceEngine.FromSequential(model, InputSize, OutputSize);

            // ONNX Runtime: native CPU execution provider, idiomatic single-input Run.
            _ort = new InferenceSession(modelPath);
            var inputName = _ort.InputMetadata.Keys.First();
            var dims = _ort.InputMetadata[inputName].Dimensions.Select(d => d <= 0 ? 1 : d).ToArray();
            var tensor = new DenseTensor<float>(_input, dims);
            _ortInputs = [NamedOnnxValue.CreateFromTensor(inputName, tensor)];
        }

        [Benchmark(Baseline = true)]
        public float[] Overfit()
        {
            _overfit.Run(_input, _output);
            return _output;
        }

        [Benchmark]
        public float OnnxRuntime()
        {
            using var results = _ort.Run(_ortInputs);
            return results[0].AsTensor<float>().GetValue(0);
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _overfit?.Dispose();
            _ort?.Dispose();
        }
    }
}
