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
    /// The "real-sized model" counterpart to <see cref="OnnxRuntimeComparisonBenchmark"/> (a tiny MNIST CNN where
    /// ORT's per-call overhead dominates). A full ImageNet CNN (VGG-16 ≈ 15.5 GFLOPs/inference) is
    /// <b>compute-dominated</b>, so this actually pits Overfit's Conv/GEMM kernels against ONNX Runtime's native
    /// MLAS — the honest test of "did an ORT bump overtake us on real work". The graph has no skip connections but
    /// is imported through the DAG importer (<see cref="OnnxGraphImporter"/>) all the same.
    /// <para>
    /// Provide the model (random weights are fine — timing is weight-independent):
    /// <c>python Scripts/export_cnn_onnx.py --arch vgg16 --out C:\onnxmodels\cnn.onnx</c>, then
    /// <c>dotnet run -c Release --project Sources/Benchmark -- --filter "*LargeCnnComparison*"</c>.
    /// Override the path with the <c>OVERFIT_CNN_ONNX</c> environment variable. (ResNet-50 is not usable yet —
    /// its 3x3 stride-2 overlapping MaxPool is unsupported by the importer.)
    /// </para>
    /// </summary>
    [MemoryDiagnoser]
    public class LargeCnnComparisonBenchmark
    {
        private const int InputSize = 3 * 224 * 224; // 150,528
        private const int OutputSize = 1000;
        private const string DefaultModelPath = @"C:\onnxmodels\cnn.onnx";

        private InferenceEngine _overfit = null!;
        private InferenceSession _ort = null!;
        private float[] _input = null!;
        private float[] _output = null!;
        private List<NamedOnnxValue> _ortInputs = null!;

        [GlobalSetup]
        public void Setup()
        {
            var modelPath = Environment.GetEnvironmentVariable("OVERFIT_CNN_ONNX") ?? DefaultModelPath;
            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException(
                    $"CNN ONNX not found at '{modelPath}'. Export it first: " +
                    "python Scripts/export_cnn_onnx.py --arch vgg16 --out " + DefaultModelPath, modelPath);
            }

            _input = new float[InputSize];
            _output = new float[OutputSize];
            var rng = new Random(1234);
            for (var i = 0; i < InputSize; i++)
            {
                _input[i] = (float)rng.NextDouble();
            }

            // Overfit: DAG importer + zero-allocation Run on caller-owned buffers.
            var model = OnnxGraphImporter.Load(modelPath, InputSize, OutputSize);
            model.Eval();
            var backend = new OnnxGraphInferenceBackend(model);
            _overfit = InferenceEngine.FromBackend(backend);

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
