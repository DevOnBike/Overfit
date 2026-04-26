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
    ///     Performance benchmark for a 3-layer MLP architecture: 784 → 256 → 128 → 10.
    ///     Evaluates the cumulative impact of layer depth on inference latency.
    /// </summary>
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    public class MultiLayerInferenceBenchmark
    {
        private const int InputSize = 784;
        private const string OnnxPath = "benchmark_mlp3.onnx";
        private const string BinPath = "benchmark_mlp3.bin";

        private float[] _inputData;
        private AutogradNode _inputNode;
        private NamedOnnxValue[] _onnxInputs;

        private InferenceSession _onnxSession;
        private Sequential _overfitModel;

        // Zmiana na TensorStorage
        private TensorStorage<float> _overfitInputTensor;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);
            _inputData = Enumerable.Range(0, InputSize).Select(_ => (float)rnd.NextDouble()).ToArray();

            // ONNX Setup
            if (File.Exists(OnnxPath))
            {
                _onnxSession = new InferenceSession(OnnxPath);
                var tensor = new DenseTensor<float>(_inputData, [1, InputSize]);
                _onnxInputs = [NamedOnnxValue.CreateFromTensor("input", tensor)];
            }

            // Overfit shape MUST exactly mirror what prepare-onnx.py exported, otherwise
            // we'd be benchmarking different graphs against each other. The .bin file
            // is a flat float dump from the same PyTorch model that produced the .onnx,
            // so loading it into a structurally-identical Sequential is what makes the
            // ONNX-vs-Overfit comparison apples-to-apples.
            _overfitModel = new Sequential(
                new LinearLayer(InputSize, 256),
                new ReluActivation(),
                new LinearLayer(256, 128),
                new ReluActivation(),
                new LinearLayer(128, 10));

            // Earlier versions auto-saved random weights when the .bin was missing.
            // That hid bugs (silent comparison against fresh-init weights instead of
            // PyTorch-exported ones) so we now fail loudly and tell the operator how to
            // regenerate the model files in one shot.
            if (!File.Exists(BinPath))
            {
                throw new InvalidOperationException(
                    $"Missing {BinPath}. Generate the model files by running:\n" +
                    "  pip install torch onnx onnxruntime\n" +
                    "  python prepare-onnx.py");
            }

            _overfitModel.Load(BinPath);
            _overfitModel.Eval();

            _overfitInputTensor = new TensorStorage<float>(InputSize, clearMemory: false);
            _inputData.AsSpan().CopyTo(_overfitInputTensor.AsSpan());
            _inputNode = new AutogradNode(_overfitInputTensor, new TensorShape(1, InputSize), false);

            for (var i = 0; i < 200; i++)
            {
                _overfitModel.Forward(null, _inputNode);
            }
        }

        /// <summary>
        ///     Benchmarks ONNX Runtime on a 3-layer MLP. Requires an external .onnx file.
        /// </summary>
        [Benchmark(Baseline = true)]
        public float OnnxRuntime_3Layer()
        {
            if (_onnxSession == null)
            {
                throw new InvalidOperationException(
                $"Missing {OnnxPath}. Generate the model files by running:\n" +
                "  pip install torch onnx onnxruntime\n" +
                "  python prepare-onnx.py");
            }

            using var results = _onnxSession.Run(_onnxInputs);
            return results.First().AsTensor<float>()[0];
        }

        /// <summary>
        ///     Benchmarks Overfit on a 3-layer MLP using the optimized zero-allocation path.
        /// </summary>
        [Benchmark]
        public float Overfit_3Layer_ZeroAlloc()
        {
            return _overfitModel.Forward(null, _inputNode).DataView.AsReadOnlySpan()[0];
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _onnxSession?.Dispose();
            _overfitModel?.Dispose();
            _overfitInputTensor?.Dispose();
            _inputNode?.Dispose();
        }
    }
}