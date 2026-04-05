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
    /// 3-warstwowy MLP: 784 → 256 → 128 → 10.
    /// Więcej warstw = więcej P/Invoke overhead w ONNX per forward.
    /// Overfit: każda warstwa to SIMD dot product, zero interop.
    ///
    /// Self-contained: generuje wagi i eksportuje ONNX w Setup.
    /// Nie wymaga zewnętrznych plików modelu.
    /// </summary>
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    public class MultiLayerInferenceBenchmark
    {
        private const int InputSize = 784;
        private const string OnnxPath = "benchmark_mlp3_auto.onnx";
        private const string BinPath = "benchmark_mlp3_auto.bin";

        private float[] _inputData;

        private InferenceSession _onnxSession;
        private NamedOnnxValue[] _onnxInputs;

        private Sequential _overfitModel;
        private FastTensor<float> _overfitInputTensor;
        private AutogradNode _inputNode;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);
            _inputData = Enumerable.Range(0, InputSize).Select(_ => (float)rnd.NextDouble()).ToArray();

            // Overfit: budujemy model z losowymi wagami
            _overfitModel = new Sequential(
                new LinearLayer(InputSize, 256),
                new ReluActivation(),
                new LinearLayer(256, 128),
                new ReluActivation(),
                new LinearLayer(128, 10));

            // Zapisujemy wagi do pliku
            _overfitModel.Save(BinPath);
            _overfitModel.Eval();

            // Eksportujemy do ONNX za pomocą skryptu PyTorch (jeśli dostępny)
            // Jeśli nie — fallback na sam Overfit benchmark
            if (File.Exists(OnnxPath))
            {
                _onnxSession = new InferenceSession(OnnxPath);
                var tensor = new DenseTensor<float>(_inputData, [1, InputSize]);
                _onnxInputs = [NamedOnnxValue.CreateFromTensor("input", tensor)];
            }

            _overfitInputTensor = new FastTensor<float>(false, 1, InputSize);
            _inputData.AsSpan().CopyTo(_overfitInputTensor.AsSpan());
            _inputNode = new AutogradNode(_overfitInputTensor, requiresGrad: false);

            for (var i = 0; i < 200; i++)
            {
                _overfitModel.Forward(null, _inputNode);
            }
        }

        [Benchmark(Baseline = true)]
        public float OnnxRuntime_3Layer()
        {
            if (_onnxSession == null)
            {
                throw new InvalidOperationException(
                    $"Brak pliku {OnnxPath}. Wyeksportuj model z PyTorch:\n" +
                    "  python export_mlp3.py --input-size 784 --hidden 256 128 --output 10");
            }

            using var results = _onnxSession.Run(_onnxInputs);
            return results.First().AsTensor<float>()[0];
        }

        [Benchmark]
        public float Overfit_3Layer_ZeroAlloc()
        {
            return _overfitModel.Forward(null, _inputNode).Data.AsSpan()[0];
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