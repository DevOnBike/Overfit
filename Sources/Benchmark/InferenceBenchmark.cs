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
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser] // Kluczowe dla śledzenia tych 2.63 KB
    [DisassemblyDiagnoser(maxDepth: 2)]
    public class InferenceBenchmark
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;

        private float[] _inputData;

        // --- ONNX Runtime ---
        private InferenceSession _onnxSession;

        // --- Overfit ---
        private Sequential _overfitModel;
        private FastTensor<float> _overfitInputTensor;
        private AutogradNode _inputNode; // Pre-alokowany węzeł wejściowy

        [GlobalSetup]
        public void Setup()
        {
            // 1. Generujemy dane wejściowe
            var rnd = new Random(42);
            _inputData = Enumerable.Range(0, InputSize).Select(_ => (float)rnd.NextDouble()).ToArray();

            // 2. Setup ONNX
            _onnxSession = new InferenceSession("benchmark_model.onnx");

            // 3. Setup Overfit (Twoje prawdziwe API)
            _overfitModel = new Sequential(
                new LinearLayer(InputSize, OutputSize)
            );

            // Ładujemy wagi z pliku binarnego
            _overfitModel.Load("benchmark_model.bin");

            // 4. PRZYGOTOWANIE ZERO-ALLOCATION
            // Alokujemy tensor i węzeł RAZ. W pętli benchmarku będziemy go tylko używać.
            _overfitInputTensor = new FastTensor<float>(false, 1, InputSize);
            _inputData.AsSpan().CopyTo(_overfitInputTensor.AsSpan());

            // Tworzymy węzeł wejściowy z requiresGrad: false, aby nie alokował tensora na gradienty
            _inputNode = new AutogradNode(_overfitInputTensor, requiresGrad: false);
        }

        [Benchmark(Baseline = true)]
        public float[] OnnxRuntimeInference()
        {
            // ONNX zawsze będzie tu alokował (DenseTensor + NamedOnnxValue + Results)
            var tensor = new DenseTensor<float>(_inputData, new[] { 1, InputSize });
            var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor("input", tensor) };

            using var results = _onnxSession.Run(inputs);

            return results.First().AsTensor<float>().ToArray();
        }

        [Benchmark]
        public float OverfitPureCSharp()
        {
            // PRAWIDŁOWY TEST:
            // Przekazujemy pre-alokowany _inputNode. 
            // Jeśli Twoja implementacja TensorMath.Linear (wywoływana przez Sequential) 
            // nie tworzy wewnątrz nowych obiektów 'new', zobaczysz tutaj 0 bajtów.
            var outputNode = _overfitModel.Forward(null, _inputNode);

            // Wyciągamy wynik prosto z Data.AsSpan()
            return outputNode.Data.AsSpan()[0];
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