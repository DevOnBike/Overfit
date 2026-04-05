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
    [MemoryDiagnoser]
    [DisassemblyDiagnoser(maxDepth: 2)] // Opcjonalnie: pokaże wygenerowany kod asemblera
    public class InferenceBenchmark
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;

        private float[] _inputData;

        // --- ONNX Runtime ---
        private InferenceSession _onnxSession;

        // --- Overfit (Prawdziwy silnik) ---
        private Sequential _overfitModel;
        private FastTensor<float> _overfitInputTensor;

        [GlobalSetup]
        public void Setup()
        {
            // 1. Generujemy fikcyjne dane wejściowe dla równego startu
            var rnd = new Random(42);
            _inputData = Enumerable.Range(0, InputSize).Select(_ => (float)rnd.NextDouble()).ToArray();

            // 2. Setup ONNX (Wymaga plików .onnx i .onnx.data w folderze wyjściowym)
            _onnxSession = new InferenceSession("benchmark_model.onnx");

            // 3. Setup Overfit (Twoje prawdziwe API)
            _overfitModel = new Sequential(
                new LinearLayer(InputSize, OutputSize)
            );

            // Ładujemy prawdziwe wagi wyciągnięte z ONNX (wymaga metody Load() z użyciem BinaryReader)
            _overfitModel.Load("benchmark_model.bin");

            // 4. Pre-alokacja tensora wejściowego dla Zero-Allocation
            _overfitInputTensor = new FastTensor<float>(false, 1, InputSize);
            _inputData.AsSpan().CopyTo(_overfitInputTensor.AsSpan());
        }

        [Benchmark(Baseline = true)]
        public float[] OnnxRuntimeInference()
        {
            // ALOKACJA 1: DenseTensor alokuje na stercie
            var tensor = new DenseTensor<float>(_inputData, new[] { 1, InputSize });

            // ALOKACJA 2: Tablica wejść dla natywnego API
            var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor("input", tensor) };

            // Bariera P/Invoke (Przejście C# -> C++)
            using var results = _onnxSession.Run(inputs);

            // ALOKACJA 3: Zrzut z pamięci C++ na nową tablicę C#
            return results.First().AsTensor<float>().ToArray();
        }

        [Benchmark]
        public float OverfitPureCSharp()
        {
            // PRAWDZIWY TEST: Uderzamy prosto w API Twojej biblioteki.

            // requiresGrad: false zapobiega niepotrzebnej alokacji tensora 'Grad' 
            // wewnątrz konstruktora AutogradNode podczas inferencji!
            var inputNode = new AutogradNode(_overfitInputTensor, requiresGrad: false);

            // Forward Pass: Przekazujemy null jako graf, wyłączając historię do AutoDiff
            var outputNode = _overfitModel.Forward(null, inputNode);

            // Zwracamy pierwszą wartość bezpośrednio z właściwości 'Data',
            // wyciągając ją ze struktury Span (omijamy całkowicie Garbage Collectora)
            return outputNode.Data.AsSpan()[0];
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _onnxSession?.Dispose();
            _overfitInputTensor?.Dispose();
            _overfitModel?.Dispose();
        }
    }
}