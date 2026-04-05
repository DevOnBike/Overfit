// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;
using DevOnBike.Overfit.Core;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Benchmarks
{
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    [DisassemblyDiagnoser(maxDepth: 2)]
    public class InferenceBenchmark
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;

        private float[] _inputData;

        // --- ONNX Runtime ---
        private InferenceSession _onnxSession;

        // --- Overfit (Czysty C#) ---
        private FastTensor<float> _overfitInputTensor;
        private FastTensor<float> _weights;
        private FastTensor<float> _bias;
        private FastTensor<float> _overfitOutputTensor; // Pre-alokowany bufor wyjściowy

        [GlobalSetup]
        public void Setup()
        {
            // 1. Generujemy fikcyjne dane wejściowe
            var rnd = new Random(42);
            _inputData = Enumerable.Range(0, InputSize).Select(_ => (float)rnd.NextDouble()).ToArray();

            // 2. Setup ONNX (Wymaga pliku benchmark_model.onnx w folderze wyjściowym)
            _onnxSession = new InferenceSession("benchmark_model.onnx");

            // 3. Setup Overfit (Pre-alokacja całej pamięci na start - Zero GC później)
            _overfitInputTensor = new FastTensor<float>(false, 1, InputSize);
            _inputData.AsSpan().CopyTo(_overfitInputTensor.AsSpan());

            _weights = new FastTensor<float>(false, InputSize, OutputSize);
            _bias = new FastTensor<float>(false, 1, OutputSize);
            _overfitOutputTensor = new FastTensor<float>(false, 1, OutputSize);

            // Wypełniamy wagi testowymi wartościami
            _weights.AsSpan().Fill(0.1f);
            _bias.AsSpan().Fill(0.01f);
        }

        [Benchmark(Baseline = true)]
        public float[] OnnxRuntimeInference()
        {
            // ALOKACJA 1: DenseTensor
            var tensor = new DenseTensor<float>(_inputData, new[] { 1, InputSize });

            // ALOKACJA 2: Tablica wejść dla API
            var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor("input", tensor) };

            // Bariera P/Invoke (Przejście C# -> C++)
            using var results = _onnxSession.Run(inputs);

            // ALOKACJA 3: Zrzut z pamięci C++ na nową tablicę zarządzaną
            return results.First().AsTensor<float>().ToArray();
        }

        [Benchmark]
        public float OverfitPureCSharp()
        {
            // ZERO-ALLOCATION
            // Przekazujemy pre-alokowany tensor wyjściowy. Brak jakichkolwiek "new" w ciele funkcji.
            LinearForwardFast(_overfitInputTensor, _weights, _bias, _overfitOutputTensor);

            // Zwracamy pierwszą wartość jako dowód (wyciągnięcie ze Span to typ wartościowy, brak GC)
            return _overfitOutputTensor.AsSpan()[0];
        }

        /// <summary>
        /// Wysoce zoptymalizowana, płaska pętla dla warstwy Linear.
        /// W rzeczywistości to odpowiednik Twojej metody z TensorMath.cs.
        /// </summary>
        private void LinearForwardFast(
            FastTensor<float> input,
            FastTensor<float> weights,
            FastTensor<float> bias,
            FastTensor<float> output)
        {
            var inSpan = input.AsReadOnlySpan();
            var wSpan = weights.AsReadOnlySpan();
            var bSpan = bias.AsReadOnlySpan();
            var outSpan = output.AsSpan();

            // Pętla po neuronach wyjściowych (10)
            for (int outIdx = 0; outIdx < OutputSize; outIdx++)
            {
                float sum = bSpan[outIdx];

                // Pętla po wejściach (784) - idealne miejsce na przyszłą wektoryzację Vector<float>
                for (int inIdx = 0; inIdx < InputSize; inIdx++)
                {
                    // weights w pamięci to płaska tablica: [InputSize * OutputSize]
                    sum += inSpan[inIdx] * wSpan[inIdx * OutputSize + outIdx];
                }

                outSpan[outIdx] = sum;
            }
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _onnxSession?.Dispose();
            _overfitInputTensor?.Dispose();
            _weights?.Dispose();
            _bias?.Dispose();
            _overfitOutputTensor?.Dispose();
        }
    }
}