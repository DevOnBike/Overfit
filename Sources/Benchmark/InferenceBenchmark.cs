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
using System;
using System.Linq;

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
        private InferenceSession _onnxSession;

        // --- Overfit Bebechy ---
        private Sequential _overfitModel;
        private FastTensor<float> _overfitInputTensor;
        private AutogradNode _inputNode; // PRE-ALOKOWANY WĘZEŁ

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);
            _inputData = Enumerable.Range(0, InputSize).Select(_ => (float)rnd.NextDouble()).ToArray();

            // Setup ONNX
            _onnxSession = new InferenceSession("benchmark_model.onnx");

            // Setup Overfit
            _overfitModel = new Sequential(new LinearLayer(InputSize, OutputSize));
            _overfitModel.Load("benchmark_model.bin");
            _overfitModel.Eval(); // Przełączamy w tryb inferencji!

            // KROK KRYTYCZNY: Alokujemy wszystko przed startem pomiaru
            _overfitInputTensor = new FastTensor<float>(false, 1, InputSize);
            _inputData.AsSpan().CopyTo(_overfitInputTensor.AsSpan());

            // Tworzymy węzeł raz. requiresGrad: false wyłącza alokację tensora gradientów.
            _inputNode = new AutogradNode(_overfitInputTensor, requiresGrad: false);
        }

        [Benchmark(Baseline = true)]
        public float[] OnnxRuntimeInference()
        {
            var tensor = new DenseTensor<float>(_inputData, new[] { 1, InputSize });
            var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor("input", tensor) };
            using var results = _onnxSession.Run(inputs);
            return results.First().AsTensor<float>().ToArray();
        }

        [Benchmark]
        public float OverfitPureCSharp()
        {
            // Pętla benchmarku woła TERAZ tylko to. 
            // Jeśli tu wciąż są alokacje, to Twój silnik robi "new" w środku Forward().
            var outputNode = _overfitModel.Forward(null, _inputNode);

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