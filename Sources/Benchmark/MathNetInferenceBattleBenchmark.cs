// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using MathNet.Numerics.LinearAlgebra;

namespace Benchmarks
{
    /// <summary>
    ///     Performance comparison between ONNX Runtime and Overfit engine.
    ///     Evaluates execution speed and memory allocation overhead during inference.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class MathNetInferenceBattleBenchmark
    {
        private const int SwarmSize = 100_000;
        private const int InputSize = 4;
        private const int OutputSize = 2;
        private const int GenomeSize = 10; // 8 wag + 2 biasy

        // --- DANE OVERFIT (Zero-Alloc, surowe tablice) ---
        private float[] _overfitInputs;
        private float[] _overfitOutputs;
        private float[] _overfitBrain;

        // --- DANE MATHNET ---
        private Matrix<float> _mathNetInputs;
        private Matrix<float> _mathNetWeights;
        private Vector<float> _mathNetBiases;

        [GlobalSetup]
        public void Setup()
        {
            var rng = new Random(42);

            // 1. Setup dla Overfit
            _overfitInputs = new float[SwarmSize * InputSize];
            _overfitOutputs = new float[SwarmSize * OutputSize];
            _overfitBrain = new float[GenomeSize];

            for (var i = 0; i < _overfitInputs.Length; i++)
            {
                _overfitInputs[i] = (float)rng.NextDouble();
            }
            for (var i = 0; i < GenomeSize; i++)
            {
                _overfitBrain[i] = (float)rng.NextDouble();
            }

            // 2. Setup dla MathNet
            // Tworzymy macierz 100_000 x 4 dla wejść
            _mathNetInputs = Matrix<float>.Build.DenseOfColumnMajor(SwarmSize, InputSize, _overfitInputs);

            // Wagi to macierz 4 x 2
            float[] weights = { _overfitBrain[0], _overfitBrain[4], _overfitBrain[1], _overfitBrain[5],
                                _overfitBrain[2], _overfitBrain[6], _overfitBrain[3], _overfitBrain[7] };
            _mathNetWeights = Matrix<float>.Build.DenseOfColumnMajor(InputSize, OutputSize, weights);

            // Biasy
            _mathNetBiases = Vector<float>.Build.Dense(new[] { _overfitBrain[8], _overfitBrain[9] });
        }

        // ==============================================================
        // TEST 1: Podejście MathNet.Numerics (Standard inżynieryjny w C#)
        // ==============================================================
        [Benchmark]
        public void Infer_MathNet()
        {
            // Mnożenie macierzy: (100_000 x 4) * (4 x 2) = Macierz (100_000 x 2)
            // UWAGA: To tutaj MathNet alokuje nową macierz w pamięci dla wyniku!
            var result = _mathNetInputs * _mathNetWeights;

            // Dodanie biasu i aktywacja Tanh (In-place, żeby chociaż trochę pomóc MathNetowi)
            result.MapIndexedInplace((row, col, val) => MathF.Tanh(val + _mathNetBiases[col]));
        }

        // ==============================================================
        // TEST 2: Podejście Overfit (Twoje brutalnie szybkie Span/SIMD)
        // ==============================================================
        [Benchmark(Baseline = true)]
        public void Infer_Overfit_ZeroAlloc()
        {
            ReadOnlySpan<float> inputs = _overfitInputs;
            ReadOnlySpan<float> population = _overfitBrain;
            Span<float> outputs = _overfitOutputs;

            for (var i = 0; i < SwarmSize; i++)
            {
                var iIdx = i * InputSize;
                var oIdx = i * OutputSize;

                var x0 = inputs[iIdx + 0];
                var x1 = inputs[iIdx + 1];
                var x2 = inputs[iIdx + 2];
                var x3 = inputs[iIdx + 3];

                var y0 = x0 * population[0] + x1 * population[1] + x2 * population[2] + x3 * population[3] + population[8];
                var y1 = x0 * population[4] + x1 * population[5] + x2 * population[6] + x3 * population[7] + population[9];

                outputs[oIdx + 0] = MathF.Tanh(y0);
                outputs[oIdx + 1] = MathF.Tanh(y1);
            }
        }
    }
}