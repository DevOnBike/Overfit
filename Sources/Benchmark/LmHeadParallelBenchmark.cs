// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace Benchmarks
{
    /// <summary>
    /// Compares sequential vs parallel LM head projection.
    ///
    /// LM head: hidden[768] × W[768 × vocabSize] → logits[vocabSize]
    ///
    /// GPT-2 Small: vocabSize = 50257, dModel = 768.
    /// That's 38.6 M multiply-add ops per token — the single largest
    /// per-token cost (~27% of total 15 ms budget).
    ///
    /// ProjectParallel splits the output dimension across all CPU cores
    /// via Parallel.For. Each thread works on a disjoint output slice
    /// with no sharing — embarrassingly parallel.
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark --filter "*LmHeadParallel*"
    ///
    /// Expected on AMD Ryzen 9 9950X3D (32 cores):
    ///   Sequential: ~3.5–4.0 ms
    ///   Parallel:   ~0.2–0.4 ms  →  ~10× speedup
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class LmHeadParallelBenchmark
    {
        private const int DModel    = 768;
        private const int VocabSize = 50_257;   // GPT-2 vocab

        private float[] _hidden  = null!;
        private float[] _weights = null!;
        private float[] _logits  = null!;

        [GlobalSetup]
        public void Setup()
        {
            var rng = new Random(42);

            _hidden  = new float[DModel];
            _weights = new float[DModel * VocabSize];
            _logits  = new float[VocabSize];

            for (var i = 0; i < _hidden.Length;  i++)
            {
                _hidden[i]  = (float)(rng.NextDouble() * 2 - 1);
            }
            for (var i = 0; i < _weights.Length; i++)
            {
                _weights[i] = (float)(rng.NextDouble() * 0.02);
            }

            // Warmup
            Project_Sequential();
            Project_Parallel();
        }

        [Benchmark(Baseline = true)]
        public float Project_Sequential()
        {
            SingleTokenProjectionKernel.Project(
                _hidden,
                _weights,
                ReadOnlySpan<float>.Empty,
                _logits,
                DModel,
                VocabSize);

            return _logits[0];
        }

        [Benchmark]
        public float Project_Parallel()
        {
            SingleTokenProjectionKernel.ProjectParallel(
                _hidden,
                _weights,
                ReadOnlySpan<float>.Empty,
                _logits,
                DModel,
                VocabSize);

            return _logits[0];
        }
    }
}
