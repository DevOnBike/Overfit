// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Kernels;

namespace Benchmarks
{
    /// <summary>
    /// Benchmark: LinearKernels.BackwardInput i AccumulateWeightGrad
    /// przed i po unsafe Parallel.For.
    ///
    /// Reprezentatywne wymiary dla GPT-1 training batch=8, SeqLen=256:
    ///   B*T = 2048 (batch × seqLen)
    ///   FFN W1: [2048, 256] @ [256, 1024]
    ///   FFN W2: [2048, 1024] @ [1024, 256]
    ///   MHA Q:  [2048, 256] @ [256, 32]
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark --filter "*LinearBackward*"
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    [MemoryDiagnoser]
    public class LinearBackwardBenchmark
    {
        // FFN W1 backward: batchSize=2048, inputSize=256, outputSize=1024
        private float[] _gradOutput2048x1024 = null!;
        private float[] _weights256x1024 = null!;
        private float[] _gradInput2048x256 = null!;
        private float[] _input2048x256 = null!;
        private float[] _gradWeights256x1024 = null!;

        // MHA Q backward: batchSize=2048, inputSize=256, outputSize=32
        private float[] _gradOutput2048x32 = null!;
        private float[] _weights256x32 = null!;
        private float[] _gradInput2048x256b = null!;
        private float[] _gradWeights256x32 = null!;

        [GlobalSetup]
        public void Setup()
        {
            var rng = new Random(42);

            void Fill(float[] arr)
            {
                for (var i = 0; i < arr.Length; i++)
                {
                    arr[i] = (float)(rng.NextDouble() - 0.5);
                }
            }

            _gradOutput2048x1024 = new float[2048 * 1024];
            Fill(_gradOutput2048x1024);
            _weights256x1024 = new float[256 * 1024];
            Fill(_weights256x1024);
            _gradInput2048x256 = new float[2048 * 256];
            _input2048x256 = new float[2048 * 256];
            Fill(_input2048x256);
            _gradWeights256x1024 = new float[256 * 1024];

            _gradOutput2048x32 = new float[2048 * 32];
            Fill(_gradOutput2048x32);
            _weights256x32 = new float[256 * 32];
            Fill(_weights256x32);
            _gradInput2048x256b = new float[2048 * 256];
            _gradWeights256x32 = new float[256 * 32];
        }

        // ── FFN W1: [2048, 256] backward ────────────────────────────────────

        [Benchmark(Description = "BackwardInput FFN [2048,256,1024]")]
        public void BackwardInput_FFN_2048x256x1024()
        {
            Array.Clear(_gradInput2048x256, 0, _gradInput2048x256.Length);
            LinearKernels.BackwardInput(
                _gradOutput2048x1024,
                _weights256x1024,
                _gradInput2048x256,
                batchSize: 2048,
                inputSize: 256,
                outputSize: 1024);
        }

        [Benchmark(Description = "AccumulateWeightGrad FFN [2048,256,1024]")]
        public void AccumulateWeightGrad_FFN_2048x256x1024()
        {
            Array.Clear(_gradWeights256x1024, 0, _gradWeights256x1024.Length);
            LinearKernels.AccumulateWeightGrad(
                _input2048x256,
                _gradOutput2048x1024,
                _gradWeights256x1024,
                batchSize: 2048,
                inputSize: 256,
                outputSize: 1024);
        }

        // ── MHA Q: [2048, 32] backward ──────────────────────────────────────

        [Benchmark(Description = "BackwardInput MHA Q [2048,256,32]")]
        public void BackwardInput_MHA_2048x256x32()
        {
            Array.Clear(_gradInput2048x256b, 0, _gradInput2048x256b.Length);
            LinearKernels.BackwardInput(
                _gradOutput2048x32,
                _weights256x32,
                _gradInput2048x256b,
                batchSize: 2048,
                inputSize: 256,
                outputSize: 32);
        }

        [Benchmark(Description = "AccumulateWeightGrad MHA Q [2048,256,32]")]
        public void AccumulateWeightGrad_MHA_2048x256x32()
        {
            Array.Clear(_gradWeights256x32, 0, _gradWeights256x32.Length);
            LinearKernels.AccumulateWeightGrad(
                _input2048x256,
                _gradOutput2048x32,
                _gradWeights256x32,
                batchSize: 2048,
                inputSize: 256,
                outputSize: 32);
        }
    }
}
