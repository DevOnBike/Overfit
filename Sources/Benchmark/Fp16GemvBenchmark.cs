// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;

namespace Benchmarks
{
    /// <summary>
    /// FP16-resident spike: is keeping weights as <see cref="Half"/> (½ the RAM of F32) competitive in SPEED
    /// for a decode GEMV? Compares a representative projection GEMV (2048×2048) three ways — F32 weights
    /// (today's <c>quantize:false</c> path), Half weights widened per-element (scalar), and Half weights widened
    /// in bulk via <c>TensorPrimitives.ConvertToSingle</c> (F16C on x86). The RAM halving is a given; this
    /// answers whether the widen cost is affordable. NOTE: desktop x86/F16C is a PROXY — the real target is
    /// mobile ARM (FP16 SIMD under Mono/CoreCLR), which must be measured separately before any engine build.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    [MemoryDiagnoser]
    public class Fp16GemvBenchmark
    {
        private const int Rows = 2048;
        private const int Cols = 2048;

        private float[] _wF32 = null!;
        private Half[] _wF16 = null!;
        private float[] _x = null!;
        private float[] _out = null!;
        private float[] _scratch = null!;

        [GlobalSetup]
        public void Setup()
        {
            var rng = new Random(1);
            _wF32 = new float[Rows * Cols];
            _wF16 = new Half[Rows * Cols];
            for (var i = 0; i < _wF32.Length; i++)
            {
                var v = (float)(rng.NextDouble() - 0.5);
                _wF32[i] = v;
                _wF16[i] = (Half)v;
            }

            _x = new float[Cols];
            for (var i = 0; i < Cols; i++)
            {
                _x[i] = (float)(rng.NextDouble() - 0.5);
            }

            _out = new float[Rows];
            _scratch = new float[Cols];
        }

        [Benchmark(Baseline = true)]
        public void F32()
        {
            for (var r = 0; r < Rows; r++)
            {
                _out[r] = TensorPrimitives.Dot(_wF32.AsSpan(r * Cols, Cols), _x);
            }
        }

        [Benchmark]
        public void F16_ScalarWiden()
        {
            for (var r = 0; r < Rows; r++)
            {
                var row = _wF16.AsSpan(r * Cols, Cols);
                var sum = 0f;
                for (var c = 0; c < Cols; c++)
                {
                    sum += (float)row[c] * _x[c];
                }
                _out[r] = sum;
            }
        }

        [Benchmark]
        public void F16_BulkWiden()
        {
            for (var r = 0; r < Rows; r++)
            {
                var row = _wF16.AsSpan(r * Cols, Cols);
                TensorPrimitives.ConvertToSingle(row, _scratch);
                _out[r] = TensorPrimitives.Dot(_scratch, _x);
            }
        }
    }
}
