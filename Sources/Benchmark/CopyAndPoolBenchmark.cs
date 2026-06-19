// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using BenchmarkDotNet.Attributes;
using DevOnBike.Overfit.Tensors;

namespace Benchmarks
{
    /// <summary>
    /// Answers two micro-questions used across the codebase: (a) is <c>Array.Copy</c>, span
    /// <c>CopyTo</c>, or <c>Buffer.BlockCopy</c> fastest for float[] copies, and (b) does the
    /// <see cref="PooledBuffer{T}"/> <c>using</c> wrapper cost anything vs. raw
    /// <c>ArrayPool.Rent</c>/<c>Return</c> with try/finally.
    /// </summary>
    [MemoryDiagnoser]
    public class CopyAndPoolBenchmark
    {
        [Params(256, 8192, 1_048_576)]
        public int Size;

        private float[] _src = [];
        private float[] _dst = [];

        [GlobalSetup]
        public void Setup()
        {
            _src = new float[Size];
            _dst = new float[Size];
            for (var i = 0; i < Size; i++)
            {
                _src[i] = i * 0.5f;
            }
        }

        // ── (a) copy variants ────────────────────────────────────────────────
        [Benchmark(Baseline = true)]
        public void ArrayCopy() => Array.Copy(_src, _dst, Size);

        [Benchmark]
        public void SpanCopyTo() => _src.AsSpan(0, Size).CopyTo(_dst);

        [Benchmark]
        public void BufferBlockCopy() => Buffer.BlockCopy(_src, 0, _dst, 0, Size * sizeof(float));

        // ── (b) pool rental: raw try/finally vs the PooledBuffer using-wrapper ──
        [Benchmark]
        public float RentTryFinally()
        {
            var a = ArrayPool<float>.Shared.Rent(Size);
            try
            {
                a[0] = 1f;
                return a[Size - 1];
            }
            finally
            {
                ArrayPool<float>.Shared.Return(a);
            }
        }

        [Benchmark]
        public float PooledBufferUsing()
        {
            using var buf = new PooledBuffer<float>(Size, clearMemory: false);
            buf.Span[0] = 1f;
            return buf.Span[Size - 1];
        }
    }
}
