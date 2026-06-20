// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using BenchmarkDotNet.Attributes;
using DevOnBike.Overfit.Tensors;

#pragma warning disable RS0030  // Benchmark project — direct ArrayPool access is fine, the ban only targets Main.

namespace Benchmarks
{
    /// <summary>
    /// Head-to-head <c>Rent</c>+<c>Return</c> cycle measurement: the .NET default
    /// <see cref="ArrayPool{T}.Shared"/> vs the project's <c>OverfitPool&lt;T&gt;.Shared</c>
    /// (tuned <c>ArrayPool&lt;T&gt;.Create(maxArrayLength: 64M elements, maxArraysPerBucket: 1024)</c>)
    /// vs the using-scoped <see cref="PooledBuffer{T}"/> wrapper. The wrapper benchmark uses
    /// <c>clearMemory: false</c> to match raw Rent semantics (no zero-fill).
    ///
    /// Key sizes (in <c>float</c> elements):
    /// <list type="bullet">
    /// <item>1 K, 64 K — small; both pools should be similar (~10 ns).</item>
    /// <item>1 M — at the default ArrayPool's per-bucket cap for floats; the next size starts to fall through.</item>
    /// <item>4 M, 16 M — well above the default cap; default <c>ArrayPool.Shared</c> stops pooling these
    /// and allocates a fresh array per Rent (large alloc → GC pressure). <c>OverfitPool</c> keeps pooling.</item>
    /// </list>
    /// Expected: OverfitPool stays flat ~10 ns / 0 B across all sizes; ArrayPool.Shared spikes to
    /// microsecond / megabyte range above its cap.
    /// </summary>
    [ShortRunJob]
    [MemoryDiagnoser]
    public class PoolComparisonBenchmark
    {
        // 1K, 64K, 1M elements (around the float cap), 4M, 16M (above cap).
        [Params(1_024, 65_536, 1_048_576, 4_194_304, 16_777_216)]
        public int Size;

        // Equivalent of the deleted `OverfitPool<float>.Shared` — ConfigurableArrayPool with the
        // same parameters. Kept here so the benchmark can demonstrate WHY OverfitPool was deleted
        // (lock-per-bucket vs ArrayPool.Shared's TLS per-CPU caches).
        private static readonly ArrayPool<float> _configurablePool = ArrayPool<float>.Create(maxArrayLength: 1024 * 1024 * 64, maxArraysPerBucket: 1024);

        // Warm up each pool/size combination so the very first Rent (which may allocate fresh) isn't
        // measured. BDN does its own warmup but priming here makes sure the bucket is populated.
        [GlobalSetup]
        public void Prime()
        {
            var a1 = ArrayPool<float>.Shared.Rent(Size);
            ArrayPool<float>.Shared.Return(a1);
            var a2 = _configurablePool.Rent(Size);
            _configurablePool.Return(a2);
        }

        [Benchmark(Baseline = true, Description = "ArrayPool<float>.Shared (.NET default)")]
        public float ArrayPoolShared()
        {
            var a = ArrayPool<float>.Shared.Rent(Size);
            try
            {
                return a[0];
            }
            finally { ArrayPool<float>.Shared.Return(a); }
        }

        [Benchmark(Description = "ArrayPool.Create(64M, 1024) — like the deleted OverfitPool")]
        public float ConfigurablePool()
        {
            var a = _configurablePool.Rent(Size);
            try
            {
                return a[0];
            }
            finally { _configurablePool.Return(a); }
        }

        [Benchmark(Description = "PooledBuffer<float> (using-scoped wrapper, ArrayPool<T>.Shared-backed)")]
        public float PooledBufferUsing()
        {
            using var buf = new PooledBuffer<float>(Size, clearMemory: false);
            return buf.Span[0];
        }
    }
}
