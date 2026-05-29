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
    /// Multi-thread Rent+Return contention measurement — the one hypothetical that could still favour
    /// <c>OverfitPool</c> (lock-per-bucket <c>ConfigurableArrayPool</c>) over the single-thread-superior
    /// <see cref="ArrayPool{T}.Shared"/> (TLS per-CPU caches via
    /// <c>TlsOverPerCoreLockedStacksArrayPool</c>): under heavy concurrent rent traffic, threads might
    /// contend on the same per-CPU stack (when scheduled on the same core), or fall through to the
    /// shared pool's slower path.
    ///
    /// Each operation issues <c>Iterations</c> Rent+Return cycles spread across <c>Threads</c> workers
    /// via <see cref="Parallel.For"/>. Measured as the WHOLE operation (all threads done), so a slower
    /// pool under contention will report higher mean time.
    ///
    /// Sizes chosen to cover both hot-path small (BN scratch-sized) and weight-scratch large.
    /// </summary>
    [ShortRunJob]
    [MemoryDiagnoser]
    public class PoolComparisonMultiThreadBenchmark
    {
        // Small (BN-scratch sized) + medium + large (weight-scratch).
        [Params(1_024, 65_536, 1_048_576)]
        public int Size;

        // 4 = below physical core count; 16 = matches physical cores (this box); 32 = SMT/oversubscribe.
        [Params(4, 16, 32)]
        public int Threads;

        // Each thread does this many Rent+Return cycles per operation — amortises BDN per-op overhead
        // and creates real contention pressure on the pool.
        private const int OpsPerThread = 64;

        // Equivalent of the deleted `OverfitPool<float>.Shared` — ConfigurableArrayPool with the
        // same parameters. The reason this benchmark exists is to document why OverfitPool was
        // deleted: lock-per-bucket serialises 32 threads vs ArrayPool.Shared's lock-free TLS path.
        private static readonly ArrayPool<float> _configurablePool = ArrayPool<float>.Create(maxArrayLength: 1024 * 1024 * 64, maxArraysPerBucket: 1024);

        [GlobalSetup]
        public void Prime()
        {
            // Warm both pools at the configured size so the first Rent in the benchmark body is hot.
            var a1 = ArrayPool<float>.Shared.Rent(Size); ArrayPool<float>.Shared.Return(a1);
            var a2 = _configurablePool.Rent(Size); _configurablePool.Return(a2);
        }

        [Benchmark(Baseline = true, Description = "ArrayPool<float>.Shared (TLS per-CPU)")]
        public void ArrayPoolShared_Concurrent()
        {
            Parallel.For(0, Threads, _ =>
            {
                for (var i = 0; i < OpsPerThread; i++)
                {
                    var a = ArrayPool<float>.Shared.Rent(Size);
                    a[0] = 1.0f; // touch one element to defeat dead-code elimination
                    ArrayPool<float>.Shared.Return(a);
                }
            });
        }

        [Benchmark(Description = "ArrayPool.Create(64M, 1024) — like the deleted OverfitPool")]
        public void ConfigurablePool_Concurrent()
        {
            Parallel.For(0, Threads, _ =>
            {
                for (var i = 0; i < OpsPerThread; i++)
                {
                    var a = _configurablePool.Rent(Size);
                    a[0] = 1.0f;
                    _configurablePool.Return(a);
                }
            });
        }

        [Benchmark(Description = "PooledBuffer<float> (using-scoped, .Shared-backed)")]
        public void PooledBuffer_Concurrent()
        {
            Parallel.For(0, Threads, _ =>
            {
                for (var i = 0; i < OpsPerThread; i++)
                {
                    using var buf = new PooledBuffer<float>(Size, clearMemory: false);
                    buf.Span[0] = 1.0f;
                }
            });
        }
    }
}
