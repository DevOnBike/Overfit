// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Tensors
{
    /// <summary>
    /// The project's single pooled-buffer wrapper over <c>ArrayPool&lt;T&gt;.Shared</c> (raw <c>.Shared</c>
    /// elsewhere in Main is RS0030-banned — this is the one sanctioned audit point). A plain struct, so the same
    /// type serves every lifetime:
    /// <list type="bullet">
    /// <item><b>method-local scratch</b>: <c>using var buf = new PooledBuffer&lt;float&gt;(n, clearMemory: false);</c>
    /// — auto-returns at scope end (works inside a <c>Parallel.For</c> lambda body too);</item>
    /// <item><b>class field</b>: rent in a constructor, <c>Dispose()</c> in the owner's Dispose (e.g.
    /// <c>FastTensor</c>, <c>TensorStorage</c>);</item>
    /// <item><b>closure-captured</b>: written from a worker lambda — capture the struct, take its
    /// <see cref="Span"/> inside the lambda body.</item>
    /// </list>
    ///
    /// <para>Exposes <see cref="Span"/> / <see cref="Memory"/>; returns the rented array to the pool on
    /// <see cref="Dispose"/> (idempotent / null-safe; <c>default</c> is a valid empty instance).</para>
    ///
    /// <para><b>Ownership discipline:</b> keep it in ONE owning field/local and dispose exactly once. Never copy it
    /// by value (a copy shares the same array, and disposing both would double-return and corrupt the pool).</para>
    ///
    /// <para><b>Span is a property</b> (a non-ref struct can't hold a <c>Span&lt;T&gt;</c> field). The JIT folds the
    /// getter to a plain field-read — measured identical to a cached Span field (~0.99×) — so just use
    /// <c>buf.Span</c>; if you touch it in a very tight inner loop, cache <c>var s = buf.Span;</c> once.</para>
    ///
    /// <para><b>Benchmark (2026-05-29, Ryzen 9 9950X3D):</b> Rent+Return on <c>ArrayPool.Shared</c> ~4 ns regardless
    /// of size (pools to 16M+ floats at 0 alloc); scales linearly across 16–32 threads via TLS per-CPU caches.</para>
    /// </summary>
    [SuppressMessage("ApiDesign", "RS0030", Justification = "Project wrapper around ArrayPool<T>.Shared — the ban targets direct callers in Main.")]
    public struct PooledBuffer<T> : IDisposable
        where T : struct
    {
        private T[]? _rented;
        private readonly int _length;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public PooledBuffer(int requiredSize, bool clearMemory = true)
        {
            _rented = ArrayPool<T>.Shared.Rent(requiredSize);
            _length = requiredSize;

            if (clearMemory)
            {
                _rented.AsSpan(0, requiredSize).Clear();
            }
        }

        /// <summary>The buffer as a span of exactly the requested size (empty once disposed).</summary>
        public readonly Span<T> Span
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _rented is null ? default : _rented.AsSpan(0, _length);
        }

        /// <summary>The buffer as memory of exactly the requested size (empty once disposed).</summary>
        public readonly Memory<T> Memory => _rented is null ? default : _rented.AsMemory(0, _length);

        /// <summary>Logical length (what was requested), not the pool's possibly-larger rented length.</summary>
        public readonly int Length => _length;

        /// <summary>False once disposed (or for a <c>default</c> instance).</summary>
        public readonly bool IsAllocated => _rented is not null;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Dispose()
        {
            var array = _rented;
            _rented = null;

            if (array is not null)
            {
                ArrayPool<T>.Shared.Return(array);
            }
        }
    }
}
