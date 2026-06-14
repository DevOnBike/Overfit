// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using System.Diagnostics.CodeAnalysis;

namespace DevOnBike.Overfit.Tensors
{
    /// <summary>
    /// The project's sanctioned interface to <see cref="ArrayPool{T}.Shared"/>. Raw
    /// <c>ArrayPool&lt;T&gt;.Shared</c> elsewhere in Main is banned (RS0030, see
    /// <c>BannedSymbols.txt</c>) so all rentals go through this single audit point — gives one swap
    /// point across the codebase.
    ///
    /// <para>Two usage modes:</para>
    /// <list type="number">
    /// <item><b>Scoped (default)</b>: <c>using var buf = new PooledBuffer&lt;float&gt;(n, clearMemory: false);</c>
    /// — ref-struct, can't escape its scope, auto-returns. Works inside <c>Parallel.For</c> lambdas
    /// (a local ref struct inside a lambda body is allowed; only capturing one from outside is not).</item>
    /// <item><b>Class-lifetime</b>: <see cref="RentArray"/> + <see cref="ReturnArray"/> for callers
    /// that rent in a constructor / method and return in <c>Dispose</c> (e.g. <c>FastTensor</c>,
    /// <c>TensorStorage</c>, complex loader patterns). Same pool, no scope wrapper.</item>
    /// </list>
    ///
    /// <para><b>Benchmark (2026-05-29, Ryzen 9 9950X3D, .NET 10):</b></para>
    /// <list type="bullet">
    /// <item>Single-thread Rent+Return: <c>.Shared</c> ~4 ns regardless of size (pools to 16M+ floats
    /// at 0 alloc). Wrapper overhead 0% (matches raw <c>.Shared</c> to noise).</item>
    /// <item>Multi-thread (16–32 threads): <c>.Shared</c> scales linearly via TLS per-CPU caches
    /// (4.9 ns/op at 32 threads — same as single-thread). <see cref="PooledBuffer{T}"/> wrapper
    /// overhead 1.03–1.14× — also linear.</item>
    /// </list>
    /// </summary>
    [SuppressMessage("ApiDesign", "RS0030", Justification = "Project wrapper around ArrayPool<T>.Shared — the ban targets direct callers in Main.")]
    public readonly ref struct PooledBuffer<T> where T : struct
    {
        public readonly Span<T> Span;

        private readonly T[] _rented;

        public PooledBuffer(int requiredSize, bool clearMemory = true)
        {
            _rented = ArrayPool<T>.Shared.Rent(requiredSize);
            Span = _rented.AsSpan(0, requiredSize);

            if (clearMemory)
            {
                Span.Clear();
            }
        }

        public void Dispose()
        {
            if (_rented != null)
            {
                ArrayPool<T>.Shared.Return(_rented);
            }
        }

        /// <summary>
        /// Class-lifetime rental — call when the ref-struct <c>using</c> scope doesn't fit (e.g.
        /// rent in a ctor, return in Dispose). The returned array length is at least
        /// <paramref name="minimumLength"/>; caller slices to the exact length it needs. Pair with
        /// <see cref="ReturnArray"/> exactly once.
        /// </summary>
        public static T[] RentArray(int minimumLength)
        {
            return ArrayPool<T>.Shared.Rent(minimumLength);
        }

        /// <summary>Returns an array previously obtained from <see cref="RentArray"/> to the pool.</summary>
        public static void ReturnArray(T[] array)
        {
            ArrayPool<T>.Shared.Return(array);
        }
    }
}
