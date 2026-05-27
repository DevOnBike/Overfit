// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;

namespace DevOnBike.Overfit.Runtime
{
    /// <summary>
    /// A scoped rental from <see cref="ArrayPool{T}.Shared"/> with a <c>using</c> lifetime — replaces the
    /// noisy <c>Rent(...) / try { … } finally { Return(...) }</c> trio with one line:
    /// <code>
    /// using var buf = new PooledArray&lt;float&gt;(n);
    /// // … use buf.Span (exactly n elements) …
    /// </code>
    /// A <see langword="ref struct"/> so it cannot escape its scope or be boxed/stored — the rented array
    /// is always returned at end of scope. <see cref="Span"/> is the requested length (the pool may hand
    /// back a larger array). Zero overhead vs. raw Rent/Return (the wrapper inlines away).
    /// </summary>
    public readonly ref struct PooledArray<T>
    {
        private readonly T[] _array;

        /// <summary>The rented buffer, sliced to exactly the requested length.</summary>
        public readonly Span<T> Span;

        public PooledArray(int length)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(length);

            _array = ArrayPool<T>.Shared.Rent(length);
            Span = _array.AsSpan(0, length);
        }

        public void Dispose()
        {
            ArrayPool<T>.Shared.Return(_array);
        }
    }
}
