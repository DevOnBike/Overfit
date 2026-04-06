// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Core
{
    /// <summary>
    /// A high-speed memory wrapper that uses ArrayPool to avoid GC allocations.
    /// Exposes a precise Span that excludes pool-induced padding.
    /// </summary>
    public sealed class FastBuffer<T> : IDisposable where T : struct
    {
        private T[] _rented;
        private bool _disposed;

        public int Length { get; }

        public FastBuffer(int length)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);

            Length = length;

            _rented = ArrayPool<T>.Shared.Rent(length);
            _rented.AsSpan(0, length).Clear();
        }

        public ref T this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                return ref _rented![index];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<T> AsSpan()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _rented.AsSpan(0, Length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<T> AsReadOnlySpan()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _rented.AsSpan(0, Length);
        }

        public void Clear() => AsSpan().Clear();

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, true))
            {
                return;
            }

            var rented = Interlocked.Exchange(ref _rented, null);

            if (rented != null)
            {
                ArrayPool<T>.Shared.Return(rented);
            }
        }
    }
}