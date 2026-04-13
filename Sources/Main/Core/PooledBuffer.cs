// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;

namespace DevOnBike.Overfit.Core
{
    public readonly ref struct PooledBuffer<T> : IDisposable where T : struct
    {
        public readonly Span<T> Span { get; }

        private readonly T[] _rented;

        public PooledBuffer(int requiredSize, bool clearMemory = false)
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
                // Oddajemy oryginalną, surową tablicę z powrotem
                ArrayPool<T>.Shared.Return(_rented, clearArray: false);
            }
        }
    }
}