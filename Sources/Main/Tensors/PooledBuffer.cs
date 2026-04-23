// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;

namespace DevOnBike.Overfit.Tensors
{
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
    }
}