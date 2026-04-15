// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;

namespace DevOnBike.Overfit.Tensors
{
    public unsafe readonly ref struct NativeBuffer<T> where T : unmanaged
    {
        public readonly Span<T> Span;
        private readonly void* _ptr;

        public NativeBuffer(int size, bool clearMemory = true)
        {
            if (size <= 0)
            {
                _ptr = null;
                Span = Span<T>.Empty;
                
                return;
            }

            var byteSize = (nuint)size * (nuint)sizeof(T);
            var paddedByteSize = (byteSize + 63) & ~(nuint)63;

            _ptr = NativeMemory.AlignedAlloc(paddedByteSize, 64);

            Span = new Span<T>(_ptr, size);

            if (clearMemory)
            {
                Span.Clear();
            }
        }

        public void Dispose()
        {
            if (_ptr != null)
            {
                NativeMemory.AlignedFree(_ptr);
            }
        }
    }
}