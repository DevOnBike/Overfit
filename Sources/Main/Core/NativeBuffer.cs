// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;

namespace DevOnBike.Overfit.Core
{
    public unsafe readonly ref struct NativeBuffer<T> where T : unmanaged
    {
        public readonly Span<T> Span;

        private readonly void* _ptr;

        public NativeBuffer(int size, bool clearMemory = true)
        {
            _ptr = clearMemory ? NativeMemory.AllocZeroed((nuint)size, (nuint)sizeof(T)) : NativeMemory.Alloc((nuint)size, (nuint)sizeof(T));

            Span = new Span<T>(_ptr, size);
        }

        public void Dispose()
        {
            if (_ptr != null)
            {
                NativeMemory.Free(_ptr);
            }
        }
    }
}