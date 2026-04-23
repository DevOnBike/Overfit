// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DevOnBike.Overfit.Tensors.Core
{
    /// <summary>
    /// 64-byte aligned unmanaged buffer.
    ///
    /// Can be used either as:
    /// - one large tensor backing store, or
    /// - a linear arena allocator for sub-allocations.
    ///
    /// Notes:
    /// - not thread-safe
    /// - disposing invalidates all previously returned spans/pointers
    /// - sub-allocations can optionally be aligned to a byte boundary
    /// </summary>
    public sealed unsafe class NativeBufferManaged<T> : IDisposable where T : unmanaged
    {
        private const nuint BaseAlignmentBytes = 64;

        public int Size { get; }

        private void* _ptr;
        private int _offset;
        private bool _disposed;

        public NativeBufferManaged(int size, bool clearMemory = true)
        {
            if (size < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(size), "Size cannot be negative.");
            }

            if (size == 0)
            {
                Size = 0;
                _ptr = null;
                _offset = 0;
                _disposed = false;

                return;
            }

            Size = size;

            var byteSize = checked((nuint)size * (nuint)sizeof(T));
            var paddedByteSize = AlignUp(byteSize, BaseAlignmentBytes);

            _ptr = NativeMemory.AlignedAlloc(paddedByteSize, BaseAlignmentBytes);

            if (_ptr is null)
            {
                throw new OutOfMemoryException($"Failed to allocate {paddedByteSize} bytes of aligned unmanaged memory.");
            }

            _offset = 0;
            _disposed = false;

            if (clearMemory)
            {
                GetSpan().Clear();
            }
        }

        /// <summary>
        /// Returns a span covering the entire buffer.
        /// </summary>
        public Span<T> GetSpan()
        {
            ThrowIfDisposed();

            return _ptr == null ? [] : new Span<T>(_ptr, Size);
        }

        /// <summary>
        /// Allocates a contiguous region and returns a pointer to its start.
        /// The returned region is aligned to at least <paramref name="alignmentBytes"/>.
        ///
        /// alignmentBytes must be:
        /// - positive
        /// - a power of two
        /// - a multiple of sizeof(T)
        /// </summary>
        public void* Allocate(int length, int alignmentBytes = (int)BaseAlignmentBytes)
        {
            ThrowIfDisposed();

            if (length < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(length), "Length cannot be negative.");
            }

            if (length == 0)
            {
                return (byte*)_ptr + (nuint)(_offset * sizeof(T));
            }

            ValidateAlignment(alignmentBytes);

            var elementSize = sizeof(T);
            var currentByteOffset = checked((nuint)_offset * (nuint)elementSize);
            var alignedByteOffset = AlignUp(currentByteOffset, (nuint)alignmentBytes);
            var alignedElementOffset = checked((int)(alignedByteOffset / (nuint)elementSize));
            var nextOffset = checked(alignedElementOffset + length);

            if (nextOffset > Size)
            {
                throw new OutOfMemoryException($"NativeBuffer exhausted. Capacity: {Size}, required end offset: {nextOffset}.");
            }

            _offset = nextOffset;

            return (byte*)_ptr + alignedByteOffset;
        }

        /// <summary>
        /// Allocates a contiguous region and returns it as a span.
        /// </summary>
        public Span<T> AllocateSpan(int length, int alignmentBytes = (int)BaseAlignmentBytes)
        {
            var ptr = Allocate(length, alignmentBytes);

            return length == 0 ? [] : new Span<T>(ptr, length);
        }

        /// <summary>
        /// Resets the arena bump pointer to zero.
        /// Does not clear memory.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void ResetOffset()
        {
            ThrowIfDisposed();

            _offset = 0;
        }

        /// <summary>
        /// Returns the current arena offset in elements.
        /// </summary>
        public int CurrentOffset
        {
            get
            {
                ThrowIfDisposed();

                return _offset;
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            var ptr = _ptr;
            _ptr = null;
            _offset = 0;
            _disposed = true;

            if (ptr != null)
            {
                NativeMemory.AlignedFree(ptr);
            }

            GC.SuppressFinalize(this);
        }

        ~NativeBufferManaged()
        {
            if (_ptr != null)
            {
                NativeMemory.AlignedFree(_ptr);
                _ptr = null;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ValidateAlignment(int alignmentBytes)
        {
            if (alignmentBytes <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(alignmentBytes), "Alignment must be positive.");
            }

            if ((alignmentBytes & (alignmentBytes - 1)) != 0)
            {
                throw new ArgumentException("Alignment must be a power of two.", nameof(alignmentBytes));
            }

            if (alignmentBytes % sizeof(T) != 0)
            {
                throw new ArgumentException($"Alignment must be a multiple of sizeof({typeof(T).Name}) = {sizeof(T)}.", nameof(alignmentBytes));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static nuint AlignUp(nuint value, nuint alignment)
        {
            return (value + (alignment - 1)) & ~(alignment - 1);
        }
    }
}