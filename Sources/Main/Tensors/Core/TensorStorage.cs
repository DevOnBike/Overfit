// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Diagnostics;

namespace DevOnBike.Overfit.Tensors.Core
{
    /// <summary>
    /// Pure DOD physical memory storage.
    /// It has no geometry, shape, rank or math semantics.
    /// </summary>
    public sealed unsafe class TensorStorage<T> : IDisposable
        where T : unmanaged
    {
        private T[]? _data;
        private int _disposed;
        private void* _nativePtr;

        internal NativeBufferManaged<T>? _buffer;
        internal readonly bool _isBorrowedMemory;

        public readonly int Length;

        /// <summary>
        /// Standard pooled managed storage.
        /// Reports only context-free infrastructure telemetry:
        /// storage created, pooled created, elements, estimated bytes.
        /// </summary>
        public TensorStorage(int length, bool clearMemory = true)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(length);

            Length = length;
            _data = OverfitPool<T>.Shared.Rent(length);

            if (clearMemory)
            {
                _data.AsSpan(0, length).Clear();
            }

            OverfitTelemetry.RecordTensorStorageCreated(
                length,
                Unsafe.SizeOf<T>(),
                borrowed: false);
        }

        /// <summary>
        /// Borrowed arena storage.
        /// Reports only context-free infrastructure telemetry:
        /// storage created, borrowed created, elements, estimated bytes.
        /// </summary>
        public TensorStorage(NativeBufferManaged<T> buffer, int length)
        {
            ArgumentNullException.ThrowIfNull(buffer);
            ArgumentOutOfRangeException.ThrowIfNegative(length);

            Length = length;
            _buffer = buffer;
            _nativePtr = buffer.Allocate(length);
            _isBorrowedMemory = true;

            OverfitTelemetry.RecordTensorStorageCreated(
                length,
                Unsafe.SizeOf<T>(),
                borrowed: true);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<T> AsSpan()
        {
            ObjectDisposedException.ThrowIf(_disposed == 1, this);

            return _isBorrowedMemory
                ? new Span<T>(_nativePtr, Length)
                : _data!.AsSpan(0, Length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<T> AsReadOnlySpan()
        {
            return AsSpan();
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
            {
                OverfitTelemetry.RecordTensorStorageDisposed(_isBorrowedMemory);

                if (!_isBorrowedMemory && _data != null)
                {
                    OverfitPool<T>.Shared.Return(_data);
                }

                _data = null;
                _nativePtr = null;
                _buffer = null;
            }
        }
    }
}