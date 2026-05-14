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
        private readonly bool _pooled;

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
            _pooled = true;

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
        /// Private ctor used by <see cref="Unpooled"/>. Wraps a caller-owned, exact-sized
        /// array. The array is allocated via <c>new T[length]</c> and will be released to
        /// the GC when this storage is disposed — never returned to <see cref="OverfitPool{T}"/>.
        /// </summary>
        private TensorStorage(T[] data)
        {
            ArgumentNullException.ThrowIfNull(data);

            Length = data.Length;
            _data = data;
            _pooled = false;

            OverfitTelemetry.RecordTensorStorageCreated(
                Length,
                Unsafe.SizeOf<T>(),
                borrowed: false);
        }

        /// <summary>
        /// Creates a TensorStorage backed by a GC-managed, exact-sized array (no pool).
        /// Use this for LONG-LIVED storage (e.g. model weights) where pool retention would
        /// waste memory: the pool keeps rented arrays alive in its buckets even after
        /// Dispose, and rounds requests up to the next power-of-2 bucket size.
        ///
        /// For inference scratch buffers (short-lived, frequently reused), keep using the
        /// pooled constructor.
        /// </summary>
        public static TensorStorage<T> Unpooled(int length, bool clearMemory = false)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(length);

            var arr = new T[length];
            // new T[] is zero-initialized by the runtime for primitive types, so clearMemory is a no-op.
            // Parameter kept for API symmetry with the pooled constructor.
            _ = clearMemory;

            return new TensorStorage<T>(arr);
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

        /// <summary>
        /// Returns a Memory&lt;T&gt; view over the data without copying.
        /// </summary>
        public Memory<T> AsMemory()
        {
            return _data!.AsMemory(0, Length);
        }

        /// <summary>
        /// Creates a TensorStorage wrapping a copy of the provided array.
        /// For internal use — enables test helpers to create weight structs from raw arrays.
        /// </summary>
        internal static TensorStorage<T> FromArray(T[] source)
        {
            var ts = new TensorStorage<T>(source.Length, clearMemory: false);
            source.AsSpan().CopyTo(ts.AsSpan());
            return ts;
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
            {
                OverfitTelemetry.RecordTensorStorageDisposed(_isBorrowedMemory);

                // Only return to pool if rented from pool. Unpooled storage is GC-managed.
                if (!_isBorrowedMemory && _pooled && _data != null)
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