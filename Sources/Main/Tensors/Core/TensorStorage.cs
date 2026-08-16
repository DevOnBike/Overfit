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
        // _pooledBuf backs the pooled ctor; _data backs the Unpooled (GC-owned) ctor; _nativePtr the borrowed
        // ctor. Exactly one is live per instance, selected by _pooled / _isBorrowedMemory.
        private PooledBuffer<T> _pooledBuf;
        private T[]? _data;
        private int _disposed;
        private void* _nativePtr;
        private readonly bool _pooled;

        internal NativeBufferManaged<T>? _buffer;
        internal readonly bool _isBorrowedMemory;

        public readonly int Length;

        /// <summary>
        /// Which of the three backings this storage has. Derived from the two flags rather than stored
        /// again, so it cannot disagree with them.
        /// </summary>
        internal TensorStorageKind Kind => _isBorrowedMemory
            ? TensorStorageKind.Borrowed
            : _pooled ? TensorStorageKind.Pooled : TensorStorageKind.Unpooled;

        /// <summary>
        /// Standard pooled managed storage.
        /// Reports only context-free infrastructure telemetry:
        /// storage created, pooled created, elements, estimated bytes.
        /// </summary>
        public TensorStorage(int length, bool clearMemory = true)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(length);

            Length = length;
            _pooledBuf = new PooledBuffer<T>(length, clearMemory);
            _pooled = true;

            OverfitTelemetry.RecordTensorStorageCreated(length, Unsafe.SizeOf<T>(), borrowed: false);
        }

        /// <summary>
        /// Private ctor used by <see cref="Unpooled"/>. Wraps a caller-owned, exact-sized
        /// array. The array is allocated via <c>new T[length]</c> and will be released to
        /// the GC when this storage is disposed — never returned to the pool.
        /// </summary>
        private TensorStorage(T[] data)
        {
            ArgumentNullException.ThrowIfNull(data);

            Length = data.Length;
            _data = data;
            _pooled = false;

            // Unpooled, not pooled. It reported itself as pooled because the counter took one bool, so
            // every model weight loaded through this path landed in the pool-churn series.
            OverfitTelemetry.RecordTensorStorageCreated(
                Length, Unsafe.SizeOf<T>(), TensorStorageKind.Unpooled);
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

            OverfitTelemetry.RecordTensorStorageCreated(length, Unsafe.SizeOf<T>(), borrowed: true);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<T> AsSpan()
        {
            ObjectDisposedException.ThrowIf(_disposed == 1, this);

            if (_isBorrowedMemory)
            {
                return new Span<T>(_nativePtr, Length);
            }

            return _pooled ? _pooledBuf.Span : _data!.AsSpan(0, Length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<T> AsReadOnlySpan()
        {
            return AsSpan();
        }

        /// <summary>
        /// A <see cref="Memory{T}"/> view over the data, without copying.
        ///
        /// <para><b>Two defects lived here, and the loud one was the less dangerous.</b> The method did not
        /// branch on <see cref="_isBorrowedMemory"/> as <see cref="AsSpan"/> does, so on native-backed
        /// storage it fell into <c>_data!</c> — which is null there — and threw a
        /// <see cref="NullReferenceException"/> from an operator that reads like a field access. It also
        /// skipped the disposed check its sibling performs first, and <b>that</b> is the one to worry about:
        /// a disposed pooled storage would hand out a <see cref="Memory{T}"/> over an array already returned
        /// to <c>ArrayPool</c>, so the caller reads and writes memory that something else now owns, with no
        /// exception at any point.</para>
        ///
        /// <para>Native storage cannot produce a <see cref="Memory{T}"/> at all — <see cref="Memory{T}"/>
        /// requires an object or a pinned owner, and a raw pointer is neither. Refusing it is the honest
        /// answer; use <see cref="AsSpan"/> there, which is what every caller inside the library does.</para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">This storage has been disposed.</exception>
        /// <exception cref="NotSupportedException">The storage is backed by borrowed native memory.</exception>
        public Memory<T> AsMemory()
        {
            ObjectDisposedException.ThrowIf(_disposed == 1, this);

            if (_isBorrowedMemory)
            {
                throw new NotSupportedException(
                    "This storage is backed by borrowed native memory, which cannot be exposed as Memory<T>. "
                    + "Use AsSpan().");
            }

            return _pooled ? _pooledBuf.Memory : _data!.AsMemory(0, Length);
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
                OverfitTelemetry.RecordTensorStorageDisposed(Kind);

                // Only return to pool if rented from pool. Unpooled storage is GC-managed.
                if (!_isBorrowedMemory && _pooled)
                {
                    _pooledBuf.Dispose();
                }

                _data = null;
                _nativePtr = null;
                _buffer = null;
            }
        }
    }
}