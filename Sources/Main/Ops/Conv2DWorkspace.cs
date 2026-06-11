// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Ops
{
    /// <summary>
    /// Reusable convolution workspace owned by ComputationGraph.
    ///
    /// Holds the per-worker im2col / col2im / partial-weight-gradient scratch the
    /// Conv2D kernels write into, so the convolution backward hot path performs no
    /// per-call allocation.
    ///
    /// Each buffer type is a <b>single contiguous</b> <see cref="TensorStorage{T}"/>:
    /// worker <c>w</c> owns the slice <c>[w * length, length)</c>. Contiguity lets the
    /// kernels pin the whole buffer with one <c>fixed</c> statement and hand a base
    /// pointer to <c>OverfitParallel</c> — a per-worker array-of-storages could not
    /// be pinned with a single, statically-shaped <c>fixed</c>.
    /// </summary>
    internal sealed class Conv2DWorkspace : IDisposable
    {
        private TensorStorage<float> _colBuffer;
        private TensorStorage<float> _dColBuffer;
        private TensorStorage<float> _partialWeightGradientBuffer;

        private int _workerCount;
        private int _colLength;
        private int _partialWeightGradientLength;

        private bool _disposed;

        public int WorkerCount
        {
            get
            {
                ThrowIfDisposed();
                return _workerCount;
            }
        }

        public int ColLength
        {
            get
            {
                ThrowIfDisposed();
                return _colLength;
            }
        }

        public int PartialWeightGradientLength
        {
            get
            {
                ThrowIfDisposed();
                return _partialWeightGradientLength;
            }
        }

        /// <summary>Whole im2col buffer — <c>workerCount * colLength</c> floats. Pin this for the kernels.</summary>
        public Span<float> ColBuffer
        {
            get
            {
                ThrowIfDisposed();
                return _colBuffer.AsSpan();
            }
        }

        /// <summary>Whole col2im gradient buffer — <c>workerCount * colLength</c> floats.</summary>
        public Span<float> DColBuffer
        {
            get
            {
                ThrowIfDisposed();
                return _dColBuffer.AsSpan();
            }
        }

        /// <summary>Whole partial-weight-gradient buffer — <c>workerCount * partialWeightGradientLength</c> floats.</summary>
        public Span<float> PartialWeightGradientBuffer
        {
            get
            {
                ThrowIfDisposed();
                return _partialWeightGradientBuffer.AsSpan();
            }
        }

        public void Ensure(
            int workerCount,
            int colLength,
            int partialWeightGradientLength)
        {
            ThrowIfDisposed();

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(workerCount);

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(colLength);

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(partialWeightGradientLength);

            if (_workerCount == workerCount &&
                _colLength == colLength &&
                _partialWeightGradientLength == partialWeightGradientLength)
            {
                return;
            }

            // Dispose the previous shape's buffers before re-allocating for the new one.
            _colBuffer?.Dispose();
            _dColBuffer?.Dispose();
            _partialWeightGradientBuffer?.Dispose();

            _workerCount = workerCount;
            _colLength = colLength;
            _partialWeightGradientLength = partialWeightGradientLength;

            _colBuffer = new TensorStorage<float>(workerCount * colLength, clearMemory: false);
            _dColBuffer = new TensorStorage<float>(workerCount * colLength, clearMemory: false);
            _partialWeightGradientBuffer = new TensorStorage<float>(
                workerCount * partialWeightGradientLength,
                clearMemory: false);
        }

        /// <summary>The partial-weight-gradient slice owned by <paramref name="workerId"/>.</summary>
        public Span<float> GetPartialWeightGradient(int workerId)
        {
            ThrowIfDisposed();
            ValidateWorkerId(workerId);
            return _partialWeightGradientBuffer
                .AsSpan()
                .Slice(workerId * _partialWeightGradientLength, _partialWeightGradientLength);
        }

        public void ClearPartialWeightGradients()
        {
            ThrowIfDisposed();
            _partialWeightGradientBuffer.AsSpan().Clear();
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;

            _colBuffer?.Dispose();
            _dColBuffer?.Dispose();
            _partialWeightGradientBuffer?.Dispose();
        }

        private void ValidateWorkerId(int workerId)
        {
            if ((uint)workerId >= (uint)_workerCount)
            {
                throw new ArgumentOutOfRangeException(nameof(workerId));
            }
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }
}
