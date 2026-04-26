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
    /// The goal is to avoid per-backward allocations:
    /// - no ConcurrentBag
    /// - no per-call TensorStorage creation for partial gradients
    /// - no PooledBuffer in the convolution backward hot path
    /// </summary>
    internal sealed class Conv2DWorkspace : IDisposable
    {
        private TensorStorage<float>[] _colBuffers = [];
        private TensorStorage<float>[] _dColBuffers = [];
        private TensorStorage<float>[] _partialWeightGradients = [];

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

        public void Ensure(
            int workerCount,
            int colLength,
            int partialWeightGradientLength)
        {
            ThrowIfDisposed();

            if (workerCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(workerCount));
            }

            if (colLength <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(colLength));
            }

            if (partialWeightGradientLength <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(partialWeightGradientLength));
            }

            if (_workerCount == workerCount &&
                _colLength == colLength &&
                _partialWeightGradientLength == partialWeightGradientLength)
            {
                return;
            }

            DisposeBuffers();

            _workerCount = workerCount;
            _colLength = colLength;
            _partialWeightGradientLength = partialWeightGradientLength;

            _colBuffers = new TensorStorage<float>[workerCount];
            _dColBuffers = new TensorStorage<float>[workerCount];
            _partialWeightGradients = new TensorStorage<float>[workerCount];

            for (var i = 0; i < workerCount; i++)
            {
                _colBuffers[i] = new TensorStorage<float>(colLength, clearMemory: false);
                _dColBuffers[i] = new TensorStorage<float>(colLength, clearMemory: false);
                _partialWeightGradients[i] = new TensorStorage<float>(partialWeightGradientLength, clearMemory: false);
            }
        }

        public Span<float> GetColBuffer(int workerId)
        {
            ThrowIfDisposed();
            ValidateWorkerId(workerId);
            return _colBuffers[workerId].AsSpan();
        }

        public Span<float> GetDColBuffer(int workerId)
        {
            ThrowIfDisposed();
            ValidateWorkerId(workerId);
            return _dColBuffers[workerId].AsSpan();
        }

        public Span<float> GetPartialWeightGradient(int workerId)
        {
            ThrowIfDisposed();
            ValidateWorkerId(workerId);
            return _partialWeightGradients[workerId].AsSpan();
        }

        public void ClearPartialWeightGradients()
        {
            ThrowIfDisposed();

            for (var i = 0; i < _workerCount; i++)
            {
                _partialWeightGradients[i].AsSpan().Clear();
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            DisposeBuffers();
        }

        private void DisposeBuffers()
        {
            for (var i = 0; i < _colBuffers.Length; i++)
            {
                _colBuffers[i]?.Dispose();
            }

            for (var i = 0; i < _dColBuffers.Length; i++)
            {
                _dColBuffers[i]?.Dispose();
            }

            for (var i = 0; i < _partialWeightGradients.Length; i++)
            {
                _partialWeightGradients[i]?.Dispose();
            }

            _colBuffers = [];
            _dColBuffers = [];
            _partialWeightGradients = [];

            _workerCount = 0;
            _colLength = 0;
            _partialWeightGradientLength = 0;
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