// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Statistical
{
    /// <summary>
    /// A high-performance, zero-allocation circular buffer for time-series data.
    /// Thread-safe for multiple producers and consumers using .NET 9 System.Threading.Lock.
    /// </summary>
    public sealed class SlidingWindowBuffer : IDisposable
    {
        private readonly FastTensor<float> _ring; // flat: windowSize × featureCount
        private readonly FastTensor<DateTime> _timestamps; // one timestamp per slot
        private readonly int _windowSize;
        private readonly int _stepSize;
        private readonly int _featureCount;
        private readonly int _windowFloats; // = windowSize × featureCount

        // --- state ---
        private readonly Lock _lock = new(); // System.Threading.Lock — no Monitor overhead
        private int _head; // index of the next write slot
        private int _count; // number of filled slots (max = windowSize)
        private int _samplesUntilStep; // how many samples until the next window
        private bool _disposed;

        public int WindowSize => _windowSize;
        public int FeatureCount => _featureCount;

        public SlidingWindowBuffer(int windowSize, int featureCount, int stepSize)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(windowSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(featureCount);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(stepSize);

            _windowSize = windowSize;
            _featureCount = featureCount;
            _stepSize = stepSize;
            _windowFloats = windowSize * featureCount;

            _ring = new FastTensor<float>(_windowFloats, clearMemory: true);
            _timestamps = new FastTensor<DateTime>(windowSize, clearMemory: true);

            _head = 0;
            _count = 0;
            _samplesUntilStep = windowSize;
        }

        public void Add(ReadOnlySpan<float> features, DateTime timestamp)
        {
            ThrowIfDisposed();

            if (features.Length != _featureCount)
            {
                throw new ArgumentException($"Expected {_featureCount} features, got {features.Length}.");
            }

            lock (_lock)
            {
                var ringSpan = _ring.GetView().AsSpan();
                var timestampsSpan = _timestamps.GetView().AsSpan();

                features.CopyTo(ringSpan.Slice(_head * _featureCount, _featureCount));
                timestampsSpan[_head] = timestamp;

                AdvanceHead();
            }
        }

        public bool TryGetWindow(Span<float> destination, out DateTime windowEnd)
        {
            ThrowIfDisposed();

            if (destination.Length < _windowFloats)
            {
                throw new ArgumentException("Destination span is too small for the window.");
            }

            lock (_lock)
            {
                if (_count < _windowSize || _samplesUntilStep > 0)
                {
                    windowEnd = default;
                    return false;
                }

                var ringSpan = _ring.GetView().AsReadOnlySpan();

                var lastIndex = _head == 0 ? _windowSize - 1 : _head - 1;
                windowEnd = _timestamps.GetView().AsReadOnlySpan()[lastIndex];

                CopyRingToContiguous(ringSpan, destination);

                _samplesUntilStep = _stepSize;
                return true;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void CopyRingToContiguous(ReadOnlySpan<float> ringSpan, Span<float> destination)
        {
            var slotsToEnd = _windowSize - _head;

            if (slotsToEnd == _windowSize)
            {
                ringSpan.Slice(_head * _featureCount, _windowFloats).CopyTo(destination);
            }
            else
            {
                var block1 = slotsToEnd * _featureCount;
                var block2 = _windowFloats - block1;

                ringSpan.Slice(_head * _featureCount, block1).CopyTo(destination);
                ringSpan.Slice(0, block2).CopyTo(destination.Slice(block1));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void AdvanceHead()
        {
            _head++;
            
            if (_head >= _windowSize)
            {
                _head = 0;
            }

            if (_count < _windowSize)
            {
                _count++;
            }

            _samplesUntilStep--;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }

        public void Dispose()
        {
            lock (_lock)
            {
                if (_disposed)
                {
                    return;
                }
                
                _disposed = true;
            }

            _ring.Dispose();
            _timestamps.Dispose();
        }
    }
}