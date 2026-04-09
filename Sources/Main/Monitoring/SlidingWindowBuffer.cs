// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Monitoring
{
    public sealed class SlidingWindowBuffer : IDisposable
    {
        // --- memory (allocated once in the constructor from ArrayPool via FastBuffer) ---
        private readonly FastBuffer<float> _ring; // flat: windowSize × featureCount
        private readonly FastBuffer<DateTime> _timestamps; // one timestamp per slot
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

        // --- diagnostics (Interlocked — do not require _lock) ---
        private long _windowsProduced;
        private long _samplesAdded;

        public int WindowSize => _windowSize;
        public int StepSize => _stepSize;
        public int FeatureCount => _featureCount;
        public int WindowFloats => _windowFloats;
        public long WindowsProduced => Volatile.Read(ref _windowsProduced);
        public long SamplesAdded => Volatile.Read(ref _samplesAdded);

        /// <summary>How many samples are missing to fill the first window.</summary>
        public int SamplesUntilFirstWindow
        {
            get
            {
                lock (_lock)
                {
                    return Math.Max(0, _windowSize - _count);
                }
            }
        }

        /// <param name="windowSize">Number of samples in the window.</param>
        /// <param name="stepSize">How often (in samples) we produce a new window. Must be <= windowSize.</param>
        /// <param name="featureCount">Feature count — constant throughout the buffer's lifecycle.</param>
        public SlidingWindowBuffer(
            int windowSize,
            int stepSize = 1,
            int featureCount = MetricSnapshot.FeatureCount)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(windowSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(stepSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(featureCount);

            if (stepSize > windowSize)
            {
                throw new ArgumentOutOfRangeException(nameof(stepSize), $"stepSize ({stepSize}) > windowSize ({windowSize}).");
            }

            _windowSize = windowSize;
            _stepSize = stepSize;
            _featureCount = featureCount;
            _windowFloats = checked(windowSize * featureCount);

            _ring = new FastBuffer<float>(_windowFloats);
            _timestamps = new FastBuffer<DateTime>(windowSize);
            _samplesUntilStep = stepSize;
        }

        // -------------------------------------------------------------------------
        // Add
        // -------------------------------------------------------------------------

        /// <summary>
        /// Adds a sample from <see cref="MetricSnapshot"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(in MetricSnapshot snapshot)
        {
            ThrowIfDisposed();

            lock (_lock)
            {
                snapshot.WriteFeatureVector(_ring.AsSpan().Slice(_head * _featureCount, _featureCount));

                _timestamps[_head] = snapshot.Timestamp;

                AdvanceHead();
            }

            Interlocked.Increment(ref _samplesAdded);
        }

        /// <summary>
        /// Adds a manually constructed feature vector.
        /// Use for testing or replaying historical data from CSV/Parquet.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(ReadOnlySpan<float> features, DateTime timestamp)
        {
            ThrowIfDisposed();

            if (features.Length != _featureCount)
            {
                throw new ArgumentException($"Oczekiwano {_featureCount} cech, otrzymano {features.Length}.", nameof(features));
            }

            lock (_lock)
            {
                features.CopyTo(_ring.AsSpan().Slice(_head * _featureCount, _featureCount));

                _timestamps[_head] = timestamp;

                AdvanceHead();
            }

            Interlocked.Increment(ref _samplesAdded);
        }

        // -------------------------------------------------------------------------
        // TryGetWindow
        // -------------------------------------------------------------------------

        /// <summary>
        /// Copies the ready window to a flat <paramref name="destination"/> (row-major, chronologically).
        ///
        /// Preferred overload when destination is <c>fastTensor.AsSpan()</c> —
        /// copies directly to tensor memory, zero intermediate buffers.
        /// </summary>
        /// <param name="destination">Caller-owned, min. <see cref="WindowFloats"/> elements.</param>
        /// <param name="windowEnd">Timestamp of the newest sample in the window.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TryGetWindow(Span<float> destination, out DateTime windowEnd)
        {
            ThrowIfDisposed();

            if (destination.Length < _windowFloats)
            {
                throw new ArgumentException($"Destination za krótki: potrzeba {_windowFloats}, dostępne {destination.Length}.", nameof(destination));
            }

            return TryGetWindowCore(destination, out windowEnd);
        }

        /// <summary>
        /// Overload for <see cref="FastMatrix{T}"/> — preallocated matrix [windowSize × featureCount].
        /// Once filled, caller accesses via <c>matrix.Row(i)</c> in FeatureExtractor.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TryGetWindow(FastMatrix<float> destination, out DateTime windowEnd)
        {
            ThrowIfDisposed();

            ArgumentNullException.ThrowIfNull(destination);

            if (destination.Rows != _windowSize || destination.Cols != _featureCount)
            {
                throw new ArgumentException($"FastMatrix musi być [{_windowSize}×{_featureCount}], otrzymano [{destination.Rows}×{destination.Cols}].", nameof(destination));
            }

            return TryGetWindowCore(destination.AsSpan(), out windowEnd);
        }

        /// <summary>Resets the buffer — after model reload or scraping interruption.</summary>
        public void Reset()
        {
            ThrowIfDisposed();

            lock (_lock)
            {
                _head = 0;
                _count = 0;
                _samplesUntilStep = _stepSize;

                _ring.Clear();
            }

            Interlocked.Exchange(ref _windowsProduced, 0L);
        }

        // -------------------------------------------------------------------------
        // Private
        // -------------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool TryGetWindowCore(Span<float> destination, out DateTime windowEnd)
        {
            lock (_lock)
            {
                if (_count < _windowSize || _samplesUntilStep > 0)
                {
                    windowEnd = default;
                    return false;
                }

                CopyWindowChronological(destination);

                var lastIndex = _head == 0 ? _windowSize - 1 : _head - 1;

                windowEnd = _timestamps[lastIndex];
                _samplesUntilStep = _stepSize;
            }

            Interlocked.Increment(ref _windowsProduced);

            return true;
        }

        /// <summary>
        /// Copies the window in chronological order (oldest sample = row 0).
        /// Maximum 2 Span.CopyTo on ring wrap-around — no loops.
        /// Called inside _lock.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void CopyWindowChronological(Span<float> destination)
        {
            var slotsToEnd = _windowSize - _head;
            var ringSpan = _ring.AsSpan();

            if (slotsToEnd >= _windowSize)
            {
                // No wrap-around — one continuous block
                ringSpan.Slice(_head * _featureCount, _windowFloats).CopyTo(destination);
            }
            else
            {
                // Wrap-around — two blocks
                var block1 = slotsToEnd * _featureCount;
                var block2 = _windowFloats - block1;

                ringSpan.Slice(_head * _featureCount, block1).CopyTo(destination);
                ringSpan.Slice(0, block2).CopyTo(destination.Slice(block1));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void AdvanceHead()
        {
            _head = (_head + 1) % _windowSize;

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