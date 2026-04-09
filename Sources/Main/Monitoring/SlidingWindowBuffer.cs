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
        // --- pamięć (alokowana raz w konstruktorze z ArrayPool przez FastBuffer) ---
        private readonly FastBuffer<float> _ring; // płaski: windowSize × featureCount
        private readonly FastBuffer<DateTime> _timestamps; // jeden timestamp per slot
        private readonly int _windowSize;
        private readonly int _stepSize;
        private readonly int _featureCount;
        private readonly int _windowFloats; // = windowSize × featureCount

        // --- stan ---
        private readonly Lock _lock = new(); // System.Threading.Lock — bez Monitor overhead
        private int _head; // indeks slotu następnego zapisu
        private int _count; // liczba wypełnionych slotów (max = windowSize)
        private int _samplesUntilStep; // ile próbek do następnego okna
        private bool _disposed;

        // --- diagnostyka (Interlocked — nie wymagają _lock) ---
        private long _windowsProduced;
        private long _samplesAdded;

        public int WindowSize => _windowSize;
        public int StepSize => _stepSize;
        public int FeatureCount => _featureCount;
        public int WindowFloats => _windowFloats;
        public long WindowsProduced => Volatile.Read(ref _windowsProduced);
        public long SamplesAdded => Volatile.Read(ref _samplesAdded);

        /// <summary>Ile próbek brakuje do zapełnienia pierwszego okna.</summary>
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

        /// <param name="windowSize">Liczba próbek w oknie.</param>
        /// <param name="stepSize">Co ile próbek produkujemy nowe okno. Musi być ≤ windowSize.</param>
        /// <param name="featureCount">Liczba cech — stała przez cały cykl życia bufora.</param>
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
        /// Dodaje próbkę z <see cref="MetricSnapshot"/>.
        /// Zero alokacji — WriteFeatureVector zapisuje bezpośrednio do slotu pierścienia.
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
        /// Dodaje ręcznie skonstruowany wektor cech.
        /// Używaj przy testach lub przy replay historycznych danych z CSV/Parquet.
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
        /// Kopiuje gotowe okno do flat <paramref name="destination"/> (row-major, chronologicznie).
        ///
        /// Preferowany overload gdy destination to <c>fastTensor.AsSpan()</c> —
        /// kopiuje bezpośrednio do pamięci tensora, zero pośrednich buforów.
        /// </summary>
        /// <param name="destination">Caller-owned, min. <see cref="WindowFloats"/> elementów.</param>
        /// <param name="windowEnd">Timestamp najnowszej próbki w oknie.</param>
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
        /// Overload dla <see cref="FastMatrix{T}"/> — prealokowana macierz [windowSize × featureCount].
        /// Po wypełnieniu caller ma dostęp przez <c>matrix.Row(i)</c> w FeatureExtractor.
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

        /// <summary>Resetuje bufor — po przeładowaniu modelu lub przerwie w scrapingu.</summary>
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
        // Prywatne
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
        /// Kopiuje okno w kolejności chronologicznej (najstarsza próbka = wiersz 0).
        /// Maksymalnie 2 Span.CopyTo przy zawinięciu pierścienia — bez pętli.
        /// Wywoływana wewnątrz _lock.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void CopyWindowChronological(Span<float> destination)
        {
            var slotsToEnd = _windowSize - _head;
            var ringSpan = _ring.AsSpan();

            if (slotsToEnd >= _windowSize)
            {
                // Brak zawinięcia — jeden ciągły blok
                ringSpan.Slice(_head * _featureCount, _windowFloats).CopyTo(destination);
            }
            else
            {
                // Zawinięcie — dwa bloki
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