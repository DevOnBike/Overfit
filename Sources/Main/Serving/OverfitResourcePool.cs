// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Concurrent;
using System.Diagnostics;

namespace DevOnBike.Overfit.Serving
{
    /// <summary>
    /// A fixed-size, thread-safe checkout pool — the primitive behind a session-per-request server. A request
    /// rents one item (e.g. an <c>OverfitClient</c> with its own KV cache), uses it exclusively, and returns it.
    /// Concurrency is bounded by the pool size; callers beyond that wait up to a timeout and are otherwise
    /// rejected (so an overloaded server sheds load instead of unbounded queueing). The model weights are shared
    /// (memory-mapped) across the pooled clients; only the per-session scratch + KV cache is per-item, which is
    /// exactly why decode must NOT run concurrently on one engine but is safe across independent ones.
    ///
    /// <para>Rent via <see cref="TryRent"/> and dispose the returned <see cref="Lease"/> (a <c>using</c>) to
    /// return the item. <see cref="Metrics"/> exposes load: active/available, total rented/rejected, peak
    /// concurrency, and mean queue-wait.</para>
    /// </summary>
    public sealed class OverfitResourcePool<T> : IDisposable
    {
        private readonly T[] _items;
        private readonly ConcurrentBag<T> _available;
        private readonly SemaphoreSlim _slots;

        private readonly bool _ownsItems;
        private long _totalRented;
        private long _totalRejected;
        private long _waitTicksTotal;
        private int _active;
        private int _peakActive;
        private bool _disposed;

        /// <summary>Creates a pool over <paramref name="items"/>. The item count is the maximum concurrency.
        /// When <paramref name="ownsItems"/> is true (default) each <see cref="IDisposable"/> item is disposed
        /// with the pool; pass false to wrap caller-owned items (e.g. a pool-of-1 around an externally-owned
        /// client) without taking over their lifetime.</summary>
        public OverfitResourcePool(IReadOnlyList<T> items, bool ownsItems = true)
        {
            ArgumentNullException.ThrowIfNull(items);
            if (items.Count == 0)
            {
                throw new ArgumentException("A resource pool needs at least one item.", nameof(items));
            }

            _ownsItems = ownsItems;

            _items = new T[items.Count];
            _available = [];
            for (var i = 0; i < items.Count; i++)
            {
                _items[i] = items[i];
                _available.Add(items[i]);
            }
            _slots = new SemaphoreSlim(items.Count, items.Count);
        }

        /// <summary>Maximum concurrent rentals (pool size).</summary>
        public int Size => _items.Length;

        /// <summary>Items currently checked out.</summary>
        public int ActiveCount => Volatile.Read(ref _active);

        /// <summary>Items available to rent right now.</summary>
        public int AvailableCount => _slots.CurrentCount;

        /// <summary>A point-in-time snapshot of pool load.</summary>
        public PoolMetrics Metrics
        {
            get
            {
                var rented = Interlocked.Read(ref _totalRented);
                var waitTicks = Interlocked.Read(ref _waitTicksTotal);
                var meanWaitMs = rented > 0
                    ? waitTicks * 1000.0 / Stopwatch.Frequency / rented
                    : 0d;
                return new PoolMetrics(
                    Size,
                    ActiveCount,
                    AvailableCount,
                    rented,
                    Interlocked.Read(ref _totalRejected),
                    Volatile.Read(ref _peakActive),
                    meanWaitMs);
            }
        }

        /// <summary>
        /// Rents one item, waiting up to <paramref name="timeout"/> for a free slot. Returns <c>true</c> with a
        /// <see cref="Lease"/> the caller must dispose; returns <c>false</c> (and counts a rejection) if no item
        /// frees up in time — the server maps that to a 503 / "busy". Honours <paramref name="cancellationToken"/>
        /// (a cancelled wait throws <see cref="OperationCanceledException"/> and is NOT counted as a rejection,
        /// so a client that disconnects before a slot opens doesn't look like overload).
        /// </summary>
        public bool TryRent(TimeSpan timeout, CancellationToken cancellationToken, out Lease lease)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            var start = Stopwatch.GetTimestamp();
            if (!_slots.Wait(timeout, cancellationToken))
            {
                Interlocked.Increment(ref _totalRejected);
                lease = default;
                return false;
            }

            Interlocked.Add(ref _waitTicksTotal, Stopwatch.GetTimestamp() - start);

            // A slot was acquired, so an item is guaranteed available.
            _available.TryTake(out var item);
            Interlocked.Increment(ref _totalRented);
            var active = Interlocked.Increment(ref _active);
            UpdatePeak(active);

            lease = new Lease(this, item!);
            return true;
        }

        private void UpdatePeak(int active)
        {
            int peak;
            while (active > (peak = Volatile.Read(ref _peakActive)))
            {
                if (Interlocked.CompareExchange(ref _peakActive, active, peak) == peak)
                {
                    break;
                }
            }
        }

        private void Return(T item)
        {
            _available.Add(item);
            Interlocked.Decrement(ref _active);
            _slots.Release();
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;

            if (_ownsItems)
            {
                foreach (var item in _items)
                {
                    if (item is IDisposable disposable)
                    {
                        disposable.Dispose();
                    }
                }
            }
            _slots.Dispose();
        }

        /// <summary>A checked-out item. Dispose (a <c>using</c>) returns it to the pool.</summary>
        public readonly struct Lease : IDisposable
        {
            private readonly OverfitResourcePool<T> _pool;

            internal Lease(OverfitResourcePool<T> pool, T value)
            {
                _pool = pool;
                Value = value;
            }

            /// <summary>The rented item — valid until this lease is disposed.</summary>
            public T Value
            {
                get;
            }

            public void Dispose() => _pool?.Return(Value);
        }
    }
}
