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
        // OVERFIT040: THE RULE IS RIGHT HERE, AND THIS SUPPRESSION RECORDS A DECISION, NOT A DEFERRAL.
        //
        // `_slots.Wait(timeout, cancellationToken)` really does block the calling thread for up to `timeout`,
        // and the caller that matters is a request thread: `OverfitInferenceService.CompleteChat` passes
        // `RentTimeout = TimeSpan.FromSeconds(30)`, and the CLI's `--sessions` defaults to 1
        // (`Cli/Program.cs` `DefaultValueFactory = _ => 1`, `Cli/Commands.cs` `Serve(..., int sessions = 1)`),
        // so at any concurrency above one every request but one waits here, each on its own request thread.
        // Nothing about this site is fine.
        //
        // CORRECTION (XC-26, 2026-08-12): this block used to call that "the largest held thread in the
        // server". It is not — it is the largest wait BEFORE any work starts. The same thread is then held
        // for the generation as well, and for as long as that takes:
        // `Server.AspNet/AspNetResponseSink.cs` records "the request thread that entered the endpoint is
        // held for the whole generation. That is the server's design."
        //
        // WHY IT IS NOT FIXED HERE — DECIDED under XC-26 on 2026-08-12, not deferred. The fix would be a
        // `TryRentAsync` on this type, and it would change which resource is held while waiting (a
        // continuation instead of a thread), not the wait itself: the number of concurrent completions is
        // bounded by `Size` either way, and the request thread is held for the whole generation regardless.
        // What it would buy is isolating unrelated endpoints from a chat burst — and that has never been
        // observed on this server, so it is a hypothesis. Against it stands a permanent cost: `out lease`
        // cannot cross an `await`, so the shape would have to change, and this is public API of the shipped
        // `DevOnBike.Overfit` package — it would be the library's first asynchronous primitive, and public
        // API cannot be withdrawn. Deciding not to add it can be.
        //
        // WHAT WOULD REOPEN IT, as a condition rather than a promise: a chat burst at `--sessions 1` that
        // measurably delays an unrelated endpoint (`GET /v1/models`, `GET /metrics`) while
        // `dotnet_threadpool_queue_length` — which the server already exports — rises. That measurement has
        // not been run, here or anywhere; any statement about what an asynchronous rent would gain is a
        // performance claim and belongs to a measurement, not to this comment.
        //
        // WORTH NOTING, because it is why XC-26 exists at all: the rule reaches this method only because
        // `SemaphoreSlim.Wait` has a `WaitAsync`. It CANNOT reach `CompleteChat`, whose block is the same
        // block one frame up, because `TryRent` has no async sibling for it to match on. CORRECTION
        // (2026-08-13): this used to say the analyzer is silent at "the more expensive of the two sites",
        // which nothing supports — it is the SAME block seen one frame up, so the two are not cheaper and
        // dearer. What is true is worse: the rule is silent exactly where a fix would have to be made,
        // since `CompleteChat` is the method that would have to become task-returning, and loud here, where
        // the wait cannot be fixed alone. So this diagnostic is the only automated signal pointing at the
        // whole shape. Removing it by pragma without saying so would delete that signal.
#pragma warning disable OVERFIT040
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
#pragma warning restore OVERFIT040

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
