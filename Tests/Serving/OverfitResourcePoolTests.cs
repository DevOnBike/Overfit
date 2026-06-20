// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Concurrent;
using DevOnBike.Overfit.Serving;

namespace DevOnBike.Overfit.Tests.Serving
{
    /// <summary>
    /// The session-per-request pool primitive: exclusive checkout, load-shedding when saturated, cancellation,
    /// metrics, and disposal of pooled items. Model-free and deterministic.
    /// </summary>
    public sealed class OverfitResourcePoolTests
    {
        private sealed class Tracked : IDisposable
        {
            public int Id;
            public bool Disposed;
            public void Dispose() => Disposed = true;
        }

        private static OverfitResourcePool<Tracked> PoolOf(int size)
        {
            var items = new List<Tracked>(size);
            for (var i = 0; i < size; i++)
            {
                items.Add(new Tracked { Id = i });
            }
            return new OverfitResourcePool<Tracked>(items);
        }

        [Fact]
        public void RentAndReturn_TracksAvailability()
        {
            using var pool = PoolOf(2);
            Assert.Equal(2, pool.Size);
            Assert.Equal(2, pool.AvailableCount);

            Assert.True(pool.TryRent(TimeSpan.Zero, default, out var a));
            Assert.True(pool.TryRent(TimeSpan.Zero, default, out var b));
            Assert.Equal(0, pool.AvailableCount);
            Assert.Equal(2, pool.ActiveCount);
            Assert.NotSame(a.Value, b.Value);   // distinct items checked out

            a.Dispose();
            Assert.Equal(1, pool.AvailableCount);
            b.Dispose();
            Assert.Equal(2, pool.AvailableCount);
            Assert.Equal(0, pool.ActiveCount);
        }

        [Fact]
        public void TryRent_WhenExhausted_RejectsAndCountsIt()
        {
            using var pool = PoolOf(1);
            Assert.True(pool.TryRent(TimeSpan.Zero, default, out var held));

            // No item free → reject after the (zero) timeout.
            Assert.False(pool.TryRent(TimeSpan.Zero, default, out _));
            Assert.Equal(1, pool.Metrics.TotalRejected);

            held.Dispose();
            Assert.True(pool.TryRent(TimeSpan.Zero, default, out var again));   // freed → succeeds
            again.Dispose();
            Assert.Equal(1, pool.Metrics.TotalRejected);                        // no new rejection
        }

        [Fact]
        public void TryRent_BlocksUntilReturned()
        {
            using var pool = PoolOf(1);
            Assert.True(pool.TryRent(TimeSpan.Zero, default, out var held));

            var gotIt = false;
            var waiter = Task.Run(() =>
            {
                gotIt = pool.TryRent(TimeSpan.FromSeconds(5), default, out var l);
                l.Dispose();
            });

            Assert.False(waiter.Wait(200));   // still blocked — no slot
            held.Dispose();                    // free it
            Assert.True(waiter.Wait(5000));    // waiter now proceeds
            Assert.True(gotIt);
        }

        [Fact]
        public void TryRent_Cancelled_ThrowsAndIsNotARejection()
        {
            using var pool = PoolOf(1);
            Assert.True(pool.TryRent(TimeSpan.Zero, default, out var held));

            using var cts = new CancellationTokenSource();
            cts.Cancel();
            Assert.Throws<OperationCanceledException>(() => pool.TryRent(TimeSpan.FromSeconds(5), cts.Token, out _));
            Assert.Equal(0, pool.Metrics.TotalRejected);   // a disconnect is not overload

            held.Dispose();
        }

        [Fact]
        public void Metrics_ReportRentedAndPeak()
        {
            using var pool = PoolOf(3);
            Assert.True(pool.TryRent(TimeSpan.Zero, default, out var a));
            Assert.True(pool.TryRent(TimeSpan.Zero, default, out var b));
            a.Dispose();
            Assert.True(pool.TryRent(TimeSpan.Zero, default, out var c));

            var m = pool.Metrics;
            Assert.Equal(3, m.TotalRented);
            Assert.Equal(2, m.PeakActive);   // a+b were concurrent before a was returned
            Assert.Equal(2, m.Active);       // b and c are held now (a was returned)

            b.Dispose();
            c.Dispose();
        }

        [Fact]
        public void Dispose_DisposesPooledItems()
        {
            var items = new List<Tracked> { new() { Id = 0 }, new() { Id = 1 } };
            var pool = new OverfitResourcePool<Tracked>(items);
            pool.Dispose();
            Assert.All(items, t => Assert.True(t.Disposed));
        }

        [Fact]
        public void Concurrent_RentReturn_NeverDoubleRentsOrExceedsSize()
        {
            const int size = 4;
            using var pool = PoolOf(size);
            var inUse = new ConcurrentDictionary<int, byte>();
            var maxActive = 0;
            var violation = false;

            Parallel.For(0, 2000, _ =>
            {
                if (!pool.TryRent(TimeSpan.FromSeconds(5), default, out var lease))
                {
                    return;
                }
                using (lease)
                {
                    // Exclusive ownership: this item must not already be held.
                    if (!inUse.TryAdd(lease.Value.Id, 0))
                    {
                        violation = true;
                    }
                    var active = pool.ActiveCount;
                    if (active > Volatile.Read(ref maxActive))
                    {
                        Interlocked.Exchange(ref maxActive, active);
                    }
                    if (active > size)
                    {
                        violation = true;
                    }
                    inUse.TryRemove(lease.Value.Id, out byte _);
                }
            });

            Assert.False(violation);
            Assert.True(maxActive <= size);
            Assert.Equal(size, pool.AvailableCount);   // everything returned
        }
    }
}
