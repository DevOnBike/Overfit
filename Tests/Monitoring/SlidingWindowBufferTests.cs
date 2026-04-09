// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Threading.Channels;
using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Monitoring;

namespace DevOnBike.Overfit.Tests.Monitoring
{
    public sealed class SlidingWindowBufferTests
    {
        // -------------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------------

        private static SlidingWindowBuffer Make(int windowSize, int stepSize = 1, int featureCount = 2)
            => new(windowSize, stepSize, featureCount);

        private static float[] Dst(SlidingWindowBuffer buf) => new float[buf.WindowFloats];

        /// Adds a feature vector using the current UTC timestamp.
        private static void Push(SlidingWindowBuffer buf, params float[] values)
            => buf.Add(values.AsSpan(), DateTime.UtcNow);

        // -------------------------------------------------------------------------
        // Constructor
        // -------------------------------------------------------------------------

        [Fact]
        public void Constructor_WhenWindowSizeIsZero_ThenThrowsArgumentOutOfRange()
            => Assert.Throws<ArgumentOutOfRangeException>(() => Make(windowSize: 0));

        [Fact]
        public void Constructor_WhenStepSizeIsZero_ThenThrowsArgumentOutOfRange()
            => Assert.Throws<ArgumentOutOfRangeException>(() => Make(windowSize: 3, stepSize: 0));

        [Fact]
        public void Constructor_WhenFeatureCountIsZero_ThenThrowsArgumentOutOfRange()
            => Assert.Throws<ArgumentOutOfRangeException>(
            () => new SlidingWindowBuffer(windowSize: 3, featureCount: 0));

        [Fact]
        public void Constructor_WhenStepSizeExceedsWindowSize_ThenThrowsArgumentOutOfRange()
            => Assert.Throws<ArgumentOutOfRangeException>(
            () => Make(windowSize: 3, stepSize: 4));

        [Fact]
        public void Constructor_WhenValidArgs_ThenWindowFloatsEqualsWindowSizeTimesFeatureCount()
        {
            using var buf = Make(windowSize: 6, featureCount: 8);
            Assert.Equal(48, buf.WindowFloats);
        }

        [Fact]
        public void Constructor_WhenCreated_ThenSamplesUntilFirstWindowEqualsWindowSize()
        {
            using var buf = Make(windowSize: 5, featureCount: 1);
            Assert.Equal(5, buf.SamplesUntilFirstWindow);
        }

        // -------------------------------------------------------------------------
        // Add — argument validation
        // -------------------------------------------------------------------------

        [Fact]
        public void Add_WhenFeatureVectorTooShort_ThenThrowsArgumentException()
        {
            using var buf = Make(windowSize: 2, featureCount: 3);
            Assert.Throws<ArgumentException>(
            () => buf.Add(stackalloc float[]
            {
                1f, 2f
            }, DateTime.UtcNow));
        }

        [Fact]
        public void Add_WhenFeatureVectorTooLong_ThenThrowsArgumentException()
        {
            using var buf = Make(windowSize: 2, featureCount: 2);
            Assert.Throws<ArgumentException>(
            () => buf.Add(stackalloc float[]
            {
                1f, 2f, 3f
            }, DateTime.UtcNow));
        }

        [Fact]
        public void Add_WhenDisposed_ThenThrowsObjectDisposedException()
        {
            var buf = Make(windowSize: 2);
            buf.Dispose();
            Assert.Throws<ObjectDisposedException>(() => Push(buf, 1f, 2f));
        }

        // -------------------------------------------------------------------------
        // Add — counters
        // -------------------------------------------------------------------------

        [Fact]
        public void Add_WhenCalledOnce_ThenSamplesAddedIsOne()
        {
            using var buf = Make(windowSize: 3, featureCount: 1);
            Push(buf, 1f);
            Assert.Equal(1L, buf.SamplesAdded);
        }

        [Fact]
        public void Add_WhenCalledMultipleTimes_ThenSamplesAddedMatchesCallCount()
        {
            using var buf = Make(windowSize: 3, featureCount: 1);
            Push(buf, 1f);
            Push(buf, 2f);
            Push(buf, 3f);
            Assert.Equal(3L, buf.SamplesAdded);
        }

        [Theory]
        [InlineData(5, 0, 5)]
        [InlineData(5, 1, 4)]
        [InlineData(5, 4, 1)]
        [InlineData(5, 5, 0)]
        public void Add_WhenNSamplesAdded_ThenSamplesUntilFirstWindowDecrementsCorrectly(
            int windowSize, int added, int expected)
        {
            using var buf = Make(windowSize, featureCount: 1);
            for (var i = 0; i < added; i++) { Push(buf, (float)i); }
            Assert.Equal(expected, buf.SamplesUntilFirstWindow);
        }

        // -------------------------------------------------------------------------
        // TryGetWindow(Span) — readiness
        // -------------------------------------------------------------------------

        [Fact]
        public void TryGetWindow_WhenBufferNotYetFull_ThenReturnsFalse()
        {
            using var buf = Make(windowSize: 3, featureCount: 1);
            var dst = Dst(buf);

            Push(buf, 1f);
            Push(buf, 2f);

            Assert.False(buf.TryGetWindow(dst, out _));
        }

        [Fact]
        public void TryGetWindow_WhenBufferExactlyFull_ThenReturnsTrue()
        {
            using var buf = Make(windowSize: 3, featureCount: 1);
            var dst = Dst(buf);

            Push(buf, 1f);
            Push(buf, 2f);
            Push(buf, 3f);

            Assert.True(buf.TryGetWindow(dst, out _));
        }

        [Fact]
        public void TryGetWindow_WhenCalledTwiceWithoutNewAdd_ThenSecondCallReturnsFalse()
        {
            using var buf = Make(windowSize: 2, featureCount: 1);
            var dst = Dst(buf);

            Push(buf, 1f);
            Push(buf, 2f);
            buf.TryGetWindow(dst, out _);

            Assert.False(buf.TryGetWindow(dst, out _));
        }

        [Fact]
        public void TryGetWindow_WhenWindowConsumedAndOneMoreAdded_ThenReturnsTrue()
        {
            using var buf = Make(windowSize: 2, featureCount: 1);
            var dst = Dst(buf);

            Push(buf, 1f);
            Push(buf, 2f);
            buf.TryGetWindow(dst, out _);

            Push(buf, 3f);

            Assert.True(buf.TryGetWindow(dst, out _));
        }

        // -------------------------------------------------------------------------
        // TryGetWindow(Span) — argument validation
        // -------------------------------------------------------------------------

        [Fact]
        public void TryGetWindow_WhenDestinationTooShort_ThenThrowsArgumentException()
        {
            using var buf = Make(windowSize: 3, featureCount: 2);
            var dst = new float[5]; // needs 6
            Assert.Throws<ArgumentException>(() => buf.TryGetWindow(dst, out _));
        }

        [Fact]
        public void TryGetWindow_WhenDisposed_ThenThrowsObjectDisposedException()
        {
            var buf = Make(windowSize: 2);
            var dst = Dst(buf);
            buf.Dispose();
            Assert.Throws<ObjectDisposedException>(() => buf.TryGetWindow(dst, out _));
        }

        // -------------------------------------------------------------------------
        // TryGetWindow(Span) — WindowsProduced counter
        // -------------------------------------------------------------------------

        [Fact]
        public void TryGetWindow_WhenFirstWindowProduced_ThenWindowsProducedIsOne()
        {
            using var buf = Make(windowSize: 2, featureCount: 1);
            var dst = Dst(buf);

            Push(buf, 1f);
            Push(buf, 2f);
            buf.TryGetWindow(dst, out _);

            Assert.Equal(1L, buf.WindowsProduced);
        }

        [Fact]
        public void TryGetWindow_WhenReturnsFalse_ThenWindowsProducedDoesNotIncrement()
        {
            using var buf = Make(windowSize: 3, featureCount: 1);
            var dst = Dst(buf);

            Push(buf, 1f);
            Push(buf, 2f);
            buf.TryGetWindow(dst, out _); // false

            Assert.Equal(0L, buf.WindowsProduced);
        }

        // -------------------------------------------------------------------------
        // TryGetWindow(Span) — WindowEnd timestamp
        // -------------------------------------------------------------------------

        [Fact]
        public void TryGetWindow_WhenWindowReady_ThenWindowEndEqualsTimestampOfLastSample()
        {
            using var buf = Make(windowSize: 2, featureCount: 1);
            var dst = Dst(buf);
            var t1 = new DateTime(2026, 1, 1, 12, 0, 0, DateTimeKind.Utc);
            var t2 = new DateTime(2026, 1, 1, 12, 0, 10, DateTimeKind.Utc);

            buf.Add(stackalloc float[]
            {
                1f
            }, t1);
            buf.Add(stackalloc float[]
            {
                2f
            }, t2);
            buf.TryGetWindow(dst, out var windowEnd);

            Assert.Equal(t2, windowEnd);
        }

        [Fact]
        public void TryGetWindow_WhenNotReady_ThenWindowEndIsDefault()
        {
            using var buf = Make(windowSize: 3, featureCount: 1);
            var dst = Dst(buf);

            Push(buf, 1f);
            buf.TryGetWindow(dst, out var windowEnd);

            Assert.Equal(default, windowEnd);
        }

        // -------------------------------------------------------------------------
        // TryGetWindow(Span) — chronological order, no ring wrap
        // -------------------------------------------------------------------------

        [Fact]
        public void TryGetWindow_WhenNoRingWrap_ThenDataIsChronologicalOldestFirst()
        {
            using var buf = Make(windowSize: 3, featureCount: 1);
            var dst = Dst(buf);

            Push(buf, 10f);
            Push(buf, 20f);
            Push(buf, 30f);
            buf.TryGetWindow(dst, out _);

            Assert.Equal(10f, dst[0]);
            Assert.Equal(20f, dst[1]);
            Assert.Equal(30f, dst[2]);
        }

        [Fact]
        public void TryGetWindow_WhenMultipleFeatures_ThenFeaturesAreNotCrossContaminated()
        {
            using var buf = Make(windowSize: 2, featureCount: 3);
            var dst = Dst(buf);

            Push(buf, 1f, 2f, 3f);
            Push(buf, 4f, 5f, 6f);
            buf.TryGetWindow(dst, out _);

            // row 0 = oldest sample
            Assert.Equal(1f, dst[0]);
            Assert.Equal(2f, dst[1]);
            Assert.Equal(3f, dst[2]);
            // row 1 = newest sample
            Assert.Equal(4f, dst[3]);
            Assert.Equal(5f, dst[4]);
            Assert.Equal(6f, dst[5]);
        }

        // -------------------------------------------------------------------------
        // TryGetWindow(Span) — chronological order, ring wrap
        // -------------------------------------------------------------------------

        [Fact]
        public void TryGetWindow_WhenRingWrapsOnce_ThenDataRemainsChronological()
        {
            using var buf = Make(windowSize: 3, featureCount: 1);
            var dst = Dst(buf);

            Push(buf, 10f);
            Push(buf, 20f);
            Push(buf, 30f);
            buf.TryGetWindow(dst, out _); // consume [10,20,30]

            Push(buf, 40f); // head=1, ring wraps
            buf.TryGetWindow(dst, out _);

            Assert.Equal(20f, dst[0]);
            Assert.Equal(30f, dst[1]);
            Assert.Equal(40f, dst[2]);
        }

        [Fact]
        public void TryGetWindow_WhenRingWrapsMultipleTimes_ThenNewestSampleIsAlwaysLast()
        {
            using var buf = Make(windowSize: 4, featureCount: 1);
            var dst = Dst(buf);

            for (var v = 1f; v <= 20f; v++)
            {
                Push(buf, v);
                if (!buf.TryGetWindow(dst, out _)) { continue; }

                Assert.Equal(v, dst[3]); // newest always last
                Assert.True(dst[0] < dst[1] && dst[1] < dst[2] && dst[2] < dst[3]);
            }
        }

        [Fact]
        public void TryGetWindow_WhenWindowSize1_ThenEachAddProducesItsOwnWindow()
        {
            using var buf = Make(windowSize: 1, featureCount: 2);
            var dst = Dst(buf);

            Push(buf, 7f, 8f);
            Assert.True(buf.TryGetWindow(dst, out _));
            Assert.Equal(7f, dst[0]);
            Assert.Equal(8f, dst[1]);

            Push(buf, 9f, 10f);
            Assert.True(buf.TryGetWindow(dst, out _));
            Assert.Equal(9f, dst[0]);
            Assert.Equal(10f, dst[1]);
        }

        // -------------------------------------------------------------------------
        // TryGetWindow(Span) — step size
        // -------------------------------------------------------------------------

        [Theory]
        [InlineData(3, 1, 10, 8L)] // step=1: window every sample after full
        [InlineData(4, 2, 10, 4L)] // step=2: window every 2nd sample
        [InlineData(3, 3, 9, 3L)] // step=windowSize: no overlap
        [InlineData(6, 1, 20, 15L)] // production defaults (60s window)
        [InlineData(1, 1, 5, 5L)] // windowSize=1: every sample
        public void TryGetWindow_WhenNSamplesAdded_ThenWindowCountMatchesExpected(
            int windowSize, int stepSize, int samples, long expectedWindows)
        {
            using var buf = new SlidingWindowBuffer(windowSize, stepSize, featureCount: 1);
            var dst = Dst(buf);
            var count = 0L;

            for (var i = 0; i < samples; i++)
            {
                Push(buf, (float)i);
                if (buf.TryGetWindow(dst, out _)) { count++; }
            }

            Assert.Equal(expectedWindows, count);
            Assert.Equal(expectedWindows, buf.WindowsProduced);
        }

        [Fact]
        public void TryGetWindow_WhenStepSize2_ThenFalseAfterFirstExtraSampleThenTrueAfterSecond()
        {
            using var buf = Make(windowSize: 4, stepSize: 2, featureCount: 1);
            var dst = Dst(buf);

            Push(buf, 1f);
            Push(buf, 2f);
            Push(buf, 3f);
            Push(buf, 4f);
            buf.TryGetWindow(dst, out _); // consume first window

            Push(buf, 5f);
            Assert.False(buf.TryGetWindow(dst, out _));

            Push(buf, 6f);
            Assert.True(buf.TryGetWindow(dst, out _));
        }

        [Fact]
        public void TryGetWindow_WhenStepEqualsWindowSize_ThenWindowsDoNotOverlap()
        {
            using var buf = Make(windowSize: 3, stepSize: 3, featureCount: 1);
            var dst = Dst(buf);

            Push(buf, 1f);
            Push(buf, 2f);
            Push(buf, 3f);
            buf.TryGetWindow(dst, out _);
            Assert.Equal(1f, dst[0]);
            Assert.Equal(3f, dst[2]);

            Push(buf, 4f);
            Push(buf, 5f);
            Push(buf, 6f);
            buf.TryGetWindow(dst, out _);
            Assert.Equal(4f, dst[0]);
            Assert.Equal(6f, dst[2]);
        }

        // -------------------------------------------------------------------------
        // TryGetWindow(FastMatrix) — readiness & shape
        // -------------------------------------------------------------------------

        [Fact]
        public void TryGetWindowMatrix_WhenBufferFull_ThenReturnsTrue()
        {
            using var buf = Make(windowSize: 3);
            using var mat = new FastMatrix<float>(3, 2);

            Push(buf, 1f, 2f);
            Push(buf, 3f, 4f);
            Push(buf, 5f, 6f);

            Assert.True(buf.TryGetWindow(mat, out _));
        }

        [Fact]
        public void TryGetWindowMatrix_WhenMatrixIsNull_ThenThrowsArgumentNullException()
        {
            using var buf = Make(windowSize: 3);
            Assert.Throws<ArgumentNullException>(
            () => buf.TryGetWindow((FastMatrix<float>)null!, out _));
        }

        [Theory]
        [InlineData(2, 2)] // wrong rows
        [InlineData(3, 5)] // wrong cols
        [InlineData(1, 1)] // both wrong
        public void TryGetWindowMatrix_WhenShapeMismatch_ThenThrowsArgumentException(int rows, int cols)
        {
            using var buf = Make(windowSize: 3, featureCount: 2);
            using var mat = new FastMatrix<float>(rows, cols);
            Assert.Throws<ArgumentException>(() => buf.TryGetWindow(mat, out _));
        }

        [Fact]
        public void TryGetWindowMatrix_WhenDisposed_ThenThrowsObjectDisposedException()
        {
            var buf = Make(windowSize: 2);
            using var mat = new FastMatrix<float>(2, 2);
            buf.Dispose();
            Assert.Throws<ObjectDisposedException>(() => buf.TryGetWindow(mat, out _));
        }

        // -------------------------------------------------------------------------
        // TryGetWindow(FastMatrix) — data correctness
        // -------------------------------------------------------------------------

        [Fact]
        public void TryGetWindowMatrix_WhenBufferFull_ThenRow0IsOldestAndLastRowIsNewest()
        {
            using var buf = Make(windowSize: 3, featureCount: 2);
            using var mat = new FastMatrix<float>(3, 2);

            Push(buf, 10f, 11f);
            Push(buf, 20f, 21f);
            Push(buf, 30f, 31f);
            buf.TryGetWindow(mat, out _);

            Assert.Equal(10f, mat.Row(0)[0]);
            Assert.Equal(11f, mat.Row(0)[1]);
            Assert.Equal(30f, mat.Row(2)[0]);
            Assert.Equal(31f, mat.Row(2)[1]);
        }

        [Fact]
        public void TryGetWindowMatrix_WhenRingWraps_ThenDataRemainsChronological()
        {
            using var buf = Make(windowSize: 3, featureCount: 1);
            using var mat = new FastMatrix<float>(3, 1);

            Push(buf, 1f);
            Push(buf, 2f);
            Push(buf, 3f);
            buf.TryGetWindow(mat, out _); // consume

            Push(buf, 4f);
            buf.TryGetWindow(mat, out _);

            Assert.Equal(2f, mat.Row(0)[0]);
            Assert.Equal(3f, mat.Row(1)[0]);
            Assert.Equal(4f, mat.Row(2)[0]);
        }

        [Fact]
        public void TryGetWindowMatrix_WhenComparedToSpanOverload_ThenDataIsIdentical()
        {
            using var buf1 = Make(windowSize: 3, featureCount: 2);
            using var buf2 = Make(windowSize: 3, featureCount: 2);
            using var mat = new FastMatrix<float>(3, 2);

            Push(buf1, 1f, 2f);
            Push(buf1, 3f, 4f);
            Push(buf1, 5f, 6f);
            Push(buf2, 1f, 2f);
            Push(buf2, 3f, 4f);
            Push(buf2, 5f, 6f);

            var spanDst = Dst(buf1);
            buf1.TryGetWindow(spanDst, out _);
            buf2.TryGetWindow(mat, out _);

            Assert.Equal(spanDst, mat.AsSpan().ToArray());
        }

        // -------------------------------------------------------------------------
        // Reset
        // -------------------------------------------------------------------------

        [Fact]
        public void Reset_WhenCalled_ThenSamplesUntilFirstWindowResetsToWindowSize()
        {
            using var buf = Make(windowSize: 3, featureCount: 1);
            Push(buf, 1f);
            Push(buf, 2f);
            Push(buf, 3f);
            buf.Reset();
            Assert.Equal(3, buf.SamplesUntilFirstWindow);
        }

        [Fact]
        public void Reset_WhenCalled_ThenWindowsProducedResetsToZero()
        {
            using var buf = Make(windowSize: 2, featureCount: 1);
            var dst = Dst(buf);
            Push(buf, 1f);
            Push(buf, 2f);
            buf.TryGetWindow(dst, out _);

            buf.Reset();

            Assert.Equal(0L, buf.WindowsProduced);
        }

        [Fact]
        public void Reset_WhenCalled_ThenSamplesAddedIsNotReset()
        {
            using var buf = Make(windowSize: 2, featureCount: 1);
            Push(buf, 1f);
            Push(buf, 2f);
            buf.Reset();
            Assert.Equal(2L, buf.SamplesAdded);
        }

        [Fact]
        public void Reset_WhenCalled_ThenBufferAcceptsNewDataWithoutStaleValues()
        {
            using var buf = Make(windowSize: 2, featureCount: 2);
            var dst = Dst(buf);

            Push(buf, 99f, 99f);
            Push(buf, 99f, 99f);
            buf.TryGetWindow(dst, out _);
            buf.Reset();

            Push(buf, 1f, 2f);
            Push(buf, 3f, 4f);
            buf.TryGetWindow(dst, out _);

            Assert.Equal(1f, dst[0]);
            Assert.Equal(4f, dst[3]);
        }

        [Fact]
        public void Reset_WhenBufferWasPartiallyFilled_ThenNextWindowRequiresFullRefill()
        {
            using var buf = Make(windowSize: 3, featureCount: 1);
            var dst = Dst(buf);

            Push(buf, 1f);
            Push(buf, 2f); // partial fill
            buf.Reset();

            Push(buf, 7f);
            Push(buf, 8f);
            Assert.False(buf.TryGetWindow(dst, out _));

            Push(buf, 9f);
            Assert.True(buf.TryGetWindow(dst, out _));
            Assert.Equal(7f, dst[0]);
            Assert.Equal(9f, dst[2]);
        }

        [Fact]
        public void Reset_WhenDisposed_ThenThrowsObjectDisposedException()
        {
            var buf = Make(windowSize: 2);
            buf.Dispose();
            Assert.Throws<ObjectDisposedException>(() => buf.Reset());
        }

        // -------------------------------------------------------------------------
        // Dispose
        // -------------------------------------------------------------------------

        [Fact]
        public void Dispose_WhenCalledOnce_ThenDoesNotThrow()
        {
            var buf = Make(windowSize: 3);
            buf.Dispose(); // must not throw
        }

        [Fact]
        public void Dispose_WhenCalledTwice_ThenIsIdempotent()
        {
            var buf = Make(windowSize: 3);
            buf.Dispose();
            buf.Dispose(); // must not throw
        }

        // -------------------------------------------------------------------------
        // Concurrency — SamplesAdded atomicity
        // -------------------------------------------------------------------------

        [Fact]
        public void Add_WhenCalledConcurrentlyFromMultipleThreads_ThenSamplesAddedIsConsistent()
        {
            const int threads = 8;
            const int perThread = 500;

            using var buf = Make(windowSize: 4, featureCount: 1);
            using var start = new ManualResetEventSlim(false);
            var pool = new Thread[threads];

            for (var t = 0; t < threads; t++)
            {
                pool[t] = new Thread(() => {
                    start.Wait();
                    for (var i = 0; i < perThread; i++)
                    {
                        buf.Add(stackalloc float[]
                        {
                            1f
                        }, DateTime.UtcNow);
                    }
                });
                pool[t].Start();
            }

            start.Set();
            foreach (var t in pool) { t.Join(); }

            Assert.Equal((long)threads * perThread, buf.SamplesAdded);
        }

        [Fact]
        public async Task Add_WhenConcurrentProducers_ThenSamplesAddedIsConsistent()
        {
            const int producers = 4;
            const int perTask = 1_000;

            using var buf = Make(windowSize: 6, featureCount: 2);
            var barrier = new Barrier(producers);

            var tasks = Enumerable.Range(0, producers).Select(_ => Task.Run(() => {
                barrier.SignalAndWait();
                for (var i = 0; i < perTask; i++)
                {
                    buf.Add(stackalloc float[]
                    {
                        i * 1f, i * 2f
                    }, DateTime.UtcNow);
                }
            })).ToArray();

            await Task.WhenAll(tasks);

            Assert.Equal((long)producers * perTask, buf.SamplesAdded);
        }

        // -------------------------------------------------------------------------
        // Concurrency — producer → Channel → consumer pattern
        // -------------------------------------------------------------------------

        [Fact]
        public async Task TryGetWindow_WhenProducerPushesViaBoundedChannel_ThenAllWindowsDelivered()
        {
            const int samples = 100;
            const int windowSz = 6;
            var expectedWindows = (long)(samples - (windowSz - 1));

            using var buf = new SlidingWindowBuffer(windowSz, stepSize: 1, featureCount: 2);
            var channel = Channel.CreateUnbounded<float[]>();
            var received = 0L;

            var consumer = Task.Run(async () => {
                await foreach (var _ in channel.Reader.ReadAllAsync())
                {
                    Interlocked.Increment(ref received);
                }
            });

            var dst = Dst(buf);
            for (var i = 0; i < samples; i++)
            {
                buf.Add(stackalloc float[]
                {
                    i * 1f, i * 2f
                }, DateTime.UtcNow);
                if (buf.TryGetWindow(dst, out _))
                {
                    channel.Writer.TryWrite(dst.ToArray());
                }
            }
            channel.Writer.Complete();

            await consumer.WaitAsync(TimeSpan.FromSeconds(5));

            Assert.Equal(expectedWindows, buf.WindowsProduced);
            Assert.Equal(expectedWindows, Volatile.Read(ref received));
        }
    }
}