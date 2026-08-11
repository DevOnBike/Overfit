// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using DevOnBike.Overfit.Text;

namespace DevOnBike.Overfit.Tests.Core.Text
{
    /// <summary>
    /// <see cref="ValueStringBuilder"/> — the rewrite of the runtime's internal type over
    /// <c>PooledBuffer&lt;char&gt;</c>, because <c>ArrayPool&lt;char&gt;.Shared</c> is RS0030-banned here.
    ///
    /// <para>Every test below pins something whose failure is <b>silent</b>: a buffer used after it went
    /// back to the pool still reads, a stale span still renders, a lost growth still produces a string —
    /// just the wrong one, or somebody else's characters.</para>
    /// </summary>
    public sealed class ValueStringBuilderTests
    {
        [Fact]
        public void WritesWithinTheInitialBufferWithoutGrowing()
        {
            Span<char> scratch = stackalloc char[32];
            var text = new ValueStringBuilder(scratch);

            text.Append("id=");
            text.Append('7');

            Assert.Equal(4, text.Length);
            Assert.Equal("id=7", text.ToString());
        }

        /// <summary>
        /// <b>The case the type exists for.</b> Growth must move off the caller's stack buffer and keep
        /// everything already written — a growth that copies the wrong length loses a prefix and still
        /// returns a plausible-looking string.
        /// </summary>
        [Fact]
        public void GrowingPastTheInitialBufferKeepsEverythingWritten()
        {
            Span<char> scratch = stackalloc char[8];
            var text = new ValueStringBuilder(scratch);
            var expected = new string('x', 5) + new string('y', 200);

            text.Append(new string('x', 5));
            text.Append(new string('y', 200));

            Assert.Equal(expected, text.ToString());
        }

        /// <summary>Several growths in a row, so the pooled-to-pooled path is exercised and not only stack-to-pooled.</summary>
        [Fact]
        public void RepeatedGrowthIsLossless()
        {
            Span<char> scratch = stackalloc char[4];
            var text = new ValueStringBuilder(scratch);
            var expected = new System.Text.StringBuilder();

            for (var i = 0; i < 500; i++)
            {
                var piece = (i % 10).ToString(CultureInfo.InvariantCulture);

                text.Append(piece);
                expected.Append(piece);
            }

            Assert.Equal(expected.ToString(), text.ToString());
        }

        /// <summary>A single append larger than a doubling — capacity must follow the request, not the double.</summary>
        [Fact]
        public void OneAppendLargerThanADoublingFits()
        {
            Span<char> scratch = stackalloc char[4];
            var text = new ValueStringBuilder(scratch);
            var big = new string('z', 10_000);

            text.Append(big);

            Assert.Equal(big, text.ToString());
        }

        /// <summary>
        /// <b>The documented footgun, pinned so it cannot drift.</b> <c>ToString</c> disposes; the builder
        /// is <c>default</c> afterwards. If somebody "fixes" this to leave the buffer outstanding, every
        /// caller without a <c>using</c> starts leaking a rented array silently — so the behaviour is
        /// asserted rather than merely documented.
        /// </summary>
        [Fact]
        public void ToStringDisposesTheBuilder()
        {
            Span<char> scratch = stackalloc char[8];
            var text = new ValueStringBuilder(scratch);

            text.Append("abc");

            Assert.Equal("abc", text.ToString());
            Assert.Equal(0, text.Length);
            Assert.Equal(string.Empty, text.AsSpan().ToString());
        }

        [Fact]
        public void TryCopyToRefusesADestinationThatIsTooSmallAndWritesNothing()
        {
            Span<char> scratch = stackalloc char[16];
            var text = new ValueStringBuilder(scratch);

            text.Append("hello");

            Span<char> tooSmall = stackalloc char[4];
            tooSmall.Fill('#');

            Assert.False(text.TryCopyTo(tooSmall, out var written));
            Assert.Equal(0, written);

            // Nothing written, not a truncated prefix: a partly-filled buffer reporting failure is the
            // state a caller cannot recover from, because it cannot tell how far it got.
            Assert.Equal("####", tooSmall.ToString());

            text.Dispose();
        }

        /// <summary>
        /// <c>TryCopyTo</c> does NOT dispose — it is the path for callers who own their destination and may
        /// keep appending afterwards.
        /// </summary>
        [Fact]
        public void TryCopyToLeavesTheBuilderUsable()
        {
            Span<char> scratch = stackalloc char[16];
            var text = new ValueStringBuilder(scratch);

            text.Append("ab");

            Span<char> destination = stackalloc char[8];

            Assert.True(text.TryCopyTo(destination, out var written));
            Assert.Equal(2, written);

            text.Append("cd");

            Assert.Equal("abcd", text.ToString());
        }

        [Fact]
        public void AppendingNullIsANoOp()
        {
            Span<char> scratch = stackalloc char[8];
            var text = new ValueStringBuilder(scratch);

            text.Append("a");
            text.Append((string?)null);

            Assert.Equal("a", text.ToString());
        }

        /// <summary>
        /// Growth happens exactly at the boundary, not one character early or late — an off-by-one here
        /// either wastes a rental on every full buffer or writes one past the end.
        /// </summary>
        [Fact]
        public void TheInitialBufferIsFilledCompletelyBeforeGrowing()
        {
            Span<char> scratch = stackalloc char[4];
            var text = new ValueStringBuilder(scratch);

            text.Append("abcd");

            Assert.Equal(4, text.Length);

            text.Append('e');

            Assert.Equal("abcde", text.ToString());
        }

        /// <summary>Disposing twice is safe — <c>this = default</c> makes the second call a no-op.</summary>
        [Fact]
        public void DisposingTwiceIsSafe()
        {
            Span<char> scratch = stackalloc char[8];
            var text = new ValueStringBuilder(scratch);

            text.Append("x");
            text.Dispose();
            text.Dispose();

            Assert.Equal(0, text.Length);
        }
    }
}
