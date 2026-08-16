// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// <see cref="LongFact.TryParseRuntime"/> — the reason the measured runtime is a field and not a comment.
    ///
    /// <para>The values are written into 256 attributes by <c>Scripts/longfact_annotate.py</c> and read back
    /// by whatever wants to know what the gate costs. <b>If the round trip does not hold, the field is a
    /// comment with extra syntax</b>, so the cases below pin the exact notation that script emits — including
    /// the two that look alike and are not: <c>"115ms"</c> ends in <c>s</c> and must not read as seconds, and
    /// <c>"15min51s"</c> must not read as 15 minutes with the tail dropped.</para>
    ///
    /// <para>Unparseable input returns <see langword="false"/> rather than throwing. These strings are
    /// written by tooling into source; a malformed one should make a report say "unknown", not fail a run
    /// that has nothing to do with it.</para>
    /// </summary>
    public sealed class LongFactRuntimeTests
    {
        [Theory]
        [InlineData("115ms", 0.115)]
        [InlineData("1ms", 0.001)]
        [InlineData("32s", 32)]
        [InlineData("59s", 59)]
        [InlineData("4min", 240)]
        [InlineData("15min51s", 951)]
        [InlineData("2h15min", 8100)]
        [InlineData("1h", 3600)]
        public void ParsesTheNotationTheAnnotatorEmits(string runtime, double expectedSeconds)
        {
            Assert.True(LongFact.TryParseRuntime(runtime, out var value), $"failed to parse '{runtime}'");
            Assert.Equal(expectedSeconds, value.TotalSeconds, 3);
        }

        [Fact]
        public void MillisecondsAreNotReadAsSeconds()
        {
            // "115ms" ends in 's'. A seconds-first parser reads it as 115 seconds — a thousandfold error
            // that looks entirely plausible in a report, which is exactly why it gets its own test.
            Assert.True(LongFact.TryParseRuntime("115ms", out var milliseconds));
            Assert.True(LongFact.TryParseRuntime("115s", out var seconds));
            Assert.True(milliseconds < seconds);
            Assert.Equal(1000d, seconds.TotalSeconds / milliseconds.TotalSeconds, 3);
        }

        [Theory]
        // Null is deliberate — TryParseRuntime is called with whatever a LongFact carried, and its
        // `runtime` parameter is optional, so null is a real input rather than an invalid one. The
        // parameter is nullable for that reason; xUnit1012 was flagging the signature, not the data.
        [InlineData(null)]
        [InlineData("")]
        [InlineData("   ")]
        [InlineData("soon")]
        [InlineData("32")]          // no unit
        [InlineData("32sec")]       // not the notation
        [InlineData("min32")]       // unit before the number
        [InlineData("2h15")]        // trailing digits with no unit
        public void UnparseableInputReturnsFalseInsteadOfThrowing(string? runtime)
        {
            Assert.False(LongFact.TryParseRuntime(runtime, out var value));
            Assert.Equal(TimeSpan.Zero, value);
        }

        [Fact]
        public void TheAttributeStillWorksWithNoRuntimeAtAll()
        {
            // The field is optional on purpose: a test that has never been measured must not be forced to
            // carry an invented number, and every [LongFact] predating 2026-08-07 is written without one.
            var withoutRuntime = new LongFact();
            var withRuntime = new LongFact("32s");

            Assert.Null(withoutRuntime.Runtime);
            Assert.Equal("32s", withRuntime.Runtime);
        }
    }
}
