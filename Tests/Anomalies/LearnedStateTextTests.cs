// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The escaping three stores share through one file.
    ///
    /// <para><b>Written because the round-trip test that existed could not fail.</b>
    /// <c>FloorCalibratorTests.ACustomChannelSurvivesARoundTrip</c> used the name
    /// <c>myapp_queue_depth</c> — no backslash, no tab, no newline, no hash — so it exercised the branch
    /// where escaping does nothing. A test that only feeds the trivial case is not weaker evidence than a
    /// proper one; it is no evidence.</para>
    ///
    /// <para>The interesting inputs are the ones where the encoder's own output can be re-read by a later
    /// decoding pass, which is exactly what a chain of <c>string.Replace</c> calls does and what broke
    /// here.</para>
    /// </summary>
    public sealed class LearnedStateTextTests
    {
        /// <summary>
        /// The case that was silently corrupted: a literal backslash followed by <c>h</c>. Escaping doubles
        /// the backslash, and a decoder built from sequential replacements then matched <c>\h</c> at the
        /// second backslash and produced <c>a\#b</c>.
        /// </summary>
        [Theory]
        [InlineData("a\\hb")]
        [InlineData("a\\tb")]
        [InlineData("a\\nb")]
        [InlineData("\\h")]
        [InlineData("\\\\h")]
        [InlineData("trailing\\")]
        public void SequencesThatLookLikeEscapesSurviveARoundTrip(string value)
        {
            Assert.Equal(value, LearnedStateText.Unescape(LearnedStateText.Escape(value)));
        }

        [Theory]
        [InlineData("myapp_queue_depth")]
        [InlineData("with\ttab")]
        [InlineData("with\nnewline")]
        [InlineData("### labels")]
        [InlineData("kafka#lag")]
        [InlineData("")]
        [InlineData("mixed \\ # \t \n all at once")]
        public void OrdinaryAndSeparatorCarryingValuesSurviveARoundTrip(string value)
        {
            Assert.Equal(value, LearnedStateText.Unescape(LearnedStateText.Escape(value)));
        }

        /// <summary>
        /// Nothing encoded may contain a record or section separator, or the field stops being a field.
        /// <c>#</c> is included because <c>LearnedState</c> delimits its sections with <c>### labels</c> and
        /// <c>### suppressions</c>, so a name carrying that text moves a section boundary.
        /// </summary>
        [Theory]
        [InlineData("with\ttab")]
        [InlineData("with\nnewline")]
        [InlineData("### labels")]
        [InlineData("### suppressions")]
        public void EscapedTextCarriesNoSeparator(string value)
        {
            var escaped = LearnedStateText.Escape(value);

            Assert.DoesNotContain('\t', escaped);
            Assert.DoesNotContain('\n', escaped);
            Assert.DoesNotContain('#', escaped);
        }

        /// <summary>A value needing no escaping is returned unchanged, not rebuilt.</summary>
        [Fact]
        public void PlainTextIsReturnedAsIs()
        {
            const string plain = "container_cpu_usage_seconds_total";

            Assert.Same(plain, LearnedStateText.Escape(plain));
            Assert.Same(plain, LearnedStateText.Unescape(plain));
        }
    }
}
