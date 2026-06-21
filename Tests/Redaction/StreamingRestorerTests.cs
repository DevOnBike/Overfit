// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Redaction;

namespace DevOnBike.Overfit.Tests.Redaction
{
    /// <summary>
    /// The streaming restorer re-hydrates redaction placeholders across SSE chunk boundaries: a placeholder split
    /// over several fragments must still come out whole, while ordinary bracketed text passes through untouched.
    /// </summary>
    public sealed class StreamingRestorerTests
    {
        private static IReadOnlyList<RedactionMatch> EmailMatch()
        {
            return new[]
            {
                new RedactionMatch("EMAIL", "a@b.co", "[REDACTED_EMAIL_0]", 0, 6),
            };
        }

        // Feeds the placeholder one character at a time — the worst-case split — and asserts the reassembled stream
        // equals the fully-restored text.
        [Fact]
        public void Push_CharByChar_RestoresSplitPlaceholder()
        {
            var restorer = new StreamingRestorer(EmailMatch());
            const string source = "mail me at [REDACTED_EMAIL_0] please";

            var sb = new StringBuilder();
            foreach (var ch in source)
            {
                sb.Append(restorer.Push(ch.ToString()));
            }
            sb.Append(restorer.Flush());

            Assert.Equal("mail me at a@b.co please", sb.ToString());
        }

        [Fact]
        public void Push_PlaceholderSplitAcrossTwoChunks_Restores()
        {
            var restorer = new StreamingRestorer(EmailMatch());

            var part1 = restorer.Push("contact [REDACTED_EMA");
            var part2 = restorer.Push("IL_0] now");
            var tail = restorer.Flush();

            // The split half must be held back until completed — never emitted broken.
            Assert.DoesNotContain("[REDACTED_EMA", part1);
            Assert.Equal("contact ", part1);
            Assert.Equal("a@b.co now", part2 + tail);
        }

        [Fact]
        public void Push_OrdinaryBrackets_PassThroughWithoutHolding()
        {
            var restorer = new StreamingRestorer(EmailMatch());

            // Markdown-style brackets are not our placeholder prefix → must not be withheld.
            var emitted = restorer.Push("see [link](url) and ");
            Assert.Equal("see [link](url) and ", emitted);
        }

        [Fact]
        public void Push_NoMatches_IsPassThrough()
        {
            var restorer = new StreamingRestorer([]);
            Assert.Equal("anything [REDACTED_EMAIL_0]", restorer.Push("anything [REDACTED_EMAIL_0]"));
            Assert.Equal(string.Empty, restorer.Flush());
        }

        [Fact]
        public void Flush_PartialPlaceholderAtStreamEnd_EmittedVerbatim()
        {
            var restorer = new StreamingRestorer(EmailMatch());

            // Stream ends mid-placeholder: the held tail is released as-is (original value stays hidden).
            var emitted = restorer.Push("oops [REDACTED_EMA");
            var tail = restorer.Flush();

            Assert.Equal("oops ", emitted);
            Assert.Equal("[REDACTED_EMA", tail);
        }

        [Fact]
        public void Push_MultiplePlaceholders_AllRestored()
        {
            IReadOnlyList<RedactionMatch> matches =
            [
                new RedactionMatch("EMAIL", "a@b.co", "[REDACTED_EMAIL_0]", 0, 6),
                new RedactionMatch("PESEL", "44051401359", "[REDACTED_PESEL_0]", 0, 11),
            ];
            var restorer = new StreamingRestorer(matches);

            var sb = new StringBuilder();
            sb.Append(restorer.Push("from [REDACTED_EMA"));
            sb.Append(restorer.Push("IL_0], id [REDACTED_PES"));
            sb.Append(restorer.Push("EL_0] end"));
            sb.Append(restorer.Flush());

            Assert.Equal("from a@b.co, id 44051401359 end", sb.ToString());
        }
    }
}
