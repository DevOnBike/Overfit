// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using System.Text;
using DevOnBike.Overfit.Redaction;

namespace DevOnBike.Overfit.Tests.Redaction
{
    /// <summary>
    /// The streaming response scanner masks model-generated secrets/PII even when a sensitive token is split across
    /// SSE chunks — the whitespace-free token accumulates in the buffer and is scanned whole once a whitespace (or
    /// end-of-stream) flushes it.
    /// </summary>
    public sealed class StreamingResponseScannerTests
    {
        private static StreamingResponseScanner NewScanner() =>
            new(Redactor.CreateDefault(), RedactionPolicy.Default());

        [Fact]
        public void Push_SecretSplitAcrossChunks_IsMaskedWhole()
        {
            var scanner = NewScanner();
            var sb = new StringBuilder();

            sb.Append(scanner.Push("contact leaked@cor"));
            sb.Append(scanner.Push("p.io now"));
            sb.Append(scanner.Flush());

            var text = sb.ToString();
            Assert.DoesNotContain("leaked@corp.io", text);
            Assert.Contains("REDACTED-RESPONSE-EMAIL", text);
            Assert.Contains("contact", text);
            Assert.Contains("now", text);

            Assert.Single(scanner.MaskedMatches);
            Assert.Equal("EMAIL", scanner.MaskedMatches[0].Category);
        }

        [Fact]
        public void Push_CharByChar_MasksSecret_NeverLeaks()
        {
            var scanner = NewScanner();
            const string source = "email leaked@corp.io end";

            var sb = new StringBuilder();
            foreach (var ch in source)
            {
                sb.Append(scanner.Push(ch.ToString()));
            }
            sb.Append(scanner.Flush());

            var text = sb.ToString();
            Assert.DoesNotContain("leaked@corp.io", text);
            Assert.Contains("REDACTED-RESPONSE-EMAIL", text);
        }

        [Fact]
        public void Flush_SecretWithNoTrailingWhitespace_IsMasked()
        {
            var scanner = NewScanner();

            // Whole stream is one whitespace-free run ending in a secret — held until flush, then masked.
            var emitted = scanner.Push("mailto:leaked@corp.io");
            var tail = scanner.Flush();

            Assert.DoesNotContain("leaked@corp.io", emitted + tail);
            Assert.Contains("REDACTED-RESPONSE-EMAIL", emitted + tail);
        }

        [Fact]
        public void CleanText_PassesThroughUnmasked()
        {
            var scanner = NewScanner();

            var sb = new StringBuilder();
            sb.Append(scanner.Push("the capital of France "));
            sb.Append(scanner.Push("is Paris."));
            sb.Append(scanner.Flush());

            Assert.Equal("the capital of France is Paris.", sb.ToString());
            Assert.Empty(scanner.MaskedMatches);
        }
    }
}
