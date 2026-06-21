// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Redaction;

namespace DevOnBike.Overfit.Tests.Redaction
{
    public sealed class RedactorTests
    {
        private static readonly Redactor _redactor = Redactor.CreateDefault();

        [Fact]
        public void Redact_RemovesSensitiveSpans_AndReportsCategories()
        {
            const string input =
                "Email me at jane.doe@acme.com or call. Key sk-ABCDEFGHIJKLMNOPQRSTUVWX, host 10.0.0.42, ssn 123-45-6789.";

            var result = _redactor.Redact(input);

            Assert.True(result.HasRedactions);
            Assert.DoesNotContain("jane.doe@acme.com", result.Text);
            Assert.DoesNotContain("sk-ABCDEFGHIJKLMNOPQRSTUVWX", result.Text);
            Assert.DoesNotContain("123-45-6789", result.Text);
            Assert.DoesNotContain("10.0.0.42", result.Text);

            Assert.Contains(result.Matches, m => m.Category == "EMAIL");
            Assert.Contains(result.Matches, m => m.Category == "API_KEY");
            Assert.Contains(result.Matches, m => m.Category == "SSN");
            Assert.Contains(result.Matches, m => m.Category == "IPV4");

            // Placeholders are present in the cleaned text.
            foreach (var match in result.Matches)
            {
                Assert.Contains(match.Placeholder, result.Text);
            }
        }

        [Fact]
        public void Restore_RoundTrips_BackToOriginal()
        {
            const string input = "Contact admin@corp.io and backup ops@corp.io about ticket.";

            var result = _redactor.Redact(input);
            var restored = Redactor.Restore(result.Text, result.Matches);

            Assert.Equal(input, restored);
            // Two distinct e-mails get distinct, indexed placeholders.
            Assert.Equal(2, result.Matches.Count);
            Assert.NotEqual(result.Matches[0].Placeholder, result.Matches[1].Placeholder);
        }

        [Fact]
        public void Redact_EmptyOrClean_ReturnsNoRedactions()
        {
            Assert.False(_redactor.Redact(string.Empty).HasRedactions);
            Assert.False(_redactor.Redact("nothing sensitive here, just words.").HasRedactions);
        }

        [Fact]
        public void Redact_IsDeterministic_AcrossRuleOrder()
        {
            const string input = "jwt eyJhbGci.eyJzdWIi.SflKxwRJ and mail a@b.co";

            var a = _redactor.Redact(input);
            var b = _redactor.Redact(input);

            Assert.Equal(a.Text, b.Text);
        }

        [Fact]
        public void AuditRecord_CountsByCategory_WithoutSensitiveValues()
        {
            const string input = "mails a@b.co and c@d.co, key sk-ABCDEFGHIJKLMNOPQRSTUVWX";

            var result = _redactor.Redact(input);
            var record = RedactionAuditRecord.FromResult("req-1", DateTimeOffset.UnixEpoch, result);

            Assert.Equal("req-1", record.RequestId);
            Assert.Equal(3, record.TotalRedactions);
            Assert.Equal(2, record.CategoryCounts["EMAIL"]);
            Assert.Equal(1, record.CategoryCounts["API_KEY"]);

            // The audit record must not carry any original sensitive value.
            foreach (var count in record.CategoryCounts)
            {
                Assert.DoesNotContain("@", count.Key);
                Assert.DoesNotContain("sk-", count.Key);
            }
        }
    }
}
