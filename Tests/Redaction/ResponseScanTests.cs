// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using DevOnBike.Overfit.Redaction;

namespace DevOnBike.Overfit.Tests.Redaction
{
    /// <summary>
    /// <see cref="Redactor.ScanResponse"/> masks sensitive content the MODEL produced (leaked secret, RAG-echoed
    /// document) per the policy — masking Redact/Block categories, leaving Allow/Alert — and uses a placeholder
    /// distinct from the request format so a later <see cref="Redactor.Restore"/> never touches it.
    /// </summary>
    public sealed class ResponseScanTests
    {
        [Fact]
        public void ScanResponse_MasksRedactAndBlockCategories_LeavesAllow()
        {
            var redactor = Redactor.CreateDefault();
            // EMAIL → Redact (mask), API_KEY → Block (mask), IPV4 → Allow (keep).
            var policy = new RedactionPolicy(
                new Dictionary<string, RedactionAction>(StringComparer.Ordinal)
                {
                    ["EMAIL"] = RedactionAction.Redact,
                    ["API_KEY"] = RedactionAction.Block,
                    ["IPV4"] = RedactionAction.Allow,
                },
                RedactionAction.Redact);

            const string modelOutput = "mail leak@corp.io key sk-abcdef0123456789abcdef0123 host 10.0.0.5";
            var result = redactor.ScanResponse(modelOutput, policy);

            // Redact + Block content is masked away; the Allow'd IPv4 stays visible.
            Assert.DoesNotContain("leak@corp.io", result.Text);
            Assert.DoesNotContain("sk-abcdef0123456789abcdef0123", result.Text);
            Assert.Contains("10.0.0.5", result.Text);

            var categories = result.Matches.Select(m => m.Category).ToHashSet();
            Assert.Contains("EMAIL", categories);
            Assert.Contains("API_KEY", categories);
            Assert.DoesNotContain("IPV4", categories);
        }

        [Fact]
        public void ScanResponse_Placeholder_IsDistinctFromRequestFormat_AndSurvivesRestore()
        {
            var redactor = Redactor.CreateDefault();
            var policy = RedactionPolicy.Default();

            var scan = redactor.ScanResponse("contact leak@corp.io now", policy);

            // Response masks use a hyphenated [REDACTED-RESPONSE-…] token, never the request's [REDACTED_…] form.
            Assert.Contains("[REDACTED-RESPONSE-EMAIL-0]", scan.Text);
            Assert.DoesNotContain("[REDACTED_EMAIL", scan.Text);

            // A subsequent Restore that targets request matches must not rewrite the response mask.
            var requestMatches = new[]
            {
                new RedactionMatch("EMAIL", "bob@example.com", "[REDACTED_EMAIL_0]", 0, 15),
            };
            var restored = Redactor.Restore(scan.Text, requestMatches);
            Assert.Contains("[REDACTED-RESPONSE-EMAIL-0]", restored);
            Assert.DoesNotContain("bob@example.com", restored);
        }

        [Fact]
        public void ScanResponse_CleanText_IsUnchanged()
        {
            var redactor = Redactor.CreateDefault();
            var result = redactor.ScanResponse("the capital of France is Paris", RedactionPolicy.Default());

            Assert.Equal("the capital of France is Paris", result.Text);
            Assert.Empty(result.Matches);
        }
    }
}
