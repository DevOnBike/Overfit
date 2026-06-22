// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using System.Text.RegularExpressions;
using DevOnBike.Overfit.Redaction;

namespace DevOnBike.Overfit.Tests.Redaction
{
    /// <summary>
    /// Precision levers against over- and under-redaction: an <b>allowlist</b> keeps known-good values (own domain,
    /// test data) out of the placeholders, and the <b>entropy detector</b> catches random secrets that no named
    /// regex enumerates while leaving low-entropy long tokens alone.
    /// </summary>
    public sealed class AllowlistAndEntropyTests
    {
        [Fact]
        public void Allowlist_KeepsKnownGoodValue_StillRedactsTheRest()
        {
            var allowlist = new[] { new Regex(@"@acme\.com$") };
            var redactor = new Redactor(DefaultRedactionRules.All(), allowlist);

            var result = redactor.Redact("write alice@acme.com and bob@evil.com");

            // The corporate-domain address survives; the foreign one is redacted.
            Assert.Contains("alice@acme.com", result.Text);
            Assert.DoesNotContain("bob@evil.com", result.Text);
            Assert.Single(result.Matches);
            Assert.Equal("bob@evil.com", result.Matches[0].Original);
        }

        [Theory]
        [InlineData("aB3xK9mPq2wL7vN4tR8cY1zE5dF6gH0j", true)]   // 32 random-looking chars → secret
        [InlineData("0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d", true)]   // 32-hex digest → high entropy
        [InlineData("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", false)]  // 32 repeats → zero entropy
        [InlineData("abcabcabcabcabcabcabcabc", false)]          // low-entropy repetition
        [InlineData("internationalization", false)]              // long ordinary word, below threshold
        [InlineData("short", false)]                             // below the min length
        public void HasHighEntropy_SeparatesSecretsFromOrdinaryTokens(string value, bool expected)
        {
            Assert.Equal(expected, RedactionValidators.HasHighEntropy(value));
        }

        [Fact]
        public void EntropyDetector_RedactsRandomSecret_LeavesLowEntropyToken()
        {
            var redactor = new Redactor(SecretRedactionRules.All());

            var result = redactor.Redact("api secret aB3xK9mPq2wL7vN4tR8cY1zE5dF6gH0j end");
            Assert.DoesNotContain("aB3xK9mPq2wL7vN4tR8cY1zE5dF6gH0j", result.Text);
            Assert.Single(result.Matches);
            Assert.Equal("SECRET", result.Matches[0].Category);

            // A long but low-entropy token is not a secret — left untouched.
            const string benign = "path aaaaaaaaaaaaaaaaaaaaaaaa done";
            Assert.Equal(benign, redactor.Redact(benign).Text);
        }

        [Fact]
        public void EntropyDetector_WithAllowlist_SuppressesKnownHighEntropyNonSecret()
        {
            // A deployment whitelists a known build hash so it is not treated as a leaked secret.
            const string buildHash = "aB3xK9mPq2wL7vN4tR8cY1zE5dF6gH0j";
            var redactor = new Redactor(SecretRedactionRules.All(), new[] { new Regex(Regex.Escape(buildHash)) });

            var result = redactor.Redact($"build {buildHash} shipped");
            Assert.Contains(buildHash, result.Text);
            Assert.Empty(result.Matches);
        }
    }
}
