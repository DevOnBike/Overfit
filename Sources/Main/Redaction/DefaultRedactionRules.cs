// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.RegularExpressions;

namespace DevOnBike.Overfit.Redaction
{
    /// <summary>
    /// A conservative STARTER set of redaction rules — enough to stand up the gateway and exercise the pipeline.
    /// The production rule set (categories, precise patterns, allow/deny policy, locale-specific identifiers) is a
    /// per-deployment policy decision and is expected to replace or extend this. Patterns are AOT-friendly compiled
    /// regexes via <c>[GeneratedRegex]</c> (source-generated — no reflection, no runtime regex compilation).
    /// </summary>
    public static partial class DefaultRedactionRules
    {
        public static RedactionRule[] All() =>
        [
            new RedactionRule("EMAIL", EmailRegex()),
            new RedactionRule("CREDIT_CARD", CreditCardRegex(), RedactionValidators.Luhn),
            new RedactionRule("SSN", SsnRegex()),
            new RedactionRule("AWS_KEY", AwsKeyRegex()),
            new RedactionRule("API_KEY", ApiKeyRegex()),
            new RedactionRule("JWT", JwtRegex()),
            new RedactionRule("PRIVATE_KEY", PrivateKeyRegex()),
            new RedactionRule("IPV4", Ipv4Regex())
        ];

        [GeneratedRegex(@"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", RegexOptions.IgnoreCase)]
        private static partial Regex EmailRegex();

        // 13–16 digit runs allowing spaces/dashes (loose; a production rule would add a Luhn check).
        [GeneratedRegex(@"\b(?:\d[ -]*?){13,16}\b")]
        private static partial Regex CreditCardRegex();

        [GeneratedRegex(@"\b\d{3}-\d{2}-\d{4}\b")]
        private static partial Regex SsnRegex();

        [GeneratedRegex(@"\bAKIA[0-9A-Z]{16}\b")]
        private static partial Regex AwsKeyRegex();

        // OpenAI / Stripe-style secret tokens: sk-/pk-/rk- followed by a long opaque body.
        [GeneratedRegex(@"\b(?:sk|pk|rk)-[A-Za-z0-9]{20,}\b")]
        private static partial Regex ApiKeyRegex();

        [GeneratedRegex(@"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b")]
        private static partial Regex JwtRegex();

        [GeneratedRegex(@"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")]
        private static partial Regex PrivateKeyRegex();

        [GeneratedRegex(@"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")]
        private static partial Regex Ipv4Regex();
    }
}
