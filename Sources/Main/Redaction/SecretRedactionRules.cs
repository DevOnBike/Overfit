// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.RegularExpressions;

namespace DevOnBike.Overfit.Redaction
{
    /// <summary>
    /// Generic secret detection for credentials no named regex can enumerate: a loose long-token candidate matcher
    /// gated by a Shannon-entropy validator (<see cref="RedactionValidators.HasHighEntropy"/>), so random API keys,
    /// tokens and credentials get caught even without a vendor-specific pattern. Opt-in — noisier than the named
    /// detectors, so enable per deployment and pair with an allowlist to suppress known high-entropy non-secrets
    /// (asset hashes, build ids, base64 blobs…).
    /// </summary>
    public static partial class SecretRedactionRules
    {
        public static RedactionRule[] All() =>
        [
            new RedactionRule("SECRET", HighEntropyTokenRegex(), static value => RedactionValidators.HasHighEntropy(value)),
        ];

        // A contiguous 20+ char run over the alphabet secrets use (base64 / url-safe / hex / token characters).
        [GeneratedRegex(@"[A-Za-z0-9_+/=\-]{20,}")]
        private static partial Regex HighEntropyTokenRegex();
    }
}
