// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.RegularExpressions;

namespace DevOnBike.Overfit.Redaction
{
    /// <summary>
    /// Polish national-identifier detectors — each a loose <c>[GeneratedRegex]</c> candidate matcher paired with a
    /// checksum validator (<see cref="RedactionValidators"/>) so only genuine identifiers are redacted, not every
    /// digit run. Compose with <see cref="DefaultRedactionRules"/> for a PL deployment.
    /// </summary>
    public static partial class PolishRedactionRules
    {
        public static RedactionRule[] All() =>
        [
            new RedactionRule("PESEL", PeselRegex(), RedactionValidators.Pesel),
            new RedactionRule("NIP", NipRegex(), RedactionValidators.Nip),
            new RedactionRule("REGON", RegonRegex(), RedactionValidators.Regon),
            new RedactionRule("IBAN_PL", IbanRegex(), RedactionValidators.IbanPl)
        ];

        [GeneratedRegex(@"\b\d{11}\b")]
        private static partial Regex PeselRegex();

        // 10 digits, optionally grouped 3-3-2-2 with dashes.
        [GeneratedRegex(@"\b\d{3}-?\d{3}-?\d{2}-?\d{2}\b")]
        private static partial Regex NipRegex();

        [GeneratedRegex(@"\b\d{9}\b")]
        private static partial Regex RegonRegex();

        // PL + 2 check digits + 24 digits (optionally in 4-digit groups separated by spaces).
        [GeneratedRegex(@"\bPL\d{2}(?:\s?\d{4}){6}\b", RegexOptions.IgnoreCase)]
        private static partial Regex IbanRegex();
    }
}
