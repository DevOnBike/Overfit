// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.RegularExpressions;

namespace DevOnBike.Overfit.Redaction
{
    /// <summary>
    /// A named detector: a category label plus the regular expression that recognizes that class of sensitive
    /// data. Rules are data — the production set is configured per deployment policy; <see cref="DefaultRedactionRules"/>
    /// ships a conservative starter set.
    /// </summary>
    public sealed class RedactionRule
    {
        public RedactionRule(string category, Regex pattern, Func<string, bool>? validator = null)
        {
            ArgumentException.ThrowIfNullOrEmpty(category);
            ArgumentNullException.ThrowIfNull(pattern);

            Category = category;
            Pattern = pattern;
            Validator = validator;
        }

        /// <summary>Category label used for the placeholder and the audit record (e.g. <c>EMAIL</c>).</summary>
        public string Category
        {
            get;
        }

        /// <summary>Recognizer for this category.</summary>
        public Regex Pattern
        {
            get;
        }

        /// <summary>
        /// Optional precision gate run on each regex match — a checksum / structural check (PESEL, NIP, Luhn…) that
        /// rejects look-alikes. A match counts only if this returns true (or there is no validator). The loose regex
        /// finds candidates; the validator cuts the false positives (any 11 digits is not a PESEL).
        /// </summary>
        public Func<string, bool>? Validator
        {
            get;
        }
    }
}
