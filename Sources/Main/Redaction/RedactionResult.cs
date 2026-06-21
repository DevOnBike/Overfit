// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Redaction
{
    /// <summary>
    /// Outcome of redacting a piece of text: the cleaned <see cref="Text"/> (safe to forward / log) plus the
    /// list of <see cref="Matches"/> that were removed (for auditing, and for an optional restore on the way back).
    /// </summary>
    public sealed class RedactionResult
    {
        public RedactionResult(string text, IReadOnlyList<RedactionMatch> matches)
        {
            ArgumentNullException.ThrowIfNull(text);
            ArgumentNullException.ThrowIfNull(matches);

            Text = text;
            Matches = matches;
        }

        /// <summary>The redacted text — sensitive spans replaced with placeholders.</summary>
        public string Text
        {
            get;
        }

        /// <summary>The spans that were redacted, in source order.</summary>
        public IReadOnlyList<RedactionMatch> Matches
        {
            get;
        }

        /// <summary>True when at least one span was redacted.</summary>
        public bool HasRedactions => Matches.Count > 0;
    }
}
