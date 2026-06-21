// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Redaction
{
    /// <summary>
    /// One redacted span found in a piece of text: its category, the original sensitive value, the placeholder
    /// substituted for it, and where it sat in the source. <see cref="Original"/> is kept only to support an
    /// optional round-trip <see cref="Redactor.Restore"/>; it is the sensitive data and must never be logged.
    /// </summary>
    public sealed class RedactionMatch
    {
        public RedactionMatch(string category, string original, string placeholder, int start, int length)
        {
            ArgumentException.ThrowIfNullOrEmpty(category);
            ArgumentNullException.ThrowIfNull(original);
            ArgumentException.ThrowIfNullOrEmpty(placeholder);

            Category = category;
            Original = original;
            Placeholder = placeholder;
            Start = start;
            Length = length;
        }

        /// <summary>Rule category that matched (e.g. <c>EMAIL</c>, <c>API_KEY</c>).</summary>
        public string Category
        {
            get;
        }

        /// <summary>The sensitive text that was removed. Sensitive — do not log; used only for restore.</summary>
        public string Original
        {
            get;
        }

        /// <summary>The placeholder substituted in the redacted text (e.g. <c>[REDACTED_EMAIL_0]</c>).</summary>
        public string Placeholder
        {
            get;
        }

        /// <summary>Start offset of the match in the original input.</summary>
        public int Start
        {
            get;
        }

        /// <summary>Length of the matched span in the original input.</summary>
        public int Length
        {
            get;
        }
    }
}
