// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Redaction
{
    /// <summary>
    /// The outcome of applying a <see cref="RedactionPolicy"/> to a piece of text. If <see cref="Blocked"/>, the
    /// payload contains a category the policy forbids leaving the box at all (e.g. a private key) and the gateway
    /// must refuse the request — <see cref="Text"/> is not meant to be forwarded. Otherwise <see cref="Text"/> is
    /// the cleaned payload (only <see cref="RedactionAction.Redact"/> spans replaced; Allow/Alert left in place) and
    /// <see cref="RedactedMatches"/> are the spans to restore on the response.
    /// </summary>
    public sealed class RedactionDecision
    {
        public RedactionDecision(
            string text,
            IReadOnlyList<RedactionMatch> redactedMatches,
            bool blocked,
            IReadOnlyList<string> blockedCategories)
        {
            ArgumentNullException.ThrowIfNull(text);
            ArgumentNullException.ThrowIfNull(redactedMatches);
            ArgumentNullException.ThrowIfNull(blockedCategories);

            Text = text;
            RedactedMatches = redactedMatches;
            Blocked = blocked;
            BlockedCategories = blockedCategories;
        }

        /// <summary>The cleaned payload — valid only when <see cref="Blocked"/> is false.</summary>
        public string Text
        {
            get;
        }

        /// <summary>Spans actually replaced with placeholders (for restoring the response).</summary>
        public IReadOnlyList<RedactionMatch> RedactedMatches
        {
            get;
        }

        /// <summary>True when the payload contains a category the policy forbids forwarding — refuse the request.</summary>
        public bool Blocked
        {
            get;
        }

        /// <summary>The distinct categories that triggered the block (for the audit / error message).</summary>
        public IReadOnlyList<string> BlockedCategories
        {
            get;
        }
    }
}
