// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Redaction
{
    /// <summary>
    /// One audit entry for a redaction pass — what to persist for compliance. It records the <b>shape</b> of what
    /// was removed (how many of each category), never the sensitive values themselves: an audit log must be safe
    /// to retain. Built from a <see cref="RedactionResult"/> via <see cref="FromResult"/>.
    /// </summary>
    public sealed class RedactionAuditRecord
    {
        public RedactionAuditRecord(
            string requestId,
            DateTimeOffset timestamp,
            int totalRedactions,
            IReadOnlyDictionary<string, int> categoryCounts)
        {
            ArgumentNullException.ThrowIfNull(requestId);
            ArgumentNullException.ThrowIfNull(categoryCounts);

            RequestId = requestId;
            Timestamp = timestamp;
            TotalRedactions = totalRedactions;
            CategoryCounts = categoryCounts;
        }

        /// <summary>Correlates the audit entry with the proxied request.</summary>
        public string RequestId
        {
            get;
        }

        /// <summary>When the redaction happened.</summary>
        public DateTimeOffset Timestamp
        {
            get;
        }

        /// <summary>Total spans redacted across all categories.</summary>
        public int TotalRedactions
        {
            get;
        }

        /// <summary>Per-category counts (e.g. <c>EMAIL → 2</c>). Counts only — no sensitive values.</summary>
        public IReadOnlyDictionary<string, int> CategoryCounts
        {
            get;
        }

        /// <summary>Summarizes a <see cref="RedactionResult"/> into an audit record (counts per category).</summary>
        public static RedactionAuditRecord FromResult(string requestId, DateTimeOffset timestamp, RedactionResult result)
        {
            ArgumentNullException.ThrowIfNull(result);

            var counts = new Dictionary<string, int>(StringComparer.Ordinal);

            foreach (var match in result.Matches)
            {
                counts[match.Category] = counts.GetValueOrDefault(match.Category) + 1;
            }

            return new RedactionAuditRecord(requestId, timestamp, result.Matches.Count, counts);
        }
    }
}
