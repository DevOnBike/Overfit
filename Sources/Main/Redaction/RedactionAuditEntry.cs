// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Redaction
{
    /// <summary>
    /// What a redaction pass produced, as the caller reports it: <b>no timestamp</b>.
    ///
    /// <para><b>The split is the point.</b> An entry says what happened; a
    /// <see cref="RedactionAuditRecord"/> says what happened <i>and when</i>, and the "when" belongs to the
    /// <see cref="IRedactionAuditSink"/> that persists it. Before this split the gateway stamped each entry
    /// with <c>DateTimeOffset.UtcNow</c> at three separate call sites, which put an untestable wall-clock
    /// read inside private static helpers — and meant three requests audited in one pass carried three
    /// different instants for no reason a reader could name.</para>
    ///
    /// <para>A struct because this sits on the per-request path and carries no state a heap object would
    /// buy anything for. The counts dictionary is the only allocation, and it already exists.</para>
    /// </summary>
    /// <param name="RequestId">Correlates the audit entry with the proxied request.</param>
    /// <param name="TotalRedactions">Total spans redacted across all categories.</param>
    /// <param name="CategoryCounts">Per-category counts. Counts only — never the sensitive values.</param>
    public readonly record struct RedactionAuditEntry(
        string RequestId,
        int TotalRedactions,
        IReadOnlyDictionary<string, int> CategoryCounts)
    {
        /// <summary>Summarises a <see cref="RedactionResult"/> into the counts an audit log may retain.</summary>
        public static RedactionAuditEntry FromResult(string requestId, RedactionResult result)
        {
            ArgumentNullException.ThrowIfNull(requestId);
            ArgumentNullException.ThrowIfNull(result);

            var counts = new Dictionary<string, int>(StringComparer.Ordinal);

            foreach (var match in result.Matches)
            {
                counts[match.Category] = counts.GetValueOrDefault(match.Category) + 1;
            }

            return new RedactionAuditEntry(requestId, result.Matches.Count, counts);
        }
    }
}
