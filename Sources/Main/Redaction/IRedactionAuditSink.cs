// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Redaction
{
    /// <summary>
    /// Destination for redaction audit entries. Implementations persist or forward them (file, SIEM,
    /// database) — the gateway calls <see cref="Record"/> once per proxied request that produced redactions.
    /// Implementations must be thread-safe: a server may redact many requests concurrently.
    /// </summary>
    ///
    /// <remarks>
    /// <para><b>The sink stamps the entry, not the caller.</b> <see cref="Record"/> takes a
    /// <see cref="RedactionAuditEntry"/>, which carries no timestamp; the implementation supplies one from
    /// its own <c>IClock</c> when it builds the <see cref="RedactionAuditRecord"/> it persists.</para>
    ///
    /// <para>This is where the clock belongs for two reasons. It is the sink that knows what a timestamp
    /// means for its destination — a file line, a SIEM event, a database row may each want a different
    /// precision or timezone convention. And it puts the one wall-clock read behind a dependency that a test
    /// can substitute: the previous shape stamped <c>DateTimeOffset.UtcNow</c> at three call sites inside
    /// private static helpers of the gateway, where nothing could reach it.</para>
    /// </remarks>
    public interface IRedactionAuditSink
    {
        /// <summary>Records what was redacted. The implementation supplies the instant.</summary>
        void Record(in RedactionAuditEntry entry);
    }
}
