// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Redaction
{
    /// <summary>
    /// Destination for redaction audit records. Implementations persist or forward them (file, SIEM, database) —
    /// the gateway calls <see cref="Record"/> once per proxied request that produced redactions. Implementations
    /// must be thread-safe: a server may redact many requests concurrently.
    /// </summary>
    public interface IRedactionAuditSink
    {
        void Record(RedactionAuditRecord record);
    }
}
