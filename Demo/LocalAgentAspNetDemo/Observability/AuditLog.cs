// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;

namespace DevOnBike.Overfit.Demo.LocalAgent.Observability
{
    /// <summary>
    /// Append-only audit trail for the agent. Each handled request becomes one JSON line — <b>metadata only</b>
    /// (who/when/which endpoint/status/latency/model fingerprint/retrieved sources/tool called), <b>never the prompt
    /// or the answer</b>, so the audit log honours the "data never leaves the process" guarantee and is safe to
    /// retain. Writes to the file at <c>AuditLogPath</c> (config/env) if set — append-only, shared-read so it can be
    /// tailed live — and always mirrors to the structured logger. Thread-safe: many requests audit concurrently.
    /// </summary>
    public sealed class AuditLog : IDisposable
    {
        private static readonly JsonSerializerOptions Json = new() { WriteIndented = false };

        private readonly object _gate = new();
        private readonly StreamWriter? _writer;
        private readonly ILogger<AuditLog> _logger;

        public AuditLog(IConfiguration configuration, ILogger<AuditLog> logger)
        {
            _logger = logger;
            var path = configuration["AuditLogPath"];
            if (string.IsNullOrWhiteSpace(path))
            {
                logger.LogInformation(
                    "Audit: structured log only — set 'AuditLogPath' for an append-only JSONL file you can retain/tail.");
                return;
            }

            _writer = new StreamWriter(
                new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.Read)) { AutoFlush = true };
            logger.LogInformation("Audit log: {Path} (append-only JSONL, metadata only — no prompt/response content).",
                Path.GetFullPath(path));
        }

        /// <summary>Records one audit entry (an anonymous metadata object). Serialized to one JSON line.</summary>
        public void Record(object record)
        {
            var line = JsonSerializer.Serialize(record, Json);
            _logger.LogInformation("audit {Audit}", line);

            if (_writer is null)
            {
                return;
            }

            lock (_gate)
            {
                _writer.WriteLine(line);
            }
        }

        public void Dispose()
        {
            lock (_gate)
            {
                _writer?.Dispose();
            }
        }
    }
}
