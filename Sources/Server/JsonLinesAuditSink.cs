// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Redaction;

namespace DevOnBike.Overfit.Server
{
    /// <summary>
    /// Append-only JSON-lines audit sink for the redaction gateway — one self-describing JSON object per line
    /// (timestamp, request id, per-category counts, total). Records the SHAPE of what was redacted, never the
    /// sensitive values, so the log is safe to retain. JSON is hand-built (AOT-safe, no reflection / source-gen
    /// dependency). Thread-safe — a gateway redacts many requests concurrently.
    /// </summary>
    public sealed class JsonLinesAuditSink : IRedactionAuditSink, IDisposable
    {
        private readonly StreamWriter _writer;
        private readonly Lock _gate = new();

        public JsonLinesAuditSink(string path)
        {
            ArgumentException.ThrowIfNullOrEmpty(path);

            var stream = new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.Read);
            _writer = new StreamWriter(stream, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false))
            {
                AutoFlush = true
            };
        }

        public void Record(RedactionAuditRecord record)
        {
            ArgumentNullException.ThrowIfNull(record);

            var sb = new StringBuilder(128);
            sb.Append("{\"timestamp\":\"").Append(record.Timestamp.ToString("o", CultureInfo.InvariantCulture)).Append('"');
            sb.Append(",\"requestId\":\"").Append(Escape(record.RequestId)).Append('"');
            sb.Append(",\"totalRedactions\":").Append(record.TotalRedactions.ToString(CultureInfo.InvariantCulture));
            sb.Append(",\"categories\":{");

            var first = true;
            foreach (var pair in record.CategoryCounts)
            {
                if (!first)
                {
                    sb.Append(',');
                }
                first = false;
                sb.Append('"').Append(Escape(pair.Key)).Append("\":").Append(pair.Value.ToString(CultureInfo.InvariantCulture));
            }

            sb.Append("}}");

            var line = sb.ToString();
            lock (_gate)
            {
                _writer.WriteLine(line);
            }
        }

        private static string Escape(string value)
        {
            // Categories/request-ids are simple tokens, but keep the line valid JSON regardless.
            return value.Replace("\\", "\\\\", StringComparison.Ordinal).Replace("\"", "\\\"", StringComparison.Ordinal);
        }

        public void Dispose()
        {
            _writer.Dispose();
        }
    }
}
