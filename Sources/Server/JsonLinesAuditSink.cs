// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Redaction;
using DevOnBike.Overfit.Runtime;

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
        private readonly IClock _clock;
        private readonly StreamWriter _writer;
        private readonly Lock _gate = new();

        /// <param name="path">The file to append to.</param>
        /// <param name="clock">
        /// Supplies the instant every line is stamped with. Injected because this sink is the one place the
        /// gateway's audit trail reads a clock at all, and an audit line whose timestamp cannot be pinned
        /// cannot be asserted on.
        /// </param>
        public JsonLinesAuditSink(string path, IClock? clock = null)
        {
            ArgumentException.ThrowIfNullOrEmpty(path);

            _clock = clock ?? SystemClock.Instance;

            var stream = new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.Read);
            _writer = new StreamWriter(stream, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false))
            {
                AutoFlush = true
            };
        }

        // OVERFIT040 on Record: `StreamWriter.WriteLine` has a `WriteLineAsync` sibling and this method cannot
        // take it. BOUND BY THE CONTRACT AND BY THE LOCK, in that order:
        //
        //   * this implements `IRedactionAuditSink.Record`, which is declared `void` in Sources/Main. The
        //     signature is not this file's to choose, and changing a public interface in the library is a
        //     different decision from a lint sweep's;
        //   * the write is inside `lock (_gate)`, and `await` is illegal in a lock body. An asynchronous
        //     version needs an asynchronous mutex, which is a different concurrency design, not a keyword.
        //
        // WHAT IS GIVEN UP, and it is real: the gateway's request path (RedactionGateway) is asynchronous, and
        // this one `AutoFlush` write to a local append-only file happens on that request's thread, once per
        // request that redacted anything. It is a single line to a local file, but it is a synchronous island
        // in an otherwise asynchronous path, and it stays one until `IRedactionAuditSink` changes.
#pragma warning disable OVERFIT040
        public void Record(in RedactionAuditEntry entry)
#pragma warning restore OVERFIT040
        {
            ArgumentNullException.ThrowIfNull(entry.RequestId);
            ArgumentNullException.ThrowIfNull(entry.CategoryCounts);

            // The stamp happens HERE, once, from an injected clock — see IRedactionAuditSink for why.
            var record = new RedactionAuditRecord(
                entry.RequestId, _clock.UtcNow, entry.TotalRedactions, entry.CategoryCounts);

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
