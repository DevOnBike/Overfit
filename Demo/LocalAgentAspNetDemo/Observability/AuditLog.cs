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

        // A SemaphoreSlim rather than a lock because the write is awaited: `lock` cannot span an await, and
        // the write is the whole reason this type is asynchronous (see RecordAsync). SemaphoreSlim.Wait in
        // Dispose is not a task and is deliberately outside OVERFIT039 — it is a synchronisation primitive
        // with no continuation to starve.
        private readonly SemaphoreSlim _gate = new(1, 1);
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
                new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.Read))
            {
                AutoFlush = true
            };
            logger.LogInformation("Audit log: {Path} (append-only JSONL, metadata only — no prompt/response content).",
                Path.GetFullPath(path));
        }

        /// <summary>
        /// Records one audit entry (an anonymous metadata object). Serialized to one JSON line.
        ///
        /// <para><b>Asynchronous because it runs on the request path.</b> This is called once per handled
        /// request, from <c>AuditMiddleware</c>'s <c>finally</c>, and the writer has
        /// <c>AutoFlush = true</c> on a <see cref="FileStream"/> — so the synchronous version put an append
        /// AND a flush to disk on the Kestrel thread serving the caller, for every request. That is the
        /// synchronous island OVERFIT040 exists to find, and the fix is the method rather than the call: the
        /// caller is already asynchronous and awaits this, so nothing blocks anywhere.</para>
        ///
        /// <para><b>No CancellationToken on purpose.</b> The one token available at the call site is
        /// <c>HttpContext.RequestAborted</c>, and an aborted request is exactly the one whose audit record
        /// matters most — taking that token would drop the trail for cancelled and failed requests, which
        /// is the opposite of what an audit log is for.</para>
        /// </summary>
        public async Task RecordAsync(object record)
        {
            var line = JsonSerializer.Serialize(record, Json);
            _logger.LogInformation("audit {Audit}", line);

            if (_writer == null)
            {
                return;
            }

            await _gate.WaitAsync().ConfigureAwait(false);

            try
            {
                // AutoFlush is on, so this also flushes — asynchronously now, off the request thread.
                await _writer.WriteLineAsync(line).ConfigureAwait(false);
            }
            finally
            {
                _gate.Release();
            }
        }

        // OVERFIT040 — synchronous by design, and this site is a CONSEQUENCE of making Record asynchronous:
        // the gate became a SemaphoreSlim, whose Wait() has a WaitAsync() sibling, so the rule now sees this
        // method. It cannot be obeyed: IDisposable.Dispose has no await, and SemaphoreSlim.Wait is a
        // blocking synchronisation primitive with no continuation to starve — the exact class OVERFIT039
        // excludes for the same reason. Disposal happens once, at host shutdown, when no request can be
        // holding the gate.
        //
        // WHAT IS LOST: nothing measurable here; the alternative is IAsyncDisposable, which would make the
        // DI container dispose this asynchronously but buys a demo nothing.
#pragma warning disable OVERFIT040
        public void Dispose()
#pragma warning restore OVERFIT040
        {
            _gate.Wait();

            try
            {
                _writer?.Dispose();
            }
            finally
            {
                _gate.Release();
                _gate.Dispose();
            }
        }
    }
}
