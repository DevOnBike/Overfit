// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;

namespace DevOnBike.Overfit.Redaction
{
    /// <summary>
    /// Scans a model's streamed (SSE) response for secrets/PII the model itself produced and masks them on the fly —
    /// the streaming counterpart of <see cref="Redactor.ScanResponse"/>. A sensitive token (an API key, e-mail, IP…)
    /// can arrive split across several chunks, so this holds back the trailing run since the last whitespace until a
    /// later fragment completes it: because the sensitive patterns it targets are whitespace-free, the whole token
    /// always accumulates in the buffer before a whitespace flushes it, and the scan therefore sees it intact.
    ///
    /// <para>Feed each delta fragment to <see cref="Push"/> and emit the returned text; call <see cref="Flush"/> at
    /// end-of-stream. Masked spans accumulate in <see cref="MaskedMatches"/> for the audit. Limitation: patterns that
    /// legitimately contain whitespace (a card number written with spaces, a spaced IBAN) are not reliably caught
    /// mid-stream — only when they land whole inside a single flush.</para>
    /// </summary>
    public sealed class StreamingResponseScanner
    {
        // A whitespace-free token longer than this is not a credential we mask — emit it rather than buffer unbounded.
        private const int MaxHeldRun = 1024;

        private readonly Redactor _redactor;
        private readonly RedactionPolicy _policy;
        private readonly StringBuilder _pending = new();
        private readonly List<RedactionMatch> _masked = [];

        public StreamingResponseScanner(Redactor redactor, RedactionPolicy policy)
        {
            ArgumentNullException.ThrowIfNull(redactor);
            ArgumentNullException.ThrowIfNull(policy);
            _redactor = redactor;
            _policy = policy;
        }

        /// <summary>Spans masked so far across the whole stream — for the request's counts-only audit.</summary>
        public IReadOnlyList<RedactionMatch> MaskedMatches => _masked;

        /// <summary>
        /// Appends <paramref name="fragment"/> and returns the text safe to emit now (model secrets masked), holding
        /// back only a trailing partial token until a whitespace completes it.
        /// </summary>
        public string Push(string fragment)
        {
            ArgumentNullException.ThrowIfNull(fragment);

            _pending.Append(fragment);
            var buffer = _pending.ToString();

            var emitUpTo = SafeEmitBoundary(buffer);
            if (emitUpTo == 0)
            {
                return string.Empty;
            }

            var emit = buffer.Substring(0, emitUpTo);
            _pending.Clear();
            _pending.Append(buffer, emitUpTo, buffer.Length - emitUpTo);

            return ScanAndCollect(emit);
        }

        /// <summary>Scans and emits any held-back tail at end-of-stream.</summary>
        public string Flush()
        {
            if (_pending.Length == 0)
            {
                return string.Empty;
            }

            var rest = _pending.ToString();
            _pending.Clear();
            return ScanAndCollect(rest);
        }

        private string ScanAndCollect(string text)
        {
            var scan = _redactor.ScanResponse(text, _policy);
            if (scan.Matches.Count > 0)
            {
                _masked.AddRange(scan.Matches);
            }

            return scan.Text;
        }

        // Index up to which the buffer is safe to emit. Everything before the last whitespace is complete (no
        // whitespace-free token spans it); the trailing run is held — unless it has grown past MaxHeldRun, in which
        // case it is not a credential and is released to avoid unbounded buffering.
        private static int SafeEmitBoundary(string buffer)
        {
            var lastWhitespace = -1;
            for (var i = buffer.Length - 1; i >= 0; i--)
            {
                if (char.IsWhiteSpace(buffer[i]))
                {
                    lastWhitespace = i;
                    break;
                }
            }

            if (lastWhitespace < 0)
            {
                // One unbroken run so far — hold it (it may still be growing into a token) until it exceeds the cap.
                return buffer.Length > MaxHeldRun ? buffer.Length : 0;
            }

            var trailingRun = buffer.Length - 1 - lastWhitespace;
            if (trailingRun > MaxHeldRun)
            {
                return buffer.Length;
            }

            return lastWhitespace + 1;
        }
    }
}
