// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;

namespace DevOnBike.Overfit.Redaction
{
    /// <summary>
    /// Restores redaction placeholders in a token stream, where a single placeholder (e.g.
    /// <c>[REDACTED_EMAIL_0]</c>) may be split across several streamed fragments. Feed each delta fragment to
    /// <see cref="Push"/>; it returns the text that is safe to emit now, holding back only a trailing run that could
    /// still be completing into a placeholder. Call <see cref="Flush"/> at end-of-stream to release the remainder.
    ///
    /// <para>The core of the gateway's streaming (SSE) path: the upstream echoes our placeholders back token by
    /// token, and we re-hydrate them on the fly without ever buffering the whole response.</para>
    /// </summary>
    public sealed class StreamingRestorer
    {
        private readonly IReadOnlyList<RedactionMatch> _matches;
        private readonly StringBuilder _pending = new();

        public StreamingRestorer(IReadOnlyList<RedactionMatch> matches)
        {
            ArgumentNullException.ThrowIfNull(matches);
            _matches = matches;
        }

        /// <summary>
        /// Appends <paramref name="fragment"/> to the pending buffer and returns the prefix that can be emitted now
        /// with all complete placeholders restored — withholding only a trailing partial placeholder (if any) until a
        /// later fragment completes it. Returns the fragment unchanged when there is nothing to restore.
        /// </summary>
        public string Push(string fragment)
        {
            ArgumentNullException.ThrowIfNull(fragment);

            if (_matches.Count == 0)
            {
                return fragment;
            }

            _pending.Append(fragment);
            var buffer = _pending.ToString();

            var holdFrom = HoldBoundary(buffer);

            var emit = buffer.Substring(0, holdFrom);
            var hold = buffer.Substring(holdFrom);

            _pending.Clear();
            _pending.Append(hold);

            return Redactor.Restore(emit, _matches);
        }

        /// <summary>
        /// Releases any held-back tail at end-of-stream, restoring complete placeholders in it. A genuinely partial
        /// placeholder (the stream ended mid-token) is emitted verbatim — the original value stays hidden either way.
        /// </summary>
        public string Flush()
        {
            if (_pending.Length == 0)
            {
                return string.Empty;
            }

            var rest = _pending.ToString();
            _pending.Clear();
            return Redactor.Restore(rest, _matches);
        }

        // Returns the index from which the buffer must be held back because its tail could still grow into a
        // placeholder. A partial placeholder is always trailing (it has no closing ']' yet), so only the last '['
        // can open one — earlier brackets are either closed or not ours, and are safe to emit.
        private int HoldBoundary(string buffer)
        {
            var lastOpen = buffer.LastIndexOf('[');
            if (lastOpen < 0)
            {
                return buffer.Length;
            }

            var tail = buffer.AsSpan(lastOpen);

            // A closing ']' means the last bracket is already a complete token — Restore handles it, emit everything.
            if (tail.IndexOf(']') >= 0)
            {
                return buffer.Length;
            }

            // Open bracket with no close: hold it only if it is a prefix of one of our placeholders; otherwise it is
            // ordinary text (markdown, code) that will never become a placeholder, so emit it.
            return IsPlaceholderPrefix(tail) ? lastOpen : buffer.Length;
        }

        private bool IsPlaceholderPrefix(ReadOnlySpan<char> tail)
        {
            foreach (var match in _matches)
            {
                var placeholder = match.Placeholder.AsSpan();
                var n = Math.Min(tail.Length, placeholder.Length);
                if (tail.Slice(0, n).SequenceEqual(placeholder.Slice(0, n)))
                {
                    return true;
                }
            }

            return false;
        }
    }
}
