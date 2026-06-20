// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;

namespace DevOnBike.Overfit.LanguageModels.Chat
{
    /// <summary>
    /// Streaming detector for arbitrary <i>string</i> stop sequences (e.g. <c>"\nUser:"</c>),
    /// complementing the token-level stops in <c>StreamingOptions</c>. Decoded text pieces
    /// are fed in as they are generated; the detector emits the text that is safe to show
    /// and holds back any trailing run that could still grow into a stop sequence — so a
    /// partial marker split across two pieces is never shown, and the stop marker itself is
    /// never emitted.
    ///
    /// Usage in a generation loop:
    /// <code>
    /// var stops = new StopSequenceDetector("\nUser:", "&lt;|im_end|&gt;");
    /// foreach (var tokenId in session.StreamGenerate(...))
    /// {
    ///     var text = stops.Append(tokenizer.Decode(tokenId));
    ///     if (text.Length > 0) { Console.Write(text); }
    ///     if (stops.Stopped) { break; }
    /// }
    /// Console.Write(stops.Flush());
    /// </code>
    /// </summary>
    public sealed class StopSequenceDetector
    {
        private readonly string[] _stops;
        private readonly StringBuilder _buffer = new();

        public StopSequenceDetector(params string[] stopSequences)
        {
            var kept = new List<string>(stopSequences?.Length ?? 0);
            if (stopSequences is not null)
            {
                foreach (var s in stopSequences)
                {
                    if (!string.IsNullOrEmpty(s))
                    {
                        kept.Add(s);
                    }
                }
            }
            _stops = kept.ToArray();
        }

        /// <summary>True once a stop sequence has completed; further <see cref="Append"/> calls no-op.</summary>
        public bool Stopped
        {
            get; private set;
        }

        /// <summary>
        /// Appends a decoded <paramref name="piece"/> and returns the text that is now safe
        /// to emit (text that cannot be part of a future stop match). When a stop sequence
        /// completes, <see cref="Stopped"/> is set and the returned text is everything before
        /// the stop (the stop and anything after it are dropped).
        /// </summary>
        public string Append(string piece)
        {
            if (Stopped || string.IsNullOrEmpty(piece))
            {
                return string.Empty;
            }
            if (_stops.Length == 0)
            {
                return piece;
            }

            _buffer.Append(piece);
            var buf = _buffer.ToString();

            // Earliest complete stop occurrence wins.
            var stopAt = -1;
            foreach (var s in _stops)
            {
                var idx = buf.IndexOf(s, StringComparison.Ordinal);
                if (idx >= 0 && (stopAt < 0 || idx < stopAt))
                {
                    stopAt = idx;
                }
            }
            if (stopAt >= 0)
            {
                Stopped = true;
                _buffer.Clear();
                return buf[..stopAt];
            }

            // No complete stop: hold back the longest suffix that prefixes some stop.
            var hold = LongestHeldSuffix(buf);
            var emit = buf[..(buf.Length - hold)];
            _buffer.Clear();
            _buffer.Append(buf, buf.Length - hold, hold);
            return emit;
        }

        /// <summary>
        /// Returns any buffered text held back for partial-match safety. Call once when the
        /// stream ends naturally (returns empty if a stop already fired).
        /// </summary>
        public string Flush()
        {
            if (Stopped)
            {
                _buffer.Clear();
                return string.Empty;
            }
            var rest = _buffer.ToString();
            _buffer.Clear();
            return rest;
        }

        // Largest k (1..len-1 of a stop) such that buf's last k chars equal a stop's first k chars.
        private int LongestHeldSuffix(string buf)
        {
            var best = 0;
            foreach (var s in _stops)
            {
                var max = Math.Min(buf.Length, s.Length - 1);
                for (var k = max; k > best; k--)
                {
                    if (EndsWithPrefix(buf, s, k))
                    {
                        if (k > best)
                        {
                            best = k;
                        }
                        break;
                    }
                }
            }
            return best;
        }

        private static bool EndsWithPrefix(string buf, string stop, int k)
        {
            var offset = buf.Length - k;
            for (var i = 0; i < k; i++)
            {
                if (buf[offset + i] != stop[i])
                {
                    return false;
                }
            }
            return true;
        }
    }
}
