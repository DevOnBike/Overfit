// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// Escaping for the three stores that share one file through <c>LearnedState</c>.
    ///
    /// <para><b>One implementation, because there were three and all three were wrong the same way.</b>
    /// <c>FloorCalibrator</c>, <c>OperatorLabelStore</c> and <c>SuppressionStore</c> each carried a private
    /// pair built from sequential <see cref="string.Replace(string, string, StringComparison)"/> calls, and
    /// a chain of replacements is not a decoder: each pass runs over the output of the previous one, so a
    /// sequence the earlier pass produced can be matched again by a later one.</para>
    ///
    /// <para><b>The failure, concretely.</b> Take a channel named <c>a\hb</c> — a literal backslash, then
    /// <c>h</c>. Escaping doubles the backslash to <c>a\\hb</c>. Unescaping then looked for <c>\h</c> first
    /// and found it at the <i>second</i> backslash, yielding <c>a\#b</c>. The name that comes back out is
    /// not the name that went in, silently, and the calibration attaches to a key the operator never
    /// configured. The same shape corrupts a tab or a newline in any of the three stores.</para>
    ///
    /// <para>Decoding is therefore a single left-to-right pass: a backslash consumes exactly one following
    /// character and nothing the pass emits can be re-read.</para>
    ///
    /// <para><b><c>#</c> is escaped as well as tab and newline</b>, and that is not cosmetic:
    /// <c>LearnedState</c> separates its sections with <c>### labels</c> / <c>### suppressions</c> lines, so
    /// a channel name — or an operator's typed reason — containing that text moves a section boundary and
    /// silently redistributes the payload between the stores.</para>
    ///
    /// <para><b>Reading a file written before this:</b> everything round-trips except the vanishingly rare
    /// case of a literal <c>\h</c> in text written by the old code, which now decodes to <c>#</c>. Refusing
    /// to read such a file, or versioning the format for it, would cost a week of calibration to protect a
    /// two-character sequence nobody types.</para>
    /// </summary>
    internal static class LearnedStateText
    {
        /// <summary>Makes a value safe to place in a tab-separated field of a section-delimited file.</summary>
        public static string Escape(string value)
        {
            ArgumentNullException.ThrowIfNull(value);

            if (!NeedsEscaping(value))
            {
                return value;
            }

            var text = new StringBuilder(value.Length + 8);

            for (var i = 0; i < value.Length; i++)
            {
                switch (value[i])
                {
                    case '\\':
                        text.Append("\\\\");

                        break;

                    case '\t':
                        text.Append("\\t");

                        break;

                    case '\n':
                        text.Append("\\n");

                        break;

                    case '#':
                        text.Append("\\h");

                        break;

                    default:
                        text.Append(value[i]);

                        break;
                }
            }

            return text.ToString();
        }

        /// <summary>
        /// Reverses <see cref="Escape"/> in one pass.
        ///
        /// <para>A trailing lone backslash, and any escape this encoder does not produce, are emitted
        /// literally rather than treated as an error: the payload is a scratch file whose worst outcome
        /// must be a slightly wrong label, never a guard that refuses to start.</para>
        /// </summary>
        public static string Unescape(string value)
        {
            ArgumentNullException.ThrowIfNull(value);

            if (value.IndexOf('\\', StringComparison.Ordinal) < 0)
            {
                return value;
            }

            var text = new StringBuilder(value.Length);

            for (var i = 0; i < value.Length; i++)
            {
                if (value[i] != '\\' || i + 1 >= value.Length)
                {
                    text.Append(value[i]);

                    continue;
                }

                i++;

                text.Append(value[i] switch
                {
                    '\\' => '\\',
                    't' => '\t',
                    'n' => '\n',
                    'h' => '#',
                    _ => value[i],
                });
            }

            return text.ToString();
        }

        private static bool NeedsEscaping(string value)
        {
            for (var i = 0; i < value.Length; i++)
            {
                if (value[i] is '\\' or '\t' or '\n' or '#')
                {
                    return true;
                }
            }

            return false;
        }
    }
}
