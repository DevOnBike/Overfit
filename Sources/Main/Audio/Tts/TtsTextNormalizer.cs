// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;

namespace DevOnBike.Overfit.Audio.Tts
{
    /// <summary>
    /// Cleans text before it reaches the TTS model so the model speaks it well: expands numbers to words
    /// (<c>3.14</c> → "three point one four"), common symbols/abbreviations (<c>.NET</c> → "dot net", <c>e.g.</c> →
    /// "for example"), and applies a pronunciation lexicon for out-of-vocabulary / brand words the model would
    /// otherwise mangle (<c>Overfit</c> → "over fit"). The lexicon is user-extendable. Model-free and deterministic.
    /// </summary>
    public sealed class TtsTextNormalizer
    {
        // (find, replace) literal expansions applied case-insensitively before tokenization. These contain dots /
        // symbols, so they are handled here rather than in the per-word lexicon. Order matters (longest first).
        private static readonly (string Find, string Replace)[] SymbolExpansions =
        [
            (".net", " dot net "),
            ("e.g.", " for example "),
            ("i.e.", " that is "),
            ("etc.", " etcetera "),
            ("vs.", " versus "),
            ("mr.", " mister "),
            ("mrs.", " missus "),
            ("dr.", " doctor "),
            ("&", " and "),
            ("%", " percent "),
            ("@", " at "),
            ("+", " plus "),
            ("=", " equals "),
            ("#", " number "),
        ];

        // Default pronunciation lexicon (whole-word, case-insensitive). Conservative on purpose.
        private static readonly (string Word, string Say)[] DefaultLexicon =
        [
            ("overfit", "over fit"),
            ("gguf", "G G U F"),
            ("llm", "L L M"),
            ("tts", "T T S"),
            ("stt", "S T T"),
            ("cpu", "C P U"),
            ("gpu", "G P U"),
            ("api", "A P I"),
            ("sql", "S Q L"),
            ("json", "jason"),
            // Common tech acronyms — spaced capitals → Orpheus spells them out ("ai" → "ay eye"). Kept to
            // unambiguous ones (no real-word clashes like "ram"/"it"); the model reads these wrong raw.
            ("ai", "A I"),
            ("ml", "M L"),
            ("ui", "U I"),
            ("ux", "U X"),
            ("io", "I O"),
            ("os", "O S"),
            ("sdk", "S D K"),
            ("cli", "C L I"),
            ("ide", "I D E"),
            ("url", "U R L"),
            ("gpt", "G P T"),
            ("html", "H T M L"),
            ("css", "C S S"),
            ("xml", "X M L"),
        ];

        private readonly Dictionary<string, string> _lexicon;

        public TtsTextNormalizer(IReadOnlyDictionary<string, string>? extraLexicon = null)
        {
            _lexicon = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var (word, say) in DefaultLexicon)
            {
                _lexicon[word] = say;
            }
            if (extraLexicon is not null)
            {
                foreach (var kv in extraLexicon)
                {
                    _lexicon[kv.Key] = kv.Value; // caller overrides / extends
                }
            }
        }

        /// <summary>Normalizes <paramref name="text"/> for synthesis.</summary>
        public string Normalize(string text)
        {
            if (string.IsNullOrEmpty(text))
            {
                return string.Empty;
            }

            var expanded = ExpandSymbols(text);
            var sb = new StringBuilder(expanded.Length + 16);

            var i = 0;
            while (i < expanded.Length)
            {
                var c = expanded[i];
                if (char.IsDigit(c) || (c == '-' && i + 1 < expanded.Length && char.IsDigit(expanded[i + 1])))
                {
                    i = AppendNumber(sb, expanded, i);
                }
                else if (IsWordChar(c))
                {
                    i = AppendWord(sb, expanded, i);
                }
                else
                {
                    sb.Append(c);
                    i++;
                }
            }

            return CollapseWhitespace(sb.ToString());
        }

        private static bool IsWordChar(char c) => char.IsLetter(c) || c == '\'';

        private int AppendWord(StringBuilder sb, string text, int i)
        {
            var start = i;
            while (i < text.Length && IsWordChar(text[i]))
            {
                i++;
            }
            var word = text[start..i];
            sb.Append(_lexicon.TryGetValue(word, out var say) ? say : word);
            return i;
        }

        private static int AppendNumber(StringBuilder sb, string text, int i)
        {
            var start = i;
            if (text[i] == '-')
            {
                i++;
            }
            while (i < text.Length && (char.IsDigit(text[i]) || text[i] == ','))
            {
                i++;
            }

            // A decimal point only if a digit follows (otherwise it is sentence punctuation).
            var fracStart = -1;
            var fracEnd = -1;
            if (i + 1 < text.Length && text[i] == '.' && char.IsDigit(text[i + 1]))
            {
                i++;
                fracStart = i;
                while (i < text.Length && char.IsDigit(text[i]))
                {
                    i++;
                }
                fracEnd = i;
            }

            var intPart = text[start..(fracStart < 0 ? i : fracStart - 1)].Replace(",", string.Empty);
            if (long.TryParse(intPart, out var intValue))
            {
                sb.Append(EnglishNumberToWords.Convert(intValue));
            }
            else
            {
                // Too long for a long, or malformed → speak the digits individually.
                AppendDigits(sb, intPart);
            }

            if (fracStart >= 0)
            {
                sb.Append(" point");
                for (var d = fracStart; d < fracEnd; d++)
                {
                    sb.Append(' ').Append(EnglishNumberToWords.Convert(text[d] - '0'));
                }
            }

            return i;
        }

        private static void AppendDigits(StringBuilder sb, string digits)
        {
            var first = true;
            foreach (var ch in digits)
            {
                if (ch == '-')
                {
                    sb.Append("minus");
                    first = false;
                    continue;
                }
                if (!char.IsDigit(ch))
                {
                    continue;
                }
                if (!first)
                {
                    sb.Append(' ');
                }
                sb.Append(EnglishNumberToWords.Convert(ch - '0'));
                first = false;
            }
        }

        private static string ExpandSymbols(string text)
        {
            var s = text;
            foreach (var (find, replace) in SymbolExpansions)
            {
                s = ReplaceIgnoreCase(s, find, replace);
            }
            return s;
        }

        private static string ReplaceIgnoreCase(string text, string find, string replace)
        {
            var idx = text.IndexOf(find, StringComparison.OrdinalIgnoreCase);
            if (idx < 0)
            {
                return text;
            }

            var sb = new StringBuilder(text.Length);
            var pos = 0;
            while (idx >= 0)
            {
                sb.Append(text, pos, idx - pos);
                sb.Append(replace);
                pos = idx + find.Length;
                idx = text.IndexOf(find, pos, StringComparison.OrdinalIgnoreCase);
            }
            sb.Append(text, pos, text.Length - pos);
            return sb.ToString();
        }

        private static string CollapseWhitespace(string text)
        {
            var sb = new StringBuilder(text.Length);
            var inSpace = false;
            foreach (var c in text)
            {
                if (char.IsWhiteSpace(c))
                {
                    inSpace = true;
                    continue;
                }
                if (inSpace && sb.Length > 0)
                {
                    sb.Append(' ');
                }
                inSpace = false;
                sb.Append(c);
            }
            return sb.ToString();
        }
    }
}
