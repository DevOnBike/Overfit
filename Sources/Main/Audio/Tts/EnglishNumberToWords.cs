// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;

namespace DevOnBike.Overfit.Audio.Tts
{
    /// <summary>
    /// Spells an integer out in English words (cardinals) — the part of TTS text normalization that turns
    /// <c>1234</c> into "one thousand two hundred thirty four" so the model speaks it instead of guessing at the
    /// digits. Used by <see cref="TtsTextNormalizer"/>; pure, allocation-light, model-free.
    /// </summary>
    public static class EnglishNumberToWords
    {
        private static readonly string[] Ones =
        [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
        ];

        private static readonly string[] Tens =
            ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

        // Index = number of 3-digit groups from the right.
        // Index = number of 3-digit groups from the right. Six entries covers up to 999 quadrillion; long
        // reaches 9.22 quintillion, so the last two entries exist because the type does, not because anyone
        // will say them. Without them a nineteen-digit number in ordinary text threw IndexOutOfRange out of
        // POST /v1/audio/speech and out of the CLI — a crash from a number, on a text-normalisation path.
        private static readonly string[] Scales =
            ["", " thousand", " million", " billion", " trillion", " quadrillion", " quintillion"];

        /// <summary>Converts <paramref name="value"/> to English cardinal words (e.g. -42 → "minus forty two").</summary>
        public static string Convert(long value)
        {
            if (value == 0)
            {
                return "zero";
            }

            var negative = value < 0;
            // Avoid overflow on long.MinValue by working in unsigned magnitude.
            var magnitude = negative ? (ulong)(-(value + 1)) + 1UL : (ulong)value;

            var groups = new List<int>();
            while (magnitude > 0)
            {
                groups.Add((int)(magnitude % 1000));
                magnitude /= 1000;
            }

            var sb = new StringBuilder();
            if (negative)
            {
                sb.Append("minus ");
            }

            var first = true;
            for (var g = groups.Count - 1; g >= 0; g--)
            {
                if (groups[g] == 0)
                {
                    continue;
                }
                if (!first)
                {
                    sb.Append(' ');
                }
                AppendBelowThousand(sb, groups[g]);

                // Belt and braces: ulong cannot produce a group index past six, and a silent wrong reading is
                // still better than taking a request down if that ever stops being true.
                sb.Append(g < Scales.Length ? Scales[g] : string.Empty);
                first = false;
            }
            return sb.ToString();
        }

        private static void AppendBelowThousand(StringBuilder sb, int n)
        {
            var wrote = false;
            if (n >= 100)
            {
                sb.Append(Ones[n / 100]).Append(" hundred");
                n %= 100;
                wrote = true;
            }
            if (n >= 20)
            {
                if (wrote)
                {
                    sb.Append(' ');
                }
                sb.Append(Tens[n / 10]);
                if (n % 10 > 0)
                {
                    sb.Append(' ').Append(Ones[n % 10]);
                }
            }
            if (n is > 0 and < 20)
            {
                if (wrote)
                {
                    sb.Append(' ');
                }
                sb.Append(Ones[n]);
            }
        }
    }
}
