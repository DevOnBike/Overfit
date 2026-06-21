// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;

namespace DevOnBike.Overfit.Redaction
{
    /// <summary>
    /// Checksum / structural validators used as the precision gate on loose regex matches (see
    /// <see cref="RedactionRule.Validator"/>). They turn "any 11 digits" into "a real PESEL", cutting the false
    /// positives that a regex alone produces. Weight tables are <c>ReadOnlySpan&lt;int&gt;</c> collection
    /// expressions (static blobs — no per-call allocation).
    /// </summary>
    public static class RedactionValidators
    {
        /// <summary>Polish PESEL — 11 digits with a weighted mod-10 control digit.</summary>
        public static bool Pesel(string value)
        {
            var digits = Digits(value);
            if (digits.Length != 11)
            {
                return false;
            }

            ReadOnlySpan<int> weights = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3];
            var sum = 0;
            for (var i = 0; i < 10; i++)
            {
                sum += (digits[i] - '0') * weights[i];
            }

            var control = (10 - (sum % 10)) % 10;
            return control == digits[10] - '0';
        }

        /// <summary>Polish NIP (tax id) — 10 digits with a weighted mod-11 control digit.</summary>
        public static bool Nip(string value)
        {
            var digits = Digits(value);
            if (digits.Length != 10)
            {
                return false;
            }

            ReadOnlySpan<int> weights = [6, 5, 7, 2, 3, 4, 5, 6, 7];
            var sum = 0;
            for (var i = 0; i < 9; i++)
            {
                sum += (digits[i] - '0') * weights[i];
            }

            var control = sum % 11;
            return control != 10 && control == digits[9] - '0';
        }

        /// <summary>Polish REGON — 9 digits with a weighted mod-11 control digit (14-digit REGON not covered).</summary>
        public static bool Regon(string value)
        {
            var digits = Digits(value);
            if (digits.Length != 9)
            {
                return false;
            }

            ReadOnlySpan<int> weights = [8, 9, 2, 3, 4, 5, 6, 7];
            var sum = 0;
            for (var i = 0; i < 8; i++)
            {
                sum += (digits[i] - '0') * weights[i];
            }

            var control = sum % 11 % 10;
            return control == digits[8] - '0';
        }

        /// <summary>Luhn (mod-10) check for a 13–19 digit payment card number.</summary>
        public static bool Luhn(string value)
        {
            var digits = Digits(value);
            if (digits.Length is < 13 or > 19)
            {
                return false;
            }

            var sum = 0;
            var alternate = false;
            for (var i = digits.Length - 1; i >= 0; i--)
            {
                var n = digits[i] - '0';
                if (alternate)
                {
                    n *= 2;
                    if (n > 9)
                    {
                        n -= 9;
                    }
                }
                sum += n;
                alternate = !alternate;
            }

            return sum % 10 == 0;
        }

        /// <summary>Polish IBAN — <c>PL</c> + 26 digits, validated by the ISO 7064 mod-97 == 1 rule.</summary>
        public static bool IbanPl(string value)
        {
            var normalized = new StringBuilder(value.Length);
            foreach (var c in value)
            {
                if (!char.IsWhiteSpace(c) && c != '-')
                {
                    normalized.Append(char.ToUpperInvariant(c));
                }
            }

            var iban = normalized.ToString();
            if (iban.Length != 28 || iban[0] != 'P' || iban[1] != 'L')
            {
                return false;
            }

            // Move the first 4 chars (PL + check digits) to the end, then take the whole thing mod 97.
            long remainder = 0;
            for (var pass = 0; pass < iban.Length; pass++)
            {
                var c = iban[(pass + 4) % iban.Length];
                if (char.IsDigit(c))
                {
                    remainder = ((remainder * 10) + (c - '0')) % 97;
                }
                else if (c is >= 'A' and <= 'Z')
                {
                    remainder = ((remainder * 100) + (c - 'A' + 10)) % 97;
                }
                else
                {
                    return false;
                }
            }

            return remainder == 1;
        }

        private static string Digits(string value)
        {
            var sb = new StringBuilder(value.Length);
            foreach (var c in value)
            {
                if (char.IsDigit(c))
                {
                    sb.Append(c);
                }
            }
            return sb.ToString();
        }
    }
}
