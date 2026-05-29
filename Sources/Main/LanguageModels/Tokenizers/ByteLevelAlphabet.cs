// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Tokenizers
{
    /// <summary>
    /// The GPT-2 "ByteLevel" byte↔char alphabet: a reversible mapping from each of the 256 raw bytes
    /// to a single printable Unicode char, so byte-level BPE vocab/merges can be expressed as ordinary
    /// strings (e.g. space → <c>Ġ</c>). Shared by every byte-level BPE tokenizer
    /// (<see cref="HuggingFaceBpeTokenizer"/> from tokenizer.json, <see cref="GgufTokenizer"/> from the
    /// GGUF gpt2 vocab) so the table is defined once.
    /// </summary>
    internal static class ByteLevelAlphabet
    {
        /// <summary>byte (0..255) → its ByteLevel display char.</summary>
        public static char[] BuildByteToChar()
        {
            var map = new char[256];
            for (var b = 0; b < 256; b++)
            {
                map[b] = (b >= '!' && b <= '~') || (b >= '¡' && b <= '¬') || (b >= '®' && b <= 'ÿ')
                    ? (char)b
                    : (char)0;
            }
            var n = 0;
            for (var b = 0; b < 256; b++)
            {
                if (map[b] == (char)0) { map[b] = (char)(256 + n++); }
            }
            return map;
        }

        /// <summary>ByteLevel display char → byte (indexed by char code; 65536-wide).</summary>
        public static byte[] BuildCharToByte()
        {
            var forward = BuildByteToChar();
            var reverse = new byte[65536];
            for (var b = 0; b < 256; b++) { reverse[forward[b]] = (byte)b; }
            return reverse;
        }
    }
}
