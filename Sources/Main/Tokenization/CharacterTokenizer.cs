// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tokenization
{
    /// <summary>
    /// Character-level tokenizer.
    ///
    /// Maps each unique character to an integer token id.
    /// Useful for toy language models and demos — no external vocab file needed.
    ///
    /// Vocabulary is built from the training corpus at construction time,
    /// or loaded from a predefined character set.
    ///
    /// Special tokens:
    ///   0  = [UNK] — unknown character not in vocabulary
    ///   1  = [BOS] — beginning of sequence (optional)
    ///   2  = [EOS] — end of sequence (optional)
    ///
    /// Usage:
    ///   var tokenizer = CharacterTokenizer.FromCorpus("hello world");
    ///   int[] ids  = tokenizer.Encode("hello");
    ///   string txt = tokenizer.Decode(ids);
    /// </summary>
    public sealed class CharacterTokenizer : ITokenizer
    {
        private readonly Dictionary<char, int> _charToId;
        private readonly string[] _idToToken;

        public const int UnknownId = 0;
        public const int BosId     = 1;
        public const int EosId     = 2;

        private CharacterTokenizer(Dictionary<char, int> charToId, string[] idToToken)
        {
            _charToId = charToId;
            _idToToken = idToToken;
        }

        public int VocabSize       => _idToToken.Length;
        public int UnknownTokenId  => UnknownId;

        /// <summary>
        /// Builds a character tokenizer from a corpus.
        /// Scans all unique characters and assigns ids starting from 3
        /// (ids 0,1,2 reserved for UNK, BOS, EOS).
        /// </summary>
        public static CharacterTokenizer FromCorpus(string corpus)
        {
            var chars = new SortedSet<char>();

            for (var i = 0; i < corpus.Length; i++)
            {
                chars.Add(corpus[i]);
            }

            return FromCharSet(chars);
        }

        /// <summary>
        /// Builds a tokenizer from an explicit character set.
        /// </summary>
        public static CharacterTokenizer FromCharSet(IEnumerable<char> chars)
        {
            var charToId  = new Dictionary<char, int>();
            var idToToken = new List<string> { "[UNK]", "[BOS]", "[EOS]" };

            foreach (var c in chars)
            {
                if (!charToId.ContainsKey(c))
                {
                    charToId[c] = idToToken.Count;
                    idToToken.Add(c.ToString());
                }
            }

            return new CharacterTokenizer(charToId, idToToken.ToArray());
        }

        /// <summary>
        /// Builds a tokenizer for ASCII printable characters (32–126) + newline + tab.
        /// Vocab size = 100. Useful for English text without corpus.
        /// </summary>
        public static CharacterTokenizer Ascii()
        {
            var chars = new List<char>();

            for (var c = 32; c <= 126; c++)
            {
                chars.Add((char)c);
            }
            chars.Add('\n');
            chars.Add('\t');

            return FromCharSet(chars);
        }

        public int[] Encode(string text)
        {
            var ids = new int[text.Length];

            for (var i = 0; i < text.Length; i++)
            {
                ids[i] = _charToId.TryGetValue(text[i], out var id) ? id : UnknownId;
            }

            return ids;
        }

        public string Decode(int[] tokenIds)
        {
            var sb = new System.Text.StringBuilder(tokenIds.Length);

            foreach (var id in tokenIds)
            {
                if (id >= 3 && id < _idToToken.Length)
                {
                    sb.Append(_idToToken[id]);
                }
                else if (id == UnknownId)
                {
                    sb.Append('?');
                }
                // BOS/EOS: skip
            }

            return sb.ToString();
        }

        public string DecodeToken(int tokenId)
        {
            if (tokenId >= 0 && tokenId < _idToToken.Length)
            {
                return _idToToken[tokenId];
            }

            return "[?]";
        }

        /// <summary>Saves vocabulary to a simple text file (one token per line).</summary>
        public void Save(string path)
        {
            File.WriteAllLines(path, _idToToken);
        }

        /// <summary>Loads vocabulary from a text file saved by <see cref="Save"/>.</summary>
        public static CharacterTokenizer Load(string path)
        {
            var lines    = File.ReadAllLines(path);
            var charToId = new Dictionary<char, int>();

            for (var i = 3; i < lines.Length; i++)
            {
                if (lines[i].Length == 1)
                {
                    charToId[lines[i][0]] = i;
                }
            }

            return new CharacterTokenizer(charToId, lines);
        }
    }
}
