// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;

namespace DevOnBike.Overfit.Tokenization
{
    /// <summary>
    /// Byte-Pair Encoding (BPE) tokenizer compatible with GPT-1 and GPT-2 vocabularies.
    ///
    /// Loads vocabulary from two files:
    ///   vocab.json  — JSON map: token_string → token_id
    ///   merges.txt  — BPE merge rules, one per line: "token_a token_b"
    ///
    /// These files are available from HuggingFace for GPT-2:
    ///   https://huggingface.co/gpt2/resolve/main/vocab.json
    ///   https://huggingface.co/gpt2/resolve/main/merges.txt
    ///
    /// GPT-1 uses a similar format (40478 vocab).
    ///
    /// Usage:
    ///   var tokenizer = BytePairEncoder.Load("vocab.json", "merges.txt");
    ///   int[] ids  = tokenizer.Encode("Hello, world!");
    ///   string txt = tokenizer.Decode(ids);
    ///
    /// Implementation: greedy BPE tokenization without regex pre-tokenization.
    /// This is a simplified version — production use should add regex word splitting
    /// matching GPT-2's tiktoken patterns.
    /// </summary>
    public sealed class BytePairEncoder : ITokenizer
    {
        private readonly Dictionary<string, int> _tokenToId;
        private readonly string[] _idToToken;
        private readonly List<(string A, string B)> _merges;

        // Byte-level encoding table (matches GPT-2 byte_encoder)
        private static readonly string[] ByteEncoder = BuildByteEncoder();

        private BytePairEncoder(
            Dictionary<string, int> tokenToId,
            string[] idToToken,
            List<(string, string)> merges)
        {
            _tokenToId = tokenToId;
            _idToToken = idToToken;
            _merges    = merges;
        }

        public int VocabSize      => _idToToken.Length;
        public int UnknownTokenId => _tokenToId.TryGetValue("[UNK]", out var id) ? id : 0;

        /// <summary>
        /// Loads BPE tokenizer from vocab.json and merges.txt.
        /// </summary>
        public static BytePairEncoder Load(string vocabJsonPath, string mergesPath)
        {
            var tokenToId = ParseVocabJson(
                File.ReadAllText(vocabJsonPath));

            var idToToken = new string[tokenToId.Count];
            
            foreach (var kv in tokenToId)
            {
                idToToken[kv.Value] = kv.Key;
            }

            var merges = ParseMerges(File.ReadAllLines(mergesPath));

            return new BytePairEncoder(tokenToId, idToToken, merges);
        }

        /// <summary>
        /// Loads BPE tokenizer from in-memory strings (useful for embedding in resources).
        /// </summary>
        public static BytePairEncoder LoadFromStrings(string vocabJson, string mergesText)
        {
            var tokenToId = ParseVocabJson(vocabJson);
            var idToToken = new string[tokenToId.Count];
            
            foreach (var kv in tokenToId)
            {
                idToToken[kv.Value] = kv.Key;
            }
            
            var merges = ParseMerges(mergesText.Split('\n'));
            
            return new BytePairEncoder(tokenToId, idToToken, merges);
        }

        public int[] Encode(string text)
        {
            if (string.IsNullOrEmpty(text))
            {
                return [];
            }

            var result = new List<int>();

            // Simple whitespace pre-tokenization (no regex for no-dependency build).
            // For full GPT-2 parity, replace with tiktoken-style word splitting.
            var words = SplitWords(text);

            foreach (var word in words)
            {
                var tokens = BpeEncode(word);
                foreach (var t in tokens)
                {
                    if (_tokenToId.TryGetValue(t, out var id))
                    {
                        result.Add(id);
                    }
                    else
                    {
                        result.Add(UnknownTokenId);
                    }
                }
            }

            return result.ToArray();
        }

        public string Decode(int[] tokenIds)
        {
            var sb = new StringBuilder();

            foreach (var id in tokenIds)
            {
                if (id >= 0 && id < _idToToken.Length)
                {
                    sb.Append(DecodeToken(id));
                }
            }

            return sb.ToString();
        }

        public string DecodeToken(int tokenId)
        {
            if (tokenId < 0 || tokenId >= _idToToken.Length)
            {
                return "?";
            }

            var token = _idToToken[tokenId];

            // Decode byte-level encoding back to UTF-8
            return ByteDecode(token);
        }

        // ── Private ──────────────────────────────────────────────────────────

        private List<string> BpeEncode(string word)
        {
            // Byte-encode the word
            var bytes = Encoding.UTF8.GetBytes(word);
            var chars = new List<string>(bytes.Length);

            foreach (var b in bytes)
            {
                chars.Add(ByteEncoder[b]);
            }

            if (chars.Count <= 1)
            {
                return chars;
            }

            // Greedy BPE merge
            while (chars.Count > 1)
            {
                var bestMergeIdx = -1;
                var bestPairIdx  = int.MaxValue;

                for (var i = 0; i < chars.Count - 1; i++)
                {
                    var pair = (chars[i], chars[i + 1]);
                    var mergeIdx = FindMerge(pair);

                    if (mergeIdx >= 0 && mergeIdx < bestPairIdx)
                    {
                        bestPairIdx  = mergeIdx;
                        bestMergeIdx = i;
                    }
                }

                if (bestMergeIdx < 0)
                {
                    break;
                }

                // Apply merge
                var merged = chars[bestMergeIdx] + chars[bestMergeIdx + 1];
                
                chars[bestMergeIdx] = merged;
                chars.RemoveAt(bestMergeIdx + 1);
            }

            return chars;
        }

        private int FindMerge((string A, string B) pair)
        {
            for (var i = 0; i < _merges.Count; i++)
            {
                if (_merges[i].A == pair.A && _merges[i].B == pair.B)
                {
                    return i;
                }
            }

            return -1;
        }

        private static IEnumerable<string> SplitWords(string text)
        {
            // Simple split: whitespace-aware, preserves leading space (GPT-2 convention).
            var start = 0;

            for (var i = 1; i <= text.Length; i++)
            {
                if (i == text.Length || (char.IsWhiteSpace(text[i]) && !char.IsWhiteSpace(text[i - 1])))
                {
                    if (i < text.Length && char.IsWhiteSpace(text[i]))
                    {
                        yield return text.Substring(start, i - start);
                        start = i;
                    }
                    else if (i == text.Length)
                    {
                        yield return text.Substring(start, i - start);
                    }
                }
                else if (!char.IsWhiteSpace(text[i]) && char.IsWhiteSpace(text[i - 1]))
                {
                    yield return text.Substring(start, i - start);
                    start = i;
                }
            }
        }

        private static string ByteDecode(string token)
        {
            // Reverse byte-level encoding: Ġ→' ', etc.
            var bytes = new List<byte>(token.Length);

            for (var i = 0; i < token.Length; i++)
            {
                var c = token[i];

                // Find byte value for this encoded character
                var found = false;

                for (var b = 0; b < 256; b++)
                {
                    if (ByteEncoder[b].Length == 1 && ByteEncoder[b][0] == c)
                    {
                        bytes.Add((byte)b);
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    // Multi-char encoding or unknown — pass through as UTF-8
                    var charBytes = Encoding.UTF8.GetBytes(new[] { c });
                    bytes.AddRange(charBytes);
                }
            }

            try { return Encoding.UTF8.GetString(bytes.ToArray()); }
            catch { return token; }
        }

        private static Dictionary<string, int> ParseVocabJson(string json)
        {
            // Minimal JSON parser for {"token": id, ...} format.
            // No dependency on System.Text.Json to stay AOT-safe.
            var result = new Dictionary<string, int>(StringComparer.Ordinal);
            var pos    = json.IndexOf('{');
            if (pos < 0)
            {
                return result;
            }

            pos++;

            while (pos < json.Length)
            {
                // Skip whitespace
                while (pos < json.Length && json[pos] != '"' && json[pos] != '}')
                {
                    pos++;
                }
                if (pos >= json.Length || json[pos] == '}')
                {
                    break;
                }

                // Read key
                pos++; // skip "
                var keyStart = pos;
                while (pos < json.Length && json[pos] != '"')
                {
                    pos++;
                }
                var key = json.Substring(keyStart, pos - keyStart);
                pos++; // skip closing "

                // Skip : and whitespace
                while (pos < json.Length && (json[pos] == ':' || json[pos] == ' '))
                {
                    pos++;
                }

                // Read value (integer)
                var numStart = pos;
                while (pos < json.Length && (json[pos] >= '0' && json[pos] <= '9'))
                {
                    pos++;
                }

                if (int.TryParse(json.Substring(numStart, pos - numStart), out var id))
                {
                    result[key] = id;
                }

                // Skip comma
                while (pos < json.Length && json[pos] != '"' && json[pos] != '}')
                {
                    pos++;
                }
            }

            return result;
        }

        private static List<(string, string)> ParseMerges(string[] lines)
        {
            var merges = new List<(string, string)>(lines.Length);

            foreach (var line in lines)
            {
                var trimmed = line.Trim();
                if (trimmed.StartsWith("#") || trimmed.Length == 0)
                {
                    continue;
                }

                var spaceIdx = trimmed.IndexOf(' ');
                if (spaceIdx < 0)
                {
                    continue;
                }

                merges.Add((trimmed.Substring(0, spaceIdx), trimmed.Substring(spaceIdx + 1)));
            }

            return merges;
        }

        /// <summary>
        /// Builds the GPT-2 byte encoder table.
        /// Maps each byte (0-255) to a printable Unicode character.
        /// </summary>
        private static string[] BuildByteEncoder()
        {
            var result = new string[256];

            // Printable ASCII: '!' (33) to '~' (126)
            for (var b = 33; b <= 126; b++)
            {
                result[b] = ((char)b).ToString();
            }

            // '¡' (161) to '¬' (172)
            for (var b = 161; b <= 172; b++)
            {
                result[b] = ((char)b).ToString();
            }

            // '®' (174) to 'ÿ' (255)
            for (var b = 174; b <= 255; b++)
            {
                result[b] = ((char)b).ToString();
            }

            // Remaining bytes (0-32, 127-160, 173) map to Unicode starting at U+0100
            var n = 0;
            for (var b = 0; b < 256; b++)
            {
                if (result[b] == null)
                {
                    result[b] = ((char)(256 + n)).ToString();
                    n++;
                }
            }

            // Space (32) → 'Ġ' (U+0120), standard GPT-2 convention
            result[32] = "Ġ";

            return result;
        }
    }
}
