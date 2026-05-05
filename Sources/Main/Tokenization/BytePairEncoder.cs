// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace DevOnBike.Overfit.Tokenization
{
    /// <summary>
    /// Byte-Pair Encoding tokenizer compatible with GPT-2-style vocab.json and merges.txt files.
    ///
    /// This implementation is intentionally dependency-light:
    /// - vocab.json is parsed with System.Text.Json, because GPT-2 token strings contain escaped
    ///   JSON keys that are easy to parse incorrectly with a hand-written JSON parser.
    /// - merges.txt is loaded into a rank dictionary for efficient BPE pair lookup.
    ///
    /// This tokenizer is not part of the zero-allocation inference hot path. It is used for
    /// GPT-2 fixture conversion/inference demos and can allocate during Encode/Decode.
    /// </summary>
    public sealed class BytePairEncoder : ITokenizer
    {
        private static readonly Regex Gpt2TokenPattern = new(
            @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            RegexOptions.CultureInvariant);

        private static readonly string[] ByteEncoder = BuildByteEncoder();
        private static readonly Dictionary<char, byte> ByteDecoder = BuildByteDecoder(ByteEncoder);

        private readonly Dictionary<string, int> _tokenToId;
        private readonly string[] _idToToken;
        private readonly Dictionary<(string A, string B), int> _mergeRanks;

        private BytePairEncoder(
            Dictionary<string, int> tokenToId,
            string[] idToToken,
            Dictionary<(string A, string B), int> mergeRanks)
        {
            _tokenToId = tokenToId;
            _idToToken = idToToken;
            _mergeRanks = mergeRanks;
        }

        public int VocabSize => _idToToken.Length;

        public int UnknownTokenId =>
            _tokenToId.TryGetValue("<|endoftext|>", out var endOfText)
                ? endOfText
                : _tokenToId.TryGetValue("[UNK]", out var unknown)
                    ? unknown
                    : 0;

        public static BytePairEncoder Load(
            string vocabJsonPath,
            string mergesPath)
        {
            var vocabJson = File.ReadAllText(vocabJsonPath);
            var mergesLines = File.ReadAllLines(mergesPath);

            return LoadFromStrings(
                vocabJson,
                string.Join('\n', mergesLines));
        }

        public static BytePairEncoder LoadFromStrings(
            string vocabJson,
            string mergesText)
        {
            var tokenToId = ParseVocabJson(vocabJson);
            var idToToken = BuildIdToToken(tokenToId);
            var mergeRanks = ParseMerges(mergesText.Split('\n'));

            return new BytePairEncoder(
                tokenToId,
                idToToken,
                mergeRanks);
        }

        public int[] Encode(
            string text)
        {
            if (string.IsNullOrEmpty(text))
            {
                return [];
            }

            var result = new List<int>();

            foreach (Match match in Gpt2TokenPattern.Matches(text))
            {
                if (!match.Success || match.Value.Length == 0)
                {
                    continue;
                }

                var bpeTokens = BpeEncode(match.Value);

                foreach (var token in bpeTokens)
                {
                    result.Add(
                        _tokenToId.TryGetValue(token, out var id)
                            ? id
                            : UnknownTokenId);
                }
            }

            return result.ToArray();
        }

        public string Decode(
            int[] tokenIds)
        {
            if (tokenIds is null)
            {
                throw new ArgumentNullException(nameof(tokenIds));
            }

            var builder = new StringBuilder();

            foreach (var id in tokenIds)
            {
                if ((uint)id < (uint)_idToToken.Length)
                {
                    builder.Append(DecodeToken(id));
                }
            }

            return builder.ToString();
        }

        public string DecodeToken(
            int tokenId)
        {
            if ((uint)tokenId >= (uint)_idToToken.Length)
            {
                return "?";
            }

            var token = _idToToken[tokenId];

            if (string.IsNullOrEmpty(token))
            {
                return string.Empty;
            }

            return ByteDecode(token);
        }

        private List<string> BpeEncode(
            string text)
        {
            var bytes = Encoding.UTF8.GetBytes(text);
            var parts = new List<string>(bytes.Length);

            foreach (var b in bytes)
            {
                parts.Add(ByteEncoder[b]);
            }

            if (parts.Count <= 1)
            {
                return parts;
            }

            while (parts.Count > 1)
            {
                var bestRank = int.MaxValue;
                var bestIndex = -1;

                for (var i = 0; i < parts.Count - 1; i++)
                {
                    if (!_mergeRanks.TryGetValue((parts[i], parts[i + 1]), out var rank))
                    {
                        continue;
                    }

                    if (rank < bestRank)
                    {
                        bestRank = rank;
                        bestIndex = i;
                    }
                }

                if (bestIndex < 0)
                {
                    break;
                }

                parts[bestIndex] = parts[bestIndex] + parts[bestIndex + 1];
                parts.RemoveAt(bestIndex + 1);
            }

            return parts;
        }

        private static Dictionary<string, int> ParseVocabJson(
            string json)
        {
            var parsed = JsonSerializer.Deserialize<Dictionary<string, int>>(json);

            if (parsed is null || parsed.Count == 0)
            {
                throw new InvalidOperationException("The BPE vocab JSON is empty or invalid.");
            }

            return new Dictionary<string, int>(
                parsed,
                StringComparer.Ordinal);
        }

        private static string[] BuildIdToToken(
            Dictionary<string, int> tokenToId)
        {
            var maxId = -1;

            foreach (var id in tokenToId.Values)
            {
                if (id > maxId)
                {
                    maxId = id;
                }
            }

            if (maxId < 0)
            {
                throw new InvalidOperationException("The BPE vocabulary does not contain any valid token ids.");
            }

            var idToToken = new string[maxId + 1];

            foreach (var kv in tokenToId)
            {
                if (kv.Value < 0)
                {
                    throw new InvalidOperationException(
                        $"Token '{kv.Key}' has a negative id: {kv.Value}.");
                }

                idToToken[kv.Value] = kv.Key;
            }

            return idToToken;
        }

        private static Dictionary<(string A, string B), int> ParseMerges(
            string[] lines)
        {
            var ranks = new Dictionary<(string A, string B), int>();
            var rank = 0;

            foreach (var rawLine in lines)
            {
                var line = rawLine.Trim();

                if (line.Length == 0 || line.StartsWith("#", StringComparison.Ordinal))
                {
                    continue;
                }

                var split = line.IndexOf(' ');

                if (split <= 0 || split >= line.Length - 1)
                {
                    continue;
                }

                var left = line[..split];
                var right = line[(split + 1)..];

                ranks[(left, right)] = rank;
                rank++;
            }

            return ranks;
        }

        private static string ByteDecode(
            string token)
        {
            var bytes = new List<byte>(token.Length);

            foreach (var ch in token)
            {
                if (ByteDecoder.TryGetValue(ch, out var value))
                {
                    bytes.Add(value);
                    continue;
                }

                var fallback = Encoding.UTF8.GetBytes(ch.ToString());
                bytes.AddRange(fallback);
            }

            try
            {
                return Encoding.UTF8.GetString(bytes.ToArray());
            }
            catch
            {
                return token;
            }
        }

        private static string[] BuildByteEncoder()
        {
            var result = new string[256];

            for (var b = 33; b <= 126; b++)
            {
                result[b] = ((char)b).ToString();
            }

            for (var b = 161; b <= 172; b++)
            {
                result[b] = ((char)b).ToString();
            }

            for (var b = 174; b <= 255; b++)
            {
                result[b] = ((char)b).ToString();
            }

            var n = 0;

            for (var b = 0; b < 256; b++)
            {
                if (result[b] is null)
                {
                    result[b] = ((char)(256 + n)).ToString();
                    n++;
                }
            }

            result[32] = "Ġ";

            return result;
        }

        private static Dictionary<char, byte> BuildByteDecoder(
            string[] byteEncoder)
        {
            var result = new Dictionary<char, byte>(byteEncoder.Length);

            for (var i = 0; i < byteEncoder.Length; i++)
            {
                var encoded = byteEncoder[i];

                if (encoded.Length == 1)
                {
                    result[encoded[0]] = (byte)i;
                }
            }

            return result;
        }
    }
}
