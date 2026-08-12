// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Frozen;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using DevOnBike.Overfit.Tensors;

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

        // FrozenDictionary: built once at class init, looked up per char in
        // every Decode. ToFrozenDictionary analyzes the keys and picks a
        // specialized read-optimized layout — slower build, faster lookup.
        private static readonly FrozenDictionary<char, byte> ByteDecoder = BuildByteDecoder(ByteEncoder);

        // Both lookup tables are load-once / read-many — the BPE merge loop in
        // BpeEncode scans every adjacent pair O(parts²) times, hammering
        // _mergeRanks. FrozenDictionary trades a one-time build cost (paid in
        // LoadFromStrings) for the fastest possible steady-state lookup.
        private readonly FrozenDictionary<string, int> _tokenToId;
        private readonly string[] _idToToken;
        private readonly FrozenDictionary<(string A, string B), int> _mergeRanks;

        private BytePairEncoder(
            FrozenDictionary<string, int> tokenToId,
            string[] idToToken,
            FrozenDictionary<(string A, string B), int> mergeRanks)
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

        // OVERFIT040 for `Load` only, so the encode path below stays covered by the rule.
        //
        // THE CONSTRAINT: a vocab JSON and a merges list read once, at construction time, on the caller's own
        // thread, before any text is encoded — and the method exists purely to hand both file contents to
        // `LoadFromStrings`, which is the string-in overload a caller who already has the bytes uses instead.
        //
        // WHAT IS GIVEN UP: `Load` is public API of the shipped `DevOnBike.Overfit` package.
#pragma warning disable OVERFIT040
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
#pragma warning restore OVERFIT040

        public static BytePairEncoder LoadFromStrings(
            string vocabJson,
            string mergesText)
        {
            var tokenToId = ParseVocabJson(vocabJson);
            var idToToken = BuildIdToToken(tokenToId);
            var mergeRanks = ParseMerges(mergesText.Split('\n'));

            // Freeze the mutable build-time dictionaries into read-optimized
            // FrozenDictionary instances. _tokenToId keeps Ordinal comparison
            // — GPT-2 token keys are byte-level escaped strings, culture-aware
            // comparison would be both wrong and slower.
            return new BytePairEncoder(
                tokenToId.ToFrozenDictionary(StringComparer.Ordinal),
                idToToken,
                mergeRanks.ToFrozenDictionary());
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
            ArgumentNullException.ThrowIfNull(tokenIds);

            // Two changes, and the second is a bug fix rather than an optimisation.
            //
            // Allocation: this was a StringBuilder plus one DecodeToken STRING per id, and DecodeToken
            // itself built a List<byte>, called ToArray on it and produced another string — four
            // allocations a token to assemble text.
            //
            // CORRECTNESS: it also converted each token's bytes to text SEPARATELY, so a codepoint spread
            // across two tokens became two replacement characters. Byte-level BPE splits multi-byte
            // characters routinely, so "zazolc" with Polish diacritics came back as "za????????" — found
            // 2026-08-11 by the first test this class's decode has ever had that runs by default. The
            // bytes are now accumulated across the whole sequence and converted once, which is what
            // GgufTokenizer and QwenTokenizer already did.
            using var byteBuffer = new PooledBuffer<byte>(ByteBudget(tokenIds), clearMemory: false);
            var byteCount = 0;

            for (var i = 0; i < tokenIds.Length; i++)
            {
                var id = tokenIds[i];

                if ((uint)id < (uint)_idToToken.Length)
                {
                    byteCount += ToBytes(_idToToken[id], byteBuffer.Span.Slice(byteCount));
                }
            }

            var pending = byteBuffer.Span.Slice(0, byteCount);
            var charCount = Encoding.UTF8.GetCharCount(pending);

            using var chars = new PooledBuffer<char>(charCount, clearMemory: false);

            Encoding.UTF8.GetChars(pending, chars.Span);

            return chars.Span.Slice(0, charCount).ToString();
        }

        /// <summary>
        /// Upper bound on the bytes a sequence decodes to. Must be an upper bound: the loop writes into the
        /// rented span at a running offset, so an underestimate is an exception rather than a slow path.
        /// </summary>
        private int ByteBudget(int[] tokenIds)
        {
            var total = 0;

            for (var i = 0; i < tokenIds.Length; i++)
            {
                var id = tokenIds[i];

                if ((uint)id < (uint)_idToToken.Length && _idToToken[id] is { } token)
                {
                    total += Encoding.UTF8.GetMaxByteCount(token.Length);
                }
            }

            return total + 1;
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
            // Reflection-free parse (JsonDocument, not JsonSerializer.Deserialize<T>) so the tokenizer
            // stays Native-AOT / trim safe — the generic deserializer is RequiresUnreferencedCode +
            // RequiresDynamicCode (IL2026 / IL3050) and would break under AOT.
            using var document = JsonDocument.Parse(json);
            var root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
            {
                throw new OverfitRuntimeException("The BPE vocab JSON is empty or invalid.");
            }

            var result = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var entry in root.EnumerateObject())
            {
                result[entry.Name] = entry.Value.GetInt32();
            }

            if (result.Count == 0)
            {
                throw new OverfitRuntimeException("The BPE vocab JSON is empty or invalid.");
            }

            return result;
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
                throw new OverfitRuntimeException("The BPE vocabulary does not contain any valid token ids.");
            }

            var idToToken = new string[maxId + 1];

            foreach (var kv in tokenToId)
            {
                if (kv.Value < 0)
                {
                    throw new OverfitRuntimeException(
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

                var left = line.Substring(0, split);
                var right = line.Substring(split + 1);

                ranks[(left, right)] = rank;
                rank++;
            }

            return ranks;
        }

        private static string ByteDecode(
            string token)
        {
            using var bytes = new PooledBuffer<byte>(
                Encoding.UTF8.GetMaxByteCount(token.Length) + 1, clearMemory: false);

            var count = ToBytes(token, bytes.Span);
            var pending = bytes.Span.Slice(0, count);
            var charCount = Encoding.UTF8.GetCharCount(pending);

            using var chars = new PooledBuffer<char>(charCount, clearMemory: false);

            Encoding.UTF8.GetChars(pending, chars.Span);

            return chars.Span.Slice(0, charCount).ToString();
        }

        /// <summary>
        /// Maps a byte-level piece back to text and appends it.
        ///
        /// <para><b>The `try/catch` that used to wrap this is gone, and its removal is the point.</b> It
        /// caught everything from <c>Encoding.UTF8.GetString</c> and returned the raw token instead — but
        /// that method does not throw on malformed input, it substitutes U+FFFD, so the catch was dead code
        /// standing in for a fallback that never ran. Keeping it would have hidden a real exception from
        /// somewhere else in the block behind a silently wrong answer.</para>
        ///
        /// <para>A character outside the byte-decoder table is encoded as itself, which is what the old
        /// <c>ch.ToString()</c> fallback did — without the string per character.</para>
        /// </summary>
        private static int ToBytes(string? token, Span<byte> destination)
        {
            if (string.IsNullOrEmpty(token))
            {
                return 0;
            }

            var count = 0;

            for (var i = 0; i < token.Length; i++)
            {
                var ch = token[i];

                if (ByteDecoder.TryGetValue(ch, out var value))
                {
                    destination[count++] = value;

                    continue;
                }

                count += Encoding.UTF8.GetBytes(token.AsSpan(i, 1), destination.Slice(count));
            }

            return count;
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
                if (result[b] == null)
                {
                    result[b] = ((char)(256 + n)).ToString();
                    n++;
                }
            }

            result[32] = "Ġ";

            return result;
        }

        private static FrozenDictionary<char, byte> BuildByteDecoder(
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

            return result.ToFrozenDictionary();
        }
    }
}
