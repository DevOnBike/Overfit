// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Tokenizers
{
    /// <summary>
    /// Generic HuggingFace ByteLevel-BPE tokenizer — reads a <c>tokenizer.json</c> (vocab,
    /// merges, added/special tokens, and the pre-tokenizer's Split regex) so it works across
    /// the Llama-3 / Mistral / Qwen family without a hard-coded per-model split pattern (the
    /// limitation of <see cref="QwenTokenizer"/>, whose cl100k regex is baked in). The
    /// end-of-text id is resolved from <c>tokenizer_config.json</c> when present.
    ///
    /// Implements <see cref="ITokenizer"/> so it drives <c>ChatSession</c> directly. Validated
    /// to reproduce <see cref="QwenTokenizer"/> bit-for-bit on Qwen's own files
    /// (HuggingFaceBpeTokenizerTests); other families load but are validation-gated on a
    /// dropped model. Supports ByteLevel BPE only (the family Overfit targets) — SentencePiece
    /// / Unigram throw.
    /// </summary>
    public sealed class HuggingFaceBpeTokenizer : ITokenizer
    {
        // GPT-2 ByteLevel default split (used when tokenizer.json has no explicit Split regex).
        private const string Gpt2SplitPattern =
            @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

        private static readonly char[] _byteToChar = ByteLevelAlphabet.BuildByteToChar();
        private static readonly byte[] _charToByte = ByteLevelAlphabet.BuildCharToByte();

        private readonly Dictionary<string, int> _vocab;
        private readonly string[] _decoder;
        private readonly Dictionary<(int, int), int> _mergeRanks;
        private readonly Regex _splitPattern;
        private readonly Dictionary<string, int> _specialTokens;
        private readonly HashSet<int> _specialTokenIds;
        private readonly int _eosId;
        private readonly int _unkId;
        private readonly bool _addPrefixSpace;

        private HuggingFaceBpeTokenizer(
            Dictionary<string, int> vocab,
            string[] decoder,
            Dictionary<(int, int), int> mergeRanks,
            Dictionary<string, int> specialTokens,
            Regex splitPattern,
            int eosId,
            int unkId,
            bool addPrefixSpace)
        {
            _vocab = vocab;
            _decoder = decoder;
            _mergeRanks = mergeRanks;
            _specialTokens = specialTokens;
            _specialTokenIds = new HashSet<int>(specialTokens.Values);
            _splitPattern = splitPattern;
            _eosId = eosId;
            _unkId = unkId;
            _addPrefixSpace = addPrefixSpace;
        }

        // ── ITokenizer surface ──────────────────────────────────────────────
        public int VocabularySize => _decoder.Length;
        public int EndOfTextTokenId => _eosId;
        public int UnknownTokenId => _unkId;
        public bool SupportsZeroAllocationEncode => false;
        public bool SupportsZeroAllocationDecode => false;

        public int CountTokens(ReadOnlySpan<char> text) => Encode(new string(text)).Length;

        public int Encode(ReadOnlySpan<char> text, Span<int> destination)
        {
            var tokens = Encode(new string(text));
            if (destination.Length < tokens.Length)
            {
                throw new ArgumentException(
                    $"Destination ({destination.Length}) is smaller than the encoded token count ({tokens.Length}).",
                    nameof(destination));
            }
            tokens.CopyTo(destination);
            return tokens.Length;
        }

        public int Decode(ReadOnlySpan<int> tokens, Span<char> destination)
        {
            var text = DecodeToString(tokens);
            if (destination.Length < text.Length)
            {
                throw new ArgumentException(
                    $"Destination ({destination.Length}) is smaller than the decoded char count ({text.Length}).",
                    nameof(destination));
            }
            text.AsSpan().CopyTo(destination);
            return text.Length;
        }

        // ── Load ────────────────────────────────────────────────────────────

        /// <summary>Loads from a directory containing <c>tokenizer.json</c>, or directly from a tokenizer.json path.</summary>
        public static HuggingFaceBpeTokenizer Load(string pathOrDirectory)
        {
            var dir = Directory.Exists(pathOrDirectory) ? pathOrDirectory : Path.GetDirectoryName(pathOrDirectory);
            var jsonPath = Directory.Exists(pathOrDirectory)
                ? Path.Combine(pathOrDirectory, "tokenizer.json")
                : pathOrDirectory;
            if (!File.Exists(jsonPath))
            {
                throw new FileNotFoundException($"tokenizer.json not found at: {jsonPath}");
            }

            using var stream = File.OpenRead(jsonPath);
            using var doc = JsonDocument.Parse(stream);
            var root = doc.RootElement;

            var model = root.GetProperty("model");
            var type = model.TryGetProperty("type", out var t) ? t.GetString() : "BPE";
            if (type != "BPE")
            {
                throw new NotSupportedException($"Only ByteLevel BPE tokenizers are supported; got '{type}'.");
            }

            var vocab = ReadVocab(model.GetProperty("vocab"), out var decoder);
            var mergeRanks = ReadMerges(model.GetProperty("merges"), vocab);
            var specialTokens = ReadAddedTokens(root, ref decoder);
            var split = BuildSplitRegex(root);
            var addPrefixSpace = ReadAddPrefixSpace(root);

            // EOS / UNK from tokenizer_config.json (preferred) then tokenizer.json model.
            var (eos, unk) = ResolveSpecialIds(dir, specialTokens, vocab, model);

            return new HuggingFaceBpeTokenizer(vocab, decoder, mergeRanks, specialTokens, split, eos, unk, addPrefixSpace);
        }

        // ── Encode / Decode ─────────────────────────────────────────────────

        /// <summary>Encodes text to token IDs, recognising special tokens in the input.</summary>
        public int[] Encode(string text)
        {
            if (_addPrefixSpace && text.Length > 0 && text[0] != ' ')
            {
                text = " " + text;
            }

            var tokens = new List<int>();
            foreach (var (piece, isSpecial) in SplitOnSpecialTokens(text))
            {
                if (isSpecial)
                {
                    tokens.Add(_specialTokens[piece]);
                }
                else
                {
                    foreach (Match m in _splitPattern.Matches(piece))
                    {
                        BpeEncode(m.Value, tokens);
                    }
                }
            }
            return tokens.ToArray();
        }

        public string DecodeToString(ReadOnlySpan<int> tokens)
        {
            var bytes = new List<byte>();
            var sb = new StringBuilder();

            foreach (var id in tokens)
            {
                if (id < 0 || id >= _decoder.Length || _decoder[id] is null) { continue; }
                var piece = _decoder[id];

                if (_specialTokenIds.Contains(id))
                {
                    if (bytes.Count > 0) { sb.Append(Encoding.UTF8.GetString(bytes.ToArray())); bytes.Clear(); }
                    sb.Append(piece);
                }
                else
                {
                    foreach (var ch in piece) { bytes.Add(_charToByte[ch]); }
                }
            }
            if (bytes.Count > 0) { sb.Append(Encoding.UTF8.GetString(bytes.ToArray())); }
            return sb.ToString();
        }

        // ── BPE core (GPT-2 ByteLevel) ──────────────────────────────────────

        private void BpeEncode(string text, List<int> output)
        {
            if (text.Length == 0) { return; }

            var utf8 = Encoding.UTF8.GetBytes(text);
            var charBuf = new char[utf8.Length];
            for (var i = 0; i < utf8.Length; i++) { charBuf[i] = _byteToChar[utf8[i]]; }

            var ids = new List<int>(charBuf.Length);
            for (var i = 0; i < charBuf.Length; i++)
            {
                var ch = charBuf[i].ToString();
                ids.Add(_vocab.TryGetValue(ch, out var id) ? id : _unkId);
            }

            while (ids.Count > 1)
            {
                var bestRank = int.MaxValue;
                var bestIndex = -1;
                for (var i = 0; i < ids.Count - 1; i++)
                {
                    if (_mergeRanks.TryGetValue((ids[i], ids[i + 1]), out var rank) && rank < bestRank)
                    {
                        bestRank = rank;
                        bestIndex = i;
                    }
                }
                if (bestIndex < 0) { break; }

                var merged = _decoder[ids[bestIndex]] + _decoder[ids[bestIndex + 1]];
                ids[bestIndex] = _vocab.TryGetValue(merged, out var mid) ? mid : _unkId;
                ids.RemoveAt(bestIndex + 1);
            }

            output.AddRange(ids);
        }

        private List<(string Text, bool IsSpecial)> SplitOnSpecialTokens(string text)
        {
            var result = new List<(string, bool)>();
            if (_specialTokens.Count == 0) { result.Add((text, false)); return result; }

            var escaped = new string[_specialTokens.Count];
            var ei = 0;
            foreach (var key in _specialTokens.Keys) { escaped[ei++] = Regex.Escape(key); }
            var regex = new Regex(string.Join("|", escaped));

            var pos = 0;
            foreach (Match m in regex.Matches(text))
            {
                if (m.Index > pos) { result.Add((text[pos..m.Index], false)); }
                result.Add((m.Value, true));
                pos = m.Index + m.Length;
            }
            if (pos < text.Length) { result.Add((text[pos..], false)); }
            return result;
        }

        // ── tokenizer.json parsing ──────────────────────────────────────────

        private static Dictionary<string, int> ReadVocab(JsonElement vocabJson, out string[] decoder)
        {
            var vocab = new Dictionary<string, int>();
            var maxId = 0;
            foreach (var kv in vocabJson.EnumerateObject())
            {
                var id = kv.Value.GetInt32();
                vocab[kv.Name] = id;
                if (id > maxId) { maxId = id; }
            }
            decoder = new string[maxId + 1];
            foreach (var kv in vocab) { decoder[kv.Value] = kv.Key; }
            return vocab;
        }

        // HF merges come as either "a b" strings (older) or ["a","b"] pairs (newer).
        private static Dictionary<(int, int), int> ReadMerges(JsonElement mergesJson, Dictionary<string, int> vocab)
        {
            var ranks = new Dictionary<(int, int), int>(mergesJson.GetArrayLength());
            var rank = 0;
            foreach (var merge in mergesJson.EnumerateArray())
            {
                string? left, right;
                if (merge.ValueKind == JsonValueKind.Array)
                {
                    left = merge[0].GetString();
                    right = merge[1].GetString();
                }
                else
                {
                    var parts = merge.GetString()!.Split(' ');
                    left = parts.Length == 2 ? parts[0] : null;
                    right = parts.Length == 2 ? parts[1] : null;
                }
                if (left is not null && right is not null
                    && vocab.TryGetValue(left, out var a) && vocab.TryGetValue(right, out var b))
                {
                    ranks[(a, b)] = rank++;
                }
            }
            return ranks;
        }

        private static Dictionary<string, int> ReadAddedTokens(JsonElement root, ref string[] decoder)
        {
            var specialTokens = new Dictionary<string, int>();

            if (!root.TryGetProperty("added_tokens", out var added))
            {
                return specialTokens;
            }

            foreach (var tok in added.EnumerateArray())
            {
                var content = tok.GetProperty("content").GetString()!;
                var id = tok.GetProperty("id").GetInt32();
                specialTokens[content] = id;

                if (id >= decoder.Length)
                {
                    var extended = new string[id + 1];

                    Array.Copy(decoder, extended, decoder.Length);

                    decoder = extended;
                }
                decoder[id] = content;
            }
            return specialTokens;
        }

        // Walks the pre_tokenizer for a Split's Regex pattern; falls back to the GPT-2 default.
        private static Regex BuildSplitRegex(JsonElement root)
        {
            var pattern = root.TryGetProperty("pre_tokenizer", out var pre)
                ? FindSplitPattern(pre)
                : null;

            pattern ??= Gpt2SplitPattern;

            try
            {
                return new Regex(pattern, RegexOptions.Compiled);
            }
            catch (ArgumentException)
            {
                // The file's pattern uses syntax .NET rejects — fall back to the GPT-2 default.
                return new Regex(Gpt2SplitPattern, RegexOptions.Compiled);
            }
        }

        private static string? FindSplitPattern(JsonElement node)
        {
            var type = node.TryGetProperty("type", out var t) ? t.GetString() : null;

            if (type == "Split" && node.TryGetProperty("pattern", out var pat)
                && pat.TryGetProperty("Regex", out var rx))
            {
                return rx.GetString();
            }

            if (type == "Sequence" && node.TryGetProperty("pretokenizers", out var seq))
            {
                foreach (var child in seq.EnumerateArray())
                {
                    var found = FindSplitPattern(child);
                    if (found is not null) { return found; }
                }
            }

            return null;
        }

        private static bool ReadAddPrefixSpace(JsonElement root)
        {
            if (root.TryGetProperty("pre_tokenizer", out var pre) && FindAddPrefixSpace(pre) is { } v)
            {
                return v;
            }
            return false;
        }

        private static bool? FindAddPrefixSpace(JsonElement node)
        {
            var type = node.TryGetProperty("type", out var t) ? t.GetString() : null;
            if (type == "ByteLevel" && node.TryGetProperty("add_prefix_space", out var aps))
            {
                return aps.ValueKind == JsonValueKind.True;
            }
            if (type == "Sequence" && node.TryGetProperty("pretokenizers", out var seq))
            {
                foreach (var child in seq.EnumerateArray())
                {
                    if (FindAddPrefixSpace(child) is { } v) { return v; }
                }
            }
            return null;
        }

        // EOS/UNK: tokenizer_config.json eos_token/unk_token (string or {content}) → id; else heuristics.
        private static (int eos, int unk) ResolveSpecialIds(string? dir, Dictionary<string, int> specialTokens, Dictionary<string, int> vocab, JsonElement model)
        {
            string? eosText = null, unkText = null;

            if (dir is not null)
            {
                var cfgPath = Path.Combine(dir, "tokenizer_config.json");

                if (File.Exists(cfgPath))
                {
                    using var cfg = JsonDocument.Parse(File.ReadAllBytes(cfgPath));
                    eosText = ReadTokenString(cfg.RootElement, "eos_token");
                    unkText = ReadTokenString(cfg.RootElement, "unk_token");
                }
            }

            var eos = ResolveId(eosText, specialTokens, vocab, fallback: -1);
            var unk = ResolveId(unkText, specialTokens, vocab, fallback: -1);

            // model.unk_token (BPE field) as a secondary source for UNK.
            if (unk < 0 && model.TryGetProperty("unk_token", out var mu) && mu.ValueKind == JsonValueKind.String)
            {
                unk = ResolveId(mu.GetString(), specialTokens, vocab, fallback: -1);
            }
            if (eos < 0) { eos = unk >= 0 ? unk : 0; }
            if (unk < 0) { unk = eos; }
            return (eos, unk);
        }

        private static string? ReadTokenString(JsonElement root, string key)
        {
            if (!root.TryGetProperty(key, out var v))
            {
                return null;
            }

            return v.ValueKind switch
            {
                JsonValueKind.String => v.GetString(),
                JsonValueKind.Object => v.TryGetProperty("content", out var c) ? c.GetString() : null,
                _ => null,
            };
        }

        private static int ResolveId(string? token, Dictionary<string, int> specialTokens, Dictionary<string, int> vocab, int fallback)
        {
            if (token is null) { return fallback; }
            if (specialTokens.TryGetValue(token, out var sid)) { return sid; }
            if (vocab.TryGetValue(token, out var vid)) { return vid; }
            return fallback;
        }

    }
}
