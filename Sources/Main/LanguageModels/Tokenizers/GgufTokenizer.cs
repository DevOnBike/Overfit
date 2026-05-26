// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;
using DevOnBike.Overfit.LanguageModels.Loading;

namespace DevOnBike.Overfit.LanguageModels.Tokenizers
{
    /// <summary>
    /// Tokenizer reconstructed from the vocabulary embedded in a GGUF file (the
    /// <c>tokenizer.ggml.*</c> metadata), so models can be tokenized with no side-loaded
    /// <c>tokenizer.json</c> / <c>tokenizer.model</c>. Two algorithms, dispatched on
    /// <c>tokenizer.ggml.model</c>:
    /// <list type="bullet">
    /// <item><b>SentencePiece (SPM)</b> — <c>model == "llama"</c> (Llama-2, Mistral, Mixtral):
    /// whitespace escaped to <c>▁</c>, the score-driven greedy bigram merge from llama.cpp's
    /// <c>llm_tokenizer_spm</c>, and <c>&lt;0xNN&gt;</c> byte fallback for OOV chars.</item>
    /// <item><b>Byte-level BPE</b> — <c>model == "gpt2"</c> (Qwen, Llama-3, GPT-2): bytes mapped to
    /// the GPT-2 <see cref="ByteLevelAlphabet"/>, pre-tokenized by a regex chosen from
    /// <c>tokenizer.ggml.pre</c>, then merged by merge rank.</item>
    /// </list>
    /// </summary>
    public sealed class GgufTokenizer
    {
        // GGUF tokenizer.ggml.token_type values (llama_token_type).
        private const int TypeControl = 3;
        private const int TypeUserDefined = 4;
        private const int TypeByte = 6;

        private const char SpaceMarker = '▁';   // ▁

        // GPT-2 ByteLevel default split.
        private const string Gpt2SplitPattern =
            @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

        // cl100k / tiktoken split shared by qwen2 / llama-bpe / most modern byte-level BPE.
        private const string Cl100kSplitPattern =
            @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

        private readonly string _model;       // "llama" (SPM) or "gpt2" (byte-level BPE)
        private readonly string[] _tokens;
        private readonly int[] _tokenTypes;
        private readonly float[] _scores;
        private readonly Dictionary<string, int> _tokenToId;
        private readonly HashSet<int> _specialIds;
        private readonly bool _addSpacePrefix;

        // SPM byte fallback.
        private readonly int[] _idToByte;             // byte value (0..255) for <0xNN> tokens, else -1
        private readonly int[] _byteToId;             // [256] byte → <0xNN> token id, else -1

        // Byte-level BPE.
        private readonly Dictionary<(int, int), int>? _mergeRanks;
        private readonly Regex? _bpeSplit;
        private readonly Regex? _bpeSpecialSplit;
        private readonly char[]? _byteToChar;
        private readonly byte[]? _charToByte;

        public bool IsByteLevelBpe => _model == "gpt2";
        public int VocabSize => _tokens.Length;
        public int BosId { get; }
        public int EosId { get; }
        public int UnknownId { get; }
        public bool AddBosByDefault { get; }

        private GgufTokenizer(
            string model, string[] tokens, int[] tokenTypes, float[] scores, string[] merges,
            int bos, int eos, int unk, bool addBos, bool addSpacePrefix, string preType)
        {
            _model = model;
            _tokens = tokens;
            _tokenTypes = tokenTypes;
            _scores = scores;
            BosId = bos;
            EosId = eos;
            UnknownId = unk;
            AddBosByDefault = addBos;
            _addSpacePrefix = addSpacePrefix;

            _tokenToId = new Dictionary<string, int>(tokens.Length);
            _idToByte = new int[tokens.Length];
            _byteToId = new int[256];
            _specialIds = [];
            for (var b = 0; b < 256; b++) { _byteToId[b] = -1; }

            for (var id = 0; id < tokens.Length; id++)
            {
                _tokenToId.TryAdd(tokens[id], id);   // first-wins (lowest id), matches llama.cpp

                var byteValue = ParseByteToken(tokens[id], tokenTypes[id]);
                _idToByte[id] = byteValue;
                if (byteValue >= 0) { _byteToId[byteValue] = id; }

                if (tokenTypes[id] is TypeControl or TypeUserDefined) { _specialIds.Add(id); }
            }

            if (_model == "gpt2")
            {
                _byteToChar = ByteLevelAlphabet.BuildByteToChar();
                _charToByte = ByteLevelAlphabet.BuildCharToByte();
                _mergeRanks = BuildMergeRanks(merges);
                _bpeSplit = new Regex(SelectBpePattern(preType), RegexOptions.Compiled);
                _bpeSpecialSplit = BuildSpecialSplit();
            }
        }

        /// <summary>Builds a tokenizer from an open <see cref="GgufReader"/>.</summary>
        public static GgufTokenizer FromGguf(GgufReader reader)
        {
            ArgumentNullException.ThrowIfNull(reader);

            var model = reader.GetMeta("tokenizer.ggml.model", "");
            if (model != "llama" && model != "gpt2")
            {
                throw new NotSupportedException(
                    $"GGUF tokenizer model '{model}' is not supported (only SentencePiece 'llama' and " +
                    "byte-level BPE 'gpt2').");
            }

            var tokens = reader.GetMetaStringArray("tokenizer.ggml.tokens");
            var tokenTypes = reader.HasArray("tokenizer.ggml.token_type")
                ? reader.GetMetaIntArray("tokenizer.ggml.token_type")
                : new int[tokens.Length];
            var scores = reader.HasArray("tokenizer.ggml.scores")
                ? reader.GetMetaFloatArray("tokenizer.ggml.scores")
                : new float[tokens.Length];
            var merges = reader.HasArray("tokenizer.ggml.merges")
                ? reader.GetMetaStringArray("tokenizer.ggml.merges")
                : [];

            var isBpe = model == "gpt2";
            var bos = reader.GetMeta("tokenizer.ggml.bos_token_id", isBpe ? -1 : 1);
            var eos = reader.GetMeta("tokenizer.ggml.eos_token_id", 2);
            var unk = reader.GetMeta("tokenizer.ggml.unknown_token_id", 0);
            // SPM adds BOS + a leading space by default; byte-level BPE (Qwen/Llama-3) does neither.
            var addBos = reader.GetMeta("tokenizer.ggml.add_bos_token", !isBpe);
            var addSpacePrefix = reader.GetMeta("tokenizer.ggml.add_space_prefix", !isBpe);
            var preType = reader.GetMeta("tokenizer.ggml.pre", "default");

            return new GgufTokenizer(model, tokens, tokenTypes, scores, merges, bos, eos, unk,
                addBos, addSpacePrefix, preType);
        }

        /// <summary>Convenience: open the GGUF and build the tokenizer (does not retain the reader).</summary>
        public static GgufTokenizer Load(string ggufPath)
        {
            using var reader = new GgufReader(ggufPath);
            return FromGguf(reader);
        }

        /// <summary>Test-only factory — builds an SPM tokenizer from an in-memory vocab (no GGUF file).</summary>
        internal static GgufTokenizer CreateForTest(
            string[] tokens, int[] tokenTypes, float[] scores,
            int bos, int eos, int unk, bool addBos, bool addSpacePrefix)
            => new("llama", tokens, tokenTypes, scores, [], bos, eos, unk, addBos, addSpacePrefix, "default");

        /// <summary>Test-only factory — builds a byte-level BPE tokenizer from an in-memory vocab.</summary>
        internal static GgufTokenizer CreateBpeForTest(
            string[] tokens, int[] tokenTypes, string[] merges,
            int bos, int eos, int unk, bool addBos, string preType)
            => new("gpt2", tokens, tokenTypes, new float[tokens.Length], merges, bos, eos, unk,
                addBos, addSpacePrefix: false, preType);

        /// <summary>
        /// Encodes <paramref name="text"/> to token ids. <paramref name="addBos"/> defaults to the
        /// file's <c>add_bos_token</c> flag.
        /// </summary>
        public int[] Encode(string text, bool? addBos = null)
        {
            ArgumentNullException.ThrowIfNull(text);

            var output = new List<int>(text.Length + 1);
            if ((addBos ?? AddBosByDefault) && BosId >= 0) { output.Add(BosId); }

            if (_model == "gpt2") { BpeEncode(text, output); }
            else { SpmEncode(text, output); }

            return output.ToArray();
        }

        /// <summary>Decodes token ids back to text.</summary>
        public string Decode(ReadOnlySpan<int> ids)
        {
            var sb = new StringBuilder();
            var bytes = new List<byte>(ids.Length * 4);

            void Flush()
            {
                if (bytes.Count > 0) { sb.Append(Encoding.UTF8.GetString(bytes.ToArray())); bytes.Clear(); }
            }

            for (var i = 0; i < ids.Length; i++)
            {
                var id = ids[i];
                if ((uint)id >= (uint)_tokens.Length) { continue; }

                var type = _tokenTypes[id];
                if (type == TypeControl) { continue; }                 // <s> / </s> / specials — drop from text
                if (type == TypeUserDefined) { Flush(); sb.Append(_tokens[id]); continue; }   // literal special

                if (_model == "gpt2")
                {
                    var piece = _tokens[id];   // ByteLevel: each char maps back to one raw byte
                    for (var c = 0; c < piece.Length; c++) { bytes.Add(_charToByte![piece[c]]); }
                }
                else if (_idToByte[id] >= 0)
                {
                    bytes.Add((byte)_idToByte[id]);
                }
                else
                {
                    var pieceBytes = Encoding.UTF8.GetBytes(_tokens[id].Replace(SpaceMarker, ' '));
                    for (var b = 0; b < pieceBytes.Length; b++) { bytes.Add(pieceBytes[b]); }
                }
            }

            Flush();
            var text = sb.ToString();
            // SPM added a leading space at encode time — strip one to round-trip.
            if (_model != "gpt2" && _addSpacePrefix && text.Length > 0 && text[0] == ' ') { text = text[1..]; }
            return text;
        }

        /// <summary>Decodes a single token id to its display string (specials → empty).</summary>
        public string DecodeToken(int id)
        {
            if ((uint)id >= (uint)_tokens.Length) { return ""; }
            return Decode(new[] { id });
        }

        // ── SPM (SentencePiece) ──────────────────────────────────────────────

        private void SpmEncode(string text, List<int> output)
        {
            var prepared = (_addSpacePrefix ? " " : "") + text;
            prepared = prepared.Replace(' ', SpaceMarker);
            if (prepared.Length == 0) { return; }
            SpmMerge(prepared, output);
        }

        private void SpmMerge(string text, List<int> output)
        {
            // Symbols as a doubly-linked list over UTF-16 code points (surrogate pairs kept together).
            var starts = new List<int>(text.Length);
            var lens = new List<int>(text.Length);
            var prev = new List<int>(text.Length);
            var next = new List<int>(text.Length);

            for (var i = 0; i < text.Length;)
            {
                var charLen = char.IsHighSurrogate(text[i]) && i + 1 < text.Length ? 2 : 1;
                starts.Add(i);
                lens.Add(charLen);
                prev.Add(starts.Count - 2);
                next.Add(-1);
                i += charLen;
            }
            for (var s = 0; s < next.Count; s++) { next[s] = s + 1 < next.Count ? s + 1 : -1; }

            var queue = new PriorityQueue<(int left, int right, int size), (float score, int left)>(BigramComparer.Instance);

            void TryAddBigram(int left, int right)
            {
                if (left < 0 || right < 0) { return; }
                var merged = text.Substring(starts[left], lens[left] + lens[right]);
                if (_tokenToId.TryGetValue(merged, out var id))
                {
                    queue.Enqueue((left, right, merged.Length), (_scores[id], left));
                }
            }

            for (var i = 1; i < starts.Count; i++) { TryAddBigram(i - 1, i); }

            while (queue.Count > 0)
            {
                var (left, right, size) = queue.Dequeue();
                if (lens[left] == 0 || lens[right] == 0 || lens[left] + lens[right] != size) { continue; }

                lens[left] += lens[right];
                lens[right] = 0;
                next[left] = next[right];
                if (next[right] >= 0) { prev[next[right]] = left; }

                TryAddBigram(prev[left], left);
                TryAddBigram(left, next[left]);
            }

            for (var i = 0; i >= 0; i = next[i])
            {
                if (lens[i] == 0) { continue; }
                SpmResegment(text.Substring(starts[i], lens[i]), output);
            }
        }

        private void SpmResegment(string piece, List<int> output)
        {
            if (_tokenToId.TryGetValue(piece, out var id))
            {
                output.Add(id);
                return;
            }

            var bytes = Encoding.UTF8.GetBytes(piece);
            for (var b = 0; b < bytes.Length; b++)
            {
                var byteId = _byteToId[bytes[b]];
                output.Add(byteId >= 0 ? byteId : UnknownId);
            }
        }

        // ── Byte-level BPE (GPT-2) ───────────────────────────────────────────

        private void BpeEncode(string text, List<int> output)
        {
            if (text.Length == 0) { return; }

            foreach (var (piece, isSpecial, specialId) in SplitOnSpecialTokens(text))
            {
                if (isSpecial)
                {
                    output.Add(specialId);
                    continue;
                }
                foreach (Match m in _bpeSplit!.Matches(piece))
                {
                    BpeMerge(m.Value, output);
                }
            }
        }

        private void BpeMerge(string text, List<int> output)
        {
            if (text.Length == 0) { return; }

            var utf8 = Encoding.UTF8.GetBytes(text);
            var ids = new List<int>(utf8.Length);
            for (var i = 0; i < utf8.Length; i++)
            {
                var ch = _byteToChar![utf8[i]].ToString();
                ids.Add(_tokenToId.TryGetValue(ch, out var id) ? id : UnknownId);
            }

            while (ids.Count > 1)
            {
                var bestRank = int.MaxValue;
                var bestIndex = -1;
                for (var i = 0; i < ids.Count - 1; i++)
                {
                    if (_mergeRanks!.TryGetValue((ids[i], ids[i + 1]), out var rank) && rank < bestRank)
                    {
                        bestRank = rank;
                        bestIndex = i;
                    }
                }
                if (bestIndex < 0) { break; }

                var merged = _tokens[ids[bestIndex]] + _tokens[ids[bestIndex + 1]];
                ids[bestIndex] = _tokenToId.TryGetValue(merged, out var mid) ? mid : UnknownId;
                ids.RemoveAt(bestIndex + 1);
            }

            for (var i = 0; i < ids.Count; i++) { output.Add(ids[i]); }
        }

        private List<(string Text, bool IsSpecial, int Id)> SplitOnSpecialTokens(string text)
        {
            var result = new List<(string, bool, int)>();
            if (_bpeSpecialSplit is null) { result.Add((text, false, -1)); return result; }

            var pos = 0;
            foreach (Match m in _bpeSpecialSplit.Matches(text))
            {
                if (m.Index > pos) { result.Add((text[pos..m.Index], false, -1)); }
                result.Add((m.Value, true, _tokenToId[m.Value]));
                pos = m.Index + m.Length;
            }
            if (pos < text.Length) { result.Add((text[pos..], false, -1)); }
            return result;
        }

        private Dictionary<(int, int), int> BuildMergeRanks(string[] merges)
        {
            var ranks = new Dictionary<(int, int), int>(merges.Length);
            var rank = 0;
            for (var i = 0; i < merges.Length; i++)
            {
                var sp = merges[i].IndexOf(' ');
                if (sp <= 0 || sp >= merges[i].Length - 1) { continue; }
                var left = merges[i][..sp];
                var right = merges[i][(sp + 1)..];
                if (_tokenToId.TryGetValue(left, out var a) && _tokenToId.TryGetValue(right, out var b))
                {
                    ranks.TryAdd((a, b), rank++);
                }
            }
            return ranks;
        }

        private Regex? BuildSpecialSplit()
        {
            if (_specialIds.Count == 0) { return null; }
            var escaped = new List<string>(_specialIds.Count);
            foreach (var id in _specialIds)
            {
                // Only literal, matchable special strings (skip empties / unused placeholders).
                if (!string.IsNullOrEmpty(_tokens[id])) { escaped.Add(Regex.Escape(_tokens[id])); }
            }
            if (escaped.Count == 0) { return null; }
            return new Regex(string.Join("|", escaped), RegexOptions.Compiled);
        }

        private static string SelectBpePattern(string preType)
            => preType is "gpt-2" or "gpt2" or "olmo"
                ? Gpt2SplitPattern
                : Cl100kSplitPattern;   // qwen2 / llama-bpe / tekken / default

        // ── Shared helpers ───────────────────────────────────────────────────

        private static int ParseByteToken(string token, int tokenType)
        {
            if (tokenType != TypeByte) { return -1; }
            if (token.Length != 6 || token[0] != '<' || token[1] != '0' || token[2] != 'x' || token[5] != '>')
            {
                return -1;
            }
            var hi = HexValue(token[3]);
            var lo = HexValue(token[4]);
            if (hi < 0 || lo < 0) { return -1; }
            return (hi << 4) | lo;
        }

        private static int HexValue(char c) => c switch
        {
            >= '0' and <= '9' => c - '0',
            >= 'A' and <= 'F' => c - 'A' + 10,
            >= 'a' and <= 'f' => c - 'a' + 10,
            _ => -1,
        };

        private sealed class BigramComparer : IComparer<(float score, int left)>
        {
            public static readonly BigramComparer Instance = new();

            public int Compare((float score, int left) a, (float score, int left) b)
            {
                if (a.score != b.score) { return a.score > b.score ? -1 : 1; }
                return a.left.CompareTo(b.left);
            }
        }
    }
}
