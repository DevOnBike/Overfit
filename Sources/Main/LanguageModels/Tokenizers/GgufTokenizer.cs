// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using System.Text;
using DevOnBike.Overfit.LanguageModels.Loading;

namespace DevOnBike.Overfit.LanguageModels.Tokenizers
{
    /// <summary>
    /// Tokenizer reconstructed from the vocabulary embedded in a GGUF file (the
    /// <c>tokenizer.ggml.*</c> metadata), so Llama-family models (Llama-2, Mistral, Mixtral) can be
    /// tokenized with no side-loaded <c>tokenizer.json</c> / <c>tokenizer.model</c>.
    ///
    /// Implements the <b>SentencePiece (SPM)</b> path (<c>tokenizer.ggml.model == "llama"</c>):
    /// whitespace is escaped to <c>▁</c> (U+2581), encoding is the score-driven greedy bigram merge
    /// from llama.cpp's <c>llm_tokenizer_spm</c>, and out-of-vocabulary characters fall back to the
    /// <c>&lt;0xNN&gt;</c> byte tokens. The byte-level BPE path (<c>model == "gpt2"</c>, used by
    /// Qwen / Llama-3) is a follow-on — those models already have a native
    /// <see cref="QwenTokenizer"/> / <see cref="HuggingFaceBpeTokenizer"/>.
    /// </summary>
    public sealed class GgufTokenizer
    {
        // GGUF tokenizer.ggml.token_type values (llama_token_type).
        private const int TypeControl = 3;
        private const int TypeByte = 6;

        private const char SpaceMarker = '▁';   // ▁

        private readonly string[] _tokens;
        private readonly int[] _tokenTypes;
        private readonly float[] _scores;
        private readonly Dictionary<string, int> _tokenToId;
        private readonly int[] _idToByte;             // byte value (0..255) for <0xNN> tokens, else -1
        private readonly int[] _byteToId;             // [256] byte → <0xNN> token id, else -1
        private readonly bool _addSpacePrefix;

        public int VocabSize => _tokens.Length;
        public int BosId { get; }
        public int EosId { get; }
        public int UnknownId { get; }
        public bool AddBosByDefault { get; }

        private GgufTokenizer(
            string[] tokens, int[] tokenTypes, float[] scores,
            int bos, int eos, int unk, bool addBos, bool addSpacePrefix)
        {
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
            for (var b = 0; b < 256; b++) { _byteToId[b] = -1; }

            for (var id = 0; id < tokens.Length; id++)
            {
                // First-wins on duplicate strings (matches llama.cpp's lowest-id preference).
                _tokenToId.TryAdd(tokens[id], id);

                var byteValue = ParseByteToken(tokens[id], tokenTypes[id]);
                _idToByte[id] = byteValue;
                if (byteValue >= 0) { _byteToId[byteValue] = id; }
            }
        }

        /// <summary>
        /// Builds a tokenizer from an open <see cref="GgufReader"/>. Throws
        /// <see cref="NotSupportedException"/> for non-SPM vocabularies (e.g. <c>gpt2</c> BPE).
        /// </summary>
        public static GgufTokenizer FromGguf(GgufReader reader)
        {
            ArgumentNullException.ThrowIfNull(reader);

            var model = reader.GetMeta("tokenizer.ggml.model", "");
            if (model != "llama")
            {
                throw new NotSupportedException(
                    $"GGUF tokenizer model '{model}' is not supported (only SentencePiece 'llama'); " +
                    "gpt2/BPE vocabularies use QwenTokenizer / HuggingFaceBpeTokenizer.");
            }

            var tokens = reader.GetMetaStringArray("tokenizer.ggml.tokens");
            var tokenTypes = reader.HasArray("tokenizer.ggml.token_type")
                ? reader.GetMetaIntArray("tokenizer.ggml.token_type")
                : new int[tokens.Length];
            var scores = reader.HasArray("tokenizer.ggml.scores")
                ? reader.GetMetaFloatArray("tokenizer.ggml.scores")
                : new float[tokens.Length];

            var bos = reader.GetMeta("tokenizer.ggml.bos_token_id", 1);
            var eos = reader.GetMeta("tokenizer.ggml.eos_token_id", 2);
            var unk = reader.GetMeta("tokenizer.ggml.unknown_token_id", 0);
            var addBos = reader.GetMeta("tokenizer.ggml.add_bos_token", true);
            var addSpacePrefix = reader.GetMeta("tokenizer.ggml.add_space_prefix", true);

            return new GgufTokenizer(tokens, tokenTypes, scores, bos, eos, unk, addBos, addSpacePrefix);
        }

        /// <summary>Convenience: open the GGUF and build the tokenizer (does not retain the reader).</summary>
        public static GgufTokenizer Load(string ggufPath)
        {
            using var reader = new GgufReader(ggufPath);
            return FromGguf(reader);
        }

        /// <summary>Test-only factory — builds a tokenizer from an in-memory SPM vocab (no GGUF file).</summary>
        internal static GgufTokenizer CreateForTest(
            string[] tokens, int[] tokenTypes, float[] scores,
            int bos, int eos, int unk, bool addBos, bool addSpacePrefix)
            => new(tokens, tokenTypes, scores, bos, eos, unk, addBos, addSpacePrefix);

        /// <summary>
        /// Encodes <paramref name="text"/> to token ids using the SPM bigram-merge algorithm.
        /// <paramref name="addBos"/> defaults to the file's <c>add_bos_token</c> flag.
        /// </summary>
        public int[] Encode(string text, bool? addBos = null)
        {
            ArgumentNullException.ThrowIfNull(text);

            var output = new List<int>(text.Length + 1);
            if (addBos ?? AddBosByDefault) { output.Add(BosId); }

            // Whitespace escaping: optional leading space (SPM add_space_prefix), then ' ' → '▁'.
            var prepared = (_addSpacePrefix ? " " : "") + text;
            prepared = prepared.Replace(' ', SpaceMarker);
            if (prepared.Length == 0) { return output.ToArray(); }

            SpmMerge(prepared, output);
            return output.ToArray();
        }

        /// <summary>Decodes token ids back to text (byte tokens reassembled, <c>▁</c> → space).</summary>
        public string Decode(ReadOnlySpan<int> ids)
        {
            var bytes = new List<byte>(ids.Length * 4);
            for (var i = 0; i < ids.Length; i++)
            {
                var id = ids[i];
                if ((uint)id >= (uint)_tokens.Length) { continue; }

                var type = _tokenTypes[id];
                if (type == TypeControl) { continue; }   // <s> / </s> / other specials — drop from text

                if (_idToByte[id] >= 0)
                {
                    bytes.Add((byte)_idToByte[id]);
                    continue;
                }

                var piece = _tokens[id].Replace(SpaceMarker, ' ');
                var pieceBytes = Encoding.UTF8.GetBytes(piece);
                for (var b = 0; b < pieceBytes.Length; b++) { bytes.Add(pieceBytes[b]); }
            }

            var text = Encoding.UTF8.GetString(bytes.ToArray());
            // SPM added a leading space at encode time — strip one to round-trip.
            if (_addSpacePrefix && text.Length > 0 && text[0] == ' ') { text = text[1..]; }
            return text;
        }

        /// <summary>Decodes a single token id to its display string (specials → empty).</summary>
        public string DecodeToken(int id)
        {
            if ((uint)id >= (uint)_tokens.Length) { return ""; }
            if (_tokenTypes[id] == TypeControl) { return ""; }
            if (_idToByte[id] >= 0) { return ""; }   // raw byte — only meaningful in a sequence
            return _tokens[id].Replace(SpaceMarker, ' ');
        }

        // ── SPM bigram merge (mirrors llama.cpp llm_tokenizer_spm) ───────────────

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
                next.Add(-1);   // patched below
                i += charLen;
            }
            for (var s = 0; s < next.Count; s++) { next[s] = s + 1 < next.Count ? s + 1 : -1; }

            // Max-heap by (score desc, left asc): top is the best bigram to merge next.
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

                // Skip stale entries: a symbol merged away has len 0, or the pair's combined size
                // no longer matches (one side already grew/shrank).
                if (lens[left] == 0 || lens[right] == 0 || lens[left] + lens[right] != size) { continue; }

                // Merge right into left.
                lens[left] += lens[right];
                lens[right] = 0;
                next[left] = next[right];
                if (next[right] >= 0) { prev[next[right]] = left; }

                TryAddBigram(prev[left], left);
                TryAddBigram(left, next[left]);
            }

            // Emit: walk the surviving symbols left→right, resolving each to an id or byte fallback.
            for (var i = 0; i >= 0; i = next[i])
            {
                if (lens[i] == 0) { continue; }
                Resegment(text.Substring(starts[i], lens[i]), output);
            }
        }

        private void Resegment(string piece, List<int> output)
        {
            if (_tokenToId.TryGetValue(piece, out var id))
            {
                output.Add(id);
                return;
            }

            // Byte fallback: emit one <0xNN> token per UTF-8 byte; <unk> if a byte token is missing.
            var bytes = Encoding.UTF8.GetBytes(piece);
            for (var b = 0; b < bytes.Length; b++)
            {
                var byteId = _byteToId[bytes[b]];
                output.Add(byteId >= 0 ? byteId : UnknownId);
            }
        }

        private static int ParseByteToken(string token, int tokenType)
        {
            // Byte tokens are "<0xNN>" with token_type == BYTE.
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
                // Higher score first; ties broken by smaller left index (llama.cpp ordering).
                if (a.score != b.score) { return a.score > b.score ? -1 : 1; }
                return a.left.CompareTo(b.left);
            }
        }
    }
}
