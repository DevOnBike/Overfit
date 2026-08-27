// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using System.Text.RegularExpressions;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Text;

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
        private readonly SpecialScan? _specialScan;
        private readonly char[]? _byteToChar;
        private readonly byte[]? _charToByte;

        public bool IsByteLevelBpe => _model == "gpt2";
        public int VocabSize => _tokens.Length;
        public int BosId
        {
            get;
        }
        public int EosId
        {
            get;
        }
        public int UnknownId
        {
            get;
        }
        public bool AddBosByDefault
        {
            get;
        }

        /// <summary>
        /// The file's <c>tokenizer.ggml.add_eos_token</c> flag, false when absent.
        ///
        /// <para><b>Read here, applied by the caller.</b> Unlike <see cref="AddBosByDefault"/> this is
        /// deliberately NOT honoured by <see cref="Encode(string, bool?)"/>. Encode is also the chat path's
        /// tokenizer, where <c>ChatTemplate</c> already renders the model's own end markers into the prompt
        /// text; appending another end-of-text on top of them would change every prompt on any model that
        /// sets the flag. The embedding path, which genuinely needs the trailing token, appends
        /// <see cref="EosId"/> itself.</para>
        ///
        /// <para>Qwen3-Embedding sets it. Measured on Qwen3-Embedding-0.6B Q8_0 against llama.cpp: with the
        /// EOS appended the cosine is 0.9994, without it 0.796.</para>
        /// </summary>
        public bool AddEosByDefault
        {
            get;
        }

        private GgufTokenizer(
            string model, string[] tokens, int[] tokenTypes, float[] scores, string[] merges,
            int bos, int eos, int unk, bool addBos, bool addSpacePrefix, string preType,
            bool addEos = false)
        {
            ArgumentNullException.ThrowIfNull(tokens);
            ArgumentNullException.ThrowIfNull(tokenTypes);
            ArgumentNullException.ThrowIfNull(scores);

            // Three arrays that arrive as three separate GGUF metadata entries and are then indexed by one
            // loop bound. A truncated or crafted file makes the shorter ones raise IndexOutOfRangeException
            // deep inside the constructor — not the OverfitFormatException every sibling loading path
            // produces, so a caller that handles malformed models does not handle this one.
            //
            // `scores` is absent for byte-level BPE vocabularies and is supplied as a zero array of the
            // right length by the caller, so equality is the correct bound for all three rather than a
            // "long enough" test that would let a silently short one through.
            if (tokenTypes.Length != tokens.Length || scores.Length != tokens.Length)
            {
                throw new OverfitFormatException(
                    $"Tokenizer arrays disagree: {tokens.Length} tokens, {tokenTypes.Length} token types, "
                    + $"{scores.Length} scores. All three are indexed by token id and must be the same "
                    + "length.");
            }

            _model = model;
            _tokens = tokens;
            _tokenTypes = tokenTypes;
            _scores = scores;
            BosId = bos;
            EosId = eos;
            UnknownId = unk;
            AddBosByDefault = addBos;
            AddEosByDefault = addEos;
            _addSpacePrefix = addSpacePrefix;

            _tokenToId = new Dictionary<string, int>(tokens.Length);
            _idToByte = new int[tokens.Length];
            _byteToId = new int[256];
            _specialIds = [];
            for (var b = 0; b < 256; b++)
            {
                _byteToId[b] = -1;
            }

            for (var id = 0; id < tokens.Length; id++)
            {
                _tokenToId.TryAdd(tokens[id], id);   // first-wins (lowest id), matches llama.cpp

                var byteValue = ParseByteToken(tokens[id], tokenTypes[id]);
                _idToByte[id] = byteValue;
                if (byteValue >= 0)
                {
                    _byteToId[byteValue] = id;
                }

                if (tokenTypes[id] is TypeControl or TypeUserDefined)
                {
                    _specialIds.Add(id);
                }
            }

            if (_model == "gpt2")
            {
                _byteToChar = ByteLevelAlphabet.BuildByteToChar();
                _charToByte = ByteLevelAlphabet.BuildCharToByte();
                _mergeRanks = BuildMergeRanks(merges);
                _bpeSplit = new Regex(SelectBpePattern(preType), RegexOptions.Compiled);
                _specialScan = BuildSpecialScan();
            }
        }

        /// <summary>Builds a tokenizer from an open <see cref="GgufReader"/>.</summary>
        public static GgufTokenizer FromGguf(GgufReader reader)
        {
            ArgumentNullException.ThrowIfNull(reader);

            var model = reader.GetMeta("tokenizer.ggml.model", "");
            if (model != "llama" && model != "gpt2")
            {
                throw new OverfitRuntimeException(
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
            // Read, not applied — see AddEosByDefault for why the default is false and why Encode ignores it.
            var addEos = reader.GetMeta("tokenizer.ggml.add_eos_token", false);

            return new GgufTokenizer(model, tokens, tokenTypes, scores, merges, bos, eos, unk,
                addBos, addSpacePrefix, preType, addEos);
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
            if ((addBos ?? AddBosByDefault) && BosId >= 0)
            {
                output.Add(BosId);
            }

            if (_model == "gpt2")
            {
                BpeEncode(text, output);
            }

            if (!(_model == "gpt2"))
            {
                SpmEncode(text, output);
            }

            return output.ToArray();
        }

        /// <summary>Decodes token ids back to text.</summary>
        public string Decode(ReadOnlySpan<int> ids)
        {
            // Pooled rather than stack: the length is set by whatever the model generated, and a
            // variable-length stackalloc is an OVERFIT026 build error here for exactly that reason.
            var text = new ValueStringBuilder(CharBudget(ids));

            try
            {
                DecodeInto(ids, ref text);

                return Trimmed(text.AsSpan(), _model, _addSpacePrefix).ToString();
            }
            finally
            {
                text.Dispose();
            }
        }

        /// <summary>
        /// Decodes into a caller-owned buffer and returns the number of characters written. Allocates
        /// nothing once the pool is warm — the house pattern, and the one <c>ITokenizer.Decode</c> declares.
        ///
        /// <para><b>Why this overload exists.</b> The string overload above is called once per generated
        /// token by <c>ChatSession</c>'s incremental detokenizer, over the whole run each time. Every one of
        /// those calls used to allocate a <c>StringBuilder</c>, a <c>List&lt;byte&gt;</c>, an array from
        /// <c>ToArray</c>, a string per flush and a string per token — in an engine whose stated property is
        /// that decode allocates nothing.</para>
        /// </summary>
        /// <exception cref="ArgumentException">
        /// <paramref name="destination"/> is shorter than the decoded text. It reports the required length
        /// rather than truncating, because a silently shortened reply is indistinguishable from a model that
        /// stopped early.
        /// </exception>
        public int Decode(ReadOnlySpan<int> ids, Span<char> destination)
        {
            var text = new ValueStringBuilder(CharBudget(ids));

            try
            {
                DecodeInto(ids, ref text);

                var body = Trimmed(text.AsSpan(), _model, _addSpacePrefix);

                if (body.Length > destination.Length)
                {
                    throw new ArgumentException(
                        $"Destination holds {destination.Length} char(s); the decoded text needs "
                        + $"{body.Length}.",
                        nameof(destination));
                }

                body.CopyTo(destination);

                return body.Length;
            }
            finally
            {
                text.Dispose();
            }
        }

        /// <summary>Decodes a single token id to its display string (specials → empty).</summary>
        public string DecodeToken(int id)
        {
            if ((uint)id >= (uint)_tokens.Length)
            {
                return "";
            }

            // A stack span rather than `new[] { id }`: this is called per token by diagnostics and by
            // callers that stream, and an int[1] per token is a pure allocation with no purpose.
            Span<int> one = stackalloc int[1];
            one[0] = id;

            return Decode(one);
        }

        /// <summary>
        /// SPM adds a leading space at encode time, so one is stripped to round-trip. Shared by both
        /// overloads rather than duplicated: the two differed by this rule once and the span path silently
        /// returned text one character longer than the string path.
        /// </summary>
        private static ReadOnlySpan<char> Trimmed(ReadOnlySpan<char> text, string model, bool addSpacePrefix)
        {
            var strip = model != "gpt2" && addSpacePrefix && text.Length > 0 && text[0] == ' ';

            return strip ? text.Slice(1) : text;
        }

        /// <summary>
        /// The decode itself. Bytes are accumulated and flushed as UTF-8 whenever a piece arrives that is
        /// not raw bytes — a byte-level token can be half a codepoint, so decoding each one separately
        /// produces replacement characters where the original had one letter.
        /// </summary>
        private void DecodeInto(ReadOnlySpan<int> ids, ref ValueStringBuilder text)
        {
            using var byteBuffer = new PooledBuffer<byte>(ByteBudget(ids), clearMemory: false);

            // One rental for the whole loop. A `stackalloc char[piece.Length]` inside it is both an
            // OVERFIT026 error (variable element count) and a CA2014 one (stackalloc in a loop) — and the
            // analyzers are right: the loop runs once per token, so the stack would grow with the reply.
            using var pieceBuffer = new PooledBuffer<char>(LongestPiece(ids), clearMemory: false);
            var bytes = byteBuffer.Span;
            var byteCount = 0;

            for (var i = 0; i < ids.Length; i++)
            {
                var id = ids[i];

                if ((uint)id >= (uint)_tokens.Length)
                {
                    continue;
                }

                var type = _tokenTypes[id];

                if (type == TypeControl)
                {
                    // <s> / </s> / specials — drop from text
                    continue;
                }

                if (type == TypeUserDefined)
                {
                    // literal special
                    Flush(ref text, bytes, ref byteCount);
                    text.Append(_tokens[id]);

                    continue;
                }

                var piece = _tokens[id];

                if (_model == "gpt2")
                {
                    // ByteLevel: each char maps back to one raw byte
                    for (var c = 0; c < piece.Length; c++)
                    {
                        bytes[byteCount++] = _charToByte![piece[c]];
                    }

                    continue;
                }

                if (_idToByte[id] >= 0)
                {
                    bytes[byteCount++] = (byte)_idToByte[id];

                    continue;
                }

                // The space marker is substituted while copying rather than by piece.Replace(...), which
                // allocated a string for every non-byte token on every call.
                var pieceChars = pieceBuffer.Span.Slice(0, piece.Length);

                for (var c = 0; c < piece.Length; c++)
                {
                    pieceChars[c] = piece[c] == SpaceMarker ? ' ' : piece[c];
                }

                byteCount += Encoding.UTF8.GetBytes(pieceChars, bytes.Slice(byteCount));
            }

            Flush(ref text, bytes, ref byteCount);
        }

        private static void Flush(ref ValueStringBuilder text, Span<byte> bytes, ref int byteCount)
        {
            if (byteCount == 0)
            {
                return;
            }

            var pending = bytes.Slice(0, byteCount);
            var charCount = Encoding.UTF8.GetCharCount(pending);

            using var chars = new PooledBuffer<char>(charCount, clearMemory: false);

            Encoding.UTF8.GetChars(pending, chars.Span);
            text.Append(chars.Span.Slice(0, charCount));

            byteCount = 0;
        }

        /// <summary>
        /// An upper bound on the bytes the accumulator can hold, so it is rented once instead of grown.
        ///
        /// <para>Must be an upper bound and not an estimate: every write below indexes into it directly, so
        /// an underestimate is an <see cref="IndexOutOfRangeException"/> on some model's vocabulary rather
        /// than a slow path. Byte-level pieces contribute one byte per char; <c>&lt;0xNN&gt;</c> tokens one
        /// byte each; everything else at most <c>GetMaxByteCount</c> of its characters.</para>
        /// </summary>
        private int ByteBudget(ReadOnlySpan<int> ids)
        {
            var chars = 0;

            for (var i = 0; i < ids.Length; i++)
            {
                var id = ids[i];

                if ((uint)id < (uint)_tokens.Length)
                {
                    chars += _tokens[id].Length;
                }
            }

            // +ids.Length covers the one-byte-per-id path; the max-byte-count covers the widest UTF-8
            // expansion of everything else. One rental, and it is never short.
            return Encoding.UTF8.GetMaxByteCount(chars) + ids.Length + 1;
        }

        /// <summary>
        /// An upper bound on the decoded character count, used to rent once instead of growing. Every
        /// token's piece is at most its own length in characters, so their sum bounds the result; the
        /// builder still grows correctly if this were ever short, so it is a sizing hint and not a
        /// contract — unlike <see cref="ByteBudget"/>, which is indexed into directly.
        /// </summary>
        private int CharBudget(ReadOnlySpan<int> ids)
        {
            var chars = 0;

            for (var i = 0; i < ids.Length; i++)
            {
                var id = ids[i];

                if ((uint)id < (uint)_tokens.Length)
                {
                    chars += _tokens[id].Length;
                }
            }

            return chars + 1;
        }

        /// <summary>The longest piece in this sequence, so the substitution scratch is rented once.</summary>
        private int LongestPiece(ReadOnlySpan<int> ids)
        {
            var longest = 1;

            for (var i = 0; i < ids.Length; i++)
            {
                var id = ids[i];

                if ((uint)id < (uint)_tokens.Length && _tokens[id].Length > longest)
                {
                    longest = _tokens[id].Length;
                }
            }

            return longest;
        }

        // ── SPM (SentencePiece) ──────────────────────────────────────────────

        private void SpmEncode(string text, List<int> output)
        {
            var prepared = (_addSpacePrefix ? " " : "") + text;
            prepared = prepared.Replace(' ', SpaceMarker);
            if (prepared.Length == 0)
            {
                return;
            }
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
            for (var s = 0; s < next.Count; s++)
            {
                next[s] = s + 1 < next.Count ? s + 1 : -1;
            }

            var queue = new PriorityQueue<(int left, int right, int size), (float score, int left)>(BigramComparer.Instance);

            void TryAddBigram(int left, int right)
            {
                if (left < 0 || right < 0)
                {
                    return;
                }
                var merged = text.Substring(starts[left], lens[left] + lens[right]);
                if (_tokenToId.TryGetValue(merged, out var id))
                {
                    queue.Enqueue((left, right, merged.Length), (_scores[id], left));
                }
            }

            for (var i = 1; i < starts.Count; i++)
            {
                TryAddBigram(i - 1, i);
            }

            while (queue.Count > 0)
            {
                var (left, right, size) = queue.Dequeue();
                if (lens[left] == 0 || lens[right] == 0 || lens[left] + lens[right] != size)
                {
                    continue;
                }

                lens[left] += lens[right];
                lens[right] = 0;
                next[left] = next[right];
                if (next[right] >= 0)
                {
                    prev[next[right]] = left;
                }

                TryAddBigram(prev[left], left);
                TryAddBigram(left, next[left]);
            }

            for (var i = 0; i >= 0; i = next[i])
            {
                if (lens[i] == 0)
                {
                    continue;
                }
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
            if (text.Length == 0)
            {
                return;
            }

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
            if (text.Length == 0)
            {
                return;
            }

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
                if (bestIndex < 0)
                {
                    break;
                }

                var merged = _tokens[ids[bestIndex]] + _tokens[ids[bestIndex + 1]];
                ids[bestIndex] = _tokenToId.TryGetValue(merged, out var mid) ? mid : UnknownId;
                ids.RemoveAt(bestIndex + 1);
            }

            for (var i = 0; i < ids.Count; i++)
            {
                output.Add(ids[i]);
            }
        }

        // Longest-match scan instead of a giant regex alternation: models like Orpheus add tens of thousands of
        // special tokens (every <custom_token_N>), which a Regex of N escaped alternatives cannot handle. Scanning
        // at each candidate start char and probing substring lengths against a hash set is O(text · maxSpecialLen)
        // — independent of the special-vocab size — and longest-match is the correct (HF) semantics.
        private List<(string Text, bool IsSpecial, int Id)> SplitOnSpecialTokens(string text)
        {
            var result = new List<(string, bool, int)>();
            if (_specialScan is not { } scan)
            {
                result.Add((text, false, -1));
                return result;
            }

            var segStart = 0;
            var pos = 0;
            while (pos < text.Length)
            {
                if (scan.FirstChars.Contains(text[pos]))
                {
                    var maxTry = Math.Min(scan.MaxLen, text.Length - pos);
                    for (var len = maxTry; len >= 1; len--)
                    {
                        var candidate = text.Substring(pos, len);
                        if (scan.Strings.Contains(candidate))
                        {
                            if (pos > segStart)
                            {
                                result.Add((text.Substring(segStart, pos - segStart), false, -1));
                            }
                            result.Add((candidate, true, _tokenToId[candidate]));
                            pos += len;
                            segStart = pos;
                            goto matched;
                        }
                    }
                }
                pos++;
            matched:
                ;
            }
            if (segStart < text.Length)
            {
                result.Add((text.Substring(segStart), false, -1));
            }
            return result;
        }

        private Dictionary<(int, int), int> BuildMergeRanks(string[] merges)
        {
            var ranks = new Dictionary<(int, int), int>(merges.Length);
            var rank = 0;
            for (var i = 0; i < merges.Length; i++)
            {
                var sp = merges[i].IndexOf(' ');
                if (sp <= 0 || sp >= merges[i].Length - 1)
                {
                    continue;
                }
                var left = merges[i].Substring(0, sp);
                var right = merges[i].Substring(sp + 1);
                if (_tokenToId.TryGetValue(left, out var a) && _tokenToId.TryGetValue(right, out var b))
                {
                    ranks.TryAdd((a, b), rank++);
                }
            }
            return ranks;
        }

        private SpecialScan? BuildSpecialScan()
        {
            if (_specialIds.Count == 0)
            {
                return null;
            }
            var strings = new HashSet<string>(_specialIds.Count, StringComparer.Ordinal);
            var firstChars = new HashSet<char>();
            var maxLen = 0;
            foreach (var id in _specialIds)
            {
                // Only literal, matchable special strings (skip empties / unused placeholders).
                var s = _tokens[id];
                if (string.IsNullOrEmpty(s))
                {
                    continue;
                }
                strings.Add(s);
                firstChars.Add(s[0]);
                if (s.Length > maxLen)
                {
                    maxLen = s.Length;
                }
            }
            return strings.Count == 0 ? null : new SpecialScan(strings, firstChars, maxLen);
        }

        // Precomputed structures for the longest-match special-token scan (see SplitOnSpecialTokens).
        private readonly struct SpecialScan
        {
            public SpecialScan(HashSet<string> strings, HashSet<char> firstChars, int maxLen)
            {
                Strings = strings;
                FirstChars = firstChars;
                MaxLen = maxLen;
            }

            public HashSet<string> Strings
            {
                get;
            }
            public HashSet<char> FirstChars
            {
                get;
            }
            public int MaxLen
            {
                get;
            }
        }

        private static string SelectBpePattern(string preType)
            => preType is "gpt-2" or "gpt2" or "olmo"
                ? Gpt2SplitPattern
                : Cl100kSplitPattern;   // qwen2 / llama-bpe / tekken / default

        // ── Shared helpers ───────────────────────────────────────────────────

        private static int ParseByteToken(string token, int tokenType)
        {
            if (tokenType != TypeByte)
            {
                return -1;
            }
            if (token.Length != 6 || token[0] != '<' || token[1] != '0' || token[2] != 'x' || token[5] != '>')
            {
                return -1;
            }
            var hi = HexValue(token[3]);
            var lo = HexValue(token[4]);
            if (hi < 0 || lo < 0)
            {
                return -1;
            }
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
                if (a.score != b.score)
                {
                    return a.score > b.score ? -1 : 1;
                }
                return a.left.CompareTo(b.left);
            }
        }
    }
}
