// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using DevOnBike.Overfit.Text;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.LanguageModels.Tokenizers
{
    /// <summary>
    /// Qwen2.5 tokenizer — ByteLevel BPE compatible with HuggingFace tokenizer.json format.
    ///
    /// Load from the HuggingFace snapshot directory:
    ///   var tok = QwenTokenizer.Load(@"C:\Users\user\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B\snapshots\abc123\");
    ///
    /// Or copy tokenizer.json (+ tokenizer_config.json) next to your binary and load directly:
    ///   var tok = QwenTokenizer.Load("tokenizer.json");
    ///
    /// Known special token IDs:
    ///   151643 = &lt;|endoftext|&gt;  (EOS / BOS)
    ///   151644 = &lt;|im_start|&gt;
    ///   151645 = &lt;|im_end|&gt;
    /// </summary>
    public sealed class QwenTokenizer
    {
        // ── Publicly useful constants ──────────────────────────────────────
        public const int EndOfText = 151643;   // <|endoftext|>
        public const int ImStart = 151644;   // <|im_start|>
        public const int ImEnd = 151645;   // <|im_end|>

        // ── State ──────────────────────────────────────────────────────────
        private readonly Dictionary<string, int> _vocab;        // piece → id
        private readonly string[] _decoder;      // id   → piece
        private readonly Dictionary<(int, int), int> _mergeRanks;  // (a,b) → rank
        private readonly Regex _splitPattern;
        private readonly HashSet<int> _specialTokenIds;
        private readonly Dictionary<string, int> _specialTokens;

        // GPT-2 byte ↔ char mapping (256 printable surrogates)
        private static readonly char[] _byteToChar = BuildByteToChar();
        private static readonly byte[] _charToByte = BuildCharToByte();

        private const string DefaultSystemMessage = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

        // ── Construction ───────────────────────────────────────────────────

        private QwenTokenizer(
            Dictionary<string, int> vocab,
            string[] decoder,
            Dictionary<(int, int), int> mergeRanks,
            Dictionary<string, int> specialTokens)
        {
            _vocab = vocab;
            _decoder = decoder;
            _mergeRanks = mergeRanks;
            _specialTokens = specialTokens;
            _specialTokenIds = new HashSet<int>(specialTokens.Values);

            // Qwen2.5 pre-tokenizer regex (GPT-4 / tiktoken cl100k_base pattern)
            _splitPattern = new Regex(
                @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                RegexOptions.Compiled);
        }

        /// <summary>Load tokenizer from a directory containing tokenizer.json, or directly from tokenizer.json path.</summary>
        public static QwenTokenizer Load(string pathOrDirectory)
        {
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
            if (model.GetProperty("type").GetString() != "BPE")
            {
                throw new OverfitRuntimeException("Only BPE tokenizers are supported.");
            }

            // ── Vocab ──────────────────────────────────────────────────────
            var vocabJson = model.GetProperty("vocab");
            var vocab = new Dictionary<string, int>();
            foreach (var kv in vocabJson.EnumerateObject())
            {
                vocab[kv.Name] = kv.Value.GetInt32();
            }

            var maxId = 0;
            foreach (var id in vocab.Values)
            {
                if (id > maxId)
                {
                    maxId = id;
                }
            }
            var decoder = new string[maxId + 1];
            foreach (var kv in vocab)
            {
                decoder[kv.Value] = kv.Key;
            }

            // ── Merges ─────────────────────────────────────────────────────
            var mergesJson = model.GetProperty("merges");
            var mergeRanks = new Dictionary<(int, int), int>(mergesJson.GetArrayLength());
            var rank = 0;
            foreach (var merge in mergesJson.EnumerateArray())
            {
                var parts = merge.GetString()!.Split(' ');
                if (parts.Length == 2
                    && vocab.TryGetValue(parts[0], out var a)
                    && vocab.TryGetValue(parts[1], out var b))
                {
                    mergeRanks[(a, b)] = rank++;
                }
            }

            // ── Special tokens ─────────────────────────────────────────────
            //
            // The decoder is grown to `id + 1` at 8 bytes an entry, and `id` is a number chosen by whoever
            // wrote tokenizer.json — so a declared 2^31-1 is a 16 GB allocation request the caller cannot
            // catch. Bounded here against the FILE'S OWN LENGTH, the same way the binary loaders bound a
            // declared count: a token cannot be encoded in fewer than a few bytes of JSON, so the document
            // cannot describe more ids than it has room for.
            //
            // Four bytes is a deliberate under-estimate of the floor, not a guess at the typical: the real
            // Qwen tokenizer measures **46.4 bytes per token** (7,031,645 B for 151,665 tokens), so the bound
            // clears its highest id by **11.6x** and caps the allocation at 14 MB whatever the file claims.
            //
            // The obvious bound — `vocab.Count + added.Count` — was measured and REJECTED: on that same file
            // it clears the highest added id by exactly ONE (151,664 against 151,665), so any tokenizer with
            // a single gap in its added-token ids would be refused. Rejecting a valid model is worse than the
            // allocation this guards against.
            const long MinBytesPerToken = 4;
            var maxDecoderEntries = stream.Length / MinBytesPerToken;

            var specialTokens = new Dictionary<string, int>();
            if (root.TryGetProperty("added_tokens", out var added))
            {
                foreach (var tok in added.EnumerateArray())
                {
                    var content = tok.GetProperty("content").GetString()!;
                    var id = tok.GetProperty("id").GetInt32();

                    if (id < 0 || id + 1L > maxDecoderEntries)
                    {
                        throw new OverfitFormatException(
                            $"'{jsonPath}' declares added token id {id}, which would size the decoder at "
                            + $"{id + 1L} entries. The file is {stream.Length} bytes and a token needs at "
                            + $"least {MinBytesPerToken}, so it cannot describe more than {maxDecoderEntries} "
                            + "ids.");
                    }

                    specialTokens[content] = id;
                    // Extend decoder if needed
                    if (id >= decoder.Length)
                    {
                        var extended = new string[id + 1];
                        decoder.AsSpan().CopyTo(extended);
                        decoder = extended;
                    }
                    decoder[id] = content;
                }
            }

            return new QwenTokenizer(vocab, decoder, mergeRanks, specialTokens);
        }

        // ── Public API ─────────────────────────────────────────────────────

        public int VocabSize => _decoder.Length;

        /// <summary>Encode text to token IDs. Special tokens in the input are recognised.</summary>
        public int[] Encode(string text, bool addBos = false)
        {
            var tokens = new List<int>();
            if (addBos)
            {
                tokens.Add(EndOfText);
            }

            // Split on special tokens first
            var parts = SplitOnSpecialTokens(text);
            foreach (var (piece, isSpecial) in parts)
            {
                if (isSpecial)
                {
                    tokens.Add(_specialTokens[piece]);
                }

                if (!(isSpecial))
                {
                    foreach (Match m in _splitPattern.Matches(piece))
                    {
                        tokens.AddRange(BpeEncode(m.Value));
                    }
                }
            }

            return tokens.ToArray();
        }

        /// <summary>Decode a sequence of token IDs to text.</summary>
        public string Decode(ReadOnlySpan<int> tokens)
        {
            var text = new ValueStringBuilder(CharBudget(tokens));

            try
            {
                DecodeInto(tokens, ref text);

                return text.AsSpan().ToString();
            }
            finally
            {
                text.Dispose();
            }
        }

        /// <summary>
        /// Decodes into a caller-owned buffer and returns the characters written; allocates nothing once
        /// the pool is warm.
        ///
        /// <para>The reason this exists is <c>ChatSession</c>'s incremental detokenizer, which decodes the
        /// WHOLE generated run once per token. Through the string overload that was a
        /// <c>List&lt;byte&gt;</c>, a <c>StringBuilder</c>, an array from <c>ToArray</c> and two strings per
        /// call, over a growing sequence.</para>
        /// </summary>
        /// <exception cref="ArgumentException">
        /// <paramref name="destination"/> is shorter than the decoded text. It names the required length
        /// rather than truncating: a silently shortened reply looks like a model that stopped early.
        /// </exception>
        public int Decode(ReadOnlySpan<int> tokens, Span<char> destination)
        {
            var text = new ValueStringBuilder(CharBudget(tokens));

            try
            {
                DecodeInto(tokens, ref text);

                if (!text.TryCopyTo(destination, out var written))
                {
                    throw new ArgumentException(
                        $"Destination holds {destination.Length} char(s); the decoded text needs "
                        + $"{text.Length}.",
                        nameof(destination));
                }

                return written;
            }
            finally
            {
                text.Dispose();
            }
        }

        /// <summary>
        /// Byte-level pieces are accumulated and flushed as UTF-8 only when a special token interrupts
        /// them or the sequence ends — a piece can hold half a codepoint, and converting each separately
        /// yields a replacement character where the text had a letter.
        /// </summary>
        private void DecodeInto(ReadOnlySpan<int> tokens, ref ValueStringBuilder text)
        {
            using var byteBuffer = new PooledBuffer<byte>(ByteBudget(tokens), clearMemory: false);
            var bytes = byteBuffer.Span;
            var byteCount = 0;

            for (var i = 0; i < tokens.Length; i++)
            {
                var id = tokens[i];

                if (id < 0 || id >= _decoder.Length || _decoder[id] == null)
                {
                    continue;
                }

                var piece = _decoder[id];

                // Looked up once. The previous shape asked the set twice per token, in an if and then in
                // its negation.
                if (_specialTokenIds.Contains(id))
                {
                    Flush(ref text, bytes, ref byteCount);
                    text.Append(piece);

                    continue;
                }

                for (var c = 0; c < piece.Length; c++)
                {
                    bytes[byteCount++] = _charToByte[piece[c]];
                }
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
        /// Upper bound on the accumulated bytes. <b>Must be an upper bound</b>: the loop indexes into the
        /// rented span directly, so an underestimate is an <see cref="IndexOutOfRangeException"/> on some
        /// vocabulary rather than a slower path. Byte-level pieces contribute exactly one byte per
        /// character, so their total length is the bound.
        /// </summary>
        private int ByteBudget(ReadOnlySpan<int> tokens) => CharBudget(tokens);

        /// <summary>Total piece length across the sequence — bounds both the bytes and the characters.</summary>
        private int CharBudget(ReadOnlySpan<int> tokens)
        {
            var total = 0;

            for (var i = 0; i < tokens.Length; i++)
            {
                var id = tokens[i];

                if (id >= 0 && id < _decoder.Length && _decoder[id] is { } piece)
                {
                    total += piece.Length;
                }
            }

            return total + 1;
        }

        /// <summary>Decode a single token ID (for streaming output).</summary>
        public string DecodeToken(int id)
        {
            if (id < 0 || id >= _decoder.Length || _decoder[id] == null)
            {
                return string.Empty;
            }

            var piece = _decoder[id];
            if (_specialTokenIds.Contains(id))
            {
                return piece;
            }

            // Rented rather than `new byte[piece.Length]`: this is the per-token streaming entry point, so
            // that array was one allocation per token for no reason. The returned string stays — it is the
            // method's product.
            using var bytes = new PooledBuffer<byte>(piece.Length, clearMemory: false);

            for (var i = 0; i < piece.Length; i++)
            {
                bytes.Span[i] = _charToByte[piece[i]];
            }

            return Encoding.UTF8.GetString(bytes.Span.Slice(0, piece.Length));
        }

        public bool IsSpecialToken(int id) => _specialTokenIds.Contains(id);

        // ── Chat template helpers ──────────────────────────────────────────

        /// <summary>
        /// Build a Qwen2.5-Instruct chat prompt.
        /// Format:  &lt;|im_start|&gt;system\n{system}&lt;|im_end|&gt;\n
        ///          &lt;|im_start|&gt;user\n{user}&lt;|im_end|&gt;\n
        ///          &lt;|im_start|&gt;assistant\n
        /// </summary>
        public int[] BuildChatPrompt(string userMessage, string systemPrompt = DefaultSystemMessage)
        {
            var text = $"<|im_start|>system\n{systemPrompt}<|im_end|>\n" +
                       $"<|im_start|>user\n{userMessage}<|im_end|>\n" +
                       $"<|im_start|>assistant\n";
            return Encode(text);
        }

        // ── BPE core ───────────────────────────────────────────────────────

        private int[] BpeEncode(string text)
        {
            if (text.Length == 0)
            {
                return [];
            }

            // Convert to byte-level characters
            var utf8 = Encoding.UTF8.GetBytes(text);
            var charBuf = new char[utf8.Length];
            for (var i = 0; i < utf8.Length; i++)
            {
                charBuf[i] = _byteToChar[utf8[i]];
            }
            var chars = new string(charBuf);

            // Initial token sequence: one token per character
            var tokens = new int[chars.Length];
            for (var i = 0; i < chars.Length; i++)
            {
                var ch = chars[i].ToString();
                tokens[i] = _vocab.TryGetValue(ch, out var id) ? id : EndOfText;
            }

            // Apply BPE merges greedily (lowest rank = highest priority)
            return ApplyMerges(tokens);
        }

        private int[] ApplyMerges(int[] tokens)
        {
            var ids = new List<int>(tokens);

            while (ids.Count > 1)
            {
                // Find lowest-rank (highest priority) adjacent pair
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

                if (bestIndex < 0)
                {
                    break;
                }

                // Merge the pair: find the merged token ID
                var pieceA = _decoder[ids[bestIndex]];
                var pieceB = _decoder[ids[bestIndex + 1]];
                var merged = pieceA + pieceB;
                var mergedId = _vocab.TryGetValue(merged, out var mid) ? mid : EndOfText;

                ids[bestIndex] = mergedId;
                ids.RemoveAt(bestIndex + 1);
            }

            return ids.ToArray();
        }

        // ── Special token splitting ────────────────────────────────────────

        private List<(string Text, bool IsSpecial)> SplitOnSpecialTokens(string text)
        {
            var result = new List<(string, bool)>();
            if (_specialTokens.Count == 0)
            {
                result.Add((text, false));
                return result;
            }

            var escaped = new string[_specialTokens.Count];
            var ei = 0;
            foreach (var key in _specialTokens.Keys)
            {
                escaped[ei++] = Regex.Escape(key);
            }
            var pattern = string.Join("|", escaped);
            var regex = new Regex(pattern);
            var pos = 0;

            foreach (Match m in regex.Matches(text))
            {
                if (m.Index > pos)
                {
                    result.Add((text.Substring(pos, m.Index - pos), false));
                }
                result.Add((m.Value, true));
                pos = m.Index + m.Length;
            }

            if (pos < text.Length)
            {
                result.Add((text.Substring(pos), false));
            }

            return result;
        }

        // ── Byte ↔ char mapping (GPT-2 style) ────────────────────────────

        private static char[] BuildByteToChar()
        {
            var map = new char[256];
            var n = 0;

            // Printable ASCII: assign directly
            for (var b = 0; b < 256; b++)
            {
                if ((b >= '!' && b <= '~') || (b >= '¡' && b <= '¬') || (b >= '®' && b <= 'ÿ'))
                {
                    map[b] = (char)b;
                }

                if (!((b >= '!' && b <= '~') || (b >= '¡' && b <= '¬') || (b >= '®' && b <= 'ÿ')))
                {
                    map[b] = (char)0; // placeholder
                }
            }

            // Remaining bytes: assign to Unicode starting at U+0100
            for (var b = 0; b < 256; b++)
            {
                if (map[b] == (char)0)
                {
                    map[b] = (char)(256 + n++);
                }
            }

            // Space (0x20) maps to Ġ (U+0120)
            map[0x20] = '\u0120';

            return map;
        }

        private static byte[] BuildCharToByte()
        {
            var forward = BuildByteToChar();
            var reverse = new byte[65536];
            for (var b = 0; b < 256; b++)
            {
                reverse[forward[b]] = (byte)b;
            }
            return reverse;
        }
    }
}
