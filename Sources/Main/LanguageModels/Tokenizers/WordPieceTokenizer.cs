// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Tokenizers
{
    /// <summary>
    /// BERT-style WordPiece tokenizer (the one shipped with <c>bert-base-uncased</c> and therefore the
    /// MiniLM / sentence-transformers encoder family). Reads a plain <c>vocab.txt</c> (one token per line,
    /// line index = id) and reproduces the reference pipeline: text cleaning → optional lowercasing +
    /// accent stripping → whitespace + punctuation splitting (BasicTokenizer) → greedy longest-match
    /// subword splitting with <c>##</c> continuation prefixes (WordpieceTokenizer). <see cref="Encode"/>
    /// wraps the result in <c>[CLS] … [SEP]</c> by default — exactly the input the <c>BertEncoder</c>
    /// embedding model expects.
    ///
    /// Implements <see cref="ITokenizer"/> for ecosystem consistency, but this is an encoder (embedding)
    /// tokenizer, not a generative one — <see cref="EndOfTextTokenId"/> maps to <c>[SEP]</c>.
    /// SentencePiece / Unigram are out of scope (WordPiece only).
    /// </summary>
    public sealed class WordPieceTokenizer : ITokenizer
    {
        private const string ContinuationPrefix = "##";

        private readonly Dictionary<string, int> _vocab;
        private readonly string[] _inverse;
        private readonly bool _doLowerCase;
        private readonly int _maxInputCharsPerWord;

        private readonly int _unkId;
        private readonly int _clsId;
        private readonly int _sepId;
        private readonly int _padId;

        public WordPieceTokenizer(
            IReadOnlyDictionary<string, int> vocab,
            bool doLowerCase = true,
            int maxInputCharsPerWord = 100,
            string unkToken = "[UNK]",
            string clsToken = "[CLS]",
            string sepToken = "[SEP]",
            string padToken = "[PAD]")
        {
            ArgumentNullException.ThrowIfNull(vocab);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxInputCharsPerWord);

            _vocab = new Dictionary<string, int>(vocab.Count, StringComparer.Ordinal);
            var maxId = -1;
            foreach (var kv in vocab)
            {
                _vocab[kv.Key] = kv.Value;
                if (kv.Value > maxId) { maxId = kv.Value; }
            }

            _inverse = new string[maxId + 1];
            foreach (var kv in _vocab)
            {
                if ((uint)kv.Value < (uint)_inverse.Length) { _inverse[kv.Value] = kv.Key; }
            }

            _doLowerCase = doLowerCase;
            _maxInputCharsPerWord = maxInputCharsPerWord;

            _unkId = RequireToken(unkToken);
            _clsId = RequireToken(clsToken);
            _sepId = RequireToken(sepToken);
            _padId = _vocab.TryGetValue(padToken, out var pad) ? pad : 0;
        }

        public int VocabularySize => _inverse.Length;

        public int EndOfTextTokenId => _sepId;

        public int UnknownTokenId => _unkId;

        public int ClassifierTokenId => _clsId;

        public int SeparatorTokenId => _sepId;

        public int PaddingTokenId => _padId;

        public bool SupportsZeroAllocationEncode => false;

        public bool SupportsZeroAllocationDecode => false;

        /// <summary>Loads a tokenizer from a HuggingFace <c>vocab.txt</c> (line index = token id).</summary>
        public static WordPieceTokenizer FromVocabFile(string path, bool doLowerCase = true)
        {
            ArgumentException.ThrowIfNullOrEmpty(path);
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("WordPiece vocab file not found.", path);
            }

            var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
            var id = 0;
            using (var reader = new StreamReader(path, Encoding.UTF8))
            {
                string? line;
                while ((line = reader.ReadLine()) is not null)
                {
                    // vocab.txt tokens never contain trailing whitespace; a token is the line verbatim
                    // minus the platform newline (already stripped by ReadLine). Blank lines map too.
                    vocab[line] = id;
                    id++;
                }
            }

            if (vocab.Count == 0)
            {
                throw new InvalidDataException($"WordPiece vocab file '{path}' is empty.");
            }

            return new WordPieceTokenizer(vocab, doLowerCase);
        }

        /// <summary>
        /// Tokenizes <paramref name="text"/> to subword ids, wrapping in <c>[CLS] … [SEP]</c> when
        /// <paramref name="addSpecialTokens"/> is true (the default — what the encoder consumes).
        /// </summary>
        public int[] Encode(string text, bool addSpecialTokens = true)
        {
            ArgumentNullException.ThrowIfNull(text);
            var ids = new List<int>(text.Length + 2);
            if (addSpecialTokens) { ids.Add(_clsId); }
            TokenizeInto(text, ids);
            if (addSpecialTokens) { ids.Add(_sepId); }
            return ids.ToArray();
        }

        public int CountTokens(ReadOnlySpan<char> text)
        {
            var ids = new List<int>();
            TokenizeInto(text.ToString(), ids);
            return ids.Count + 2; // [CLS] … [SEP]
        }

        public int Encode(ReadOnlySpan<char> text, Span<int> destination)
        {
            var encoded = Encode(text.ToString(), addSpecialTokens: true);
            if (encoded.Length > destination.Length)
            {
                throw new ArgumentException(
                    $"Destination too small: need {encoded.Length}, have {destination.Length}.",
                    nameof(destination));
            }

            encoded.AsSpan().CopyTo(destination);
            return encoded.Length;
        }

        public int Decode(ReadOnlySpan<int> tokens, Span<char> destination)
        {
            var s = DecodeToString(tokens);
            if (s.Length > destination.Length)
            {
                throw new ArgumentException(
                    $"Destination too small: need {s.Length}, have {destination.Length}.",
                    nameof(destination));
            }

            s.AsSpan().CopyTo(destination);
            return s.Length;
        }

        public string DecodeToString(ReadOnlySpan<int> tokens)
        {
            var sb = new StringBuilder();
            foreach (var id in tokens)
            {
                if ((uint)id >= (uint)_inverse.Length) { continue; }
                var tok = _inverse[id];
                if (tok is null || id == _clsId || id == _sepId || id == _padId) { continue; }

                if (tok.StartsWith(ContinuationPrefix, StringComparison.Ordinal))
                {
                    sb.Append(tok, ContinuationPrefix.Length, tok.Length - ContinuationPrefix.Length);
                }
                else
                {
                    if (sb.Length > 0) { sb.Append(' '); }
                    sb.Append(tok);
                }
            }

            return sb.ToString();
        }

        /// <summary>BasicTokenizer (clean + lowercase/strip-accents + split) → WordPiece, appending ids.</summary>
        private void TokenizeInto(string text, List<int> ids)
        {
            var words = new List<string>();
            BasicTokenize(text, words);
            foreach (var word in words)
            {
                WordPiece(word, ids);
            }
        }

        private void BasicTokenize(string text, List<string> output)
        {
            // 1. clean: drop control / replacement chars, normalize all whitespace to a single space.
            var cleaned = new StringBuilder(text.Length);
            foreach (var ch in text)
            {
                if (ch == '\0' || ch == '�' || IsControl(ch)) { continue; }
                cleaned.Append(IsWhitespace(ch) ? ' ' : ch);
            }

            // 2. optional lowercase + accent strip.
            var prepared = cleaned.ToString();
            if (_doLowerCase)
            {
                prepared = StripAccents(prepared.ToLowerInvariant());
            }

            // 3. split on whitespace, then split punctuation out as standalone tokens.
            var current = new StringBuilder();
            foreach (var ch in prepared)
            {
                if (ch == ' ')
                {
                    FlushWord(current, output);
                    continue;
                }

                if (IsPunctuation(ch))
                {
                    FlushWord(current, output);
                    output.Add(ch.ToString());
                    continue;
                }

                current.Append(ch);
            }

            FlushWord(current, output);
        }

        private static void FlushWord(StringBuilder current, List<string> output)
        {
            if (current.Length > 0)
            {
                output.Add(current.ToString());
                current.Clear();
            }
        }

        private void WordPiece(string word, List<int> ids)
        {
            if (word.Length > _maxInputCharsPerWord)
            {
                ids.Add(_unkId);
                return;
            }

            var start = 0;
            var pieces = new List<int>();
            var isBad = false;
            while (start < word.Length)
            {
                var end = word.Length;
                var matchId = -1;
                while (start < end)
                {
                    var sub = start > 0
                        ? string.Concat(ContinuationPrefix, word.AsSpan(start, end - start))
                        : word.Substring(start, end - start);
                    if (_vocab.TryGetValue(sub, out var found))
                    {
                        matchId = found;
                        break;
                    }

                    end--;
                }

                if (matchId < 0)
                {
                    isBad = true;
                    break;
                }

                pieces.Add(matchId);
                start = end;
            }

            if (isBad)
            {
                ids.Add(_unkId);
            }
            else
            {
                ids.AddRange(pieces);
            }
        }

        private int RequireToken(string token)
        {
            if (!_vocab.TryGetValue(token, out var id))
            {
                throw new InvalidDataException($"WordPiece vocabulary is missing the required special token '{token}'.");
            }

            return id;
        }

        private static string StripAccents(string text)
        {
            var decomposed = text.Normalize(NormalizationForm.FormD);
            var sb = new StringBuilder(decomposed.Length);
            foreach (var ch in decomposed)
            {
                if (CharUnicodeInfo.GetUnicodeCategory(ch) != UnicodeCategory.NonSpacingMark)
                {
                    sb.Append(ch);
                }
            }

            return sb.ToString().Normalize(NormalizationForm.FormC);
        }

        private static bool IsWhitespace(char ch)
        {
            if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') { return true; }
            return CharUnicodeInfo.GetUnicodeCategory(ch) == UnicodeCategory.SpaceSeparator;
        }

        private static bool IsControl(char ch)
        {
            if (ch == '\t' || ch == '\n' || ch == '\r') { return false; }
            var cat = CharUnicodeInfo.GetUnicodeCategory(ch);
            return cat == UnicodeCategory.Control || cat == UnicodeCategory.Format;
        }

        private static bool IsPunctuation(char ch)
        {
            // BERT treats all ASCII non-alphanumeric printable chars as punctuation, plus any Unicode P* category.
            if ((ch >= '!' && ch <= '/') || (ch >= ':' && ch <= '@') ||
                (ch >= '[' && ch <= '`') || (ch >= '{' && ch <= '~'))
            {
                return true;
            }

            var cat = CharUnicodeInfo.GetUnicodeCategory(ch);
            return cat is UnicodeCategory.ConnectorPunctuation
                or UnicodeCategory.DashPunctuation
                or UnicodeCategory.OpenPunctuation
                or UnicodeCategory.ClosePunctuation
                or UnicodeCategory.InitialQuotePunctuation
                or UnicodeCategory.FinalQuotePunctuation
                or UnicodeCategory.OtherPunctuation;
        }
    }
}
