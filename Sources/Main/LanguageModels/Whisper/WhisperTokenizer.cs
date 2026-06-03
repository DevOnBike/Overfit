// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;

namespace DevOnBike.Overfit.LanguageModels.Whisper
{
    /// <summary>
    /// Whisper BPE tokenizer built from a loaded <see cref="WhisperModel"/>'s vocab. Decodes token ids →
    /// text: the whisper.cpp ggml vocab stores each token as its RAW UTF-8 text piece (a leading space is a
    /// literal <c>0x20</c>, not GPT-2 byte-level <c>Ġ</c>), so decoding concatenates the raw token bytes and
    /// UTF-8 decodes once (so multi-byte characters split across tokens join correctly). The special-token
    /// ids (sot/eot/translate/transcribe/timestamps/language) are computed from the config exactly as
    /// whisper.cpp does (they are not stored in the file).
    /// </summary>
    public sealed class WhisperTokenizer
    {
        private readonly IReadOnlyList<string> _vocab;

        public WhisperTokenizer(WhisperModel model) : this(model.Config, model.Vocab) { }

        public WhisperTokenizer(WhisperConfig config, IReadOnlyList<string> vocab)
        {
            _vocab = vocab;

            // whisper.cpp struct defaults (valid for English-only), then the multilingual adjustment.
            int eot = 50256, sot = 50257, translate = 50357, transcribe = 50358,
                solm = 50359, prev = 50360, nosp = 50361, not_ = 50362, beg = 50363;

            if (config.IsMultilingual)
            {
                eot++;
                sot++;
                var dt = config.NumLanguages - 98;
                translate += dt; transcribe += dt; solm += dt; prev += dt; nosp += dt; not_ += dt; beg += dt;
            }

            EndOfTranscript = eot;
            StartOfTranscript = sot;
            Translate = translate;
            Transcribe = transcribe;
            StartOfLm = solm;
            StartOfPrev = prev;
            NoSpeech = nosp;
            NoTimestamps = not_;
            TimestampBegin = beg;
            LanguageCount = config.NumLanguages;
        }

        public int EndOfTranscript { get; }
        public int StartOfTranscript { get; }
        public int Translate { get; }
        public int Transcribe { get; }
        public int StartOfLm { get; }
        public int StartOfPrev { get; }
        public int NoSpeech { get; }
        public int NoTimestamps { get; }

        /// <summary>First timestamp token (<c>&lt;|0.00|&gt;</c>); subsequent timestamps are this + n.</summary>
        public int TimestampBegin { get; }

        public int LanguageCount { get; }

        /// <summary>Language token for the language at <paramref name="index"/> in Whisper's fixed language
        /// order (index 0 = English). Tokens sit at <c>StartOfTranscript + 1 + index</c>.</summary>
        public int LanguageTokenAt(int index)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(index);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, LanguageCount);
            return StartOfTranscript + 1 + index;
        }

        /// <summary>Language token for an ISO code (e.g. "en", "pl") in Whisper's order; throws if unknown.</summary>
        public int LanguageToken(string code)
        {
            var idx = WhisperLanguages.IndexOf(code);
            if (idx < 0) { throw new ArgumentException($"Unknown Whisper language code '{code}'.", nameof(code)); }
            return LanguageTokenAt(idx);
        }

        /// <summary>True for any non-text token (end/start/language/task/timestamp markers).</summary>
        public bool IsSpecial(int tokenId) => tokenId >= EndOfTranscript || tokenId >= _vocab.Count;

        /// <summary>
        /// Decodes a token sequence to text. Special tokens (sot/eot/language/task/timestamps) are skipped
        /// when <paramref name="skipSpecial"/> is true. Text tokens contribute their raw bytes, joined and
        /// UTF-8 decoded once at the end.
        /// </summary>
        public string Decode(ReadOnlySpan<int> tokens, bool skipSpecial = true)
        {
            var bytes = new List<byte>(tokens.Length * 4);
            foreach (var id in tokens)
            {
                if (IsSpecial(id))
                {
                    if (skipSpecial) { continue; }
                    continue; // markers aren't real text
                }
                // The token's raw UTF-8 bytes (round-tripped from the stored piece); concatenate and decode once.
                bytes.AddRange(Encoding.UTF8.GetBytes(_vocab[id]));
            }
            return Encoding.UTF8.GetString(bytes.ToArray());
        }
    }
}
