// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio.Tts.Snac;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Whisper;

namespace DevOnBike.Overfit.Audio.Tts.Orpheus
{
    /// <summary>
    /// Turns a folder of voice recordings (+ transcripts) into Orpheus fine-tuning examples, entirely in managed
    /// .NET: each clip is SNAC-<see cref="Snac.Snac.Encode"/>d to audio tokens and paired with its text in the
    /// model's prompt format, producing the exact (prompt → audio-token) sequences the LM should learn. This is the
    /// data-prep half of voice cloning — the encoder makes it Python-free. (Whether you may clone a given voice is a
    /// consent/legal matter; clone only voices you own or have permission for.)
    /// </summary>
    public sealed class VoiceCloneDatasetBuilder
    {
        private const int OrpheusSampleRate = 24_000;
        private const int WhisperSampleRate = 16_000;

        private readonly Snac.Snac _snac;
        private readonly ITokenizer _tokenizer;
        private readonly int _audioTokenBase;
        private readonly int _endOfSpeechTokenId;

        /// <summary>
        /// <paramref name="snac"/> encodes clips to codes; <paramref name="tokenizer"/> is the Orpheus model's
        /// tokenizer (used to tokenize prompts and to resolve the audio-token id base);
        /// <paramref name="endOfSpeechTokenId"/> terminates each audio stream.
        /// </summary>
        public VoiceCloneDatasetBuilder(Snac.Snac snac, ITokenizer tokenizer, int endOfSpeechTokenId)
        {
            _snac = snac;
            _tokenizer = tokenizer;
            _endOfSpeechTokenId = endOfSpeechTokenId;
            _audioTokenBase = ResolveAudioTokenBase(tokenizer);
        }

        /// <summary>The vocab id of <c>&lt;custom_token_0&gt;</c> (the audio-token base) for this tokenizer.</summary>
        public int AudioTokenBase => _audioTokenBase;

        /// <summary>Builds one training example from in-memory 24 kHz mono audio and its transcript.</summary>
        public OrpheusTrainingExample BuildExample(ReadOnlySpan<float> audio24k, string text, string voice)
        {
            var codes = _snac.Encode(audio24k);
            var promptIds = Tokenize(OrpheusPrompt.Format(text, voice));
            return OrpheusTrainingSequence.Build(promptIds, codes, _audioTokenBase, _endOfSpeechTokenId);
        }

        /// <summary>
        /// Builds examples from every <c>*.wav</c> / <c>*.mp3</c> in <paramref name="directory"/>. The transcript
        /// is read from a sibling <c>.txt</c>; if absent and <paramref name="whisper"/> is supplied, the clip is
        /// auto-transcribed (review the result — STT is imperfect). All clips are resampled to 24 kHz.
        ///
        /// <para>
        /// Per-clip cleanup (on by default — real per-file recordings are rarely studio-clean): each clip is
        /// peak-normalized (quiet takes make a weak voice) and has leading/trailing silence trimmed. The trim
        /// matters most: a recording with a long pause before you start speaking would otherwise teach the model
        /// to emit that silence at the start of every generation (the "start didn't generate" failure). A short
        /// <paramref name="keepPaddingSeconds"/> of headroom is kept on each side so onsets/decays aren't clipped.
        /// </para>
        /// </summary>
        public List<OrpheusTrainingExample> BuildFromFolder(
            string directory, string voice, WhisperTranscriber? whisper = null, string language = "en",
            bool normalize = true, bool trimSilence = true,
            float silenceThreshold = 0.02f, float keepPaddingSeconds = 0.05f)
        {
            if (!Directory.Exists(directory))
            {
                throw new OverfitFormatException($"Voice dataset directory not found: {directory}");
            }

            var keepPadding = (int)(keepPaddingSeconds * OrpheusSampleRate);
            var examples = new List<OrpheusTrainingExample>();
            foreach (var path in Directory.GetFiles(directory))
            {
                if (!AudioFile.IsSupported(path))
                {
                    continue;
                }

                var raw = AudioFile.ReadMono(path, out var rate);

                var text = ResolveTranscript(path, raw, rate, whisper, language);
                if (string.IsNullOrWhiteSpace(text))
                {
                    continue;
                }

                var audio24 = rate == OrpheusSampleRate ? raw : AudioResampler.Resample(raw, rate, OrpheusSampleRate);

                // Normalize first so the silence threshold is relative to a known peak (consistent across clips),
                // then trim the leading/trailing pause the recording may carry before/after the spoken line.
                if (normalize)
                {
                    audio24 = AudioPostProcessing.PeakNormalize(audio24);
                }
                if (trimSilence)
                {
                    audio24 = AudioPostProcessing.TrimSilence(audio24, silenceThreshold, keepPadding);
                }

                examples.Add(BuildExample(audio24, text, voice));
            }

            if (examples.Count == 0)
            {
                throw new OverfitFormatException(
                    $"No usable (audio + transcript) pairs found in {directory}. Provide *.txt transcripts or a Whisper model.");
            }
            return examples;
        }

        /// <summary>
        /// Builds examples from a <b>single recording</b> of all the lines read in order, with a clear pause
        /// between each. The audio is split on silence (<see cref="AudioSegmenter"/>) and each segment is paired
        /// with the matching transcript line. Throws if the segment count does not match the line count (so you
        /// can re-record or adjust the pause).
        /// </summary>
        public List<OrpheusTrainingExample> BuildFromRecording(
            ReadOnlySpan<float> audio, int sampleRate, IReadOnlyList<string> transcripts, string voice,
            float minSilenceSeconds = 0.5f, float silenceThreshold = 0.03f)
        {
            // Peak-normalize first: quiet recordings make silence detection (and the voice) weak.
            var normalized = AudioPostProcessing.PeakNormalize(audio);

            var segments = AudioSegmenter.SplitOnSilence(
                normalized, sampleRate, amplitudeThreshold: silenceThreshold, minSilenceSeconds: minSilenceSeconds);
            if (segments.Count != transcripts.Count)
            {
                throw new OverfitFormatException(
                    $"Found {segments.Count} spoken segments but {transcripts.Count} transcript lines. Read every "
                    + $"line in order with a clear (~1 s) pause between, or tune minSilenceSeconds ({minSilenceSeconds}s) "
                    + $"/ silenceThreshold ({silenceThreshold}).");
            }

            var examples = new List<OrpheusTrainingExample>(segments.Count);
            for (var k = 0; k < segments.Count; k++)
            {
                var (start, end) = segments[k];
                var segment = normalized.AsSpan(start, end - start);
                var segment24 = sampleRate == OrpheusSampleRate
                    ? segment.ToArray()
                    : AudioResampler.Resample(segment, sampleRate, OrpheusSampleRate);
                examples.Add(BuildExample(segment24, transcripts[k], voice));
            }
            return examples;
        }

        /// <summary>
        /// Builds examples from a single recording by <b>auto-transcribing each segment with Whisper</b> instead of
        /// pairing against a fixed transcript by order. This is the robust path: it can't misalign, because every
        /// segment is paired with its own transcription — even if a silence split merges two sentences or cuts one
        /// in half. Segments whose transcription is shorter than <paramref name="minChars"/> are dropped (junk /
        /// stray fragments). STT isn't perfect, but small text errors don't stop the model learning the voice.
        /// </summary>
        public List<OrpheusTrainingExample> BuildFromRecording(
            ReadOnlySpan<float> audio, int sampleRate, string voice, WhisperTranscriber whisper,
            float minSilenceSeconds = 0.5f, float silenceThreshold = 0.03f, string language = "en", int minChars = 4)
        {
            var normalized = AudioPostProcessing.PeakNormalize(audio);
            var segments = AudioSegmenter.SplitOnSilence(
                normalized, sampleRate, amplitudeThreshold: silenceThreshold, minSilenceSeconds: minSilenceSeconds);

            var examples = new List<OrpheusTrainingExample>(segments.Count);
            foreach (var (start, end) in segments)
            {
                var segment = normalized.AsSpan(start, end - start);
                var segment16 = sampleRate == WhisperSampleRate ? segment.ToArray() : AudioResampler.Resample(segment, sampleRate, WhisperSampleRate);
                var text = whisper.Transcribe(segment16, language).Trim();
                if (text.Length < minChars)
                {
                    continue;
                }
                var segment24 = sampleRate == OrpheusSampleRate ? segment.ToArray() : AudioResampler.Resample(segment, sampleRate, OrpheusSampleRate);
                examples.Add(BuildExample(segment24, text, voice));
            }

            if (examples.Count == 0)
            {
                throw new OverfitFormatException("No usable segments after auto-transcription — check the recording / silence settings.");
            }
            return examples;
        }

        /// <summary>
        /// Convenience: read a single audio file and a transcript file (one line per utterance, in order) and build
        /// the dataset. Blank lines and a leading "N." / "N)" numbering are ignored.
        /// </summary>
        public List<OrpheusTrainingExample> BuildFromRecordingFile(string audioPath, string transcriptPath, string voice)
        {
            var audio = AudioFile.ReadMono(audioPath, out var rate);
            var lines = new List<string>();
            foreach (var raw in File.ReadAllLines(transcriptPath))
            {
                var line = CleanTranscriptLine(raw);
                if (line.Length > 0)
                {
                    lines.Add(line);
                }
            }
            return BuildFromRecording(audio, rate, lines, voice);
        }

        private static string CleanTranscriptLine(string raw)
        {
            var s = raw.Trim();
            // Strip a leading "12. " / "12) " enumeration if present.
            var dot = 0;
            while (dot < s.Length && char.IsDigit(s[dot]))
            {
                dot++;
            }
            if (dot > 0 && dot < s.Length && (s[dot] == '.' || s[dot] == ')'))
            {
                s = s[(dot + 1)..].Trim();
            }
            return s;
        }

        // Returns the transcript for a clip, or empty (→ the clip is skipped) when none can be resolved. A folder
        // routinely also holds non-dataset audio (e.g. previously-generated clone outputs) with no sibling .txt;
        // those must be skipped, not fail the whole build. With no .txt and no Whisper there's nothing to pair, so
        // the clip is skipped; if EVERY clip ends up unpaired, BuildFromFolder's count==0 guard reports it clearly.
        private static string ResolveTranscript(
            string audioPath, float[] raw, int rate, WhisperTranscriber? whisper, string language)
        {
            var txt = Path.ChangeExtension(audioPath, ".txt");
            if (File.Exists(txt))
            {
                return File.ReadAllText(txt).Trim();
            }
            if (whisper is null)
            {
                return string.Empty;
            }
            var audio16 = rate == WhisperSampleRate ? raw : AudioResampler.Resample(raw, rate, WhisperSampleRate);
            return whisper.Transcribe(audio16, language).Trim();
        }

        private int[] Tokenize(string text)
        {
            var buffer = new int[_tokenizer.CountTokens(text)];
            var n = _tokenizer.Encode(text, buffer);
            return n == buffer.Length ? buffer : buffer[..n];
        }

        // <custom_token_0> must be a single token; <custom_token_1> must be the next id (contiguous audio range).
        private static int ResolveAudioTokenBase(ITokenizer tokenizer)
        {
            var b0 = SingleTokenId(tokenizer, "<custom_token_0>");
            var b1 = SingleTokenId(tokenizer, "<custom_token_1>");
            if (b1 != b0 + 1)
            {
                throw new OverfitRuntimeException(
                    $"Audio tokens are not contiguous ({b0}, {b1}) — this does not look like an Orpheus tokenizer.");
            }
            return b0;
        }

        private static int SingleTokenId(ITokenizer tokenizer, string special)
        {
            Span<int> ids = stackalloc int[8];
            var n = tokenizer.Encode(special, ids);
            if (n != 1)
            {
                throw new OverfitRuntimeException(
                    $"Expected '{special}' to tokenize to exactly one token, got {n} — not an Orpheus tokenizer.");
            }
            return ids[0];
        }
    }
}
