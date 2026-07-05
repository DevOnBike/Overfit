// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio;

namespace DevOnBike.Overfit.LanguageModels.Whisper
{
    /// <summary>
    /// Turnkey Whisper speech-to-text — pure .NET, CPU, no GPU/Python. Loads a whisper.cpp ggml model
    /// (e.g. <c>ggml-tiny.bin</c>) and transcribes 16 kHz mono audio:
    /// <code>
    ///   var w = WhisperTranscriber.Load(@"C:\whisper\ggml-tiny.bin");
    ///   var text = w.TranscribeFile(@"C:\audio\speech.wav");
    /// </code>
    /// Pipeline: audio → log-mel (using the model's own mel filterbank for parity) → encoder → greedy decode
    /// (prompt = sot / language / transcribe / no-timestamps) → text. Scope: a single 30 s window, greedy.
    /// </summary>
    public sealed class WhisperTranscriber
    {
        private const int SamplesPerWindow = MelSpectrogram.SampleRate * 30; // 30 s (Whisper's max window)

        // Lower bound on the processed window so very short clips still give the conv stem enough frames.
        // Anti-loop: a short clip can make Whisper re-emit the same phrase instead of stopping. If the decoded
        // tail is an exact back-to-back repeat of period >= this many tokens, drop the duplicate and stop. Big
        // enough not to fire on natural short repeats ("New York, New York" = period 2).
        private const int MinRepeatPeriod = 3;

        private const int MinSamples = MelSpectrogram.SampleRate; // 1 s floor
        // Trailing silence kept after the audio. Whisper uses the post-speech silence as its "audio ended" cue
        // to emit the end-of-transcript token; trim it too tight and the decoder never stops and loops to
        // maxNewTokens (repetition hallucination). ~1 s of zeros is enough for EOT while still cutting the
        // encoder's frame count far below the full 30 s window.
        private const int TrailingSamples = MelSpectrogram.SampleRate * 3 / 2; // 1.5 s

        private readonly WhisperModel _model;
        private readonly MelSpectrogram _mel;
        private readonly WhisperEncoder _encoder;
        private readonly WhisperDecoder _decoder;
        private readonly WhisperTokenizer _tokenizer;
        private readonly float[] _window = new float[SamplesPerWindow]; // reused 30 s input window

        private WhisperTranscriber(WhisperModel model)
        {
            _model = model;
            // Use the model's own mel filterbank for bit-parity with whisper.cpp.
            _mel = new MelSpectrogram(model.MelFilterRows, model.MelFilters);
            _encoder = new WhisperEncoder(model);
            _decoder = new WhisperDecoder(model);
            _tokenizer = new WhisperTokenizer(model);
        }

        /// <summary>Loads a whisper.cpp ggml model from disk.</summary>
        public static WhisperTranscriber Load(string ggmlPath) => new(WhisperGgmlLoader.Load(ggmlPath));

        public WhisperConfig Config => _model.Config;
        public WhisperTokenizer Tokenizer => _tokenizer;

        /// <summary>Transcribes an audio file — any format with a registered <see cref="IAudioDecoder"/> (WAV
        /// and MP3 built in), any sample rate (resampled to 16 kHz), mono (stereo is downmixed by the decoders).</summary>
        public string TranscribeFile(string audioPath, string language = "en", int maxNewTokens = 224)
        {
            var samples = AudioFile.ReadMono(audioPath, out var sr);
            if (sr != MelSpectrogram.SampleRate)
            {
                samples = AudioResampler.Resample(samples, sr, MelSpectrogram.SampleRate);
            }
            return Transcribe(samples, language, maxNewTokens);
        }

        /// <summary>Transcribes mono 16 kHz <paramref name="samples"/> → text.</summary>
        public string Transcribe(ReadOnlySpan<float> samples, string language = "en", int maxNewTokens = 224)
            => Transcribe(samples, language, maxNewTokens, padToFullWindow: false);

        /// <summary>
        /// Core transcription. By default the encoder runs over only the audio actually present (rounded up to
        /// the 1 s floor + a little trailing room, capped at 30 s) instead of always padding to a full 30 s
        /// window. The encoder cost is dominated by self-attention, which is O(nCtx²) in the frame count, so a
        /// 3 s clip costs ~(3/30)² of a full window — a large speed-up for short utterances (voice input). This
        /// matches whisper.cpp's "process the real length" behaviour; it is NOT bit-identical to the 30 s-padded
        /// path (self-attention sees fewer frames), but the log-mel of the trimmed window equals the prefix of
        /// the padded one and the transcript is unchanged for normal clips. <paramref name="padToFullWindow"/>
        /// forces the legacy full-30 s behaviour (internal, for A/B parity/perf tests).
        /// </summary>
        internal string Transcribe(ReadOnlySpan<float> samples, string language, int maxNewTokens, bool padToFullWindow)
        {
            int windowLen;
            if (padToFullWindow)
            {
                windowLen = SamplesPerWindow;
            }
            else
            {
                var present = Math.Min(samples.Length, SamplesPerWindow);
                windowLen = Math.Min(SamplesPerWindow, Math.Max(present + TrailingSamples, MinSamples));
            }

            // The 30 s buffer is reused across calls; we only clear + fill (and run the mel/encoder over) the
            // first windowLen samples — the rest is left untouched and never read.
            var window = _window.AsSpan(0, windowLen);
            window.Clear();
            samples.Slice(0, Math.Min(samples.Length, windowLen)).CopyTo(window);

            var mel = _mel.LogMel(window, out var frames);
            var encoderOut = _encoder.Encode(mel, frames, out var nCtx);

            var prompt = BuildPrompt(language);
            var produced = _decoder.DecodeCached(
                encoderOut, nCtx, prompt, _tokenizer.EndOfTranscript, maxNewTokens, MinRepeatPeriod);
            return _tokenizer.Decode(produced).Trim();
        }

#pragma warning disable OVERFIT001 // Prompt token list built once per transcription (3-4 special ids), not per token.
        private int[] BuildPrompt(string language)
        {
            // English-only models have no language token; multilingual prepend it.
            if (!_model.Config.IsMultilingual)
            {
                return
                [
                    _tokenizer.StartOfTranscript,
                    _tokenizer.Transcribe,
                    _tokenizer.NoTimestamps
                ];
            }

            return
            [
                _tokenizer.StartOfTranscript,
                _tokenizer.LanguageToken(language),
                _tokenizer.Transcribe,
                _tokenizer.NoTimestamps
            ];
        }
#pragma warning restore OVERFIT001
    }
}
