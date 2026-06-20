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
        private const int SamplesPerWindow = MelSpectrogram.SampleRate * 30; // 30 s

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
        {
            // Pad/trim to a single 30 s window (Whisper's fixed input length); the window is reused across calls.
            var window = _window.AsSpan();
            window.Clear();
            samples.Slice(0, Math.Min(samples.Length, SamplesPerWindow)).CopyTo(window);

            var mel = _mel.LogMel(window, out var frames);
            var encoderOut = _encoder.Encode(mel, frames, out var nCtx);

            var prompt = BuildPrompt(language);
            var produced = _decoder.DecodeCached(encoderOut, nCtx, prompt, _tokenizer.EndOfTranscript, maxNewTokens);
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
