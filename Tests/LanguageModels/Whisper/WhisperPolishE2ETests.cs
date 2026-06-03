// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Whisper;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Whisper
{
    /// <summary>
    /// Multilingual transcription: the same tiny model transcribes a POLISH recording with
    /// <c>language: "pl"</c> — proving the language-token path works beyond English. Drop a 16 kHz mono
    /// <c>polish.wav</c> into <c>OVERFIT_WHISPER_DIR</c> (TTS-generated or recorded, then
    /// <c>ffmpeg -i in.mp3 -ar 16000 -ac 1 polish.wav</c>). <see cref="LongFact"/>.
    /// </summary>
    public sealed class WhisperPolishE2ETests
    {
        private readonly ITestOutputHelper _out;
        public WhisperPolishE2ETests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Transcribe_Polish_Recording()
        {
            var ggml = TestModelPaths.Whisper.RequireTinyGgmlPath();
            var wav = TestModelPaths.Whisper.RequirePolishWavPath();

            var whisper = WhisperTranscriber.Load(ggml);
            Assert.True(whisper.Config.IsMultilingual, "Polish needs a multilingual model (use ggml-tiny.bin, not .en).");

            var text = whisper.TranscribeFile(wav, language: "pl");
            _out.WriteLine($"polish transcription: \"{text}\"");

            Assert.False(string.IsNullOrWhiteSpace(text), "transcription is empty");
        }
    }
}
