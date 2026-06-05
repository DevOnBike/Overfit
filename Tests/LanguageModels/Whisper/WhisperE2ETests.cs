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
    /// THE PAYOFF: transcribe a real audio file with a real whisper.cpp model, end-to-end, in pure .NET on the
    /// CPU. Loads <c>ggml-tiny.bin</c> + the classic <c>jfk.wav</c> sample and checks the transcription contains
    /// the expected words. This is the FIRST real-model / real-tensor-name validation (S1–S4 were synthetic);
    /// any tensor-name / shape / mel mismatch surfaces here. <see cref="LongFact"/> — needs
    /// <c>OVERFIT_WHISPER_DIR</c> with <c>ggml-tiny.bin</c> and <c>jfk.wav</c>.
    /// </summary>
    public sealed class WhisperE2ETests
    {
        private readonly ITestOutputHelper _out;
        public WhisperE2ETests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Dump_RealTinyModel_TensorNames()
        {
            var ggml = TestModelPaths.Whisper.RequireTinyGgmlPath();
            var model = WhisperGgmlLoader.Load(ggml);
            var c = model.Config;
            _out.WriteLine($"config: nVocab={c.NVocab} aCtx={c.NAudioCtx} aState={c.NAudioState} aHead={c.NAudioHead} aLayer={c.NAudioLayer} tCtx={c.NTextCtx} tState={c.NTextState} tHead={c.NTextHead} tLayer={c.NTextLayer} nMels={c.NMels} f16={c.F16}");
            _out.WriteLine($"mel filters: {model.MelFilterRows} x {model.MelFilterCols}; vocab size {model.Vocab.Count}");
            var names = new List<string>(model.Tensors.Keys);
            names.Sort(StringComparer.Ordinal);
            _out.WriteLine($"--- {names.Count} tensors ---");
            foreach (var n in names)
            {
                var t = model.Tensors[n];
                _out.WriteLine($"{n}  [{string.Join(",", t.Shape)}]");
            }
        }

        [LongFact]
        public void Transcribe_RealTinyModel_JfkSample()
        {
            var ggml = TestModelPaths.Whisper.RequireTinyGgmlPath();
            var wav = TestModelPaths.Whisper.RequireSampleWavPath();

            var whisper = WhisperTranscriber.Load(ggml);
            _out.WriteLine($"model: nVocab={whisper.Config.NVocab} audioLayers={whisper.Config.NAudioLayer} textLayers={whisper.Config.NTextLayer} multilingual={whisper.Config.IsMultilingual}");

            var text = whisper.TranscribeFile(wav, language: "en");
            _out.WriteLine($"transcription: \"{text}\"");

            // jfk.wav: "...ask not what your country can do for you, ask what you can do for your country."
            Assert.Contains("country", text, StringComparison.OrdinalIgnoreCase);
        }
    }
}
