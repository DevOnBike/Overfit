// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Audio;
using DevOnBike.Overfit.Audio.Tts.Orpheus;
using DevOnBike.Overfit.LanguageModels.Whisper;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>
    /// Empirical validation of the acronym lexicon (ROADMAP audio #2): the normalizer spells acronyms as spaced
    /// capitals ("AI" → "A I") so Orpheus says the LETTER NAMES ("ay eye"). This closes the loop the user's way —
    /// synthesize a sentence full of acronyms, then transcribe the produced audio with our OWN Whisper and check
    /// the letters actually come back. If the "spaced capitals" convention failed (model mumbles / says a word),
    /// Whisper would not recover the acronyms. Needs <c>c:\orpheus</c> + <c>c:\snac</c> + <c>c:\whisper</c>;
    /// [LongFact].
    /// </summary>
    public sealed class OrpheusAcronymPronunciationE2ETests
    {
        private readonly ITestOutputHelper _out;

        public OrpheusAcronymPronunciationE2ETests(ITestOutputHelper output)
        {
            _out = output;
        }

        [LongFact]
        public void Acronyms_SpelledConvention_HeardBackByWhisper()
        {
            TestModelPaths.Orpheus.RequireGgufPath();
            TestModelPaths.Snac.RequireSafetensorsPath();
            var ggml = TestModelPaths.Whisper.RequireTinyGgmlPath();

            using var engine = OrpheusVoiceEngine.Load(TestModelPaths.Orpheus.GgufPath, TestModelPaths.Snac.Dir);

            // Raw text — Synthesize(normalize:true) runs TtsTextNormalizer, so "AI"/"CPU"/"GPU"/"API" are spelled.
            const string sentence = "The AI uses the CPU and the GPU through one API.";
            var audio = engine.Synthesize(sentence, voice: "tara", seed: 1);
            Assert.True(audio.Length > engine.SampleRate / 2, $"too little audio: {audio.Length} samples");

            var wav = Path.Combine(TestModelPaths.Orpheus.Dir, "e2e_acronyms.wav");
            WavWriter.WriteMono(wav, audio, engine.SampleRate, WavSampleFormat.Pcm16,
                infoComment: "Overfit acronym-pronunciation validation");

            var whisper = WhisperTranscriber.Load(ggml);
            var heard = whisper.TranscribeFile(wav, language: "en");
            _out.WriteLine($"SENT : {sentence}");
            _out.WriteLine($"HEARD: \"{heard}\"");

            // Whisper renders spoken letter-names as the acronym ("ay eye" → "AI", "see pee you" → "CPU"), often
            // with dots/spaces ("A.I.", "C P U"). Strip non-letters + uppercase, then look for the acronyms back.
            var compact = Compact(heard);
            _out.WriteLine($"compact: {compact}");
            var hits = 0;
            foreach (var acr in new[] { "AI", "CPU", "GPU", "API" })
            {
                if (compact.Contains(acr, StringComparison.Ordinal))
                {
                    hits++;
                    _out.WriteLine($"  ✓ recovered {acr}");
                }
            }

            // At least half the acronyms must survive the synth→hear round-trip for the convention to be "working".
            Assert.True(hits >= 2, $"only {hits}/4 acronyms recovered from the audio — spelling convention suspect. Heard: \"{heard}\"");
        }

        // Uppercase, letters only (drops the dots/spaces Whisper inserts between spoken letters).
        private static string Compact(string text)
        {
            var sb = new StringBuilder(text.Length);
            foreach (var c in text)
            {
                if (char.IsLetter(c))
                {
                    sb.Append(char.ToUpperInvariant(c));
                }
            }
            return sb.ToString();
        }
    }
}
