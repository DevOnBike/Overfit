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
    /// End-to-end validation of the pure-C# MP3 decoder: transcribe the SAME recording from both the MP3
    /// (<c>pl.mp3</c>, 24 kHz MPEG-2) and the WAV (<c>polish.wav</c>, 16 kHz) and check the Whisper output
    /// matches. If the from-scratch Layer III decoder is correct, both paths yield the same Polish text.
    /// <see cref="LongFactAttribute"/> — needs the real Whisper model + audio.
    /// </summary>
    public sealed class WhisperMp3E2ETests
    {
        private readonly ITestOutputHelper _out;
        public WhisperMp3E2ETests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Transcribe_Mp3_MatchesWav()
        {
            var w = WhisperTranscriber.Load(TestModelPaths.Whisper.RequireTinyGgmlPath());

            var fromWav = w.TranscribeFile(TestModelPaths.Whisper.RequirePolishWavPath(), "pl");
            var fromMp3 = w.TranscribeFile(TestModelPaths.Whisper.RequireSampleMp3Path(), "pl");

            _out.WriteLine($"WAV: {fromWav}");
            _out.WriteLine($"MP3: {fromMp3}");

            Assert.False(string.IsNullOrWhiteSpace(fromMp3), "MP3 transcription is empty");
            // The same recording → the same words. Compare on a normalized token overlap to tolerate tiny
            // boundary differences from the different sample rates / resampling.
            var wavWords = Normalize(fromWav);
            var mp3Words = Normalize(fromMp3);
            var overlap = 0;
            foreach (var word in mp3Words)
            {
                if (wavWords.Contains(word)) { overlap++; }
            }
            var ratio = mp3Words.Count == 0 ? 0.0 : (double)overlap / mp3Words.Count;
            _out.WriteLine($"word overlap: {overlap}/{mp3Words.Count} = {ratio:P0}");
            Assert.True(ratio >= 0.6, $"MP3 transcription diverges from WAV (overlap {ratio:P0})");
        }

        [LongFact]
        public void Transcribe_RepeatedCall_AllocStable()
        {
            var w = WhisperTranscriber.Load(TestModelPaths.Whisper.RequireTinyGgmlPath());
            var samples = DevOnBike.Overfit.Audio.WavReader.ReadMono(TestModelPaths.Whisper.RequirePolishWavPath(), out _);

            w.Transcribe(samples, "pl"); // warm up (JIT + first allocation of reusable mel/encoder buffers)

            var before = GC.GetAllocatedBytesForCurrentThread();
            w.Transcribe(samples, "pl");
            var after = GC.GetAllocatedBytesForCurrentThread();

            var alloc = after - before;
            _out.WriteLine($"repeated Transcribe allocates {alloc:N0} B (mel/encoder/KV-cache buffers reused; was ~21 MB before)");
            // The big per-call transients (encoder buffers, the ~18 MB cross-attention KV cache) are reused;
            // only the result itself (token list + decoded string) is left — a few KB.
            Assert.True(alloc < 64_000, $"repeated transcription allocates more than expected: {alloc:N0} B");
        }

        private static List<string> Normalize(string text)
        {
            var words = new List<string>();
            foreach (var raw in text.ToLowerInvariant().Split(' ', '\t', '\n', '\r', '.', ',', '!', '?'))
            {
                if (raw.Length > 0) { words.Add(raw); }
            }
            return words;
        }
    }
}
