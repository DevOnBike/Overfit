// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Audio;
using DevOnBike.Overfit.Audio.Mp3;
using DevOnBike.Overfit.LanguageModels.Whisper;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Whisper
{
    /// <summary>Real-time-factor characterisation: how fast the MP3 decoder and the full speech-to-text
    /// pipeline run on the dev box. The transcriber processes a fixed 30 s window, so the per-call time is the
    /// latency unit for live use.</summary>
    public sealed class WhisperSpeedTests
    {
        private readonly ITestOutputHelper _out;
        public WhisperSpeedTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Measure_Decode_And_Transcription_Speed()
        {
            // ── MP3 decode speed ──
            var mp3Bytes = File.ReadAllBytes(TestModelPaths.Whisper.RequireSampleMp3Path());
            var dec = new Mp3Decoder();
            dec.DecodeMono(mp3Bytes, out _); // warm
            var swd = Stopwatch.StartNew();
            var pcm = dec.DecodeMono(mp3Bytes, out var sr);
            swd.Stop();
            var audioSec = (double)pcm.Length / sr;
            _out.WriteLine($"MP3 decode: {audioSec:F2}s audio in {swd.Elapsed.TotalMilliseconds:F1} ms = {audioSec / swd.Elapsed.TotalSeconds:F0}x real-time");

            // ── full transcription (tiny) on a 30 s window ──
            var samples = WavReader.ReadMono(TestModelPaths.Whisper.RequirePolishWavPath(), out _);
            TimeTranscribe("tiny", TestModelPaths.Whisper.RequireTinyGgmlPath(), samples);
            if (File.Exists(TestModelPaths.Whisper.BaseGgmlPath))
            {
                TimeTranscribe("base", TestModelPaths.Whisper.BaseGgmlPath, samples);
            }
        }

        private void TimeTranscribe(string label, string modelPath, float[] samples)
        {
            var w = WhisperTranscriber.Load(modelPath);
            w.Transcribe(samples, "pl"); // warm
            var best = double.MaxValue;
            for (var r = 0; r < 3; r++)
            {
                var sw = Stopwatch.StartNew();
                w.Transcribe(samples, "pl");
                sw.Stop();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
            }
            // The pipeline pads to a fixed 30 s window, so 'best' is the cost of transcribing up to 30 s of audio.
            _out.WriteLine($"Whisper {label}: 30 s window in {best:F0} ms = {30000.0 / best:F1}x real-time (latency unit {best / 1000.0:F2} s)");
        }
    }
}
