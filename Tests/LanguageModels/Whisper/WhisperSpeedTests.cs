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

        [LongFact]
        public void Profile_Pipeline_Breakdown()
        {
            var model = WhisperGgmlLoader.Load(TestModelPaths.Whisper.RequireTinyGgmlPath());
            var mel = new MelSpectrogram(model.MelFilterRows, model.MelFilters);
            var encoder = new WhisperEncoder(model);
            var decoder = new WhisperDecoder(model);
            var tok = new WhisperTokenizer(model);

            var raw = WavReader.ReadMono(TestModelPaths.Whisper.RequirePolishWavPath(), out _);
            var window = new float[MelSpectrogram.SampleRate * 30];
            raw.AsSpan(0, Math.Min(raw.Length, window.Length)).CopyTo(window);

            // warm
            var lm = mel.LogMel(window, out var frames);
            var enc = encoder.Encode(lm, frames, out var nCtx);
            var prompt = new[] { tok.StartOfTranscript, tok.LanguageToken("pl"), tok.Transcribe, tok.NoTimestamps };
            decoder.DecodeCached(enc, nCtx, prompt, tok.EndOfTranscript, 224);

            var melMs = Best(() => mel.LogMel(window, out _));
            var encMs = Best(() => { var m = mel.LogMel(window, out var f); encoder.Encode(m, f, out _); });
            var decMs = Best(() =>
            {
                var m = mel.LogMel(window, out var f);
                var e = encoder.Encode(m, f, out var c);
                decoder.DecodeCached(e, c, prompt, tok.EndOfTranscript, 224);
            });
            var encOnly = encMs - melMs;
            var decOnly = decMs - encMs;
            _out.WriteLine($"mel (log-mel STFT) : {melMs,7:F0} ms");
            _out.WriteLine($"encoder            : {encOnly,7:F0} ms");
            _out.WriteLine($"decoder (greedy)   : {decOnly,7:F0} ms");
            _out.WriteLine($"total              : {decMs,7:F0} ms");
        }

        private static double Best(Action run)
        {
            var best = double.MaxValue;
            for (var r = 0; r < 3; r++)
            {
                var sw = Stopwatch.StartNew();
                run();
                sw.Stop();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
            }
            return best;
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
