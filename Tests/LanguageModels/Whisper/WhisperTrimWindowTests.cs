// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Audio;
using DevOnBike.Overfit.LanguageModels.Whisper;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Whisper
{
    /// <summary>
    /// Lever #1 (trim-to-actual-length): the transcriber no longer pads every clip to a full 30 s window before
    /// the encoder. Encoder self-attention is O(nCtx²) in the frame count, so a short clip should transcribe
    /// much faster than the legacy full-window path while producing the same text. This A/B pins both:
    /// <list type="bullet">
    /// <item>correctness — the trimmed transcript still contains the expected words, and matches the
    /// full-window transcript for the whole jfk clip;</item>
    /// <item>perf — on a short (3 s) slice the trimmed path is dramatically faster than the full-window one.</item>
    /// </list>
    /// <see cref="LongFact"/> — needs <c>OVERFIT_WHISPER_DIR</c> with <c>ggml-tiny.bin</c> and <c>jfk.wav</c>.
    /// </summary>
    public sealed class WhisperTrimWindowTests
    {
        private readonly ITestOutputHelper _out;
        public WhisperTrimWindowTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Trim_MatchesFullWindowText_AndIsFasterOnShortClips()
        {
            var ggml = TestModelPaths.Whisper.RequireTinyGgmlPath();
            var wav = TestModelPaths.Whisper.RequireSampleWavPath();

            var whisper = WhisperTranscriber.Load(ggml);

            var samples = AudioFile.ReadMono(wav, out var sr);
            if (sr != MelSpectrogram.SampleRate)
            {
                samples = AudioResampler.Resample(samples, sr, MelSpectrogram.SampleRate);
            }
            _out.WriteLine($"clip: {samples.Length / (double)MelSpectrogram.SampleRate:F1} s ({samples.Length} samples)");

            // Warm up the JIT / buffers so the timings reflect steady state, not first-call compilation.
            _ = whisper.Transcribe(samples, "en", 224, padToFullWindow: true);
            _ = whisper.Transcribe(samples, "en", 224, padToFullWindow: false);

            // ── whole clip: full-window vs trimmed (correctness parity) ──
            var (full, tFull) = Timed(() => whisper.Transcribe(samples, "en", 224, padToFullWindow: true));
            var (trim, tTrim) = Timed(() => whisper.Transcribe(samples, "en", 224, padToFullWindow: false));
            _out.WriteLine($"FULL  ({tFull,6:F0} ms): \"{full}\"");
            _out.WriteLine($"TRIM  ({tTrim,6:F0} ms): \"{trim}\"");

            Assert.Contains("country", trim, StringComparison.OrdinalIgnoreCase);
            Assert.Equal(full, trim); // a clean clip transcribes identically with or without 30 s padding

            // ── short 3 s slice: the voice-input case where the win is largest ──
            var threeSec = samples.AsSpan(0, Math.Min(samples.Length, MelSpectrogram.SampleRate * 3)).ToArray();
            var (_, tFull3) = Timed(() => whisper.Transcribe(threeSec, "en", 64, padToFullWindow: true));
            var (_, tTrim3) = Timed(() => whisper.Transcribe(threeSec, "en", 64, padToFullWindow: false));
            _out.WriteLine($"3 s slice — full {tFull3:F0} ms, trim {tTrim3:F0} ms → {tFull3 / Math.Max(tTrim3, 1):F1}× faster");

            Assert.True(tTrim3 < tFull3, $"trimmed ({tTrim3:F0} ms) should beat full-window ({tFull3:F0} ms) on a 3 s clip");
        }

        private static (string text, double ms) Timed(Func<string> f)
        {
            var sw = Stopwatch.StartNew();
            var text = f();
            sw.Stop();
            return (text, sw.Elapsed.TotalMilliseconds);
        }
    }
}
