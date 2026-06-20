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
    /// Performance + zero-allocation characterisation of the Whisper KV-cache decode on the REAL tiny model
    /// (<c>ggml-tiny.bin</c> + <c>jfk.wav</c>). Two claims, both measured here:
    /// (1) <see cref="WhisperDecoder.DecodeCached"/> is faster than the recompute-everything
    ///     <see cref="WhisperDecoder.Decode"/> AND produces the identical greedy token stream; and
    /// (2) the per-token <c>Step</c> is allocation-free (all scratch lives in the pre-allocated state — no
    ///     per-step string lookups, no jagged caches), proven by <c>GC.GetAllocatedBytesForCurrentThread</c>.
    /// <see cref="LongFactAttribute"/> — needs the real model; flip to <c>[Fact]</c> to run locally.
    /// </summary>
    public sealed class WhisperPerfTests
    {
        private readonly ITestOutputHelper _out;
        public WhisperPerfTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void DecodeCached_FasterThanRecompute_AndStepIsZeroAlloc()
        {
            var model = WhisperGgmlLoader.Load(TestModelPaths.Whisper.RequireTinyGgmlPath());
            var mel = new MelSpectrogram(model.MelFilterRows, model.MelFilters);
            var encoder = new WhisperEncoder(model);
            var decoder = new WhisperDecoder(model);
            var tok = new WhisperTokenizer(model);

            // Audio → 30 s window → log-mel → encoder (done once; we benchmark the decode).
            var samples = WavReader.ReadMono(TestModelPaths.Whisper.RequireSampleWavPath(), out var sr);
            Assert.Equal(MelSpectrogram.SampleRate, sr);
            var window = new float[MelSpectrogram.SampleRate * 30];
            samples.AsSpan(0, Math.Min(samples.Length, window.Length)).CopyTo(window);
            var logMel = mel.LogMel(window, out var frames);
            var encoderOut = encoder.Encode(logMel, frames, out var nCtx);

            var prompt = new[]
            {
                tok.StartOfTranscript, tok.LanguageToken("en"), tok.Transcribe, tok.NoTimestamps,
            };
            var eot = tok.EndOfTranscript;
            const int maxNew = 224;

            // ── Correctness: cached == recompute (same greedy stream) ──
            var slow = decoder.Decode(encoderOut, nCtx, prompt, eot, maxNew);
            var fast = decoder.DecodeCached(encoderOut, nCtx, prompt, eot, maxNew);
            Assert.Equal(slow, fast);
            _out.WriteLine($"tokens generated: {fast.Length}  → \"{tok.Decode(fast).Trim()}\"");

            // ── Perf: best-of-3 wall time, recompute vs KV-cache ──
            var slowMs = BestOfThree(() => decoder.Decode(encoderOut, nCtx, prompt, eot, maxNew));
            var fastMs = BestOfThree(() => decoder.DecodeCached(encoderOut, nCtx, prompt, eot, maxNew));
            _out.WriteLine($"recompute Decode   : {slowMs,8:F1} ms  ({slowMs / fast.Length:F2} ms/token)");
            _out.WriteLine($"KV-cache DecodeCached: {fastMs,6:F1} ms  ({fastMs / fast.Length:F2} ms/token)");
            _out.WriteLine($"speedup: {slowMs / fastMs:F2}×");
            Assert.True(fastMs < slowMs, $"KV-cache ({fastMs:F1} ms) should beat recompute ({slowMs:F1} ms)");

            // ── Zero-alloc: drive the per-token Step directly and measure thread-local allocations ──
            var state = decoder.CreateState(encoderOut, nCtx, prompt.Length + maxNew);
            for (var i = 0; i < prompt.Length; i++)
            {
                decoder.Step(state, prompt[i]);
            }
            // Warm up a handful of generated steps (JIT, first-touch) before measuring.
            var warm = Math.Min(5, fast.Length);
            for (var i = 0; i < warm; i++)
            {
                decoder.Step(state, fast[i]);
            }

            var measured = fast.Length - warm;
            Assert.True(measured >= 10, "need enough generated tokens to measure");
            var before = GC.GetAllocatedBytesForCurrentThread();
            for (var i = warm; i < fast.Length; i++)
            {
                decoder.Step(state, fast[i]);
            }
            var after = GC.GetAllocatedBytesForCurrentThread();

            var perStep = (double)(after - before) / measured;
            _out.WriteLine($"per-token decode Step: {perStep:F1} bytes/token over {measured} steps (total {after - before} B)");
            // Allocation-free: allow a tiny slack for measurement noise, but it must be essentially zero
            // (a single naive float[dModel] scratch would be ~1.5 KB/token at d=384 — orders of magnitude above this).
            Assert.True(perStep < 64, $"per-token Step should be ~zero-alloc, was {perStep:F1} bytes/token");
        }

        private static double BestOfThree(Action run)
        {
            var best = double.MaxValue;
            for (var r = 0; r < 3; r++)
            {
                var sw = Stopwatch.StartNew();
                run();
                sw.Stop();
                if (sw.Elapsed.TotalMilliseconds < best)
                {
                    best = sw.Elapsed.TotalMilliseconds;
                }
            }
            return best;
        }
    }
}
