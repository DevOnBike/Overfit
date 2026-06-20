// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio;
using DevOnBike.Overfit.Audio.Tts;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Diagnostic over the user's recorded voice clips (C:\myvoice\*.wav): measures the leading/trailing silence
    /// each clip carries before/after speech and how much <see cref="AudioPostProcessing.TrimSilence"/> removes.
    /// Quantifies whether the per-clip trim added to the dataset builder actually matters for these recordings.
    /// [LongFact] — needs C:\myvoice. Flip to [Fact] to run.
    /// </summary>
    public sealed class MyVoiceSilenceDiagnosticTests
    {
        private const string Dir = @"C:\myvoice";
        private const int Rate24 = 24_000;
        private const float Threshold = 0.02f;
        private const int KeepPadding = 1_200; // 0.05 s @ 24 kHz
        private readonly ITestOutputHelper _out;

        public MyVoiceSilenceDiagnosticTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Report_LeadingSilence_PerClip()
        {
            if (!Directory.Exists(Dir))
            {
                _out.WriteLine("missing C:\\myvoice");
                return;
            }

            var wavs = Directory.GetFiles(Dir, "*.wav");
            Array.Sort(wavs, StringComparer.Ordinal);

            double sumLeadMs = 0, maxLeadMs = 0, sumTrimPct = 0;
            var count = 0;
            var overHalfSecond = 0;

            foreach (var path in wavs)
            {
                var raw = AudioFile.ReadMono(path, out var rate);
                var audio = rate == Rate24 ? raw : AudioResampler.Resample(raw, rate, Rate24);
                var norm = AudioPostProcessing.PeakNormalize(audio);

                // Leading silence = first sample above threshold.
                var first = 0;
                while (first < norm.Length && MathF.Abs(norm[first]) < Threshold)
                {
                    first++;
                }
                var leadMs = first / (double)Rate24 * 1000.0;

                var trimmed = AudioPostProcessing.TrimSilence(norm, Threshold, KeepPadding);
                var trimPct = 100.0 * (norm.Length - trimmed.Length) / norm.Length;

                sumLeadMs += leadMs;
                maxLeadMs = Math.Max(maxLeadMs, leadMs);
                sumTrimPct += trimPct;
                if (leadMs > 500)
                {
                    overHalfSecond++;
                }
                count++;

                if (leadMs > 300 || trimPct > 25)
                {
                    _out.WriteLine($"{Path.GetFileName(path),-10} lead {leadMs,6:F0} ms  trimmed {trimPct,5:F1}%  " +
                        $"({norm.Length} → {trimmed.Length} samp)");
                }
            }

            _out.WriteLine($"=== {count} clips: avg lead {sumLeadMs / count:F0} ms | max {maxLeadMs:F0} ms | " +
                $"avg trimmed {sumTrimPct / count:F1}% | clips with >500 ms lead: {overHalfSecond} ===");
            Assert.True(count > 0);
        }
    }
}
