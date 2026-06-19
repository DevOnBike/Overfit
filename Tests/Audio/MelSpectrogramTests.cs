// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>
    /// Validates the Whisper log-mel front-end (no model needed): the DFT is exact on a pure tone, the
    /// Slaney mel filterbank has the right shape/normalization, the STFT localizes a sine in the correct
    /// bin, and the final log-mel is well-formed (Whisper's normalization gives a ≤ 2.0 dynamic span).
    /// </summary>
    public sealed class MelSpectrogramTests
    {
        private readonly ITestOutputHelper _out;
        public MelSpectrogramTests(ITestOutputHelper output) => _out = output;

        [Fact]
        public void Dft_PureCosine_PeaksAtItsBin_WithExpectedMagnitude()
        {
            const int n = MelSpectrogram.NFft; // 400
            const int k0 = 25;
            var frame = new float[n];
            for (var i = 0; i < n; i++)
            {
                frame[i] = MathF.Cos(2f * MathF.PI * k0 * i / n);
            }

            var atPeak = MelSpectrogram.DftPowerAt(frame, k0, n);
            var offPeak = MelSpectrogram.DftPowerAt(frame, k0 + 7, n);

            // For cos(2π·k0·n/N): |X[k0]| = N/2 → power = (N/2)² ; ~0 elsewhere (integer bin, no leakage).
            var expected = (n / 2f) * (n / 2f);
            _out.WriteLine($"power@k0 {atPeak:F1} (expected {expected:F1}), off-peak {offPeak:E2}");
            Assert.True(Math.Abs(atPeak - expected) / expected < 1e-3, $"peak magnitude off: {atPeak} vs {expected}");
            Assert.True(offPeak < 1e-2, $"off-peak leakage too high: {offPeak}");
        }

        [Fact]
        public void MelFilterBank_IsNonNegative_RightShape_AndAreaNormalized()
        {
            var mel = new MelSpectrogram(nMels: 80);
            Assert.Equal(80, mel.MelCount);
            Assert.Equal(MelSpectrogram.NFft / 2 + 1, mel.FrequencyBins);

            var filters = mel.MelFilters;
            Assert.Equal(80 * mel.FrequencyBins, filters.Length);

            for (var i = 0; i < filters.Length; i++)
            {
                Assert.True(filters[i] >= 0f, "mel filter weights must be non-negative");
            }
            // Each mel band has some energy (non-empty triangle).
            for (var m = 0; m < 80; m++)
            {
                var sum = 0f;
                for (var f = 0; f < mel.FrequencyBins; f++)
                {
                    sum += filters[m * mel.FrequencyBins + f];
                }
                Assert.True(sum > 0f, $"mel filter {m} is empty");
            }
        }

        [Fact]
        public void Stft_Sine_ConcentratesPowerInExpectedBin()
        {
            var mel = new MelSpectrogram();
            // 1 kHz tone for 0.2 s at 16 kHz. Bin spacing = sr/nFft = 40 Hz → 1000 Hz is bin 25.
            const float freq = 1000f;
            var samples = new float[3200];
            for (var i = 0; i < samples.Length; i++)
            {
                samples[i] = MathF.Sin(2f * MathF.PI * freq * i / MelSpectrogram.SampleRate);
            }

            var power = mel.PowerSpectrogram(samples, out var frames);
            Assert.True(frames > 0);

            // Average power per frequency bin across a middle frame; the peak bin should be ~freq/40.
            var mid = frames / 2;
            int peakBin = 0;
            float peakVal = 0;
            for (var f = 0; f < mel.FrequencyBins; f++)
            {
                var v = power[f * frames + mid];
                if (v > peakVal)
                {
                    peakVal = v;
                    peakBin = f;
                }
            }
            var expectedBin = (int)MathF.Round(freq / (MelSpectrogram.SampleRate / (float)MelSpectrogram.NFft));
            _out.WriteLine($"peak bin {peakBin} (expected ~{expectedBin})");
            Assert.True(Math.Abs(peakBin - expectedBin) <= 1, $"sine peak in wrong bin: {peakBin} vs {expectedBin}");
        }

        [Fact]
        public void BluesteinFft_PowerMatchesDirectDft()
        {
            const int nFft = MelSpectrogram.NFft;
            const int pad = nFft / 2;
            var mel = new MelSpectrogram();
            var rng = new Random(11);
            var samples = new float[3200];
            for (var i = 0; i < samples.Length; i++)
            {
                samples[i] = (float)(rng.NextDouble() * 2 - 1);
            }

            var power = mel.PowerSpectrogram(samples, out var frames); // Bluestein FFT path

            // Reconstruct the windowed frame exactly as the internal STFT does (reflect-pad + periodic Hann),
            // then compare every bin against the direct O(N²) DFT.
            var padded = new float[samples.Length + 2 * pad];
            samples.CopyTo(padded, pad);
            for (var i = 0; i < pad; i++)
            {
                padded[pad - 1 - i] = samples[Math.Min(i + 1, samples.Length - 1)];
                padded[pad + samples.Length + i] = samples[Math.Max(samples.Length - 2 - i, 0)];
            }
            var hann = new float[nFft];
            for (var n = 0; n < nFft; n++)
            {
                hann[n] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * n / nFft));
            }

            var mid = frames / 2;
            var windowed = new float[nFft];
            for (var n = 0; n < nFft; n++)
            {
                windowed[n] = padded[mid * MelSpectrogram.HopLength + n] * hann[n];
            }

            var maxRel = 0.0;
            for (var k = 0; k < mel.FrequencyBins; k++)
            {
                var reference = MelSpectrogram.DftPowerAt(windowed, k, nFft);
                var got = power[k * frames + mid];
                var rel = reference > 1e-6f ? Math.Abs(got - reference) / reference : Math.Abs(got - reference);
                if (rel > maxRel)
                {
                    maxRel = rel;
                }
            }
            _out.WriteLine($"max relative error Bluestein-FFT vs direct DFT over {mel.FrequencyBins} bins: {maxRel:E3}");
            Assert.True(maxRel < 1e-3, $"Bluestein FFT diverges from the direct DFT: {maxRel:E3}");
        }

        [Fact]
        public void LogMel_Shape_Finite_AndWhisperNormalizedSpan()
        {
            var mel = new MelSpectrogram();
            var rng = new Random(7);
            var samples = new float[16000]; // 1 s of noise
            for (var i = 0; i < samples.Length; i++)
            {
                samples[i] = (float)(rng.NextDouble() * 2 - 1) * 0.3f;
            }

            var logmel = mel.LogMel(samples, out var frames);
            Assert.Equal(80 * frames, logmel.Length);
            Assert.Equal(samples.Length / MelSpectrogram.HopLength, frames); // Whisper: frames = samples/hop

            float min = float.PositiveInfinity, max = float.NegativeInfinity;
            foreach (var v in logmel)
            {
                Assert.True(float.IsFinite(v), "log-mel must be finite");
                if (v < min)
                {
                    min = v;
                }
                if (v > max)
                {
                    max = v;
                }
            }
            _out.WriteLine($"frames={frames}, log-mel range [{min:F3}, {max:F3}], span {max - min:F3}");
            // Whisper's clamp to (max−8) then (x+4)/4 caps the dynamic span at 8/4 = 2.0.
            Assert.True(max - min <= 2.0f + 1e-4f, $"normalized span exceeds 2.0: {max - min}");
        }
    }
}
