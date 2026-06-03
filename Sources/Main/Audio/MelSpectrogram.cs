// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio
{
    /// <summary>
    /// Whisper-compatible log-mel spectrogram front-end: 16 kHz mono audio → <c>[nMels, frames]</c>.
    ///
    /// Matches OpenAI Whisper's pipeline (so a real Whisper GGUF sees the features it was trained on):
    /// reflect-pad by <c>nFft/2</c> → framed STFT (Hann window, hop = 160, nFft = 400) → power spectrum
    /// → mel filterbank (Slaney scale, the librosa default Whisper ships) → <c>log10</c> → dynamic-range
    /// clamp to (max − 8) → <c>(x + 4) / 4</c>. Correctness-first: a direct real DFT per frame (no FFT
    /// yet — a 30 s clip is ~3000 frames, fine; FFT is a later optimization).
    /// </summary>
    public sealed class MelSpectrogram
    {
        public const int SampleRate = 16_000;
        public const int NFft = 400;
        public const int HopLength = 160;
        public const int DefaultMelCount = 80;

        private readonly int _nMels;
        private readonly int _nFreqs;        // nFft/2 + 1
        private readonly float[] _hann;      // [nFft]
        private readonly float[] _melFilters; // [nMels * nFreqs], mel-major

        /// <summary>
        /// Builds the extractor. By default the Slaney mel filterbank is computed (librosa default). Pass
        /// <paramref name="melFilters"/> (<c>[nMels × (NFft/2+1)]</c>, e.g. <c>WhisperModel.MelFilters</c>) to
        /// use the model's own filterbank for bit-parity with whisper.cpp.
        /// </summary>
        public MelSpectrogram(int nMels = DefaultMelCount, ReadOnlySpan<float> melFilters = default)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(nMels);
            _nMels = nMels;
            _nFreqs = NFft / 2 + 1;

            // Periodic Hann window (torch.hann_window default: periodic = true → divide by N, not N-1).
            _hann = new float[NFft];
            for (var n = 0; n < NFft; n++)
            {
                _hann[n] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * n / NFft));
            }

            if (melFilters.IsEmpty)
            {
                _melFilters = BuildSlaneyMelFilters(nMels, _nFreqs, SampleRate, NFft);
            }
            else
            {
                if (melFilters.Length != nMels * _nFreqs)
                {
                    throw new ArgumentException($"melFilters must be [{nMels} × {_nFreqs}] = {nMels * _nFreqs}, got {melFilters.Length}.", nameof(melFilters));
                }
                _melFilters = melFilters.ToArray();
            }
        }

        public int MelCount => _nMels;
        public int FrequencyBins => _nFreqs;

        /// <summary>The mel filterbank, <c>[nMels, nFreqs]</c> mel-major (exposed for parity tests / overrides).</summary>
        public ReadOnlySpan<float> MelFilters => _melFilters;

        /// <summary>
        /// Computes the log-mel spectrogram of mono 16 kHz <paramref name="samples"/>. Returns a
        /// <c>[nMels × frames]</c> mel-major array (row = one mel bin across time); <paramref name="frames"/>
        /// receives the time-frame count. Whisper drops the final STFT frame, so
        /// <c>frames = samples.Length / HopLength</c> after center padding.
        /// </summary>
        public float[] LogMel(ReadOnlySpan<float> samples, out int frames)
        {
            // Center padding: reflect by nFft/2 on each side (torch.stft center=true).
            var pad = NFft / 2;
            var padded = ReflectPad(samples, pad);

            // Whisper keeps stft[..., :-1] → frames = (paddedLen - nFft)/hop, which equals samples/hop.
            frames = 1 + (padded.Length - NFft) / HopLength;
            if (frames > 0) { frames -= 1; } // drop the last frame (Whisper's stft[..., :-1])
            if (frames < 0) { frames = 0; }

            var power = new float[_nFreqs * frames]; // [freq, frame] freq-major
            ComputePowerSpectrogram(padded, frames, power);

            // mel = melFilters @ power  → [nMels, frames]; then log10 + Whisper normalize.
            var mel = new float[_nMels * frames];
            ApplyMelFilters(power, frames, mel);
            NormalizeLogMel(mel);
            return mel;
        }

        // ── STFT power spectrum (direct DFT per frame) ──

        private void ComputePowerSpectrogram(ReadOnlySpan<float> padded, int frames, Span<float> power)
        {
            Span<float> windowed = NFft <= 1024 ? stackalloc float[NFft] : new float[NFft];
            for (var t = 0; t < frames; t++)
            {
                var start = t * HopLength;
                for (var n = 0; n < NFft; n++)
                {
                    windowed[n] = padded[start + n] * _hann[n];
                }

                // Real DFT power per bin: |Σ x[n]·e^(−i2πkn/N)|².
                for (var k = 0; k < _nFreqs; k++)
                {
                    power[k * frames + t] = DftPowerAt(windowed, k, NFft);
                }
            }
        }

        /// <summary>Power <c>|X[k]|²</c> of frequency bin <paramref name="k"/> via a direct real DFT
        /// of <paramref name="frame"/> (length <paramref name="nFft"/>). Internal — DFT-correctness hook.</summary>
        internal static float DftPowerAt(ReadOnlySpan<float> frame, int k, int nFft)
        {
            float re = 0f, im = 0f;
            var w = -2f * MathF.PI * k / nFft;
            for (var n = 0; n < nFft; n++)
            {
                var ang = w * n;
                re += frame[n] * MathF.Cos(ang);
                im += frame[n] * MathF.Sin(ang);
            }
            return re * re + im * im;
        }

        /// <summary>The STFT power spectrogram <c>[nFreqs, frames]</c> (freq-major) before mel/log —
        /// internal, for parity tests.</summary>
        internal float[] PowerSpectrogram(ReadOnlySpan<float> samples, out int frames)
        {
            var pad = NFft / 2;
            var padded = ReflectPad(samples, pad);
            frames = 1 + (padded.Length - NFft) / HopLength;
            if (frames > 0) { frames -= 1; }
            if (frames < 0) { frames = 0; }
            var power = new float[_nFreqs * frames];
            ComputePowerSpectrogram(padded, frames, power);
            return power;
        }

        private void ApplyMelFilters(ReadOnlySpan<float> power, int frames, Span<float> mel)
        {
            // mel[m, t] = Σ_f melFilters[m, f] · power[f, t].
            for (var m = 0; m < _nMels; m++)
            {
                var filt = _melFilters.AsSpan(m * _nFreqs, _nFreqs);
                for (var t = 0; t < frames; t++)
                {
                    var acc = 0f;
                    for (var f = 0; f < _nFreqs; f++)
                    {
                        acc += filt[f] * power[f * frames + t];
                    }
                    mel[m * frames + t] = acc;
                }
            }
        }

        private static void NormalizeLogMel(Span<float> mel)
        {
            // log10(max(x, 1e-10)); clamp to (max − 8); (x + 4) / 4.   (Whisper's exact normalization.)
            var maxLog = float.NegativeInfinity;
            for (var i = 0; i < mel.Length; i++)
            {
                var v = MathF.Log10(MathF.Max(mel[i], 1e-10f));
                mel[i] = v;
                if (v > maxLog) { maxLog = v; }
            }
            var floor = maxLog - 8f;
            for (var i = 0; i < mel.Length; i++)
            {
                var v = MathF.Max(mel[i], floor);
                mel[i] = (v + 4f) / 4f;
            }
        }

        private static float[] ReflectPad(ReadOnlySpan<float> x, int pad)
        {
            var n = x.Length;
            var outp = new float[n + 2 * pad];
            x.CopyTo(outp.AsSpan(pad, n));
            // Reflect (without repeating the edge sample), matching np.pad(mode='reflect').
            for (var i = 0; i < pad; i++)
            {
                outp[pad - 1 - i] = x[Math.Min(i + 1, n - 1)];
                outp[pad + n + i] = x[Math.Max(n - 2 - i, 0)];
            }
            return outp;
        }

        // ── Slaney mel filterbank (librosa default; what Whisper's mel_filters were built with) ──

        private static float[] BuildSlaneyMelFilters(int nMels, int nFreqs, int sampleRate, int nFft)
        {
            var fMin = 0.0;
            var fMax = sampleRate / 2.0;

            // Mel band edges (nMels + 2 points), converted back to Hz.
            var melMin = HzToMel(fMin);
            var melMax = HzToMel(fMax);
            var hzPoints = new double[nMels + 2];
            for (var i = 0; i < nMels + 2; i++)
            {
                hzPoints[i] = MelToHz(melMin + (melMax - melMin) * i / (nMels + 1));
            }

            // FFT bin center frequencies.
            var fftFreqs = new double[nFreqs];
            for (var f = 0; f < nFreqs; f++)
            {
                fftFreqs[f] = (double)f * sampleRate / nFft;
            }

            var filters = new float[nMels * nFreqs];
            for (var m = 0; m < nMels; m++)
            {
                var lower = hzPoints[m];
                var center = hzPoints[m + 1];
                var upper = hzPoints[m + 2];
                var enorm = 2.0 / (upper - lower); // Slaney area normalization

                for (var f = 0; f < nFreqs; f++)
                {
                    var freq = fftFreqs[f];
                    var down = (freq - lower) / (center - lower);
                    var up = (upper - freq) / (upper - center);
                    var weight = Math.Max(0.0, Math.Min(down, up));
                    filters[m * nFreqs + f] = (float)(weight * enorm);
                }
            }
            return filters;
        }

        // Slaney mel scale (linear < 1000 Hz, log above) — librosa htk=false.
        private const double MelFSp = 200.0 / 3.0;        // Hz per mel below the break
        private const double MelBreakHz = 1000.0;
        private static readonly double MelBreakMel = MelBreakHz / MelFSp;     // 15.0
        private static readonly double MelLogStep = Math.Log(6.4) / 27.0;

        private static double HzToMel(double hz)
            => hz < MelBreakHz ? hz / MelFSp : MelBreakMel + Math.Log(hz / MelBreakHz) / MelLogStep;

        private static double MelToHz(double mel)
            => mel < MelBreakMel ? MelFSp * mel : MelBreakHz * Math.Exp(MelLogStep * (mel - MelBreakMel));
    }
}
