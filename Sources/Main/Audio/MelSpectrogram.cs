// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Audio
{
    /// <summary>
    /// Whisper-compatible log-mel spectrogram front-end: 16 kHz mono audio → <c>[nMels, frames]</c>.
    ///
    /// Matches OpenAI Whisper's pipeline (so a real Whisper GGUF sees the features it was trained on):
    /// reflect-pad by <c>nFft/2</c> → framed STFT (Hann window, hop = 160, nFft = 400) → power spectrum
    /// → mel filterbank (Slaney scale, the librosa default Whisper ships) → <c>log10</c> → dynamic-range
    /// clamp to (max − 8) → <c>(x + 4) / 4</c>. The per-frame 400-point DFT uses the <b>Bluestein</b>
    /// algorithm (an exact non-power-of-2 DFT via a length-1024 radix-2 FFT convolution); buffers are
    /// reused across calls, so repeated transcriptions are allocation-stable.
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

        // Reusable buffers — grown on demand, kept across LogMel calls so repeated transcriptions
        // (streaming / microphone) are allocation-stable after warm-up.
        private float[] _padded = Array.Empty<float>();
        private float[] _power = Array.Empty<float>();
        private float[] _mel = Array.Empty<float>();

        // Bluestein FFT for the (non-power-of-2) 400-point STFT: a length-N DFT via a length-M (power-of-2)
        // FFT convolution. Tables are precomputed once; the per-frame transform stackallocs its own scratch
        // (so the frame loop parallelizes without sharing mutable state).
        private readonly int _fftM;          // smallest power of 2 ≥ 2·NFft − 1 (= 1024)
        private readonly float[] _twRe, _twIm;   // forward FFT twiddles, [M/2]
        private readonly float[] _chirpRe, _chirpIm; // w[n] = exp(−iπn²/N), [NFft]
        private readonly float[] _bwRe, _bwIm;   // FFT of the symmetric chirp filter B, [M]

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

            // ── Bluestein tables ──
            var m = 1;
            while (m < 2 * NFft - 1) { m <<= 1; }
            _fftM = m;
            _twRe = new float[m / 2];
            _twIm = new float[m / 2];
            for (var k = 0; k < m / 2; k++)
            {
                var ang = -2.0 * Math.PI * k / m;
                _twRe[k] = (float)Math.Cos(ang);
                _twIm[k] = (float)Math.Sin(ang);
            }
            _chirpRe = new float[NFft];
            _chirpIm = new float[NFft];
            for (var n = 0; n < NFft; n++)
            {
                var n2 = (long)n * n % (2L * NFft);     // reduce before scaling to keep the angle small/accurate
                var ang = Math.PI * n2 / NFft;
                _chirpRe[n] = (float)Math.Cos(ang);     // exp(−iπn²/N)
                _chirpIm[n] = -(float)Math.Sin(ang);
            }
            _bwRe = new float[m];
            _bwIm = new float[m];
            for (var n = 0; n < NFft; n++)
            {
                var n2 = (long)n * n % (2L * NFft);
                var ang = Math.PI * n2 / NFft;          // B[n] = exp(+iπn²/N), symmetric: B[M−n] = B[n]
                var br = (float)Math.Cos(ang);
                var bi = (float)Math.Sin(ang);
                _bwRe[n] = br; _bwIm[n] = bi;
                if (n > 0) { _bwRe[m - n] = br; _bwIm[m - n] = bi; }
            }

            Fft(_bwRe, _bwIm, _twRe, _twIm, inverse: false);
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
            var paddedLen = samples.Length + 2 * pad;
            _padded = EnsureCapacity(_padded, paddedLen);
            ReflectPadInto(samples, pad, _padded);

            // Whisper keeps stft[..., :-1] → frames = (paddedLen - nFft)/hop, which equals samples/hop.
            frames = 1 + (paddedLen - NFft) / HopLength;
            if (frames > 0) { frames -= 1; } // drop the last frame (Whisper's stft[..., :-1])
            if (frames < 0) { frames = 0; }

            _power = EnsureCapacity(_power, _nFreqs * frames);
            _mel = EnsureCapacity(_mel, _nMels * frames);
            var power = _power.AsSpan(0, _nFreqs * frames); // [freq, frame] freq-major
            ComputePowerSpectrogram(_padded.AsSpan(0, paddedLen), frames, power);

            // mel = melFilters @ power  → [nMels, frames]; then log10 + Whisper normalize.
            var mel = _mel.AsSpan(0, _nMels * frames);
            ApplyMelFilters(power, frames, mel);
            NormalizeLogMel(mel);
            return _mel;
        }

        private static float[] EnsureCapacity(float[] buffer, int needed)
            => buffer.Length >= needed ? buffer : new float[needed];

        // ── STFT power spectrum (per-frame 400-point DFT via Bluestein FFT, parallelized over frames) ──

        private unsafe void ComputePowerSpectrogram(ReadOnlySpan<float> padded, int frames, Span<float> power)
        {
            if (frames <= 0)
            {
                return;
            }
            fixed (float* pad = padded, hann = _hann, cRe = _chirpRe, cIm = _chirpIm,
                          tRe = _twRe, tIm = _twIm, bRe = _bwRe, bIm = _bwIm, pw = power)
            {
                var ctx = new MelFrameCtx(pad, hann, cRe, cIm, tRe, tIm, bRe, bIm, pw, _fftM, NFft, HopLength, _nFreqs, frames);
                OverfitParallelFor.For(0, frames, 16, &MelFrameWorker, &ctx);
            }
        }

        private readonly unsafe struct MelFrameCtx
        {
            public readonly float* Pad, Hann, CRe, CIm, TRe, TIm, BRe, BIm, Pw;
            public readonly int M, NFft, Hop, NFreqs, Frames;
            public MelFrameCtx(float* pad, float* hann, float* cRe, float* cIm, float* tRe, float* tIm,
                float* bRe, float* bIm, float* pw, int m, int nFft, int hop, int nFreqs, int frames)
            {
                Pad = pad; Hann = hann; CRe = cRe; CIm = cIm; TRe = tRe; TIm = tIm; BRe = bRe; BIm = bIm; Pw = pw;
                M = m; NFft = nFft; Hop = hop; NFreqs = nFreqs; Frames = frames;
            }
        }

        private static unsafe void MelFrameWorker(int start, int end, void* ctxPtr)
        {
            ref var c = ref Unsafe.AsRef<MelFrameCtx>(ctxPtr);
            var m = c.M;
            Span<float> aRe = stackalloc float[m];
            Span<float> aIm = stackalloc float[m];
            var twRe = new ReadOnlySpan<float>(c.TRe, m / 2);
            var twIm = new ReadOnlySpan<float>(c.TIm, m / 2);

            for (var t = start; t < end; t++)
            {
                var s = t * c.Hop;
                // a[n] = (window · x)[n] · chirp[n], zero-padded to M.
                for (var n = 0; n < c.NFft; n++)
                {
                    var x = c.Pad[s + n] * c.Hann[n];
                    aRe[n] = x * c.CRe[n];
                    aIm[n] = x * c.CIm[n];
                }
                aRe.Slice(c.NFft, m - c.NFft).Clear();
                aIm.Slice(c.NFft, m - c.NFft).Clear();

                Fft(aRe, aIm, twRe, twIm, inverse: false);
                for (var i = 0; i < m; i++) // pointwise × FFT(B)
                {
                    var ar = aRe[i];
                    var ai = aIm[i];
                    aRe[i] = ar * c.BRe[i] - ai * c.BIm[i];
                    aIm[i] = ar * c.BIm[i] + ai * c.BRe[i];
                }
                Fft(aRe, aIm, twRe, twIm, inverse: true);

                // X[k] = chirp[k] · conv[k]; power[k] = |X[k]|².
                for (var k = 0; k < c.NFreqs; k++)
                {
                    var cr = aRe[k];
                    var ci = aIm[k];
                    var xr = cr * c.CRe[k] - ci * c.CIm[k];
                    var xi = cr * c.CIm[k] + ci * c.CRe[k];
                    c.Pw[k * c.Frames + t] = xr * xr + xi * xi;
                }
            }
        }

        /// <summary>In-place iterative radix-2 Cooley-Tukey FFT (size = power of 2 = <c>re.Length</c>).
        /// <paramref name="twRe"/>/<paramref name="twIm"/> are the forward twiddles <c>exp(−i2πk/M)</c>
        /// (<c>[M/2]</c>); <paramref name="inverse"/> conjugates them and scales by <c>1/M</c>.</summary>
        private static void Fft(Span<float> re, Span<float> im, ReadOnlySpan<float> twRe, ReadOnlySpan<float> twIm, bool inverse)
        {
            var n = re.Length;

            // bit-reversal permutation
            for (int i = 1, j = 0; i < n; i++)
            {
                var bit = n >> 1;
                for (; (j & bit) != 0; bit >>= 1) { j ^= bit; }
                j ^= bit;

                if (i < j)
                {
                    (re[i], re[j]) = (re[j], re[i]);
                    (im[i], im[j]) = (im[j], im[i]);
                }
            }

            var sign = inverse ? -1f : 1f;
            for (var len = 2; len <= n; len <<= 1)
            {
                var half = len >> 1;
                var step = n / len;
                for (var i = 0; i < n; i += len)
                {
                    for (var k = 0; k < half; k++)
                    {
                        var ti = k * step;
                        var wr = twRe[ti];
                        var wi = sign * twIm[ti];
                        var a = i + k;
                        var b = a + half;
                        var xr = re[b] * wr - im[b] * wi;
                        var xi = re[b] * wi + im[b] * wr;
                        re[b] = re[a] - xr; im[b] = im[a] - xi;
                        re[a] += xr; im[a] += xi;
                    }
                }
            }

            if (inverse)
            {
                var invN = 1f / n;
                for (var i = 0; i < n; i++) { re[i] *= invN; im[i] *= invN; }
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
            var outp = new float[x.Length + 2 * pad];
            ReflectPadInto(x, pad, outp);
            return outp;
        }

        private static void ReflectPadInto(ReadOnlySpan<float> x, int pad, Span<float> outp)
        {
            var n = x.Length;
            x.CopyTo(outp.Slice(pad, n));
            // Reflect (without repeating the edge sample), matching np.pad(mode='reflect').
            for (var i = 0; i < pad; i++)
            {
                outp[pad - 1 - i] = x[Math.Min(i + 1, n - 1)];
                outp[pad + n + i] = x[Math.Max(n - 2 - i, 0)];
            }
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
