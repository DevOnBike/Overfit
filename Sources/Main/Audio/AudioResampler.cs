// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio
{
    /// <summary>
    /// Sample-rate conversion for mono float PCM. Linear interpolation — adequate for speech fed to the
    /// Whisper frontend (which is robust to mild resampling artefacts) and dependency-free.
    /// </summary>
    public static class AudioResampler
    {
        /// <summary>Resamples mono <paramref name="src"/> from <paramref name="srcRate"/> to
        /// <paramref name="dstRate"/> Hz. Returns <paramref name="src"/> copied when the rates match.</summary>
        public static float[] Resample(ReadOnlySpan<float> src, int srcRate, int dstRate)
        {
            if (srcRate == dstRate || src.Length == 0)
            {
                return src.ToArray();
            }

            var outLen = (int)((long)src.Length * dstRate / srcRate);
            // OVERFIT001: by-contract — the resampled signal is the return value the caller owns.
#pragma warning disable OVERFIT001
            var dst = new float[outLen];
#pragma warning restore OVERFIT001
            var ratio = (double)srcRate / dstRate;
            for (var i = 0; i < outLen; i++)
            {
                var srcPos = i * ratio;
                var i0 = (int)srcPos;
                var frac = (float)(srcPos - i0);
                var a = src[i0];
                var b = i0 + 1 < src.Length ? src[i0 + 1] : a;
                dst[i] = a + (b - a) * frac;
            }
            return dst;
        }
    }
}
