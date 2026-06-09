// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Tts.Snac
{
    /// <summary>
    /// The SNAC decoder's nonlinearity: <b>Snake1d</b>, the periodic activation BigVGAN-family vocoders use to
    /// model the strong periodicity of speech/audio. Per-channel learnable α; grounded verbatim in the SNAC source
    /// (<c>x + (α + 1e-9)⁻¹ · sin(αx)²</c>). Pointwise, in-place, model-free.
    /// </summary>
    internal static class SnacActivations
    {
        // Matches snake(x, alpha) = x + (alpha + 1e-9).reciprocal() * sin(alpha*x)^2 (the +1e-9 guards 1/α).
        private const float AlphaEps = 1e-9f;

        /// <summary>
        /// Applies Snake1d in place to a channel-major <c>[channels × time]</c> tensor, with one learnable
        /// <paramref name="alpha"/> per channel (shape <c>[channels]</c>). Each element becomes
        /// <c>v + (α + 1e-9)⁻¹ · sin(α·v)²</c>.
        /// </summary>
        public static void Snake1dInPlace(Span<float> x, ReadOnlySpan<float> alpha, int channels, int time)
        {
            for (var c = 0; c < channels; c++)
            {
                var a = alpha[c];
                var invA = 1f / (a + AlphaEps);
                var baseIdx = c * time;
                for (var t = 0; t < time; t++)
                {
                    var v = x[baseIdx + t];
                    var s = MathF.Sin(a * v);
                    x[baseIdx + t] = v + (invA * s * s);
                }
            }
        }
    }
}
