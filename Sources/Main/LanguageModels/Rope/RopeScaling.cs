// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Rope
{
    /// <summary>
    /// Llama-3 "llama3" RoPE frequency scaling (NTK-by-parts) for long context — the
    /// <c>rope_scaling</c> block in a Llama-3.x <c>config.json</c>. Rescales the per-dimension
    /// RoPE frequencies once at table construction: low-frequency (long-wavelength) dims are
    /// divided by <see cref="Factor"/>, high-frequency dims are left untouched, and a smooth
    /// interpolation bridges the band between. This is what lets Llama-3.2 extend to 128k while
    /// staying accurate at short range — short-range high-frequency dims are unchanged.
    ///
    /// Port of HuggingFace transformers' <c>_compute_llama3_parameters</c>. Applied to the base
    /// frequency <c>1/θ^(2i/d)</c> per dimension; null/absent ⇒ no scaling (plain RoPE).
    /// </summary>
    public sealed record RopeScaling(
        float Factor,
        float LowFreqFactor,
        float HighFreqFactor,
        int OriginalContextLength)
    {
        /// <summary>
        /// Rescales one base RoPE frequency per the llama3 NTK-by-parts rule.
        /// <paramref name="baseFreq"/> = <c>1/θ^(2i/d)</c>; returns the scaled frequency.
        /// </summary>
        public float Apply(float baseFreq)
        {
            // Wavelength of this frequency (in token positions).
            var wavelen = 2f * MathF.PI / baseFreq;
            var lowFreqWavelen = OriginalContextLength / LowFreqFactor;
            var highFreqWavelen = OriginalContextLength / HighFreqFactor;

            if (wavelen > lowFreqWavelen)
            {
                // Low frequency (long wavelength) — scale down by the full factor.
                return baseFreq / Factor;
            }
            if (wavelen < highFreqWavelen)
            {
                // High frequency (short wavelength) — unchanged.
                return baseFreq;
            }

            // Medium band — smooth interpolation between scaled and unscaled.
            var smooth = (OriginalContextLength / wavelen - LowFreqFactor) / (HighFreqFactor - LowFreqFactor);
            return ((1f - smooth) * (baseFreq / Factor)) + (smooth * baseFreq);
        }
    }
}
