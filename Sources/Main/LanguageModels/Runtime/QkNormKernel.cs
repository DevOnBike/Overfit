// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Qwen3 QK-RMSNorm: per-head RMS-normalize a projected Q or K buffer over <c>head_dim</c> with a learnable
    /// weight, applied right after the projection and <b>before RoPE</b> (matching the reference graph). Only used
    /// by Qwen3 / Qwen3-MoE; every other arch leaves Q/K untouched.
    /// </summary>
    internal static class QkNormKernel
    {
        // Qwen3 uses rms_norm_eps = 1e-6 for all its RMSNorms, including the QK norm.
        private const float Eps = 1e-6f;

        /// <summary>
        /// RMS-normalizes each of <paramref name="rows"/> consecutive head vectors in <paramref name="buffer"/>
        /// (each <paramref name="headDim"/> long) by <paramref name="weight"/> (<c>[head_dim]</c>), in place:
        /// <c>x ← x / sqrt(mean(x²) + eps) · weight</c>.
        /// </summary>
        public static void Apply(Span<float> buffer, ReadOnlySpan<float> weight, int rows, int headDim)
        {
            for (var n = 0; n < rows; n++)
            {
                var v = buffer.Slice(n * headDim, headDim);
                var sumSq = 0f;

                for (var i = 0; i < headDim; i++)
                {
                    sumSq += v[i] * v[i];
                }

                var inv = 1f / MathF.Sqrt((sumSq / headDim) + Eps);

                for (var i = 0; i < headDim; i++)
                {
                    v[i] = v[i] * inv * weight[i];
                }
            }
        }
    }
}
