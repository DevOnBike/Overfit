// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.DeepLearning.Attention
{
    // ── Extension for test random fill ───────────────────────────────────────
    internal static class RandomExtensions
    {
        public static float[] NextFloats(this Random rng, float[] arr)
        {
            for (var i = 0; i < arr.Length; i++)
            {
                arr[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
            }
            return arr;
        }

        public static void NextFloats(this Random rng, Span<float> span)
        {
            for (var i = 0; i < span.Length; i++)
            {
                span[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
            }
        }
    }
}
