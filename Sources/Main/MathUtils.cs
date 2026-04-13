// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit
{
    /// <summary>
    ///     Static utility class providing mathematical helper functions used across the engine.
    /// </summary>
    public static class MathUtils
    {
        [ThreadStatic]
        private static Random _rng;

        /// <summary>
        ///     Gets a thread-safe <see cref="Random" /> instance, lazily initialized with a unique seed.
        ///     Uses <see cref="ThreadStaticAttribute" /> to avoid lock contention in multi-threaded training.
        /// </summary>
        private static Random Rng => _rng ??= new Random(Guid.NewGuid().GetHashCode());

        /// <summary>
        ///     Returns a random number from a standard normal distribution N(0, 1) using the Box-Muller transform.
        /// </summary>
        /// <returns>A random float following the Gaussian distribution.</returns>
        public static float NextGaussian()
        {
            const float twoPi = 2.0f * MathF.PI;

            var u1 = 1.0f - Rng.NextSingle();
            var u2 = 1.0f - Rng.NextSingle();

            return MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Sin(twoPi * u2);
        }
    }
}