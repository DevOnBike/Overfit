// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;

namespace DevOnBike.Overfit.Maths
{
    /// <summary>
    ///     Static utility class providing mathematical helper functions used across the engine.
    /// </summary>
    public static class MathUtils
    {
        const float twoPi = 2.0f * MathF.PI;

        /// <summary>
        ///     Index of the maximum value in <paramref name="values" /> — the single, vectorized source of truth for
        ///     argmax (greedy decoding, MNIST / classification heads, anomaly scoring). Delegates to
        ///     <see cref="TensorPrimitives.IndexOfMax{T}(ReadOnlySpan{T})" /> (AVX2/AVX-512, far faster than a scalar
        ///     scan on cache-resident vectors). On ties it returns the FIRST maximum — matching the scalar <c>&gt;</c>
        ///     scans this replaced. (NaN follows IEEE / TensorPrimitives semantics, which only diverges from those
        ///     scalar scans on NaN input — i.e. an already-broken model.)
        /// </summary>
        public static int ArgMax(ReadOnlySpan<float> values)
        {
            return TensorPrimitives.IndexOfMax(values);
        }

        [ThreadStatic]
        private static Random _rng;

        /// <summary>
        ///     Gets a thread-safe <see cref="Random" /> instance, lazily initialized with a unique seed.
        ///     Uses <see cref="ThreadStaticAttribute" /> to avoid lock contention in multi-threaded training.
        /// </summary>
        private static Random Rng => _rng ??= new Random(Guid.NewGuid().GetHashCode());

        /// <summary>
        ///     Seeds this thread's RNG for reproducible weight init / sampling. Per-thread
        ///     (<see cref="ThreadStaticAttribute" />), so call it on the thread that builds the model.
        ///     Intended for tests/repro — production leaves the default per-process random seed.
        /// </summary>
        public static void SetSeed(int seed)
        {
            _rng = new Random(seed);
        }

        /// <summary>
        ///     Returns a random number from a standard normal distribution N(0, 1) using the Box-Muller transform.
        /// </summary>
        /// <returns>A random float following the Gaussian distribution.</returns>
        public static float NextGaussian()
        {
            var u1 = 1.0f - Rng.NextSingle();
            var u2 = 1.0f - Rng.NextSingle();

            return MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Sin(twoPi * u2);
        }
    }
}