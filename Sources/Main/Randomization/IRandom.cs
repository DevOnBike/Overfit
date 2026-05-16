// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Randomization
{
    /// <summary>
    ///     Common surface for the engine's pseudo-random generators, so consumers —
    ///     evolutionary operators, samplers, stochastic kernels — can be written against
    ///     an abstraction instead of a concrete generator.
    /// </summary>
    /// <remarks>
    ///     The number-producing members mirror the subset of <see cref="Random"/> the
    ///     engine actually uses. <see cref="SaveState"/> / <see cref="LoadState"/> add what
    ///     <see cref="Random"/> cannot: capturing the full generator state so a run resumes
    ///     bit-identically after a checkpoint. Each implementation serialises whatever its
    ///     state needs — a single word, a SIMD vector, a lane cache — which is why the
    ///     contract is a stream pair rather than a fixed-width property.
    /// </remarks>
    public interface IRandom
    {
        /// <summary>Returns a non-negative random integer less than <see cref="int.MaxValue"/>.</summary>
        int Next();

        /// <summary>Returns a non-negative random integer less than <paramref name="maxValue"/>.</summary>
        int Next(int maxValue);

        /// <summary>Returns a random integer in [<paramref name="minValue"/>, <paramref name="maxValue"/>).</summary>
        int Next(int minValue, int maxValue);

        /// <summary>Returns a random double in [0, 1).</summary>
        double NextDouble();

        /// <summary>Returns a random float in [0, 1).</summary>
        float NextSingle();

        /// <summary>Fills <paramref name="buffer"/> with random bytes.</summary>
        void NextBytes(Span<byte> buffer);

        /// <summary>
        ///     Writes the complete generator state to <paramref name="writer"/> so that a
        ///     later <see cref="LoadState"/> resumes the stream bit-identically.
        /// </summary>
        void SaveState(BinaryWriter writer);

        /// <summary>
        ///     Restores generator state previously written by <see cref="SaveState"/>.
        /// </summary>
        void LoadState(BinaryReader reader);
    }
}
