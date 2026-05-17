// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Randomization;

namespace DevOnBike.Overfit.Evolutionary
{
    /// <summary>
    ///     Deterministic xorshift32 generator implementing <see cref="IRandom"/> and also
    ///     exposed as a <see cref="Random"/> subclass.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Being both an <see cref="IRandom"/> and a <see cref="Random"/> lets this
    ///         generator drive the evolutionary operators (which accept <see cref="IRandom"/>)
    ///         and equally serve any BCL API that expects a <see cref="Random"/>.
    ///     </para>
    ///     <para>
    ///         The built-in <see cref="Random"/> hides its internal state, so a checkpoint
    ///         cannot capture and restore it. This type's whole state is a single 32-bit
    ///         word; <see cref="SaveState"/> / <see cref="LoadState"/> persist and restore it
    ///         so a run resumes bit-identically after a save/load.
    ///     </para>
    ///     <para>
    ///         xorshift32 (Marsaglia) is a small, fast generator with a period of 2^32 - 1.
    ///         It is intended for evolutionary search — where the perturbation stream only
    ///         needs to be well-distributed and reproducible — not for cryptographic use.
    ///     </para>
    /// </remarks>
    internal sealed class SeededXorShiftRandom : Random, IRandom
    {
        // xorshift32 has a fixed point at 0: the state must never be zero. Any seed that
        // normalises to zero is replaced with this arbitrary non-zero constant.
        private const uint DefaultNonZeroSeed = 0x6D2B79F5u;

        private uint _state;

        public SeededXorShiftRandom(int seed)
        {
            _state = Normalize(unchecked((uint)seed));
        }

        /// <summary>
        ///     Writes the 32-bit generator state so <see cref="LoadState"/> can resume the
        ///     stream bit-identically.
        /// </summary>
        public void SaveState(BinaryWriter writer)
        {
            ArgumentNullException.ThrowIfNull(writer);
            writer.Write(_state);
        }

        /// <summary>
        ///     Restores generator state previously written by <see cref="SaveState"/>. A
        ///     value that normalises to zero is replaced with a fixed non-zero constant,
        ///     since zero is a fixed point of xorshift32.
        /// </summary>
        public void LoadState(BinaryReader reader)
        {
            ArgumentNullException.ThrowIfNull(reader);
            _state = Normalize(reader.ReadUInt32());
        }

        public override int Next()
        {
            // Non-negative int: drop the sign bit.
            return (int)(NextUInt32() >> 1);
        }

        public override int Next(int maxValue)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(maxValue);
            return maxValue == 0 ? 0 : (int)NextUInt32Below((uint)maxValue);
        }

        public override int Next(int minValue, int maxValue)
        {
            if (minValue > maxValue)
            {
                throw new ArgumentOutOfRangeException(nameof(minValue), "minValue cannot be greater than maxValue.");
            }

            var range = (uint)((long)maxValue - minValue);

            return range == 0u ? minValue : minValue + (int)NextUInt32Below(range);
        }

        public override double NextDouble()
        {
            return Sample();
        }

        public override float NextSingle()
        {
            // 24-bit mantissa path: uniform in [0, 1).
            return (NextUInt32() >> 8) * (1.0f / (1u << 24));
        }

        protected override double Sample()
        {
            // 53-bit double in [0, 1) assembled from two draws.
            var hi = (ulong)(NextUInt32() >> 5); // 27 bits
            var lo = (ulong)(NextUInt32() >> 6); // 26 bits
            return ((hi << 26) | lo) * (1.0 / (1UL << 53));
        }

        private uint NextUInt32()
        {
            var x = _state;

            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;

            _state = Normalize(x);

            return _state;
        }

        private uint NextUInt32Below(uint maxExclusive)
        {
            // Lemire's unbiased bounded generator.
            var product = (ulong)NextUInt32() * maxExclusive;
            var low = (uint)product;

            if (low < maxExclusive)
            {
                var threshold = unchecked(0u - maxExclusive) % maxExclusive;

                while (low < threshold)
                {
                    product = (ulong)NextUInt32() * maxExclusive;
                    low = (uint)product;
                }
            }

            return (uint)(product >> 32);
        }

        private static uint Normalize(uint seed)
        {
            return seed == 0u ? DefaultNonZeroSeed : seed;
        }
    }
}
