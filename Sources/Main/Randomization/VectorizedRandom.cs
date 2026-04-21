// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Security.Cryptography;

namespace DevOnBike.Overfit.Randomization
{
    /// <summary>
    /// Fast SIMD-oriented pseudo-random number generator producing uniform values.
    ///
    /// Designed for throughput-oriented workloads such as:
    /// - simulations,
    /// - evolutionary algorithms,
    /// - stochastic masking,
    /// - general high-volume numeric sampling.
    ///
    /// This generator is:
    /// - not cryptographically secure,
    /// - not intended for security-sensitive scenarios,
    /// - optimized for speed rather than top-tier statistical quality.
    ///
    /// Thread safety:
    /// - instances of <see cref="VectorizedRandom"/> are NOT thread-safe,
    /// - <see cref="Shared"/> is safe to use from multiple threads because it provides
    ///   one independent instance per thread via thread-local storage.
    ///
    /// Sequence semantics:
    /// - vector methods and scalar methods share the same underlying state,
    /// - but mixing scalar and vector APIs does not preserve a perfect element-by-element
    ///   sequence because vector calls may discard partially consumed scalar lanes.
    /// </summary>
    public sealed class VectorizedRandom
    {
        [ThreadStatic]
        private static VectorizedRandom _shared;

        private Vector256<uint> _state;

        // Scalar lane cache built from one vector step.
        private Vector256<uint> _laneBuffer;
        private int _laneIndex;

        /// <summary>
        /// Gets a thread-local shared instance, similar in spirit to <see cref="Random.Shared"/>.
        /// Each thread gets its own independent generator instance.
        /// </summary>
        public static VectorizedRandom Shared => _shared ??= new VectorizedRandom();

        /// <summary>
        /// Creates a new generator with either an explicit seed or a securely generated seed.
        /// </summary>
        public VectorizedRandom(uint? seed = null)
        {
            if (!Vector256.IsHardwareAccelerated)
            {
                throw new PlatformNotSupportedException("VectorizedRandom requires 256-bit SIMD hardware acceleration.");
            }

            var actualSeed = seed ?? CreateSeed();

            _state = CreateInitialState(NonZero(actualSeed));

            _laneBuffer = default;
            _laneIndex = Vector256<uint>.Count; // empty
        }

        /// <summary>
        /// Resets the generator state using a new seed without allocating a new object.
        /// Critical for deterministic noise regeneration in Evolution Strategies.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Reinitialize(uint seed)
        {
            _state = CreateInitialState(NonZero(seed));
            _laneIndex = Vector256<uint>.Count; // Invalidate the scalar lane buffer
        }

        /// <summary>
        /// Returns 8 uniformly distributed floats in [0, 1).
        /// Hot path: no branches in the PRNG step itself, just state update + bit manipulation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector256<float> NextSingleVector()
        {
            // Discard any partially consumed scalar lane buffer so vector output
            // always represents a fresh full SIMD step.
            _laneIndex = Vector256<uint>.Count;

            var s = NextUInt32VectorInternal();

            var mantissa = s & Vector256.Create(0x007FFFFFu);
            var oneFloatBits = mantissa | Vector256.Create(0x3F800000u);

            return oneFloatBits.AsSingle() - Vector256.Create(1.0f);
        }

        /// <summary>
        /// Returns a uniformly distributed float in [0, 1).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float NextSingle()
        {
            var bits = (int)((NextUInt32() & 0x007FFFFFu) | 0x3F800000u);
            return BitConverter.Int32BitsToSingle(bits) - 1.0f;
        }

        /// <summary>
        /// Returns a uniformly distributed double in [0, 1).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double NextDouble()
        {
            // Standard 53-bit construction.
            ulong a = NextUInt32() >> 5;
            ulong b = NextUInt32() >> 6;
            var value = (a << 26) | b;

            return value * (1.0 / (1UL << 53));
        }

        /// <summary>
        /// Returns a non-negative random integer that is less than <see cref="int.MaxValue"/>.
        /// Mirrors the behavior of <see cref="Random.Next()"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Next()
        {
            return (int)NextUInt32Below(int.MaxValue);
        }

        /// <summary>
        /// Returns a non-negative random integer that is less than the specified maximum.
        /// Mirrors the behavior of <see cref="Random.Next(int)"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Next(int maxValue)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(maxValue);

            if (maxValue == 0)
            {
                return 0;
            }

            return (int)NextUInt32Below((uint)maxValue);
        }

        /// <summary>
        /// Returns a random integer that is within a specified range.
        /// Mirrors the behavior of <see cref="Random.Next(int, int)"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Next(int minValue, int maxValue)
        {
            if (minValue > maxValue)
            {
                throw new ArgumentOutOfRangeException(nameof(minValue), "minValue cannot be greater than maxValue.");
            }

            if (minValue == maxValue)
            {
                return minValue;
            }

            var range = (uint)((long)maxValue - minValue);
            var sample = NextUInt32Below(range);

            return (int)(minValue + (long)sample);
        }

        /// <summary>
        /// Fills the destination span with random bytes.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void NextBytes(Span<byte> buffer)
        {
            var i = 0;

            while (i <= buffer.Length - sizeof(uint))
            {
                var value = NextUInt32();

                buffer[i] = (byte)value;
                buffer[i + 1] = (byte)(value >> 8);
                buffer[i + 2] = (byte)(value >> 16);
                buffer[i + 3] = (byte)(value >> 24);

                i += sizeof(uint);
            }

            if (i < buffer.Length)
            {
                var value = NextUInt32();

                while (i < buffer.Length)
                {
                    buffer[i++] = (byte)value;
                    value >>= 8;
                }
            }
        }

        /// <summary>
        /// Fills the destination array with random bytes.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void NextBytes(byte[] buffer)
        {
            ArgumentNullException.ThrowIfNull(buffer);

            NextBytes(buffer.AsSpan());
        }

        /// <summary>
        /// Fills the destination span with uniformly distributed floats in [0, 1).
        /// Steady-state path is allocation-free.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Fill(Span<float> destination)
        {
            var width = Vector256<float>.Count;
            var i = 0;

            for (; i <= destination.Length - width; i += width)
            {
                var v = NextSingleVector();
                v.CopyTo(destination.Slice(i, width));
            }

            if (i < destination.Length)
            {
                Span<float> tail = stackalloc float[8];
                var v = NextSingleVector();
                v.CopyTo(tail);
                tail[..(destination.Length - i)].CopyTo(destination[i..]);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private uint NextUInt32()
        {
            if (_laneIndex >= Vector256<uint>.Count)
            {
                _laneBuffer = NextUInt32VectorInternal();
                _laneIndex = 0;
            }

            return _laneBuffer.GetElement(_laneIndex++);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Vector256<uint> NextUInt32VectorInternal()
        {
            var s = _state;

            // Vectorized XorShift32
            s ^= Vector256.ShiftLeft(s, 13);
            s ^= Vector256.ShiftRightLogical(s, 17);
            s ^= Vector256.ShiftLeft(s, 5);

            _state = s;
            return s;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private uint NextUInt32Below(uint maxExclusive)
        {
            if (maxExclusive == 0)
            {
                return 0;
            }

            // Lemire-style unbiased bounded generation.
            var product = (ulong)NextUInt32() * maxExclusive;
            var low = (uint)product;

            if (low < maxExclusive)
            {
                var threshold = unchecked((uint)(0 - maxExclusive)) % maxExclusive;

                while (low < threshold)
                {
                    product = (ulong)NextUInt32() * maxExclusive;
                    low = (uint)product;
                }
            }

            return (uint)(product >> 32);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<uint> CreateInitialState(uint seed)
        {
            return Vector256.Create(
                NonZero(Mix32(unchecked(seed + (0x9E3779B9u * 1u)))),
                NonZero(Mix32(unchecked(seed + (0x9E3779B9u * 2u)))),
                NonZero(Mix32(unchecked(seed + (0x9E3779B9u * 3u)))),
                NonZero(Mix32(unchecked(seed + (0x9E3779B9u * 4u)))),
                NonZero(Mix32(unchecked(seed + (0x9E3779B9u * 5u)))),
                NonZero(Mix32(unchecked(seed + (0x9E3779B9u * 6u)))),
                NonZero(Mix32(unchecked(seed + (0x9E3779B9u * 7u)))),
                NonZero(Mix32(unchecked(seed + (0x9E3779B9u * 8u))))
            );
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint CreateSeed()
        {
            Span<byte> bytes = stackalloc byte[4];

            RandomNumberGenerator.Fill(bytes);

            var seed =
                (uint)bytes[0] |
                ((uint)bytes[1] << 8) |
                ((uint)bytes[2] << 16) |
                ((uint)bytes[3] << 24);

            return NonZero(seed);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint NonZero(uint x) => x == 0u ? 0x6D2B79F5u : x;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint Mix32(uint x)
        {
            unchecked
            {
                x ^= x >> 16;
                x *= 0x7FEB352Du;
                x ^= x >> 15;
                x *= 0x846CA68Bu;
                x ^= x >> 16;

                return x;
            }
        }
    }
}