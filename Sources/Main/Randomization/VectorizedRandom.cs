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
    /// Fast SIMD-oriented pseudo-random number generator producing uniform floats in [0, 1).
    /// Designed for throughput-oriented workloads such as simulations, evolutionary algorithms,
    /// stochastic masking, and general high-volume numeric sampling.
    ///
    /// This generator is:
    /// - not thread-safe (use one instance per thread/worker),
    /// - not cryptographically secure,
    /// - optimized for speed rather than top-tier statistical quality.
    ///
    /// Internally it keeps 8 independent uint32 streams inside a single 256-bit vector and
    /// advances them with a vectorized XorShift32 step. Floats are produced via IEEE754 bit
    /// construction: random mantissa + exponent for 1.0f, then subtract 1.0f.
    /// </summary>
    public sealed class VectorizedRandom
    {
        private Vector256<uint> _state;

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
        }

        /// <summary>
        /// Returns 8 uniformly distributed floats in [0, 1).
        /// Hot path: no branches, just state update + bit manipulation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector256<float> NextSingleVector()
        {
            var s = _state;

            // Vectorized XorShift32
            s ^= Vector256.ShiftLeft(s, 13);
            s ^= Vector256.ShiftRightLogical(s, 17);
            s ^= Vector256.ShiftLeft(s, 5);

            _state = s;

            // Build IEEE754 floats in [1.0, 2.0), then subtract 1.0f => [0.0, 1.0)
            var mantissa = s & Vector256.Create(0x007FFFFFu);
            var oneFloatBits = mantissa | Vector256.Create(0x3F800000u);

            return oneFloatBits.AsSingle() - Vector256.Create(1.0f);
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

            // Tail fallback when destination length is not a multiple of vector width
            if (i < destination.Length)
            {
                Span<float> tail = stackalloc float[8];
                
                var v = NextSingleVector();
                
                v.CopyTo(tail);
                
                tail[..(destination.Length - i)].CopyTo(destination[i..]);
            }
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