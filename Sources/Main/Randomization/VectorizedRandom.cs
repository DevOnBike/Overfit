// Copyright (c) 2026 DevOnBike.
// This file is part of DevOnBike Overfit.
// DevOnBike Overfit is licensed under the GNU AGPLv3.

using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Security.Cryptography;

namespace DevOnBike.Overfit.Randomization
{
    /// <summary>
    /// High-throughput SIMD-oriented pseudo-random number generator based on 256-bit SIMD.
    /// Optimized for zero-allocation hot paths and high sampling throughput.
    ///
    /// Intended for workloads such as:
    /// - evolutionary algorithms,
    /// - simulations,
    /// - stochastic masking,
    /// - general numeric sampling.
    ///
    /// This generator is:
    /// - not cryptographically secure,
    /// - not thread-safe per instance,
    /// - optimized for speed rather than top-tier statistical quality.
    ///
    /// Use <see cref="Shared"/> for a thread-local shared instance.
    /// </summary>
    public sealed class VectorizedRandom
    {
        private static readonly int UIntLaneCount = Vector256<uint>.Count;
        private static readonly int FloatLaneCount = Vector256<float>.Count;
        private static readonly int ByteVectorWidth = Vector256<byte>.Count;

        [ThreadStatic]
        private static VectorizedRandom? _shared;

        private Vector256<uint> _state;

        // Scalar cache built from one SIMD step.
        private Vector256<uint> _laneBuffer;
        private int _laneIndex;

        static VectorizedRandom()
        {
            if (!Vector256.IsHardwareAccelerated)
            {
                throw new PlatformNotSupportedException("VectorizedRandom requires 256-bit SIMD hardware acceleration.");
            }
        }

        /// <summary>
        /// Gets a thread-local shared instance, similar in spirit to <see cref="Random.Shared"/>.
        /// Each thread receives its own independent generator instance.
        /// </summary>
        public static VectorizedRandom Shared => _shared ??= new();

        /// <summary>
        /// Creates a new generator with either an explicit seed or a securely generated seed.
        /// </summary>
        public VectorizedRandom(uint? seed = null)
        {
            var actualSeed = seed ?? CreateSeed();
            _state = CreateInitialState(NonZero(actualSeed));

            _laneBuffer = default;
            _laneIndex = UIntLaneCount; // cache empty
        }

        /// <summary>
        /// Reinitializes the generator state without allocating a new instance.
        /// Useful for deterministic regeneration of noise in evolutionary algorithms.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Reinitialize(uint seed)
        {
            _state = CreateInitialState(NonZero(seed));
            _laneBuffer = default;
            _laneIndex = UIntLaneCount;
        }

        /// <summary>
        /// Returns 8 random floats in the range [0, 1) in a single SIMD step.
        /// This invalidates any partially consumed scalar cache.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector256<float> NextSingleVector()
        {
            // A vector call consumes a fresh SIMD step and discards any partially used scalar cache.
            _laneIndex = UIntLaneCount;

            var s = NextUInt32VectorInternal();

            var mantissa = s & Vector256.Create(0x007FFFFFu);
            var oneFloatBits = mantissa | Vector256.Create(0x3F800000u);

            return oneFloatBits.AsSingle() - Vector256.Create(1.0f);
        }

        /// <summary>
        /// Returns a random float in the range [0, 1).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float NextSingle()
        {
            var bits = (int)((NextUInt32() & 0x007FFFFFu) | 0x3F800000u);

            return BitConverter.Int32BitsToSingle(bits) - 1.0f;
        }

        /// <summary>
        /// Returns a random double in the range [0, 1).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double NextDouble()
        {
            // Standard 53-bit construction from two uint draws.
            ulong a = NextUInt32() >> 5;
            ulong b = NextUInt32() >> 6;

            return ((a << 26) | b) * (1.0 / (1UL << 53));
        }

        /// <summary>
        /// Returns a non-negative random integer that is less than <see cref="int.MaxValue"/>.
        /// Mirrors the semantics of <see cref="Random.Next()"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Next() => Next(int.MaxValue);

        /// <summary>
        /// Returns a non-negative random integer that is less than the specified maximum.
        /// Mirrors the semantics of <see cref="Random.Next(int)"/>.
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
        /// Returns a random integer within the specified range.
        /// Mirrors the semantics of <see cref="Random.Next(int, int)"/>.
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

            uint range = (uint)((long)maxValue - minValue);
            uint sample = NextUInt32Below(range);

            return (int)(minValue + (long)sample);
        }

        /// <summary>
        /// Fills the destination span with random floats in [0, 1) using SIMD.
        /// Steady-state path is allocation-free.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Fill(Span<float> destination)
        {
            var i = 0;

            for (; i <= destination.Length - FloatLaneCount; i += FloatLaneCount)
            {
                var v = NextSingleVector();
                v.CopyTo(destination.Slice(i, FloatLaneCount));
            }

            if (i < destination.Length)
            {
                Span<float> tail = stackalloc float[8];
                var v = NextSingleVector();
                v.CopyTo(tail);
                tail[..(destination.Length - i)].CopyTo(destination[i..]);
            }
        }

        /// <summary>
        /// Fills the destination span with random bytes.
        /// Uses a bulk SIMD path for 32-byte chunks and preserves scalar cache continuity.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void NextBytes(Span<byte> buffer)
        {
            var i = 0;

            // 1. Drain remaining scalar cache first to preserve sequence continuity.
            while (_laneIndex < UIntLaneCount && buffer.Length - i >= sizeof(uint))
            {
                var value = _laneBuffer.GetElement(_laneIndex++);
                WriteUInt32LittleEndian(buffer, ref i, value);
            }

            // 2. Bulk SIMD path: one PRNG step produces 32 bytes.
            while (i <= buffer.Length - ByteVectorWidth)
            {
                var v = NextUInt32VectorInternal();
                v.AsByte().CopyTo(buffer.Slice(i, ByteVectorWidth));
                i += ByteVectorWidth;
            }

            // 3. Consume remaining full uints through the scalar cache.
            while (buffer.Length - i >= sizeof(uint))
            {
                var value = NextUInt32();
                WriteUInt32LittleEndian(buffer, ref i, value);
            }

            // 4. Tail bytes.
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private uint NextUInt32()
        {
            if (_laneIndex >= UIntLaneCount)
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

            // Lemire's unbiased bounded random generation.
            ulong product = (ulong)NextUInt32() * maxExclusive;
            uint low = (uint)product;

            if (low < maxExclusive)
            {
                uint threshold = unchecked((uint)(0 - maxExclusive)) % maxExclusive;

                while (low < threshold)
                {
                    product = (ulong)NextUInt32() * maxExclusive;
                    low = (uint)product;
                }
            }

            return (uint)(product >> 32);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WriteUInt32LittleEndian(Span<byte> buffer, ref int offset, uint value)
        {
            buffer[offset] = (byte)value;
            buffer[offset + 1] = (byte)(value >> 8);
            buffer[offset + 2] = (byte)(value >> 16);
            buffer[offset + 3] = (byte)(value >> 24);

            offset += sizeof(uint);
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

            return NonZero(BinaryPrimitives.ReadUInt32LittleEndian(bytes));
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