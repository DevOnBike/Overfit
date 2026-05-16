// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary;
using DevOnBike.Overfit.Randomization;

namespace DevOnBike.Overfit.Tests.Randomization
{
    /// <summary>
    ///     Verifies the <see cref="IRandom.SaveState"/> / <see cref="IRandom.LoadState"/>
    ///     contract: a generator's state, captured and reloaded into a differently-seeded
    ///     instance, resumes the stream bit-identically.
    /// </summary>
    public sealed class RandomStateRoundTripTests
    {
        [Fact]
        public void SeededXorShiftRandom_SaveState_ResumesStreamBitIdentically()
        {
            AssertStateRoundTrip(
                () => new SeededXorShiftRandom(12345),
                () => new SeededXorShiftRandom(98765));
        }

        [Fact]
        public void VectorizedRandom_SaveState_ResumesStreamBitIdentically()
        {
            AssertStateRoundTrip(
                () => new VectorizedRandom(12345u),
                () => new VectorizedRandom(98765u));
        }

        // Advances 'original' partway — leaving a SIMD generator's lane cache half-consumed
        // — then checkpoints its state and proves a freshly (differently) seeded generator
        // reproduces the identical continuation once that state is loaded.
        private static void AssertStateRoundTrip(Func<IRandom> primary, Func<IRandom> alternate)
        {
            var original = primary();

            // Five scalar draws leave an 8-lane SIMD cache mid-buffer, so the saved state
            // must carry the lane index — not just the core generator word — to round-trip.
            for (var i = 0; i < 5; i++)
            {
                _ = original.NextSingle();
            }

            byte[] stateBytes;
            using (var stream = new MemoryStream())
            {
                var writer = new BinaryWriter(stream);
                original.SaveState(writer);
                writer.Flush();
                stateBytes = stream.ToArray();
            }

            var expected = DrawSequence(original);

            var restored = alternate();
            using (var stream = new MemoryStream(stateBytes))
            {
                var reader = new BinaryReader(stream);
                restored.LoadState(reader);
            }

            var actual = DrawSequence(restored);

            Assert.Equal(expected, actual);
        }

        // Exercises all three integer/float/double draw paths, which consume the
        // generator differently — a complete continuation fingerprint.
        private static double[] DrawSequence(IRandom rng)
        {
            var result = new double[48];

            for (var i = 0; i < 16; i++)
            {
                result[(i * 3) + 0] = rng.Next(1000);
                result[(i * 3) + 1] = rng.NextSingle();
                result[(i * 3) + 2] = rng.NextDouble();
            }

            return result;
        }
    }
}
