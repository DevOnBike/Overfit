// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Fitness;

namespace DevOnBike.Overfit.Tests
{
    public sealed class EvolutionaryCenteredRankFitnessShaperTests
    {
        [Fact]
        public void Shape_ForEmptyInput_DoesNothing()
        {
            var shaper = new CenteredRankFitnessShaper();
            Span<float> shaped = [];

            shaper.Shape(ReadOnlySpan<float>.Empty, shaped);

            Assert.Empty(shaped.ToArray());
        }

        [Fact]
        public void Shape_ForSingleValue_ProducesZero()
        {
            var shaper = new CenteredRankFitnessShaper();
            float[] raw = [42f];
            var shaped = new float[1];

            shaper.Shape(raw, shaped);

            Assert.Equal(0f, shaped[0]);
        }

        [Fact]
        public void Shape_MapsLowestToMinusHalf_AndHighestToPlusHalf()
        {
            var shaper = new CenteredRankFitnessShaper();
            float[] raw = [10f, 30f, 20f];
            var shaped = new float[3];

            shaper.Shape(raw, shaped);

            Assert.Equal(-0.5f, shaped[0], 5);
            Assert.Equal(0.5f, shaped[1], 5);
            Assert.Equal(0.0f, shaped[2], 5);
        }

        [Fact]
        public void Shape_ThrowsWhenLengthsDiffer()
        {
            var shaper = new CenteredRankFitnessShaper();
            float[] raw = [1f, 2f];
            var shaped = new float[1];

            Assert.Throws<ArgumentException>(() => shaper.Shape(raw, shaped));
        }

        [Fact]
        public void Shape_HandlesNaN_RanksAsWorst()
        {
            // float.CompareTo treats NaN as smallest: a NaN fitness should receive rank 0
            // (lowest shaped value = -0.5) and never propagate upward in the ordering.
            var shaper = new CenteredRankFitnessShaper();
            float[] raw = [10f, float.NaN, 30f, 20f];
            var shaped = new float[4];

            shaper.Shape(raw, shaped);

            // Expected ordering, ascending: NaN (-0.5), 10 (-1/6), 20 (+1/6), 30 (+0.5).
            Assert.Equal(-0.5f, shaped[1], 5);  // NaN entry gets the lowest rank
            Assert.Equal(-1f / 6f, shaped[0], 5);
            Assert.Equal(1f / 6f, shaped[3], 5);
            Assert.Equal(0.5f, shaped[2], 5);
        }

        [Fact]
        public void Shape_IsAllocationStable_AfterWarmup()
        {
            var shaper = new CenteredRankFitnessShaper();
            var raw = new float[256];
            var shaped = new float[256];
            var rng = new Random(42);

            for (var i = 0; i < raw.Length; i++)
            {
                raw[i] = rng.NextSingle();
            }

            shaper.Shape(raw, shaped); // warmup: one-time ranking-buffer allocation

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < 5_000; i++)
            {
                shaper.Shape(raw, shaped);
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            Assert.True(allocated <= 512, $"Shape allocated {allocated} bytes.");
        }
    }
}