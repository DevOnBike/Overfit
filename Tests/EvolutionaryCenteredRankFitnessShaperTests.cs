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
    }
}
