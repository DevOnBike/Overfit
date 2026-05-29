// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Training;

namespace DevOnBike.Overfit.Tests.Training
{
    /// <summary>Value checks for the <see cref="LearningRateSchedule"/> curves.</summary>
    public sealed class LearningRateScheduleTests
    {
        [Fact]
        public void Cosine_StartsAtMax_EndsAtMin_MidwayBetween()
        {
            Assert.Equal(0.01f, LearningRateSchedule.Cosine(0, 101, 0.01f, 0.001f), 5);
            Assert.Equal(0.001f, LearningRateSchedule.Cosine(100, 101, 0.01f, 0.001f), 5);
            Assert.Equal(0.0055f, LearningRateSchedule.Cosine(50, 101, 0.01f, 0.001f), 4); // midpoint = (max+min)/2
        }

        [Fact]
        public void LinearWarmup_Ramps_ThenHolds()
        {
            Assert.Equal(0.0025f, LearningRateSchedule.LinearWarmup(0, 4, 0.01f), 5); // (0+1)/4
            Assert.Equal(0.0075f, LearningRateSchedule.LinearWarmup(2, 4, 0.01f), 5); // (2+1)/4
            Assert.Equal(0.01f, LearningRateSchedule.LinearWarmup(4, 4, 0.01f), 5);   // held
            Assert.Equal(0.01f, LearningRateSchedule.LinearWarmup(99, 4, 0.01f), 5);
        }

        [Fact]
        public void WarmupCosine_PeaksAtWarmupBoundary_ThenAnneals()
        {
            const int total = 100, warmup = 10;
            var atWarmupEnd = LearningRateSchedule.WarmupCosine(warmup - 1, total, warmup, 0.01f, 0f);
            var afterPeak = LearningRateSchedule.WarmupCosine(warmup, total, warmup, 0.01f, 0f);
            var atEnd = LearningRateSchedule.WarmupCosine(total - 1, total, warmup, 0.01f, 0f);

            Assert.Equal(0.01f, atWarmupEnd, 5);     // ramp reaches max at the boundary
            Assert.Equal(0.01f, afterPeak, 5);       // cosine starts at max
            Assert.Equal(0f, atEnd, 5);              // anneals to min
        }

        [Fact]
        public void Cosine_Rejects_InvalidTotalSteps()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => LearningRateSchedule.Cosine(0, 0, 0.01f));
        }
    }
}
