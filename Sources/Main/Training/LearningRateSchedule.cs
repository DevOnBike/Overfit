// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Training
{
    /// <summary>
    /// Learning-rate schedules over training <b>time</b> (step → learning rate) — the curve that decays
    /// the rate as training proceeds so the model settles instead of bouncing. Pure functions: call once
    /// per step and assign to <c>optimizer.LearningRate</c>. (For multiplying the base rate by the
    /// replica count in data-parallel training, see <see cref="DataParallelLearningRate"/> — a separate
    /// concern.)
    /// </summary>
    public static class LearningRateSchedule
    {
        /// <summary>
        /// Cosine anneal: <paramref name="maxLr"/> at step 0 smoothly down to <paramref name="minLr"/> at
        /// step <c>totalSteps − 1</c>. The default decay schedule for most training.
        /// </summary>
        public static float Cosine(int step, int totalSteps, float maxLr, float minLr = 0f)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(step);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(totalSteps);
            if (totalSteps == 1) { return minLr; }

            var progress = Math.Clamp(step / (float)(totalSteps - 1), 0f, 1f);
            return minLr + 0.5f * (maxLr - minLr) * (1f + MathF.Cos(MathF.PI * progress));
        }

        /// <summary>
        /// Linear warmup: ramps from 0 up to <paramref name="targetLr"/> over the first
        /// <paramref name="warmupSteps"/> steps, then holds <paramref name="targetLr"/>. Stabilises the
        /// first steps of training (large models / large batches).
        /// </summary>
        public static float LinearWarmup(int step, int warmupSteps, float targetLr)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(step);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(warmupSteps);
            return step >= warmupSteps ? targetLr : targetLr * (step + 1) / warmupSteps;
        }

        /// <summary>
        /// Warmup-then-cosine: linear warmup 0 → <paramref name="maxLr"/> over the first
        /// <paramref name="warmupSteps"/> steps, then a cosine anneal <paramref name="maxLr"/> →
        /// <paramref name="minLr"/> over the remaining steps. The standard transformer schedule.
        /// </summary>
        public static float WarmupCosine(int step, int totalSteps, int warmupSteps, float maxLr, float minLr = 0f)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(step);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(totalSteps);
            ArgumentOutOfRangeException.ThrowIfNegative(warmupSteps);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(warmupSteps, totalSteps);

            if (step < warmupSteps)
            {
                return maxLr * (step + 1) / warmupSteps;
            }
            return Cosine(step - warmupSteps, totalSteps - warmupSteps, maxLr, minLr);
        }
    }
}
