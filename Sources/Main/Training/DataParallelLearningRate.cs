// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Training
{
    /// <summary>
    /// Learning-rate scaling rules for data-parallel training. <see cref="DataParallelTrainer"/>
    /// <b>averages</b> the replica gradients, so a step looks like a single replica's — to actually
    /// benefit from the N× larger global batch you scale the base learning rate up by the worker count.
    /// Pick the rule by worker count: <see cref="Linear"/> (Goyal et al. 2017) is the textbook choice
    /// for large worker counts; <see cref="Sqrt"/> is gentler and more stable for the small counts
    /// typical on a single machine.
    /// </summary>
    public static class DataParallelLearningRate
    {
        /// <summary>
        /// Linear scaling rule (Goyal et al., "Accurate, Large Minibatch SGD", 2017):
        /// <c>baseLearningRate × workerCount</c>. Matches the N× larger global batch but can be
        /// unstable without warmup at high worker counts.
        /// </summary>
        public static float Linear(float baseLearningRate, int workerCount)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(workerCount);
            return baseLearningRate * workerCount;
        }

        /// <summary>
        /// Square-root scaling rule: <c>baseLearningRate × √workerCount</c>. More conservative than
        /// <see cref="Linear"/> — the safer default for the small worker counts (≈ core count) of
        /// single-machine data parallelism.
        /// </summary>
        public static float Sqrt(float baseLearningRate, int workerCount)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(workerCount);
            return baseLearningRate * MathF.Sqrt(workerCount);
        }
    }
}
