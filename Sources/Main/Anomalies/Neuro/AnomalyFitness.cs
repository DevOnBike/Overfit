// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Neuro
{
    /// <summary>
    /// Scores one candidate genome against labelled windows — the fitness function an evolutionary strategy
    /// maximises. Pure and allocation-free: it walks the samples, applies the genome's own evolved threshold, and
    /// counts the confusion matrix.
    ///
    /// <para>The whole reason this exists is that these objectives are <b>not differentiable</b>
    /// (see <see cref="AnomalyObjective"/>), so they cannot be a backprop loss. A gradient-trained detector has
    /// to optimise a surrogate (MSE / cross-entropy) and then have its threshold tuned by hand afterwards,
    /// leaving a gap between what was optimised and what is judged. Evolution closes that gap by optimising the
    /// judged metric itself.</para>
    /// </summary>
    public static class AnomalyFitness
    {
        /// <summary>
        /// Fitness of <paramref name="genome"/> — higher is better for every objective (Cost is returned negated).
        /// </summary>
        /// <param name="genome">Candidate parameters, laid out per <see cref="AnomalyMlp"/>.</param>
        /// <param name="features">Flat samples: <paramref name="sampleCount"/> × <paramref name="inputs"/>.</param>
        /// <param name="labels">1 = anomaly, 0 = normal; length <paramref name="sampleCount"/>.</param>
        /// <param name="options">Objective and its knobs.</param>
        public static float Evaluate(
            ReadOnlySpan<float> genome,
            ReadOnlySpan<float> features,
            ReadOnlySpan<byte> labels,
            int sampleCount,
            int inputs,
            int hidden,
            in AnomalyFitnessOptions options)
        {
            var tp = 0;
            var fp = 0;
            var fn = 0;

            for (var i = 0; i < sampleCount; i++)
            {
                var window = features.Slice(i * inputs, inputs);
                var flagged = AnomalyMlp.IsAnomaly(genome, window, inputs, hidden);
                var actual = labels[i] != 0;

                if (flagged)
                {
                    if (actual)
                    {
                        tp++;
                        continue;
                    }

                    fp++;
                    continue;
                }

                if (actual)
                {
                    fn++;
                }
            }

            return options.Objective switch
            {
                AnomalyObjective.Cost => -((fn * options.MissCost) + (fp * options.FalseAlarmCost)),
                AnomalyObjective.RecallAtFalseAlarmBudget => fp > options.FalseAlarmBudget ? 0f : Recall(tp, fn),
                _ => F1(tp, fp, fn),
            };
        }

        /// <summary>F1 with the degenerate cases pinned to 0 — a detector that never fires must not look perfect.</summary>
        public static float F1(int tp, int fp, int fn)
        {
            if (tp == 0)
            {
                return 0f;
            }
            var precision = tp / (float)(tp + fp);
            var recall = tp / (float)(tp + fn);
            return 2f * precision * recall / (precision + recall);
        }

        private static float Recall(int tp, int fn) => tp + fn == 0 ? 0f : tp / (float)(tp + fn);
    }
}
