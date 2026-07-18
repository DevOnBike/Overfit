// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Neuro;
using DevOnBike.Overfit.Evolutionary.Runtime;
using DevOnBike.Overfit.Evolutionary.Storage;
using DevOnBike.Overfit.Evolutionary.Strategies;

namespace DevOnBike.Overfit.Tests.Anomalies.Neuro
{
    /// <summary>
    /// Proves the point of neuroevolving a detector: the objective you evolve is the objective you get, including
    /// objectives that are step functions of a threshold and therefore have no usable gradient. Deterministic —
    /// fixed data seed and a fixed ES seed — so these are regression gates, not flaky "it usually works" checks.
    /// </summary>
    public sealed class AnomalyNeuroEvolutionTests
    {
        private const int Inputs = 4;
        private const int Hidden = 5;
        private const int Samples = 300;

        /// <summary>
        /// Imbalanced synthetic data (~8 % anomalies) — the regime where accuracy is a useless objective because
        /// "never fire" already scores 92 %. Anomalies are shifted and noisier than normal windows.
        /// </summary>
        private static (float[] Features, byte[] Labels) MakeData(int seed)
        {
            var rng = new Random(seed);
            var features = new float[Samples * Inputs];
            var labels = new byte[Samples];

            for (var i = 0; i < Samples; i++)
            {
                var anomaly = rng.NextDouble() < 0.08;
                labels[i] = anomaly ? (byte)1 : (byte)0;
                for (var j = 0; j < Inputs; j++)
                {
                    var noise = (float)((rng.NextDouble() - 0.5) * (anomaly ? 1.2 : 0.4));
                    features[(i * Inputs) + j] = (anomaly ? 1.6f : 0f) + noise;
                }
            }
            return (features, labels);
        }

        private static float[] Evolve(
            float[] features, byte[] labels, AnomalyFitnessOptions options, int generations = 60)
        {
            var evaluator = new AnomalyPopulationEvaluator(features, labels, Inputs, Hidden, options);
            var noise = new PrecomputedNoiseTable(length: 8192, seed: 3);
            using var es = new OpenAiEsStrategy(
                populationSize: 24,
                parameterCount: evaluator.GenomeSize,
                sigma: 0.35f,
                learningRate: 0.12f,
                noiseTable: noise,
                seed: 11);

            // Without this the search centre starts at all-zeros and only noise drives it: fitness stalls, and
            // with a fixed seed two different objectives can even land on a bit-identical best genome.
            es.Initialize();

            using var runner = new EvolutionRunner(es, evaluator);
            for (var g = 0; g < generations; g++)
            {
                runner.RunGeneration();
            }
            return es.GetBestParameters().ToArray();
        }

        private static (int Tp, int Fp, int Fn) Confusion(float[] genome, float[] features, byte[] labels)
        {
            int tp = 0, fp = 0, fn = 0;
            for (var i = 0; i < labels.Length; i++)
            {
                var flagged = AnomalyMlp.IsAnomaly(genome, features.AsSpan(i * Inputs, Inputs), Inputs, Hidden);
                var actual = labels[i] != 0;
                if (flagged && actual) { tp++; }
                else if (flagged) { fp++; }
                else if (actual) { fn++; }
            }
            return (tp, fp, fn);
        }

        [Fact]
        public void EvolvingF1_FindsARealDetector_NotTheNeverFireDegenerate()
        {
            var (features, labels) = MakeData(seed: 42);

            var genome = Evolve(features, labels, new AnomalyFitnessOptions { Objective = AnomalyObjective.F1 });
            var (tp, fp, fn) = Confusion(genome, features, labels);

            // "Never fire" scores 92% accuracy on this data and F1 = 0. Anything above 0.5 F1 means the search
            // found genuine signal rather than exploiting the imbalance.
            Assert.True(tp > 0, "detector never fires — collapsed to the degenerate solution");
            Assert.True(AnomalyFitness.F1(tp, fp, fn) > 0.5f,
                $"F1 too low: tp={tp} fp={fp} fn={fn}");
        }

        [Fact]
        public void AsymmetricCost_BuysRecall_BecauseTheObjectiveSaysMissesAreExpensive()
        {
            var (features, labels) = MakeData(seed: 42);

            // A miss costs 20x a false alarm -> the search should accept more false alarms to miss less.
            var costly = Evolve(features, labels, new AnomalyFitnessOptions
            {
                Objective = AnomalyObjective.Cost, MissCost = 20f, FalseAlarmCost = 1f,
            });
            var balanced = Evolve(features, labels, new AnomalyFitnessOptions { Objective = AnomalyObjective.F1 });

            var (_, _, costlyFn) = Confusion(costly, features, labels);
            var (_, _, balancedFn) = Confusion(balanced, features, labels);

            // This is the claim that gradient training cannot express: the operating point follows the cost model.
            Assert.True(costlyFn <= balancedFn,
                $"expensive-miss objective should miss no more than F1 did: {costlyFn} vs {balancedFn}");
        }

        [Fact]
        public void FalseAlarmBudget_IsRespected_AsAHardConstraint()
        {
            var (features, labels) = MakeData(seed: 7);
            const int budget = 6;

            var genome = Evolve(features, labels, new AnomalyFitnessOptions
            {
                Objective = AnomalyObjective.RecallAtFalseAlarmBudget, FalseAlarmBudget = budget,
            });

            var (tp, fp, _) = Confusion(genome, features, labels);

            // Exceeding the budget scores exactly 0, so any genome the search kept must be inside it — a
            // constraint, not a penalty term. (tp > 0 guards against the trivial "never fire" which also fits.)
            Assert.True(fp <= budget, $"false alarms {fp} exceeded the hard budget {budget}");
            Assert.True(tp > 0, "degenerate: never fires");
        }

        [Fact]
        public void ThresholdIsEvolved_NotHardcoded()
        {
            var (features, labels) = MakeData(seed: 42);

            var f1 = Evolve(features, labels, new AnomalyFitnessOptions { Objective = AnomalyObjective.F1 });
            var costly = Evolve(features, labels, new AnomalyFitnessOptions
            {
                Objective = AnomalyObjective.Cost, MissCost = 20f, FalseAlarmCost = 1f,
            });

            // The threshold is the last gene, optimised jointly with the weights. Two objectives that disagree
            // about the cost of a miss should not land on the same operating point.
            Assert.NotEqual(AnomalyMlp.Threshold(f1), AnomalyMlp.Threshold(costly));
        }

        [Fact]
        public void GenomeSize_CoversWeightsBiasAndTheThreshold_WithNoRedundantOutputBias()
        {
            // W1(5x4) + b1(5) + W2(5) + threshold(1). No output bias: it would be redundant with the threshold
            // and would leave the threshold gene under no selection pressure (see AnomalyMlp docs).
            Assert.Equal(31, AnomalyMlp.GenomeSize(Inputs, Hidden));
        }
    }
}
