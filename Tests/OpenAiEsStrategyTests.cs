// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Fitness;
using DevOnBike.Overfit.Evolutionary.Storage;
using DevOnBike.Overfit.Evolutionary.Strategies;

namespace DevOnBike.Overfit.Tests
{
    public sealed class OpenAiEsStrategyTests
    {
        [Fact]
        public void Constructor_ThrowsOnOddPopulationSize()
        {
            var noise = new PrecomputedNoiseTable(length: 1024, seed: 1);

            Assert.Throws<ArgumentException>(() => new OpenAiEsStrategy(
                populationSize: 7,
                parameterCount: 4,
                sigma: 0.1f,
                learningRate: 0.01f,
                noiseTable: noise));
        }

        [Fact]
        public void Constructor_ThrowsOnNonPositiveSigma()
        {
            var noise = new PrecomputedNoiseTable(length: 1024, seed: 1);

            Assert.Throws<ArgumentOutOfRangeException>(() => new OpenAiEsStrategy(
                populationSize: 8,
                parameterCount: 4,
                sigma: 0f,
                learningRate: 0.01f,
                noiseTable: noise));
        }

        [Fact]
        public void Constructor_ThrowsOnNonPositiveLearningRate()
        {
            var noise = new PrecomputedNoiseTable(length: 1024, seed: 1);

            Assert.Throws<ArgumentOutOfRangeException>(() => new OpenAiEsStrategy(
                populationSize: 8,
                parameterCount: 4,
                sigma: 0.1f,
                learningRate: -0.01f,
                noiseTable: noise));
        }

        [Fact]
        public void Constructor_ThrowsWhenNoiseTableIsTooShort()
        {
            var noise = new PrecomputedNoiseTable(length: 10, seed: 1);

            Assert.Throws<ArgumentException>(() => new OpenAiEsStrategy(
                populationSize: 8,
                parameterCount: 32,
                sigma: 0.1f,
                learningRate: 0.01f,
                noiseTable: noise));
        }

        [Fact]
        public void Ask_ProducesAntitheticPairsAroundMean()
        {
            const int populationSize = 8;
            const int parameterCount = 4;

            var noise = new PrecomputedNoiseTable(length: 1024, seed: 1);
            using var es = new OpenAiEsStrategy(
                populationSize,
                parameterCount,
                sigma: 0.5f,
                learningRate: 0.01f,
                noiseTable: noise,
                seed: 42);

            es.Initialize(min: 0.25f, max: 0.25f); // μ = [0.25, 0.25, 0.25, 0.25]

            var pop = new float[populationSize * parameterCount];
            es.Ask(pop);

            var mu = es.Mean;

            // For each antithetic pair p, (child[2p] + child[2p+1]) / 2 must equal μ,
            // because the pair is (μ + σε, μ − σε).
            for (var p = 0; p < populationSize / 2; p++)
            {
                for (var j = 0; j < parameterCount; j++)
                {
                    var plus = pop[(2 * p * parameterCount) + j];
                    var minus = pop[(((2 * p) + 1) * parameterCount) + j];
                    var avg = 0.5f * (plus + minus);

                    Assert.Equal(mu[j], avg, 4);
                }
            }
        }

        [Fact]
        public void GetBestParameters_BeforeFirstTell_IsEmpty()
        {
            var noise = new PrecomputedNoiseTable(length: 1024, seed: 1);
            using var es = new OpenAiEsStrategy(
                populationSize: 8,
                parameterCount: 4,
                sigma: 0.1f,
                learningRate: 0.01f,
                noiseTable: noise);

            Assert.True(es.GetBestParameters().IsEmpty);
        }

        [Fact]
        public void Tell_RecordsBestFitnessAndBestCandidate()
        {
            const int populationSize = 8;
            const int parameterCount = 4;

            var noise = new PrecomputedNoiseTable(length: 1024, seed: 1);
            using var es = new OpenAiEsStrategy(
                populationSize,
                parameterCount,
                sigma: 0.1f,
                learningRate: 0.01f,
                noiseTable: noise,
                seed: 42);

            es.Initialize(min: 0f, max: 0f);

            var pop = new float[populationSize * parameterCount];
            es.Ask(pop);

            // Artificial fitness vector: index 3 is the best, index 7 is the worst.
            float[] fitness = [1f, 2f, 3f, 10f, 1f, 1f, 1f, -5f];

            es.Tell(fitness);

            Assert.Equal(10f, es.BestFitness, 4);

            // The recorded best candidate must equal the actual genome at index 3 in the
            // population we sampled.
            var expected = pop.AsSpan(3 * parameterCount, parameterCount).ToArray();
            var actual = es.GetBestParameters().ToArray();

            Assert.Equal(expected, actual);
        }

        [Fact]
        public void Tell_IncrementsGeneration()
        {
            var noise = new PrecomputedNoiseTable(length: 1024, seed: 1);
            using var es = new OpenAiEsStrategy(
                populationSize: 8,
                parameterCount: 4,
                sigma: 0.1f,
                learningRate: 0.01f,
                noiseTable: noise,
                seed: 42);

            es.Initialize();
            var pop = new float[8 * 4];
            es.Ask(pop);
            var fitness = new float[8];

            es.Tell(fitness);
            Assert.Equal(1, es.Generation);

            es.Ask(pop);
            es.Tell(fitness);
            Assert.Equal(2, es.Generation);
        }

        [Fact]
        public void Tell_ConvergesOnQuadraticObjective()
        {
            // Integration test: optimise f(θ) = −‖θ‖² with ES. The global maximum is at
            // θ = 0, so ‖μ‖² should decrease monotonically in expectation. Start from a
            // non-trivial vector, run 100 generations, and assert meaningful convergence.
            const int parameterCount = 4;
            const int populationSize = 64;

            var noise = new PrecomputedNoiseTable(length: 1 << 16, seed: 2024);
            using var es = new OpenAiEsStrategy(
                populationSize,
                parameterCount,
                sigma: 0.1f,
                learningRate: 0.05f,
                noiseTable: noise,
                shaper: new CenteredRankFitnessShaper(),
                seed: 123);

            es.Initialize(min: 1f, max: 1f); // μ = [1, 1, 1, 1], ‖μ‖² = 4

            var pop = new float[populationSize * parameterCount];
            var fitness = new float[populationSize];

            for (var gen = 0; gen < 100; gen++)
            {
                es.Ask(pop);

                for (var i = 0; i < populationSize; i++)
                {
                    var sumSq = 0f;
                    for (var j = 0; j < parameterCount; j++)
                    {
                        var v = pop[(i * parameterCount) + j];
                        sumSq += v * v;
                    }

                    fitness[i] = -sumSq;
                }

                es.Tell(fitness);
            }

            var mu = es.Mean;
            var finalNormSq = 0f;
            for (var i = 0; i < parameterCount; i++)
            {
                finalNormSq += mu[i] * mu[i];
            }

            // Initial ‖μ‖² = 4. After 100 gens we expect convergence well below 1.
            Assert.True(finalNormSq < 1f, $"Expected ‖μ‖² < 1, got {finalNormSq}.");
        }

        [Fact]
        public void Tell_HandlesNaNFitness_WithoutPoisoningMean()
        {
            // A NaN in the fitness vector must not corrupt μ. Centered-rank shaping ranks
            // NaN as worst, so the gradient update uses only the finite samples.
            const int populationSize = 8;
            const int parameterCount = 3;

            var noise = new PrecomputedNoiseTable(length: 1024, seed: 1);
            using var es = new OpenAiEsStrategy(
                populationSize,
                parameterCount,
                sigma: 0.1f,
                learningRate: 0.01f,
                noiseTable: noise,
                seed: 42);

            es.Initialize(min: 0f, max: 0f);

            var pop = new float[populationSize * parameterCount];
            es.Ask(pop);

            float[] fitness = [1f, 2f, float.NaN, 3f, 1f, float.NaN, 2f, 1f];

            es.Tell(fitness);

            var mu = es.Mean;
            for (var j = 0; j < parameterCount; j++)
            {
                Assert.False(float.IsNaN(mu[j]), $"μ[{j}] became NaN after Tell with NaN fitness.");
                Assert.False(float.IsInfinity(mu[j]), $"μ[{j}] became infinite after Tell with NaN fitness.");
            }
        }

        [Fact]
        public void Tell_IsAllocationStable_AfterWarmup()
        {
            const int populationSize = 32;
            const int parameterCount = 16;

            var noise = new PrecomputedNoiseTable(length: 1 << 14, seed: 1);
            using var es = new OpenAiEsStrategy(
                populationSize,
                parameterCount,
                sigma: 0.1f,
                learningRate: 0.01f,
                noiseTable: noise,
                shaper: new CenteredRankFitnessShaper(),
                seed: 42);

            es.Initialize();

            var pop = new float[populationSize * parameterCount];
            var fitness = new float[populationSize];
            var rng = new Random(1);

            // Warmup: populate pooled ranking buffer inside the shaper and JIT paths.
            for (var i = 0; i < fitness.Length; i++)
            {
                fitness[i] = rng.NextSingle();
            }

            es.Ask(pop);
            es.Tell(fitness);

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var iter = 0; iter < 1_000; iter++)
            {
                es.Ask(pop);

                for (var i = 0; i < fitness.Length; i++)
                {
                    fitness[i] = rng.NextSingle();
                }

                es.Tell(fitness);
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            var perCall = allocated / 1_000.0;

            Assert.True(perCall <= 64, $"Ask+Tell allocated {perCall:F1} bytes per iteration ({allocated} total).");
        }

        [Fact]
        public void AskOrTell_AfterDispose_Throws()
        {
            var noise = new PrecomputedNoiseTable(length: 1024, seed: 1);
            var es = new OpenAiEsStrategy(
                populationSize: 8,
                parameterCount: 4,
                sigma: 0.1f,
                learningRate: 0.01f,
                noiseTable: noise);

            es.Dispose();

            var pop = new float[8 * 4];
            Assert.Throws<ObjectDisposedException>(() => es.Ask(pop));
        }
    }
}