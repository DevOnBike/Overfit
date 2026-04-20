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
            //
            // Pinned to plain SGD so the learning rate and expected convergence curve are
            // decoupled from Adam's bias-correction schedule — the Adam variant has a
            // separate convergence test below.
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
                seed: 123,
                useAdam: false);

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
        public void Tell_WithAdam_FirstStepMagnitudeMatchesLearningRate()
        {
            // Direct test of Adam's core scale-invariance property. After the very first
            // Adam step with fresh moments:
            //
            //   m₁ = (1−β₁)·g
            //   v₁ = (1−β₂)·g²
            //   m̂₁ = m₁ / (1−β₁) = g
            //   v̂₁ = v₁ / (1−β₂) = g²
            //   step = lr · m̂₁ / (√v̂₁ + ε) = lr · g / (|g| + ε) ≈ lr · sign(g)
            //
            // So after one generation each component of (μ − μ₀) has magnitude ≈ lr,
            // regardless of the gradient's scale. That is the property Adam provides and
            // the property users rely on when tuning its learning rate. We assert it
            // directly: per-component |μ_j − μ_j,0| ∈ [0.5·lr, 1.5·lr] after step 1.
            //
            // This test does NOT depend on the problem structure, the seed, or SGD. It is
            // a pure contract test for the Adam update rule as implemented.
            const int parameterCount = 4;
            const int populationSize = 64;
            const float lr = 0.01f;

            var noise = new PrecomputedNoiseTable(length: 1 << 16, seed: 2024);
            using var adam = new OpenAiEsStrategy(
                populationSize, parameterCount, sigma: 0.1f, learningRate: lr,
                noiseTable: noise, seed: 123, useAdam: true);

            adam.Initialize(min: 1f, max: 1f);
            var muBefore = adam.Mean.ToArray();

            var pop = new float[populationSize * parameterCount];
            var fitness = new float[populationSize];

            adam.Ask(pop);

            // Use the quadratic bowl so that the ES gradient estimate has non-trivial magnitude
            // in every dimension — otherwise m̂ would be zero in a component and the step would
            // be zero there, which is not the scale-invariance regime we're probing.
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

            adam.Tell(fitness);

            var muAfter = adam.Mean;

            for (var j = 0; j < parameterCount; j++)
            {
                var delta = MathF.Abs(muAfter[j] - muBefore[j]);

                Assert.InRange(delta, 0.5f * lr, 1.5f * lr);
            }
        }

        [Fact]
        public void Tell_WithAdam_MeanStaysFinite_OnHighGradientMagnitude()
        {
            // Guard against Adam's moment accumulators amplifying a pathological gradient.
            // If the denominator √v̂ + ε approaches zero while m̂ is large, the step can blow
            // up. Feed wildly varying fitness for several generations and assert μ stays
            // finite throughout.
            const int parameterCount = 8;
            const int populationSize = 16;

            var noise = new PrecomputedNoiseTable(length: 1 << 14, seed: 5);
            using var es = new OpenAiEsStrategy(
                populationSize, parameterCount, sigma: 0.1f, learningRate: 0.1f,
                noiseTable: noise, seed: 42, useAdam: true);

            es.Initialize();
            var pop = new float[populationSize * parameterCount];
            var fitness = new float[populationSize];
            var rng = new Random(99);

            for (var gen = 0; gen < 50; gen++)
            {
                es.Ask(pop);

                // Alternate between very large and very small fitness values each generation.
                var scale = (gen & 1) == 0 ? 1e6f : 1e-6f;
                for (var i = 0; i < populationSize; i++)
                {
                    fitness[i] = (rng.NextSingle() - 0.5f) * scale;
                }

                es.Tell(fitness);
            }

            var mu = es.Mean;
            for (var j = 0; j < parameterCount; j++)
            {
                Assert.False(float.IsNaN(mu[j]), $"μ[{j}] is NaN.");
                Assert.False(float.IsInfinity(mu[j]), $"μ[{j}] is infinite.");
            }
        }

        [Fact]
        public void Initialize_ResetsAdamMomentsAcrossTrainingRuns()
        {
            // Calling Initialize after a completed training run must wipe m/v/step so the
            // second run behaves like a fresh start. Without this reset, leftover moment
            // accumulators from the first run would distort the first few Adam updates.
            const int parameterCount = 4;
            const int populationSize = 16;

            var noise = new PrecomputedNoiseTable(length: 1024, seed: 1);
            using var es = new OpenAiEsStrategy(
                populationSize, parameterCount, sigma: 0.1f, learningRate: 0.05f,
                noiseTable: noise, seed: 123, useAdam: true);

            // First run — train down to small ‖μ‖² and accumulate non-trivial moments.
            es.Initialize(min: 1f, max: 1f);
            var pop = new float[populationSize * parameterCount];
            var fitness = new float[populationSize];

            for (var gen = 0; gen < 30; gen++)
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

            // Second run — Initialize must behave as if the first never happened.
            es.Initialize(min: 1f, max: 1f);
            var muAfterReinit = es.Mean.ToArray();

            Assert.Equal(0, es.Generation);
            for (var j = 0; j < parameterCount; j++)
            {
                Assert.Equal(1f, muAfterReinit[j], 6);
            }

            // Run one step with a known gradient direction; the first Adam update with
            // fresh moments has a predictable sign. With leftover moments, the sign could
            // flip in the first step.
            es.Ask(pop);
            for (var i = 0; i < populationSize; i++)
            {
                // Fitness rewards smaller ‖θ‖² — gradient points toward origin, so μ must
                // move toward origin on every step under a healthy optimizer.
                var sumSq = 0f;
                for (var j = 0; j < parameterCount; j++)
                {
                    var v = pop[(i * parameterCount) + j];
                    sumSq += v * v;
                }
                fitness[i] = -sumSq;
            }
            es.Tell(fitness);

            var muAfterOneStep = es.Mean;
            var movedTowardOrigin = true;
            for (var j = 0; j < parameterCount; j++)
            {
                if (muAfterOneStep[j] >= 1f)
                {
                    movedTowardOrigin = false;
                    break;
                }
            }
            Assert.True(movedTowardOrigin, "First Adam step after re-Initialize did not move μ toward the origin.");
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