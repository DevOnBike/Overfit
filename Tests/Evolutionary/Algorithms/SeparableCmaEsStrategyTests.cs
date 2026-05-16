// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Strategies;

namespace DevOnBike.Overfit.Tests.Evolutionary.Algorithms
{
    public sealed class SeparableCmaEsStrategyTests
    {
        [Fact]
        public void Constructor_ThrowsOnPopulationSizeOfOne()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new SeparableCmaEsStrategy(
                populationSize: 1,
                parameterCount: 4,
                initialSigma: 0.3f,
                seed: 0));
        }

        [Fact]
        public void Constructor_ThrowsOnNonPositiveParameterCount()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new SeparableCmaEsStrategy(
                populationSize: 8,
                parameterCount: 0,
                initialSigma: 0.3f,
                seed: 0));
        }

        [Fact]
        public void Constructor_ThrowsOnNonPositiveSigma()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new SeparableCmaEsStrategy(
                populationSize: 8,
                parameterCount: 4,
                initialSigma: 0f,
                seed: 0));
        }

        [Fact]
        public void DefaultPopulationSize_MatchesLiteratureFormula()
        {
            // 4 + floor(3 * ln(n)).
            Assert.Equal(4, SeparableCmaEsStrategy.DefaultPopulationSize(1));
            Assert.Equal(8, SeparableCmaEsStrategy.DefaultPopulationSize(4));
            Assert.Equal(10, SeparableCmaEsStrategy.DefaultPopulationSize(8));
        }

        [Fact]
        public void Tell_BeforeAsk_Throws()
        {
            using var es = new SeparableCmaEsStrategy(8, 4, 0.3f, seed: 0);

            Assert.Throws<InvalidOperationException>(() => es.Tell(new float[8]));
        }

        [Fact]
        public void GetBestParameters_BeforeFirstTell_IsEmpty()
        {
            using var es = new SeparableCmaEsStrategy(8, 4, 0.3f, seed: 0);

            Assert.True(es.GetBestParameters().IsEmpty);
        }

        [Fact]
        public void Ask_ProducesFinitePopulationOfCorrectLength()
        {
            const int populationSize = 8;
            const int parameterCount = 4;

            using var es = new SeparableCmaEsStrategy(populationSize, parameterCount, 0.3f, seed: 42);
            es.Initialize(min: 0f, max: 0f);

            var pop = new float[populationSize * parameterCount];
            es.Ask(pop);

            foreach (var v in pop)
            {
                Assert.False(float.IsNaN(v), "NaN in sampled population.");
                Assert.False(float.IsInfinity(v), "Inf in sampled population.");
            }
        }

        [Fact]
        public void Tell_RecordsBestFitnessAndBestCandidate()
        {
            const int populationSize = 8;
            const int parameterCount = 4;

            using var es = new SeparableCmaEsStrategy(populationSize, parameterCount, 0.3f, seed: 42);
            es.Initialize(min: 0f, max: 0f);

            var pop = new float[populationSize * parameterCount];
            es.Ask(pop);

            // Index 3 is the unambiguous best.
            float[] fitness = [1f, 2f, 3f, 10f, 1f, 1f, 1f, -5f];
            es.Tell(fitness);

            Assert.Equal(10f, es.BestFitness, 4);

            var expected = pop.AsSpan(3 * parameterCount, parameterCount).ToArray();
            var actual = es.GetBestParameters().ToArray();

            Assert.Equal(expected, actual);
        }

        [Fact]
        public void Tell_IncrementsGeneration()
        {
            using var es = new SeparableCmaEsStrategy(8, 4, 0.3f, seed: 42);
            es.Initialize();

            var pop = new float[8 * 4];
            var fitness = new float[8];

            es.Ask(pop);
            es.Tell(fitness);
            Assert.Equal(1, es.Generation);

            es.Ask(pop);
            es.Tell(fitness);
            Assert.Equal(2, es.Generation);
        }

        [Fact]
        public void Tell_ConvergesOnSphereObjective()
        {
            // Maximise f(theta) = -||theta||^2. Optimum at the origin. CMA-ES converges
            // log-linearly on the sphere: after a couple hundred generations ||mean||^2
            // should be deep below 1e-3 and the step size should have collapsed.
            const int parameterCount = 4;
            const int populationSize = 16;

            using var es = new SeparableCmaEsStrategy(
                populationSize, parameterCount, initialSigma: 0.5f, seed: 123);
            es.Initialize(min: 3f, max: 3f); // mean = [3,3,3,3], ||mean||^2 = 36

            var pop = new float[populationSize * parameterCount];
            var fitness = new float[populationSize];

            for (var gen = 0; gen < 250; gen++)
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

            var mean = es.Mean;
            var finalNormSq = 0f;
            for (var j = 0; j < parameterCount; j++)
            {
                finalNormSq += mean[j] * mean[j];
            }

            Assert.True(finalNormSq < 1e-3f, $"Expected ||mean||^2 < 1e-3, got {finalNormSq}.");
            Assert.True(es.Sigma < 0.5f, $"Expected sigma to shrink below 0.5, got {es.Sigma}.");
        }

        [Fact]
        public void Tell_SolvesRosenbrockValley()
        {
            // The 2-D Rosenbrock function has a curved, correlated valley with its
            // minimum (f = 0) at (1, 1). It is the classic CMA-ES stress test — a
            // diagonal model cannot exploit the correlation, but separable CMA-ES still
            // tracks the valley down to a near-zero objective.
            const int parameterCount = 2;
            const int populationSize = 16;

            using var es = new SeparableCmaEsStrategy(
                populationSize, parameterCount, initialSigma: 0.3f, seed: 2024);
            es.Initialize(min: -1f, max: -1f); // start at (-1, -1)

            var pop = new float[populationSize * parameterCount];
            var fitness = new float[populationSize];

            for (var gen = 0; gen < 1000; gen++)
            {
                es.Ask(pop);

                for (var i = 0; i < populationSize; i++)
                {
                    var x = pop[i * parameterCount];
                    var y = pop[(i * parameterCount) + 1];
                    fitness[i] = -Rosenbrock(x, y);
                }

                es.Tell(fitness);
            }

            var mean = es.Mean;
            var objectiveAtMean = Rosenbrock(mean[0], mean[1]);

            Assert.True(objectiveAtMean < 0.05f,
                $"Expected Rosenbrock(mean) < 0.05, got {objectiveAtMean} at ({mean[0]}, {mean[1]}).");
        }

        [Fact]
        public void Tell_IsInvariantToMonotoneFitnessTransform()
        {
            // Every CMA-ES update depends only on the RANKING of the population, so
            // rescaling the fitness values must leave the trajectory bit-identical.
            // Two strategies with the same seed — one fed f, the other fed 4f — must
            // produce an identical mean. Scaling by a power of two is exact in IEEE
            // float, so it preserves both the order AND the ties of the fitness vector;
            // a gradient-based ES would scale its step by 4 and diverge here.
            const int parameterCount = 4;
            const int populationSize = 12;

            using var plain = new SeparableCmaEsStrategy(populationSize, parameterCount, 0.4f, seed: 7);
            using var transformed = new SeparableCmaEsStrategy(populationSize, parameterCount, 0.4f, seed: 7);

            plain.Initialize(min: 1f, max: 1f);
            transformed.Initialize(min: 1f, max: 1f);

            var popA = new float[populationSize * parameterCount];
            var popB = new float[populationSize * parameterCount];
            var fitnessA = new float[populationSize];
            var fitnessB = new float[populationSize];

            for (var gen = 0; gen < 40; gen++)
            {
                plain.Ask(popA);
                transformed.Ask(popB);

                for (var i = 0; i < populationSize; i++)
                {
                    var sumSq = 0f;
                    for (var j = 0; j < parameterCount; j++)
                    {
                        var v = popA[(i * parameterCount) + j];
                        sumSq += v * v;
                    }

                    fitnessA[i] = -sumSq;
                    fitnessB[i] = 4f * -sumSq; // exact power-of-two rescale
                }

                plain.Tell(fitnessA);
                transformed.Tell(fitnessB);
            }

            var meanPlain = plain.Mean;
            var meanTransformed = transformed.Mean;

            for (var j = 0; j < parameterCount; j++)
            {
                Assert.True(
                    meanPlain[j] == meanTransformed[j],
                    $"mean[{j}]: plain={meanPlain[j]} != transformed={meanTransformed[j]}");
            }
        }

        [Fact]
        public void Checkpoint_RoundTrip_RestoresStateAndContinuationIsIdentical()
        {
            const int parameterCount = 5;
            const int populationSize = 10;

            using var original = new SeparableCmaEsStrategy(populationSize, parameterCount, 0.4f, seed: 99);
            original.Initialize(min: 0.5f, max: 0.5f);

            var pop = new float[populationSize * parameterCount];
            var fitness = new float[populationSize];
            var rng = new Random(5);

            for (var gen = 0; gen < 12; gen++)
            {
                original.Ask(pop);
                for (var i = 0; i < populationSize; i++)
                {
                    fitness[i] = rng.NextSingle();
                }

                original.Tell(fitness);
            }

            // Save between cycles (after a completed Tell).
            using var stream = new MemoryStream();
            var writer = new BinaryWriter(stream);
            original.Save(writer);
            writer.Flush();
            stream.Position = 0;

            using var restored = new SeparableCmaEsStrategy(populationSize, parameterCount, 0.4f, seed: 12345);
            var reader = new BinaryReader(stream);
            restored.Load(reader);

            Assert.Equal(original.Generation, restored.Generation);
            Assert.Equal(original.Sigma, restored.Sigma, 6);
            Assert.Equal(original.BestFitness, restored.BestFitness, 6);
            Assert.Equal(original.Mean.ToArray(), restored.Mean.ToArray());
            Assert.Equal(original.GetBestParameters().ToArray(), restored.GetBestParameters().ToArray());

            // Continuation must be bit-identical: same RNG state + same distribution
            // state => same population => same fitness => same mean.
            var popOriginal = new float[populationSize * parameterCount];
            var popRestored = new float[populationSize * parameterCount];

            original.Ask(popOriginal);
            restored.Ask(popRestored);

            Assert.Equal(popOriginal, popRestored);

            var sharedFitness = new float[populationSize];
            var rng2 = new Random(77);
            for (var i = 0; i < populationSize; i++)
            {
                sharedFitness[i] = rng2.NextSingle();
            }

            original.Tell(sharedFitness);
            restored.Tell(sharedFitness);

            Assert.Equal(original.Mean.ToArray(), restored.Mean.ToArray());
        }

        [Fact]
        public void Tell_HandlesNaNFitness_WithoutPoisoningMean()
        {
            const int populationSize = 8;
            const int parameterCount = 3;

            using var es = new SeparableCmaEsStrategy(populationSize, parameterCount, 0.3f, seed: 42);
            es.Initialize(min: 0f, max: 0f);

            var pop = new float[populationSize * parameterCount];
            es.Ask(pop);

            float[] fitness = [1f, 2f, float.NaN, 3f, 1f, float.NaN, 2f, 1f];
            es.Tell(fitness);

            var mean = es.Mean;
            for (var j = 0; j < parameterCount; j++)
            {
                Assert.False(float.IsNaN(mean[j]), $"mean[{j}] became NaN.");
                Assert.False(float.IsInfinity(mean[j]), $"mean[{j}] became infinite.");
            }
        }

        [Fact]
        public void Tell_IsAllocationStable_AfterWarmup()
        {
            const int populationSize = 32;
            const int parameterCount = 16;

            using var es = new SeparableCmaEsStrategy(populationSize, parameterCount, 0.3f, seed: 42);
            es.Initialize();

            var pop = new float[populationSize * parameterCount];
            var fitness = new float[populationSize];
            var rng = new Random(1);

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
        public void Ask_AfterDispose_Throws()
        {
            var es = new SeparableCmaEsStrategy(8, 4, 0.3f, seed: 0);
            es.Dispose();

            Assert.Throws<ObjectDisposedException>(() => es.Ask(new float[8 * 4]));
        }

        private static float Rosenbrock(float x, float y)
        {
            var a = 1f - x;
            var b = y - (x * x);
            return (a * a) + (100f * b * b);
        }
    }
}
