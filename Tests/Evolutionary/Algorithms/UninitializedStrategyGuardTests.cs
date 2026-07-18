// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Storage;
using DevOnBike.Overfit.Evolutionary.Strategies;
using DevOnBike.Overfit.Exceptions;

namespace DevOnBike.Overfit.Tests.Evolutionary.Algorithms
{
    /// <summary>
    /// Pins the guard against a silent-failure mode that is genuinely hard to spot: skipping <c>Initialize()</c>
    /// leaves the search centre / population at all zeros, so the run does not crash — it just converges badly.
    /// With a fixed seed it can even return a bit-identical "best" genome for two different objectives, which
    /// reads like a design flaw in the fitness function rather than a missing setup call. Failing loudly at
    /// <c>Ask()</c> is the whole point, so it needs a test.
    /// </summary>
    public sealed class UninitializedStrategyGuardTests
    {
        private const int Population = 8;
        private const int Parameters = 6;

        [Fact]
        public void OpenAiEs_Ask_ThrowsWhenNotInitialized()
        {
            var noise = new PrecomputedNoiseTable(length: 4096, seed: 1);
            using var es = new OpenAiEsStrategy(
                populationSize: Population, parameterCount: Parameters,
                sigma: 0.1f, learningRate: 0.05f, noiseTable: noise, seed: 1);

            var buffer = new float[Population * Parameters];

            var ex = Assert.Throws<OverfitRuntimeException>(() => es.Ask(buffer));
            Assert.Contains("Initialize", ex.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void SeparableCmaEs_Ask_ThrowsWhenNotInitialized()
        {
            using var cma = new SeparableCmaEsStrategy(
                populationSize: Population, parameterCount: Parameters, initialSigma: 0.1f, seed: 1);

            var buffer = new float[Population * Parameters];

            Assert.Throws<OverfitRuntimeException>(() => cma.Ask(buffer));
        }

        [Fact]
        public void Ask_SucceedsAfterInitialize()
        {
            var noise = new PrecomputedNoiseTable(length: 4096, seed: 1);
            using var es = new OpenAiEsStrategy(
                populationSize: Population, parameterCount: Parameters,
                sigma: 0.1f, learningRate: 0.05f, noiseTable: noise, seed: 1);

            es.Initialize();

            var buffer = new float[Population * Parameters];
            es.Ask(buffer);

            // The centre is seeded from a non-degenerate range, so the sampled population is not all zeros —
            // this is exactly what the uninitialised path silently failed to provide.
            Assert.Contains(buffer, static v => v != 0f);
        }
    }
}
