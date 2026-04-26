// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Fitness;
using DevOnBike.Overfit.Evolutionary.Storage;
using DevOnBike.Overfit.Evolutionary.Strategies;

namespace Benchmarks
{
    /// <summary>
    /// OpenAI-ES strategy benchmarks.
    ///
    /// Notes:
    ///
    /// - Ask measures population generation.
    /// - AskThenTell measures one complete strategy cycle.
    /// - Standalone Tell is intentionally not benchmarked here because Tell requires
    ///   a preceding Ask generation state. Benchmarking Tell alone produces invalid
    ///   NA results unless every invocation is prepared with Ask, which would no
    ///   longer isolate Tell cleanly under OperationsPerInvoke.
    ///
    /// These benchmarks are allocation checks for strategy buffers. They are not
    /// inference benchmarks.
    /// </summary>
    public abstract class OpenAiEsStrategyBenchmarkBase
    {
        private PrecomputedNoiseTable _noiseTable = null!;
        private OpenAiEsStrategy _strategy = null!;
        private float[] _population = null!;
        private float[] _fitness = null!;

        protected void SetupCore(
            int populationSize,
            int parameterCount,
            bool useAdam)
        {
            PopulationSize = populationSize;
            ParameterCount = parameterCount;
            UseAdam = useAdam;

            var noiseLength = Math.Max(
                parameterCount * 4096,
                1 << 20);

            _noiseTable = new PrecomputedNoiseTable(
                noiseLength,
                seed: 12345);

            _strategy = new OpenAiEsStrategy(
                populationSize: populationSize,
                parameterCount: parameterCount,
                sigma: 0.05f,
                learningRate: 0.01f,
                noiseTable: _noiseTable,
                shaper: new CenteredRankFitnessShaper(),
                seed: 123,
                useAdam: useAdam);

            _strategy.Initialize();

            _population = new float[populationSize * parameterCount];
            _fitness = new float[populationSize];

            FillFitness(_fitness);

            // Warmup outside measured region.
            for (var i = 0; i < 16; i++)
            {
                _strategy.Ask(_population);
                _strategy.Tell(_fitness);
            }
        }

        protected int PopulationSize { get; private set; }

        protected int ParameterCount { get; private set; }

        protected bool UseAdam { get; private set; }

        protected float AskCore(
            int iterations)
        {
            var checksum = 0f;

            for (var i = 0; i < iterations; i++)
            {
                _strategy.Ask(_population);

                checksum += _population[0];
                checksum += _population[_population.Length - 1];
            }

            return checksum;
        }

        protected float AskThenTellCore(
            int iterations)
        {
            var checksum = 0f;

            for (var i = 0; i < iterations; i++)
            {
                _strategy.Ask(_population);
                _strategy.Tell(_fitness);

                checksum += _population[0];
                checksum += _population[_population.Length - 1];
            }

            return checksum;
        }

        private static void FillFitness(
            float[] fitness)
        {
            for (var i = 0; i < fitness.Length; i++)
            {
                // Deterministic non-flat fitness distribution.
                fitness[i] = (i % 17) - 8;
            }
        }
    }

}