// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Evolutionary.Fitness;
using DevOnBike.Overfit.Evolutionary.Storage;
using DevOnBike.Overfit.Evolutionary.Strategies;
using static Benchmarks.GenerationalGeneticAlgorithmBenchmarks;

namespace Benchmarks
{
    [Config(typeof(BenchmarkConfig))]
    public class OpenAiEsStrategyBenchmarks
    {
        [Params(256, 1024)]
        public int PopulationSize { get; set; }

        [Params(64, 256, 1024)]
        public int ParameterCount { get; set; }

        [Params(false, true)]
        public bool UseAdam { get; set; }

        private PrecomputedNoiseTable _noiseTable = null!;
        private OpenAiEsStrategy _strategy = null!;
        private float[] _population = null!;
        private float[] _fitness = null!;

        [GlobalSetup]
        public void Setup()
        {
            // Duży zapas, żeby SampleOffset nie wpadał stale w ten sam obszar.
            var noiseLength = Math.Max(ParameterCount * 4096, 1 << 20);

            _noiseTable = new PrecomputedNoiseTable(noiseLength, seed: 12345);
            _strategy = new OpenAiEsStrategy(
                populationSize: PopulationSize,
                parameterCount: ParameterCount,
                sigma: 0.05f,
                learningRate: 0.01f,
                noiseTable: _noiseTable,
                shaper: new CenteredRankFitnessShaper(),
                seed: 123,
                useAdam: UseAdam);

            _strategy.Initialize();

            _population = new float[PopulationSize * ParameterCount];
            _fitness = new float[PopulationSize];

            for (var i = 0; i < _fitness.Length; i++)
            {
                _fitness[i] = (i % 17) - 8;
            }
        }

        [Benchmark]
        public void Ask()
        {
            _strategy.Ask(_population);
        }

        [Benchmark]
        public void Tell()
        {
            _strategy.Tell(_fitness);
        }

        [Benchmark]
        public void AskThenTell()
        {
            _strategy.Ask(_population);
            _strategy.Tell(_fitness);
        }
    }
}