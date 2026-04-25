// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;
using DevOnBike.Overfit.Evolutionary.Abstractions;
using DevOnBike.Overfit.Evolutionary.Fitness;
using DevOnBike.Overfit.Evolutionary.Mutation;
using DevOnBike.Overfit.Evolutionary.Selection;
using DevOnBike.Overfit.Evolutionary.Storage;
using DevOnBike.Overfit.Evolutionary.Strategies;

namespace Benchmarks
{
    /// <summary>
    ///     Side-by-side per-generation cost of the two <see cref="IEvolutionAlgorithm"/>
    ///     implementations currently in the module: the elitist generational GA with Gaussian
    ///     mutation, and OpenAI-style Evolution Strategies with a shared noise table and
    ///     antithetic sampling. Both paths use the centered-rank shaper.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         This benchmark intentionally covers only the algorithmic core (Ask + Tell) and
    ///         keeps fitness synthetic and random. Real training pipelines spend the bulk of
    ///         their wallclock inside fitness evaluation — see <c>ParallelPopulationEvaluator</c>
    ///         for the parallelism layer that dominates real runs. The numbers here answer one
    ///         narrow question: given the same population and genome sizes, which algorithm
    ///         has a cheaper per-generation step?
    ///     </para>
    ///     <para>
    ///         The two algorithms differ structurally in what they do per generation:
    ///         <list type="bullet">
    ///             <item>GA Ask = memcpy of the current population; GA Tell = rank + elite
    ///                   selection + per-element Gaussian sampling for each child.</item>
    ///             <item>ES Ask = per-genome synthesis of <c>μ ± σε</c> using a shared noise
    ///                   table; ES Tell = rank + weighted noise accumulation + μ update.</item>
    ///         </list>
    ///         Consequently Ask is cheaper for GA and Tell tends to be cheaper for ES; the
    ///         Ask+Tell total is what matters in practice.
    ///     </para>
    /// </remarks>
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    public class EvolutionaryAlgorithmBenchmarks
    {
        private const int NoiseTableLength = 1 << 20; // 1M floats = 4 MB, fits any tested parameterCount.
        private const int Seed = 7919;

        private IEvolutionAlgorithm _algorithm = null!;
        private INoiseTable? _noiseTable;
        private float[] _population = null!;
        private float[] _fitness = null!;
        private Random _fitnessRng = null!;

        [Params(256, 1024)]
        public int PopulationSize { get; set; }

        [Params(64, 256)]
        public int ParameterCount { get; set; }

        [Params(Algorithm.GA, Algorithm.ES)]
        public Algorithm Strategy { get; set; }

        public enum Algorithm
        {
            GA,
            ES,
        }

        [GlobalSetup]
        public void Setup()
        {
            _fitnessRng = new Random(Seed);

            _algorithm = Strategy switch
            {
                Algorithm.GA => BuildGeneticAlgorithm(),
                Algorithm.ES => BuildEvolutionStrategies(),
                _ => throw new InvalidOperationException($"Unknown strategy {Strategy}."),
            };

            _population = new float[PopulationSize * ParameterCount];
            _fitness = new float[PopulationSize];

            RandomizeFitness();

            // One warm-up generation so pooled buffers (shaper ranking, ES bookkeeping) are
            // sized before the first measured call.
            _algorithm.Ask(_population);
            _algorithm.Tell(_fitness);
        }

        // Reshuffle fitness before every Tell invocation so the sorter inside the shaper never
        // sees the same permutation twice. Without this the JIT + branch predictor collude to
        // make worst-case sorts look like best-case.
        [IterationSetup(Target = nameof(Tell))]
        public void ShuffleFitnessBeforeTell()
        {
            RandomizeFitness();
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _algorithm.Dispose();

            // ES owns the noise table through us (we built it); GA never touched it.
            if (_noiseTable is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }

        [Benchmark]
        public void Ask()
        {
            _algorithm.Ask(_population);
        }

        [Benchmark]
        public void Tell()
        {
            _algorithm.Tell(_fitness);
        }

        private GenerationalGeneticAlgorithm BuildGeneticAlgorithm()
        {
            var ga = new GenerationalGeneticAlgorithm(
                populationSize: PopulationSize,
                parameterCount: ParameterCount,
                eliteFraction: 0.1f,
                selectionOperator: new TruncationSelectionOperator(),
                mutationOperator: new GaussianMutationOperator(),
                fitnessShaper: new CenteredRankFitnessShaper(),
                seed: Seed);

            ga.Initialize(min: -0.25f, max: 0.25f);
            return ga;
        }

        private OpenAiEsStrategy BuildEvolutionStrategies()
        {
            _noiseTable = new PrecomputedNoiseTable(NoiseTableLength, seed: Seed);

            var es = new OpenAiEsStrategy(
                populationSize: PopulationSize,
                parameterCount: ParameterCount,
                sigma: 0.1f,
                learningRate: 0.01f,
                noiseTable: _noiseTable,
                shaper: new CenteredRankFitnessShaper(),
                seed: Seed);

            es.Initialize(min: -0.25f, max: 0.25f);
            return es;
        }

        private void RandomizeFitness()
        {
            for (var i = 0; i < _fitness.Length; i++)
            {
                _fitness[i] = _fitnessRng.NextSingle() * 1000f;
            }
        }
    }
}