using DevOnBike.Overfit.Evolutionary.Abstractions;
using DevOnBike.Overfit.Evolutionary.Storage;
using DevOnBike.Overfit.Maths;

namespace DevOnBike.Overfit.Evolutionary.Strategies
{
    /// <summary>
    ///     (mu + lambda)-style generational genetic algorithm with truncation elitism
    ///     and pluggable selection, mutation and fitness-shaping operators.
    ///     Publishes the standard Ask/Tell black-box API so the same loop can be driven
    ///     from any fitness evaluator.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         All per-generation state lives in a pooled <see cref="EvolutionWorkspace"/>;
    ///         steady-state Ask/Tell calls perform zero managed allocations.
    ///     </para>
    ///     <para>
    ///         Ranking uses an O(n log k) partial sort so the cost of extracting the elite
    ///         set scales with the elite count rather than the population size. NaN fitness
    ///         values are tolerated: they always rank as worst and can never displace a
    ///         valid individual from the elite set.
    ///     </para>
    ///     <para>
    ///         Child creation is sequential by design. The per-child work (selection + mutation)
    ///         is measured in hundreds of nanoseconds for typical genomes, which is far below
    ///         the threshold at which parallel dispatch overhead pays off. Parallelism is instead
    ///         placed at the fitness-evaluation boundary, where each genome evaluation routinely
    ///         takes microseconds to milliseconds and the dispatch cost is amortized over real
    ///         work. See <c>ParallelPopulationEvaluator</c> for that integration point.
    ///     </para>
    /// </remarks>
    public sealed class GenerationalGeneticAlgorithm : IEvolutionAlgorithm
    {
        private readonly EvolutionWorkspace _workspace;
        private readonly ISelectionOperator _selectionOperator;
        private readonly IMutationOperator _mutationOperator;
        private readonly IFitnessShaper? _fitnessShaper;
        private readonly Random _rng;
        private readonly int _eliteCount;
        private readonly float[] _bestParameters;

        private float _bestFitness;
        private bool _disposed;
        private bool _hasFitness;

        public GenerationalGeneticAlgorithm(
            int populationSize,
            int parameterCount,
            float eliteFraction,
            ISelectionOperator selectionOperator,
            IMutationOperator mutationOperator,
            IFitnessShaper? fitnessShaper = null,
            int? seed = null)
        {
            if (populationSize <= 1)
            {
                throw new ArgumentOutOfRangeException(nameof(populationSize), "Population size must be greater than 1.");
            }

            if (parameterCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(parameterCount), "Parameter count must be greater than 0.");
            }

            if (eliteFraction is <= 0f or > 1f)
            {
                throw new ArgumentOutOfRangeException(nameof(eliteFraction), "Elite fraction must be in range (0, 1].");
            }

            _selectionOperator = selectionOperator ?? throw new ArgumentNullException(nameof(selectionOperator));
            _mutationOperator = mutationOperator ?? throw new ArgumentNullException(nameof(mutationOperator));
            _fitnessShaper = fitnessShaper;
            _rng = seed.HasValue ? new Random(seed.Value) : new Random();

            PopulationSize = populationSize;
            ParameterCount = parameterCount;
            Generation = 0;

            _eliteCount = Math.Max(1, (int)MathF.Round(populationSize * eliteFraction));
            _workspace = new EvolutionWorkspace(populationSize, parameterCount, clearMemory: false);

            _bestParameters = new float[parameterCount];
            _bestFitness = float.NaN;
            _hasFitness = false;
        }

        public int PopulationSize { get; }

        public int ParameterCount { get; }

        public int Generation { get; private set; }

        public float BestFitness
        {
            get
            {
                ThrowIfDisposed();
                return _bestFitness;
            }
        }

        public void Initialize(float min = -0.3f, float max = 0.3f)
        {
            ThrowIfDisposed();

            if (min > max)
            {
                throw new ArgumentException("min cannot be greater than max.");
            }

            var population = _workspace.Population.GetView().AsSpan();

            for (var i = 0; i < population.Length; i++)
            {
                population[i] = (float)(_rng.NextDouble() * (max - min) + min);
            }

            _hasFitness = false;
            _bestFitness = float.NaN;
            Generation = 0;
        }

        public void Ask(Span<float> populationMatrix)
        {
            ThrowIfDisposed();

            var population = _workspace.Population.GetView().AsReadOnlySpan();

            if (populationMatrix.Length != population.Length)
            {
                throw new ArgumentException(
                    $"populationMatrix length must be {population.Length}.",
                    nameof(populationMatrix));
            }

            population.CopyTo(populationMatrix);
        }

        public void Tell(ReadOnlySpan<float> fitness)
        {
            ThrowIfDisposed();

            if (fitness.Length != PopulationSize)
            {
                throw new ArgumentException(
                    $"fitness length must be {PopulationSize}.",
                    nameof(fitness));
            }

            fitness.CopyTo(_workspace.Fitness.GetView().AsSpan());
            _hasFitness = true;

            RankPopulation();
            StoreBestGenome();
            RebuildPopulation();

            Generation++;
        }

        public ReadOnlySpan<float> GetBestParameters()
        {
            ThrowIfDisposed();

            if (!_hasFitness)
            {
                return ReadOnlySpan<float>.Empty;
            }

            return _bestParameters;
        }

        private void RankPopulation()
        {
            var rankingFitness = _workspace.Fitness.GetView().AsReadOnlySpan();

            if (_fitnessShaper is not null)
            {
                var shapedFitness = _workspace.ShapedFitness.GetView().AsSpan();
                _fitnessShaper.Shape(rankingFitness, shapedFitness);
                rankingFitness = _workspace.ShapedFitness.GetView().AsReadOnlySpan();
            }

            var ranking = _workspace.Ranking;

            for (var i = 0; i < ranking.Length; i++)
            {
                ranking[i] = i;
            }

            // O(n log k) partial sort. Only the top _eliteCount positions end up ordered;
            // the remaining population is left in an unspecified order, which is fine because
            // we only read from EliteIndices below. NaN fitness values are tolerated.
            PartialSort.TopKDescending(ranking, rankingFitness, _eliteCount);

            var elite = _workspace.EliteIndices.AsSpan(0, _eliteCount);
            ranking.AsSpan(0, _eliteCount).CopyTo(elite);
        }

        private void StoreBestGenome()
        {
            var bestIndex = _workspace.EliteIndices[0];
            ValidateGenomeIndex(bestIndex);

            _bestFitness = _workspace.Fitness.GetView().AsReadOnlySpan()[bestIndex];

            var population = _workspace.Population.GetView().AsReadOnlySpan();
            var bestGenome = population.Slice(bestIndex * ParameterCount, ParameterCount);
            bestGenome.CopyTo(_bestParameters);
        }

        private void RebuildPopulation()
        {
            var currentPopulation = _workspace.Population.GetView().AsReadOnlySpan();
            var nextPopulation = _workspace.NextPopulation.GetView().AsSpan();
            var eliteIndices = _workspace.EliteIndices.AsSpan(0, _eliteCount);

            CopyElites(currentPopulation, nextPopulation, eliteIndices);
            CreateChildren(currentPopulation, nextPopulation, eliteIndices);

            _workspace.SwapPopulations();
        }

        private void CopyElites(
            ReadOnlySpan<float> currentPopulation,
            Span<float> nextPopulation,
            ReadOnlySpan<int> eliteIndices)
        {
            for (var i = 0; i < eliteIndices.Length; i++)
            {
                var sourceIndex = eliteIndices[i];
                ValidateGenomeIndex(sourceIndex);

                var sourceGenome = currentPopulation.Slice(sourceIndex * ParameterCount, ParameterCount);
                var targetGenome = nextPopulation.Slice(i * ParameterCount, ParameterCount);

                sourceGenome.CopyTo(targetGenome);
            }
        }

        private void CreateChildren(
            ReadOnlySpan<float> currentPopulation,
            Span<float> nextPopulation,
            ReadOnlySpan<int> eliteIndices)
        {
            if (eliteIndices.Length == 0)
            {
                throw new InvalidOperationException("Elite set cannot be empty.");
            }

            for (var i = _eliteCount; i < PopulationSize; i++)
            {
                var parentIndex = _selectionOperator.SelectParent(eliteIndices, _rng);
                ValidateGenomeIndex(parentIndex);

                var parentGenome = currentPopulation.Slice(parentIndex * ParameterCount, ParameterCount);
                var childGenome = nextPopulation.Slice(i * ParameterCount, ParameterCount);

                _mutationOperator.Mutate(parentGenome, childGenome, _rng);
            }
        }

        private void ValidateGenomeIndex(int index)
        {
            if ((uint)index >= (uint)PopulationSize)
            {
                throw new InvalidOperationException($"Invalid genome index: {index}.");
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _workspace.Dispose();
            _disposed = true;
        }

        // -----------------------------------------------------------------------------
        // Checkpoint: IEvolutionCheckpoint.Save / Load
        // Format (little-endian, BinaryWriter defaults):
        //   int32   magic           = 0x47474101 ('G','G','A',0x01)
        //   int32   schemaVersion   = 1
        //   int32   populationSize
        //   int32   parameterCount
        //   int32   eliteCount
        //   int32   generation
        //   byte    hasFitness
        //   float   bestFitness
        //   float[] bestParameters  [parameterCount]
        //   float[] population      [populationSize * parameterCount]
        //   float[] fitness         [populationSize]          (only when hasFitness == 1)
        //   float[] shapedFitness   [populationSize]          (only when hasFitness == 1)
        //   int32[] ranking         [populationSize]          (only when hasFitness == 1)
        //   int32[] eliteIndices    [eliteCount]              (only when hasFitness == 1)
        // -----------------------------------------------------------------------------

        private const int CheckpointMagic = 0x47474101;
        private const int CheckpointSchemaVersion = 1;

        public void Save(BinaryWriter writer)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(writer);

            writer.Write(CheckpointMagic);
            writer.Write(CheckpointSchemaVersion);
            writer.Write(PopulationSize);
            writer.Write(ParameterCount);
            writer.Write(_eliteCount);
            writer.Write(Generation);
            writer.Write(_hasFitness);
            writer.Write(_bestFitness);

            WriteFloats(writer, _bestParameters);
            WriteFloats(writer, _workspace.Population.GetView().AsReadOnlySpan());

            if (_hasFitness)
            {
                WriteFloats(writer, _workspace.Fitness.GetView().AsReadOnlySpan());
                WriteFloats(writer, _workspace.ShapedFitness.GetView().AsReadOnlySpan());
                WriteInts(writer, _workspace.Ranking);
                WriteInts(writer, _workspace.EliteIndices.AsSpan(0, _eliteCount));
            }
        }

        public void Load(BinaryReader reader)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(reader);

            var magic = reader.ReadInt32();

            if (magic != CheckpointMagic)
            {
                throw new InvalidDataException(
                    $"Expected magic 0x{CheckpointMagic:X8}, found 0x{magic:X8}. Stream was not produced by GenerationalGeneticAlgorithm.");
            }

            var schemaVersion = reader.ReadInt32();

            if (schemaVersion != CheckpointSchemaVersion)
            {
                throw new InvalidDataException(
                    $"Unsupported schema version {schemaVersion}; this build supports {CheckpointSchemaVersion}.");
            }

            var populationSize = reader.ReadInt32();
            var parameterCount = reader.ReadInt32();
            var eliteCount = reader.ReadInt32();

            if (populationSize != PopulationSize || parameterCount != ParameterCount || eliteCount != _eliteCount)
            {
                throw new InvalidDataException(
                    $"Checkpoint was produced for ({populationSize}, {parameterCount}, elite={eliteCount}); " +
                    $"current instance is ({PopulationSize}, {ParameterCount}, elite={_eliteCount}).");
            }

            Generation = reader.ReadInt32();
            _hasFitness = reader.ReadBoolean();
            _bestFitness = reader.ReadSingle();

            ReadFloats(reader, _bestParameters);
            ReadFloats(reader, _workspace.Population.GetView().AsSpan());

            if (_hasFitness)
            {
                ReadFloats(reader, _workspace.Fitness.GetView().AsSpan());
                ReadFloats(reader, _workspace.ShapedFitness.GetView().AsSpan());
                ReadInts(reader, _workspace.Ranking);
                ReadInts(reader, _workspace.EliteIndices.AsSpan(0, _eliteCount));
            }
        }

        private static void WriteFloats(BinaryWriter writer, ReadOnlySpan<float> values)
        {
            for (var i = 0; i < values.Length; i++)
            {
                writer.Write(values[i]);
            }
        }

        private static void WriteInts(BinaryWriter writer, ReadOnlySpan<int> values)
        {
            for (var i = 0; i < values.Length; i++)
            {
                writer.Write(values[i]);
            }
        }

        private static void ReadFloats(BinaryReader reader, Span<float> destination)
        {
            for (var i = 0; i < destination.Length; i++)
            {
                destination[i] = reader.ReadSingle();
            }
        }

        private static void ReadInts(BinaryReader reader, Span<int> destination)
        {
            for (var i = 0; i < destination.Length; i++)
            {
                destination[i] = reader.ReadInt32();
            }
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }
}