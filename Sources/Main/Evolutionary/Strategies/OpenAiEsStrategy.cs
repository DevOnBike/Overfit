using DevOnBike.Overfit.Evolutionary.Abstractions;
using DevOnBike.Overfit.Evolutionary.Fitness;

namespace DevOnBike.Overfit.Evolutionary.Strategies
{
    /// <summary>
    ///     Natural Evolution Strategies optimizer in the style of
    ///     "Evolution Strategies as a Scalable Alternative to Reinforcement Learning"
    ///     (Salimans et al., 2017). Maintains a single mean parameter vector and estimates
    ///     the search gradient each generation from population evaluations of antithetically
    ///     mirrored perturbations sampled from a shared <see cref="INoiseTable"/>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Per generation:
    ///         <list type="number">
    ///             <item>Draw <c>populationSize / 2</c> independent noise offsets into the table.</item>
    ///             <item>Emit the antithetic pair <c>(μ + σε, μ − σε)</c> for each offset.</item>
    ///             <item>Consume fitness, shape it through the supplied <see cref="IFitnessShaper"/>
    ///                   (centered-rank by default), estimate the gradient
    ///                   <c>∇ ≈ (1 / (N·σ)) · Σ_pairs (f⁺ − f⁻)·ε</c>, and update
    ///                   <c>μ ← μ + α·∇</c>.</item>
    ///         </list>
    ///         Antithetic sampling halves the variance of the gradient estimate for the same
    ///         number of evaluations, which is the main correctness advantage over the
    ///         textbook ES sampler.
    ///     </para>
    ///     <para>
    ///         Steady-state <see cref="Ask"/> and <see cref="Tell"/> perform zero managed
    ///         allocations. All per-generation state lives in arrays owned by the strategy
    ///         and sized once in the constructor.
    ///     </para>
    ///     <para>
    ///         <see cref="IEvolutionAlgorithm.GetBestParameters"/> returns the best-fitness
    ///         candidate from the most recent generation, matching the semantics of the
    ///         generational GA. <see cref="Mean"/> exposes the current mean vector μ, which
    ///         is typically the parameter vector you want to deploy after training finishes.
    ///     </para>
    /// </remarks>
    public sealed class OpenAiEsStrategy : IEvolutionAlgorithm
    {
        private readonly INoiseTable _noiseTable;
        private readonly IFitnessShaper _fitnessShaper;
        private readonly Random _rng;

        private readonly float[] _mu;
        private readonly float[] _gradient;
        private readonly float[] _shapedFitness;
        private readonly float[] _bestParameters;
        private readonly int[] _noiseOffsets;
        private readonly int _pairCount;
        private readonly float _sigma;
        private readonly float _learningRate;
        private readonly float _gradientScale;

        private float _bestFitness;
        private bool _disposed;
        private bool _hasFitness;

        public OpenAiEsStrategy(
            int populationSize,
            int parameterCount,
            float sigma,
            float learningRate,
            INoiseTable noiseTable,
            IFitnessShaper? shaper = null,
            int? seed = null)
        {
            if (populationSize <= 1)
            {
                throw new ArgumentOutOfRangeException(nameof(populationSize), "Population size must be greater than 1.");
            }

            if ((populationSize & 1) != 0)
            {
                throw new ArgumentException(
                    "Population size must be even; antithetic sampling produces pairs (θ+, θ−).",
                    nameof(populationSize));
            }

            if (parameterCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(parameterCount), "Parameter count must be greater than 0.");
            }

            if (sigma <= 0f)
            {
                throw new ArgumentOutOfRangeException(nameof(sigma), "Sigma must be positive.");
            }

            if (learningRate <= 0f)
            {
                throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
            }

            ArgumentNullException.ThrowIfNull(noiseTable);

            if (noiseTable.Length < parameterCount)
            {
                throw new ArgumentException(
                    $"Noise table length {noiseTable.Length} is smaller than parameterCount {parameterCount}.",
                    nameof(noiseTable));
            }

            _noiseTable = noiseTable;
            _fitnessShaper = shaper ?? new CenteredRankFitnessShaper();
            _rng = seed.HasValue ? new Random(seed.Value) : new Random();

            PopulationSize = populationSize;
            ParameterCount = parameterCount;
            Generation = 0;
            _pairCount = populationSize / 2;
            _sigma = sigma;
            _learningRate = learningRate;

            // Gradient normalizer per Salimans 2017: 1 / (N * σ). Precomputed once.
            _gradientScale = 1f / (populationSize * sigma);

            _mu = new float[parameterCount];
            _gradient = new float[parameterCount];
            _shapedFitness = new float[populationSize];
            _bestParameters = new float[parameterCount];
            _noiseOffsets = new int[_pairCount];
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

        /// <summary>
        ///     Current mean parameter vector (μ). This is the typical "deploy after training"
        ///     vector — not the best candidate seen, but the center of the sampling distribution
        ///     that ES has converged to.
        /// </summary>
        public ReadOnlySpan<float> Mean
        {
            get
            {
                ThrowIfDisposed();
                return _mu;
            }
        }

        public void Initialize(float min = -0.3f, float max = 0.3f)
        {
            ThrowIfDisposed();

            if (min > max)
            {
                throw new ArgumentException("min cannot be greater than max.");
            }

            for (var i = 0; i < _mu.Length; i++)
            {
                _mu[i] = (float)(_rng.NextDouble() * (max - min) + min);
            }

            _hasFitness = false;
            _bestFitness = float.NaN;
            Generation = 0;
        }

        public void Ask(Span<float> populationMatrix)
        {
            ThrowIfDisposed();

            var expectedLength = PopulationSize * ParameterCount;

            if (populationMatrix.Length != expectedLength)
            {
                throw new ArgumentException(
                    $"populationMatrix length must be {expectedLength}.",
                    nameof(populationMatrix));
            }

            var sigma = _sigma;
            var paramCount = ParameterCount;

            for (var p = 0; p < _pairCount; p++)
            {
                // Sample a single offset per antithetic pair; the two children of the pair
                // share the same noise vector (one with +σ, one with −σ).
                var offset = _noiseTable.SampleOffset(_rng, paramCount);
                _noiseOffsets[p] = offset;

                var noise = _noiseTable.GetSlice(offset, paramCount);

                var positiveChild = populationMatrix.Slice((2 * p) * paramCount, paramCount);
                var negativeChild = populationMatrix.Slice(((2 * p) + 1) * paramCount, paramCount);

                for (var j = 0; j < paramCount; j++)
                {
                    var delta = sigma * noise[j];
                    positiveChild[j] = _mu[j] + delta;
                    negativeChild[j] = _mu[j] - delta;
                }
            }
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

            // Shape raw fitness into ranks. CenteredRank is NaN-safe: degenerate fitness
            // values sink to the bottom rank and cannot poison the gradient estimate.
            _fitnessShaper.Shape(fitness, _shapedFitness);

            // Track the unshaped best for reporting, using the raw fitness (shaped values
            // are meaningful only within a generation, not across generations).
            UpdateBestCandidate(fitness);

            // Accumulate the gradient: for each antithetic pair with noise ε,
            //   Σ (f⁺ − f⁻)·ε
            // then scale once by 1 / (N·σ) at the end.
            Array.Clear(_gradient, 0, _gradient.Length);

            var paramCount = ParameterCount;

            for (var p = 0; p < _pairCount; p++)
            {
                var fPlus = _shapedFitness[2 * p];
                var fMinus = _shapedFitness[(2 * p) + 1];
                var weight = fPlus - fMinus;

                if (weight == 0f)
                {
                    continue;
                }

                var noise = _noiseTable.GetSlice(_noiseOffsets[p], paramCount);

                for (var j = 0; j < paramCount; j++)
                {
                    _gradient[j] += weight * noise[j];
                }
            }

            // Apply: μ ← μ + α · ∇, with ∇ = _gradient · _gradientScale.
            var step = _learningRate * _gradientScale;

            for (var j = 0; j < paramCount; j++)
            {
                _mu[j] += step * _gradient[j];
            }

            _hasFitness = true;
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

        private void UpdateBestCandidate(ReadOnlySpan<float> fitness)
        {
            // Find the (raw) best fitness this generation. NaN values compare false against
            // anything, so a strict > filter keeps NaN out of the best slot automatically.
            var bestLocalIndex = -1;
            var bestLocalFitness = float.NegativeInfinity;

            for (var i = 0; i < fitness.Length; i++)
            {
                var f = fitness[i];

                if (f > bestLocalFitness)
                {
                    bestLocalFitness = f;
                    bestLocalIndex = i;
                }
            }

            if (bestLocalIndex < 0)
            {
                // Every fitness was NaN or −∞. Leave previous best untouched, but publish NaN
                // so callers can detect the degenerate generation.
                _bestFitness = float.NaN;
                return;
            }

            // Reconstruct the winning candidate from μ ± σε. We know the candidate's genome
            // was never stored explicitly, but μ, σ, and the noise slice fully determine it.
            var paramCount = ParameterCount;
            var pairIndex = bestLocalIndex / 2;
            var isNegative = (bestLocalIndex & 1) == 1;
            var noise = _noiseTable.GetSlice(_noiseOffsets[pairIndex], paramCount);
            var sigma = _sigma;

            if (isNegative)
            {
                for (var j = 0; j < paramCount; j++)
                {
                    _bestParameters[j] = _mu[j] - (sigma * noise[j]);
                }
            }
            else
            {
                for (var j = 0; j < paramCount; j++)
                {
                    _bestParameters[j] = _mu[j] + (sigma * noise[j]);
                }
            }

            _bestFitness = bestLocalFitness;
        }

        public void Dispose()
        {
            // Strategy holds only managed float[] / int[] buffers; GC reclaims them.
            // The noise table is user-owned and is NOT disposed here — lifetime is caller's
            // responsibility so the same table can be shared across multiple strategies.
            _disposed = true;
        }

        // -----------------------------------------------------------------------------
        // Checkpoint: IEvolutionCheckpoint.Save / Load
        // Format (little-endian, BinaryWriter defaults):
        //   int32   magic          = 0x4F414553 ('O','A','E','S')
        //   int32   schemaVersion  = 1
        //   int32   populationSize
        //   int32   parameterCount
        //   int32   generation
        //   byte    hasFitness
        //   float   bestFitness
        //   float[] mu              [parameterCount]
        //   float[] bestParameters  [parameterCount]
        // -----------------------------------------------------------------------------

        private const int CheckpointMagic = 0x4F414553;
        private const int CheckpointSchemaVersion = 1;

        public void Save(BinaryWriter writer)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(writer);

            writer.Write(CheckpointMagic);
            writer.Write(CheckpointSchemaVersion);
            writer.Write(PopulationSize);
            writer.Write(ParameterCount);
            writer.Write(Generation);
            writer.Write(_hasFitness);
            writer.Write(_bestFitness);

            WriteFloats(writer, _mu);
            WriteFloats(writer, _bestParameters);
        }

        public void Load(BinaryReader reader)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(reader);

            var magic = reader.ReadInt32();

            if (magic != CheckpointMagic)
            {
                throw new InvalidDataException(
                    $"Expected magic 0x{CheckpointMagic:X8}, found 0x{magic:X8}. Stream was not produced by OpenAiEsStrategy.");
            }

            var schemaVersion = reader.ReadInt32();

            if (schemaVersion != CheckpointSchemaVersion)
            {
                throw new InvalidDataException(
                    $"Unsupported schema version {schemaVersion}; this build supports {CheckpointSchemaVersion}.");
            }

            var populationSize = reader.ReadInt32();
            var parameterCount = reader.ReadInt32();

            if (populationSize != PopulationSize || parameterCount != ParameterCount)
            {
                throw new InvalidDataException(
                    $"Checkpoint was produced for ({populationSize}, {parameterCount}); " +
                    $"current instance is ({PopulationSize}, {ParameterCount}).");
            }

            Generation = reader.ReadInt32();
            _hasFitness = reader.ReadBoolean();
            _bestFitness = reader.ReadSingle();

            ReadFloats(reader, _mu);
            ReadFloats(reader, _bestParameters);
        }

        private static void WriteFloats(BinaryWriter writer, ReadOnlySpan<float> values)
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

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }
}