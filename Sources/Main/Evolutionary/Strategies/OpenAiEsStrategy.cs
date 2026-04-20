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
    ///                   μ. With <paramref name="useAdam"/>=false (plain SGD),
    ///                   <c>μ ← μ + α·∇</c>; with useAdam=true (default), μ is updated using
    ///                   the Adam moment estimator — Salimans' original recipe, which typically
    ///                   converges 2–3× faster per generation at the cost of two extra
    ///                   parameter-sized buffers (m, v).</item>
    ///         </list>
    ///         Antithetic sampling halves the variance of the gradient estimate for the same
    ///         number of evaluations, which is the main correctness advantage over the
    ///         textbook ES sampler.
    ///     </para>
    ///     <para>
    ///         Sign convention: ES maximizes fitness, so the update direction is <c>+∇</c>,
    ///         not <c>−∇</c> as in standard deep-learning optimizers that minimize a loss.
    ///         The internal Adam step returns the signed step to add to μ.
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

        // Adam state. Non-null iff useAdam == true.
        private readonly bool _useAdam;
        private readonly float _beta1;
        private readonly float _beta2;
        private readonly float _epsilon;
        private readonly float[]? _m;
        private readonly float[]? _v;
        private int _adamStep;

        private float _bestFitness;
        private bool _disposed;
        private bool _hasFitness;

        /// <summary>
        ///     Creates an ES optimizer with either plain SGD or Adam as the mean-update rule.
        /// </summary>
        /// <param name="useAdam">
        ///     When true (default), applies the Adam moment estimator to the ES gradient
        ///     estimate. Matches Salimans et al.'s published recipe and typically converges
        ///     faster. When false, uses plain SGD: <c>μ ← μ + α·∇</c>.
        /// </param>
        /// <param name="beta1">Adam first-moment decay. Ignored when useAdam=false.</param>
        /// <param name="beta2">Adam second-moment decay. Ignored when useAdam=false.</param>
        /// <param name="epsilon">Adam numerical stabilizer. Ignored when useAdam=false.</param>
        public OpenAiEsStrategy(
            int populationSize,
            int parameterCount,
            float sigma,
            float learningRate,
            INoiseTable noiseTable,
            IFitnessShaper? shaper = null,
            int? seed = null,
            bool useAdam = true,
            float beta1 = 0.9f,
            float beta2 = 0.999f,
            float epsilon = 1e-8f)
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

            if (useAdam)
            {
                if (beta1 is < 0f or >= 1f)
                {
                    throw new ArgumentOutOfRangeException(nameof(beta1), "beta1 must be in [0, 1).");
                }

                if (beta2 is < 0f or >= 1f)
                {
                    throw new ArgumentOutOfRangeException(nameof(beta2), "beta2 must be in [0, 1).");
                }

                if (epsilon <= 0f)
                {
                    throw new ArgumentOutOfRangeException(nameof(epsilon), "epsilon must be positive.");
                }
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

            _useAdam = useAdam;
            _beta1 = beta1;
            _beta2 = beta2;
            _epsilon = epsilon;

            if (useAdam)
            {
                _m = new float[parameterCount];
                _v = new float[parameterCount];
                _adamStep = 0;
            }
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

        /// <summary>
        ///     True when Adam is active, false when plain SGD is active. Informational only —
        ///     not part of the strategy's behavioral contract.
        /// </summary>
        public bool UseAdam => _useAdam;

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

            // Reset Adam moments: resuming training from scratch must not inherit the m/v
            // accumulators from a previous run. Initialize is the natural "factory reset" hook.
            if (_useAdam)
            {
                Array.Clear(_m!, 0, _m!.Length);
                Array.Clear(_v!, 0, _v!.Length);
                _adamStep = 0;
            }
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
            // then scale once by 1 / (N·σ) at the end so the final gradient magnitude is
            // independent of population size and sigma choice.
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

            // Normalize: _gradient now holds the ES estimate of the search gradient.
            var scale = _gradientScale;

            for (var j = 0; j < paramCount; j++)
            {
                _gradient[j] *= scale;
            }

            if (_useAdam)
            {
                ApplyAdamStep();
            }
            else
            {
                ApplySgdStep();
            }

            _hasFitness = true;
            Generation++;
        }

        /// <summary>
        ///     Plain SGD mean update: μ ← μ + α · ∇. Sign is positive because ES maximizes.
        /// </summary>
        private void ApplySgdStep()
        {
            var lr = _learningRate;
            var paramCount = ParameterCount;

            for (var j = 0; j < paramCount; j++)
            {
                _mu[j] += lr * _gradient[j];
            }
        }

        /// <summary>
        ///     Adam mean update with bias-corrected moments. Same as the TF/PyTorch Adam,
        ///     except the sign of the step is positive (maximization) rather than negative
        ///     (minimization). All moments live in dedicated pre-allocated buffers, so the
        ///     update is zero-allocation.
        /// </summary>
        private void ApplyAdamStep()
        {
            _adamStep++;

            var m = _m!;
            var v = _v!;
            var beta1 = _beta1;
            var beta2 = _beta2;
            var epsilon = _epsilon;
            var lr = _learningRate;
            var paramCount = ParameterCount;

            // Bias correction factors. At t=1 with beta1=0.9 these amplify the step by 10×,
            // which is intentional and matches the reference Adam.
            var biasCorrection1 = 1f - MathF.Pow(beta1, _adamStep);
            var biasCorrection2 = 1f - MathF.Pow(beta2, _adamStep);

            // Hoist the per-step constant to avoid the divide-in-loop.
            var stepSize = lr / biasCorrection1;
            var invSqrtBc2 = 1f / MathF.Sqrt(biasCorrection2);

            for (var j = 0; j < paramCount; j++)
            {
                var g = _gradient[j];

                // m_t = β₁·m_{t−1} + (1−β₁)·g
                m[j] = (beta1 * m[j]) + ((1f - beta1) * g);

                // v_t = β₂·v_{t−1} + (1−β₂)·g²
                v[j] = (beta2 * v[j]) + ((1f - beta2) * g * g);

                // μ ← μ + α · m̂ / (√v̂ + ε), with both bias corrections factored into the
                // step size and denominator. Plus sign because ES maximizes.
                var denom = (MathF.Sqrt(v[j]) * invSqrtBc2) + epsilon;
                _mu[j] += stepSize * m[j] / denom;
            }
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
        // Schema version 2 (breaking change from v1: added Adam state).
        // Format (little-endian, BinaryWriter defaults):
        //   int32   magic          = 0x4F414553 ('O','A','E','S')
        //   int32   schemaVersion  = 2
        //   int32   populationSize
        //   int32   parameterCount
        //   int32   generation
        //   byte    hasFitness
        //   float   bestFitness
        //   float[] mu              [parameterCount]
        //   float[] bestParameters  [parameterCount]
        //   byte    useAdam
        //   (if useAdam:)
        //     int32   adamStep
        //     float[] m              [parameterCount]
        //     float[] v              [parameterCount]
        // -----------------------------------------------------------------------------

        private const int CheckpointMagic = 0x4F414553;
        private const int CheckpointSchemaVersion = 2;

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

            writer.Write(_useAdam);

            if (_useAdam)
            {
                writer.Write(_adamStep);
                WriteFloats(writer, _m!);
                WriteFloats(writer, _v!);
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
                    $"Expected magic 0x{CheckpointMagic:X8}, found 0x{magic:X8}. Stream was not produced by OpenAiEsStrategy.");
            }

            var schemaVersion = reader.ReadInt32();

            if (schemaVersion != CheckpointSchemaVersion)
            {
                throw new InvalidDataException(
                    $"Unsupported schema version {schemaVersion}; this build supports {CheckpointSchemaVersion}. " +
                    "Checkpoints produced by earlier builds cannot be loaded.");
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

            var streamUsedAdam = reader.ReadBoolean();

            if (streamUsedAdam != _useAdam)
            {
                throw new InvalidDataException(
                    $"Checkpoint optimizer mode (useAdam={streamUsedAdam}) does not match " +
                    $"this instance (useAdam={_useAdam}).");
            }

            if (_useAdam)
            {
                _adamStep = reader.ReadInt32();
                ReadFloats(reader, _m!);
                ReadFloats(reader, _v!);
            }
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