// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Abstractions;

namespace DevOnBike.Overfit.Evolutionary.Strategies
{
    /// <summary>
    ///     Separable CMA-ES — Covariance Matrix Adaptation Evolution Strategy with the
    ///     covariance matrix constrained to its diagonal (Ros &amp; Hansen, 2008).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Full CMA-ES adapts a dense <c>n x n</c> covariance matrix, costing
    ///         <c>O(n^2)</c> memory and an <c>O(n^3)</c> eigendecomposition per generation —
    ///         infeasible for neural-network-sized parameter vectors. The separable variant
    ///         keeps <c>C</c> diagonal: every update is <c>O(lambda * n)</c>, there is no
    ///         matrix and no eigendecomposition, and the whole strategy is allocation-free
    ///         per generation and Native-AOT clean. The diagonal model cannot capture
    ///         correlations between coordinates, so Ros &amp; Hansen apply a learning-rate
    ///         speed-up of <c>(n + 2) / 3</c> to the covariance terms — fewer free
    ///         parameters can be learned faster.
    ///     </para>
    ///     <para>
    ///         The strategy <b>maximises</b> fitness, consistent with the rest of the engine
    ///         (higher is better). Because every update depends only on the <i>ranking</i> of
    ///         the population, CMA-ES is invariant to any strictly increasing transform of the
    ///         fitness values — scale, offset and monotone reshaping leave the trajectory
    ///         bit-identical.
    ///     </para>
    ///     <para>
    ///         Sampling uses an internal deterministic xorshift32 stream plus Box-Muller, so
    ///         a given seed reproduces a run exactly. Checkpoints persist the RNG state.
    ///     </para>
    /// </remarks>
    public sealed class SeparableCmaEsStrategy : IEvolutionAlgorithm
    {
        private const int CheckpointMagic = 0x53434D41; // 'S','C','M','A'
        private const int CheckpointSchemaVersion = 1;
        private const uint DefaultNonZeroSeed = 0x6D2B79F5u;

        // Lower bound on a diagonal variance entry. Guards sqrt and keeps the
        // distribution non-degenerate if a coordinate's variance collapses.
        private const float MinVariance = 1e-20f;

        // ── Distribution state (mutated every generation) ────────────────────
        private readonly float[] _mean;       // distribution mean m
        private readonly float[] _diagC;      // diagonal of the covariance matrix
        private readonly float[] _diagD;      // per-coordinate std dev = sqrt(_diagC)
        private readonly float[] _pSigma;     // evolution path for step-size control
        private readonly float[] _pC;         // evolution path for covariance adaptation
        private float _sigma;                 // global step size

        // ── Per-generation scratch (allocated once, reused) ──────────────────
        private readonly float[] _z;          // [lambda * n] raw N(0,I) samples from Ask
        private readonly int[] _ranking;      // population indices sorted best-first
        private readonly float[] _zMean;      // weighted mean of the mu best z vectors
        private readonly float[] _weights;    // recombination weights, sum to 1

        private readonly float[] _bestParameters;

        // ── Strategy constants (computed once in the constructor) ────────────
        private readonly int _mu;             // number of parents in recombination
        private readonly float _muEff;        // effective selection mass
        private readonly float _cSigma;       // step-size path learning rate
        private readonly float _dSigma;       // step-size damping
        private readonly float _cc;           // covariance path learning rate
        private readonly float _c1;           // rank-one covariance learning rate
        private readonly float _cMu;          // rank-mu covariance learning rate
        private readonly float _chiN;         // expected length of an N(0,I) vector

        private readonly float _initialSigma;
        private readonly int _initialSeed;

        private uint _rngState;
        private float _gaussianSpare;
        private bool _hasGaussianSpare;

        private float _bestFitness;
        private bool _hasFitness;
        private bool _hasPendingPopulation;
        private bool _disposed;

        /// <summary>
        ///     Creates a separable CMA-ES strategy.
        /// </summary>
        /// <param name="populationSize">
        ///     Number of candidates sampled per generation (lambda). Must be greater than 1.
        ///     <see cref="DefaultPopulationSize"/> returns the literature default for a given
        ///     dimension.
        /// </param>
        /// <param name="parameterCount">Dimension of the search space (n). Must be positive.</param>
        /// <param name="initialSigma">Initial global step size. Must be positive.</param>
        /// <param name="seed">Seed for the deterministic internal RNG.</param>
        public SeparableCmaEsStrategy(
            int populationSize,
            int parameterCount,
            float initialSigma,
            int seed)
        {
            if (populationSize <= 1)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(populationSize), "Population size must be greater than 1.");
            }

            if (parameterCount <= 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(parameterCount), "Parameter count must be greater than 0.");
            }

            if (initialSigma <= 0f)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(initialSigma), "Initial sigma must be positive.");
            }

            PopulationSize = populationSize;
            ParameterCount = parameterCount;
            _initialSigma = initialSigma;
            _initialSeed = seed;

            _mu = populationSize / 2;

            // Recombination weights (Hansen): w_i' = ln(mu + 1/2) - ln(i), i = 1..mu,
            // normalised so the positive weights sum to 1. Computed in double for the
            // accuracy of the one-time constant setup, then stored as float.
            _weights = new float[_mu];
            var weightsRaw = new double[_mu];
            var sumW = 0.0;

            for (var i = 0; i < _mu; i++)
            {
                var w = Math.Log(_mu + 0.5) - Math.Log(i + 1);
                weightsRaw[i] = w;
                sumW += w;
            }

            var sumW2 = 0.0;

            for (var i = 0; i < _mu; i++)
            {
                var w = weightsRaw[i] / sumW;
                _weights[i] = (float)w;
                sumW2 += w * w;
            }

            // Effective selection mass: muEff = (sum w)^2 / sum w^2 = 1 / sum w^2,
            // since the weights are normalised to sum to 1.
            var muEff = 1.0 / sumW2;
            var n = (double)parameterCount;

            var cSigma = (muEff + 2.0) / (n + muEff + 5.0);
            var dSigma = 1.0
                + (2.0 * Math.Max(0.0, Math.Sqrt((muEff - 1.0) / (n + 1.0)) - 1.0))
                + cSigma;
            var cc = (4.0 + (muEff / n)) / (n + 4.0 + (2.0 * muEff / n));
            var c1 = 2.0 / (((n + 1.3) * (n + 1.3)) + muEff);
            var cMu = Math.Min(
                1.0 - c1,
                2.0 * (muEff - 2.0 + (1.0 / muEff)) / (((n + 2.0) * (n + 2.0)) + muEff));

            // sep-CMA-ES speed-up: the diagonal model has n free covariance parameters
            // instead of n(n+1)/2, so the rank-one and rank-mu terms can be learned
            // (n+2)/3 times faster (Ros & Hansen 2008). Cap so c1 + cMu <= 1.
            var sep = (n + 2.0) / 3.0;
            c1 *= sep;
            cMu *= sep;

            if (c1 + cMu > 1.0)
            {
                var rescale = 1.0 / (c1 + cMu);
                c1 *= rescale;
                cMu *= rescale;
            }

            _muEff = (float)muEff;
            _cSigma = (float)cSigma;
            _dSigma = (float)dSigma;
            _cc = (float)cc;
            _c1 = (float)c1;
            _cMu = (float)cMu;
            _chiN = (float)(Math.Sqrt(n)
                * (1.0 - (1.0 / (4.0 * n)) + (1.0 / (21.0 * n * n))));

            _mean = new float[parameterCount];
            _diagC = new float[parameterCount];
            _diagD = new float[parameterCount];
            _pSigma = new float[parameterCount];
            _pC = new float[parameterCount];
            _zMean = new float[parameterCount];
            _bestParameters = new float[parameterCount];
            _z = new float[populationSize * parameterCount];
            _ranking = new int[populationSize];

            ResetState();
        }

        /// <summary>
        ///     Literature-default population size for a given dimension:
        ///     <c>4 + floor(3 * ln(n))</c>.
        /// </summary>
        /// <param name="parameterCount">Dimension of the search space. Must be positive.</param>
        public static int DefaultPopulationSize(int parameterCount)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(parameterCount);
            return 4 + (int)(3.0 * Math.Log(parameterCount));
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

        /// <summary>True once at least one full Ask/Tell cycle has completed.</summary>
        public bool HasBest
        {
            get
            {
                ThrowIfDisposed();
                return _hasFitness;
            }
        }

        /// <summary>The current distribution mean — the strategy's running estimate of the optimum.</summary>
        public ReadOnlySpan<float> Mean
        {
            get
            {
                ThrowIfDisposed();
                return _mean;
            }
        }

        /// <summary>The current global step size. Shrinks as the search converges.</summary>
        public float Sigma
        {
            get
            {
                ThrowIfDisposed();
                return _sigma;
            }
        }

        /// <summary>
        ///     Resets the strategy and seeds the mean with a fresh uniform draw in
        ///     <paramref name="min"/>..<paramref name="max"/>. Step size, evolution paths and
        ///     the covariance diagonal are restored to their initial values, and the RNG is
        ///     rewound to the constructor seed.
        /// </summary>
        public void Initialize(float min = -0.3f, float max = 0.3f)
        {
            ThrowIfDisposed();

            if (min > max)
            {
                throw new ArgumentException("min cannot be greater than max.");
            }

            ResetState();

            var range = max - min;

            for (var j = 0; j < _mean.Length; j++)
            {
                _mean[j] = min + (range * NextUnitFloat());
            }
        }

        public void Ask(Span<float> populationMatrix)
        {
            ThrowIfDisposed();

            var expectedLength = PopulationSize * ParameterCount;

            if (populationMatrix.Length != expectedLength)
            {
                throw new ArgumentException(
                    $"populationMatrix length must be {expectedLength}.", nameof(populationMatrix));
            }

            var n = ParameterCount;

            // Each Ask is fully determined by the RNG state: drop any half-consumed
            // Box-Muller pair so the stream restarts cleanly on a checkpoint resume.
            _hasGaussianSpare = false;

            for (var k = 0; k < PopulationSize; k++)
            {
                var baseIndex = k * n;

                for (var j = 0; j < n; j++)
                {
                    var z = NextGaussian();
                    _z[baseIndex + j] = z;
                    populationMatrix[baseIndex + j] = _mean[j] + (_sigma * _diagD[j] * z);
                }
            }

            _hasPendingPopulation = true;
        }

        public void Tell(ReadOnlySpan<float> fitness)
        {
            ThrowIfDisposed();

            if (fitness.Length != PopulationSize)
            {
                throw new ArgumentException(
                    $"fitness length must be {PopulationSize}.", nameof(fitness));
            }

            if (!_hasPendingPopulation)
            {
                throw new OverfitRuntimeException("Tell() was called without a matching Ask().");
            }

            var n = ParameterCount;

            RankPopulation(fitness);

            // Best-candidate tracking uses the still-current mean/sigma/diagD — must run
            // before the recombination step overwrites them.
            UpdateBestCandidate(fitness);

            // Weighted recombination of the mu best raw samples: zMean = sum w_i * z_i.
            Array.Clear(_zMean);

            for (var i = 0; i < _mu; i++)
            {
                var w = _weights[i];
                var baseIndex = _ranking[i] * n;

                for (var j = 0; j < n; j++)
                {
                    _zMean[j] += w * _z[baseIndex + j];
                }
            }

            // New mean: m += sigma * (d (.) zMean). Uses the pre-update sigma/diagD.
            for (var j = 0; j < n; j++)
            {
                _mean[j] += _sigma * _diagD[j] * _zMean[j];
            }

            // Step-size evolution path. For diagonal C the conjugate-path scaling
            // C^(-1/2) * yMean reduces exactly to zMean.
            var pSigmaDecay = 1f - _cSigma;
            var pSigmaGain = MathF.Sqrt(_cSigma * (2f - _cSigma) * _muEff);

            for (var j = 0; j < n; j++)
            {
                _pSigma[j] = (pSigmaDecay * _pSigma[j]) + (pSigmaGain * _zMean[j]);
            }

            var pSigmaNorm = Norm(_pSigma);

            // Cumulative step-size adaptation.
            _sigma *= MathF.Exp((_cSigma / _dSigma) * ((pSigmaNorm / _chiN) - 1f));

            // Heaviside step: stall the rank-one update while the step-size path is
            // still inflated, so a large initial sigma cannot blow up the covariance.
            var updateIndex = Generation + 1;
            var pSigmaNormalizer = MathF.Sqrt(
                1f - MathF.Pow(pSigmaDecay, 2f * updateIndex));
            var hSigma =
                (pSigmaNorm / pSigmaNormalizer) < ((1.4f + (2f / (n + 1f))) * _chiN)
                    ? 1f
                    : 0f;

            // Covariance evolution path. yMean = d (.) zMean.
            var pcDecay = 1f - _cc;
            var pcGain = hSigma * MathF.Sqrt(_cc * (2f - _cc) * _muEff);

            for (var j = 0; j < n; j++)
            {
                _pC[j] = (pcDecay * _pC[j]) + (pcGain * _diagD[j] * _zMean[j]);
            }

            // Diagonal covariance update: decay + rank-one + rank-mu. The
            // (1 - hSigma) term restores the variance the stalled path would lose.
            var varianceLossCorrection = (1f - hSigma) * _cc * (2f - _cc);
            var cDecay = 1f - _c1 - _cMu + (_c1 * varianceLossCorrection);

            for (var j = 0; j < n; j++)
            {
                var rankMu = 0f;

                for (var i = 0; i < _mu; i++)
                {
                    var y = _diagD[j] * _z[(_ranking[i] * n) + j];
                    rankMu += _weights[i] * y * y;
                }

                var c = (cDecay * _diagC[j])
                    + (_c1 * _pC[j] * _pC[j])
                    + (_cMu * rankMu);

                if (c < MinVariance)
                {
                    c = MinVariance;
                }

                _diagC[j] = c;
                _diagD[j] = MathF.Sqrt(c);
            }

            _hasFitness = true;
            _hasPendingPopulation = false;
            Generation++;
        }

        public ReadOnlySpan<float> GetBestParameters()
        {
            ThrowIfDisposed();
            return _hasFitness ? _bestParameters : [];
        }

        private void RankPopulation(ReadOnlySpan<float> fitness)
        {
            var ranking = _ranking;

            for (var i = 0; i < ranking.Length; i++)
            {
                ranking[i] = i;
            }

            // Insertion sort, descending by fitness, NaN last. Population size is
            // small (tens), so O(lambda^2) is cheaper than a comparer-based sort and
            // stays allocation-free.
            for (var i = 1; i < ranking.Length; i++)
            {
                var index = ranking[i];
                var key = fitness[index];
                var j = i - 1;

                while (j >= 0 && RanksBelow(fitness[ranking[j]], key))
                {
                    ranking[j + 1] = ranking[j];
                    j--;
                }

                ranking[j + 1] = index;
            }
        }

        // True when 'candidate' should rank after 'reference' (worse), with NaN last.
        private static bool RanksBelow(float candidate, float reference)
        {
            if (float.IsNaN(candidate))
            {
                return !float.IsNaN(reference);
            }

            if (float.IsNaN(reference))
            {
                return false;
            }

            return candidate < reference;
        }

        private void UpdateBestCandidate(ReadOnlySpan<float> fitness)
        {
            var bestIndex = -1;
            var bestLocalFitness = float.NegativeInfinity;

            for (var i = 0; i < fitness.Length; i++)
            {
                var f = fitness[i];

                if (f > bestLocalFitness)
                {
                    bestLocalFitness = f;
                    bestIndex = i;
                }
            }

            if (bestIndex < 0)
            {
                return;
            }

            if (_hasFitness && bestLocalFitness.CompareTo(_bestFitness) <= 0)
            {
                return;
            }

            var n = ParameterCount;
            var baseIndex = bestIndex * n;

            for (var j = 0; j < n; j++)
            {
                _bestParameters[j] = _mean[j] + (_sigma * _diagD[j] * _z[baseIndex + j]);
            }

            _bestFitness = bestLocalFitness;
        }

        private static float Norm(ReadOnlySpan<float> values)
        {
            var sum = 0f;

            for (var i = 0; i < values.Length; i++)
            {
                sum += values[i] * values[i];
            }

            return MathF.Sqrt(sum);
        }

        private void ResetState()
        {
            Array.Clear(_mean);
            Array.Clear(_pSigma);
            Array.Clear(_pC);
            Array.Clear(_zMean);
            Array.Clear(_z);
            Array.Clear(_bestParameters);

            for (var j = 0; j < ParameterCount; j++)
            {
                _diagC[j] = 1f;
                _diagD[j] = 1f;
            }

            _sigma = _initialSigma;
            Generation = 0;
            _bestFitness = float.NaN;
            _hasFitness = false;
            _hasPendingPopulation = false;
            _hasGaussianSpare = false;
            _gaussianSpare = 0f;
            _rngState = NormalizeSeed(unchecked((uint)_initialSeed));
        }

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
            writer.Write(_initialSeed);
            writer.Write(_rngState);
            writer.Write(_sigma);
            writer.Write(_hasPendingPopulation);

            WriteFloats(writer, _mean);
            WriteFloats(writer, _diagC);
            WriteFloats(writer, _pSigma);
            WriteFloats(writer, _pC);
            WriteFloats(writer, _bestParameters);

            // The pending raw samples are only needed if a checkpoint is taken
            // between Ask and Tell; skip them on the (recommended) between-cycles save.
            if (_hasPendingPopulation)
            {
                WriteFloats(writer, _z);
            }
        }

        public void Load(BinaryReader reader)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(reader);

            var magic = reader.ReadInt32();

            if (magic != CheckpointMagic)
            {
                throw new OverfitFormatException(
                    $"Expected magic 0x{CheckpointMagic:X8}, found 0x{magic:X8}. "
                    + "Stream was not produced by SeparableCmaEsStrategy.");
            }

            var schemaVersion = reader.ReadInt32();

            if (schemaVersion != CheckpointSchemaVersion)
            {
                throw new OverfitFormatException(
                    $"Unsupported schema version {schemaVersion}; this build supports "
                    + $"{CheckpointSchemaVersion}.");
            }

            var populationSize = reader.ReadInt32();
            var parameterCount = reader.ReadInt32();

            if (populationSize != PopulationSize || parameterCount != ParameterCount)
            {
                throw new OverfitFormatException(
                    $"Checkpoint was produced for ({populationSize}, {parameterCount}); "
                    + $"current instance is ({PopulationSize}, {ParameterCount}).");
            }

            Generation = reader.ReadInt32();
            _hasFitness = reader.ReadBoolean();
            _bestFitness = reader.ReadSingle();
            _ = reader.ReadInt32(); // initial seed from checkpoint; instance keeps its own
            _rngState = NormalizeSeed(reader.ReadUInt32());
            _sigma = reader.ReadSingle();
            _hasPendingPopulation = reader.ReadBoolean();

            ReadFloats(reader, _mean);
            ReadFloats(reader, _diagC);
            ReadFloats(reader, _pSigma);
            ReadFloats(reader, _pC);
            ReadFloats(reader, _bestParameters);

            for (var j = 0; j < ParameterCount; j++)
            {
                _diagD[j] = MathF.Sqrt(_diagC[j]);
            }

            if (_hasPendingPopulation)
            {
                ReadFloats(reader, _z);
            }
            else
            {
                Array.Clear(_z);
            }

            _hasGaussianSpare = false;
            _gaussianSpare = 0f;
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

        // ── Deterministic RNG: xorshift32 + Box-Muller ───────────────────────

        private float NextGaussian()
        {
            if (_hasGaussianSpare)
            {
                _hasGaussianSpare = false;
                return _gaussianSpare;
            }

            float u1;

            do
            {
                u1 = NextUnitFloat();
            }
            while (u1 <= 1e-7f);

            var u2 = NextUnitFloat();
            var magnitude = MathF.Sqrt(-2f * MathF.Log(u1));
            var angle = 2f * MathF.PI * u2;

            _gaussianSpare = magnitude * MathF.Sin(angle);
            _hasGaussianSpare = true;

            return magnitude * MathF.Cos(angle);
        }

        private float NextUnitFloat()
        {
            // 24-bit mantissa path, deterministic and allocation-free.
            return (NextUInt32() >> 8) * (1.0f / (1u << 24));
        }

        private uint NextUInt32()
        {
            var x = _rngState;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            _rngState = NormalizeSeed(x);
            return _rngState;
        }

        private static uint NormalizeSeed(uint seed)
        {
            return seed == 0u ? DefaultNonZeroSeed : seed;
        }

        public void Dispose()
        {
            _disposed = true;
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }
}
