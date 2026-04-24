using System.Numerics;
using System.Numerics.Tensors;
using DevOnBike.Overfit.Evolutionary.Abstractions;
using DevOnBike.Overfit.Evolutionary.Fitness;

namespace DevOnBike.Overfit.Evolutionary.Strategies
{
    public sealed class OpenAiEsStrategy : IEvolutionAlgorithm
    {
        // -----------------------------------------------------------------------------
        // Constants
        // -----------------------------------------------------------------------------
        private const int CheckpointMagic = 0x4F414553;
        private const int CheckpointSchemaVersion = 2;

        // -----------------------------------------------------------------------------
        // Private Fields
        // -----------------------------------------------------------------------------
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

        // Adam state
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

        // -----------------------------------------------------------------------------
        // Constructor
        // -----------------------------------------------------------------------------
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
                throw new ArgumentException("Population size must be even for antithetic sampling.", nameof(populationSize));
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
                throw new ArgumentException($"Noise table length is smaller than parameterCount.", nameof(noiseTable));
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

        // -----------------------------------------------------------------------------
        // Properties
        // -----------------------------------------------------------------------------
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

        public ReadOnlySpan<float> Mean
        {
            get
            {
                ThrowIfDisposed();
                return _mu;
            }
        }

        public bool UseAdam => _useAdam;

        // -----------------------------------------------------------------------------
        // Methods
        // -----------------------------------------------------------------------------
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
                throw new ArgumentException($"populationMatrix length must be {expectedLength}.", nameof(populationMatrix));
            }

            var sigma = _sigma;
            var paramCount = ParameterCount;
            var muSpan = _mu.AsSpan();

            for (var p = 0; p < _pairCount; p++)
            {
                var offset = _noiseTable.SampleOffset(_rng, paramCount);
                _noiseOffsets[p] = offset;

                var noise = _noiseTable.GetSlice(offset, paramCount);
                var positiveChild = populationMatrix.Slice((2 * p) * paramCount, paramCount);
                var negativeChild = populationMatrix.Slice(((2 * p) + 1) * paramCount, paramCount);

                // SIMD: positiveChild[j] = (noise[j] * sigma) + _mu[j]
                TensorPrimitives.MultiplyAdd(noise, sigma, muSpan, positiveChild);

                // SIMD: negativeChild[j] = (noise[j] * -sigma) + _mu[j]
                TensorPrimitives.MultiplyAdd(noise, -sigma, muSpan, negativeChild);
            }
        }

        public void Tell(ReadOnlySpan<float> fitness)
        {
            ThrowIfDisposed();
            if (fitness.Length != PopulationSize)
            {
                throw new ArgumentException($"fitness length must be {PopulationSize}.", nameof(fitness));
            }

            _fitnessShaper.Shape(fitness, _shapedFitness);
            UpdateBestCandidate(fitness);

            var gradientSpan = _gradient.AsSpan();
            gradientSpan.Clear();

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

                // SIMD Accumulation: _gradient[j] += noise[j] * weight
                TensorPrimitives.MultiplyAdd(noise, weight, gradientSpan, gradientSpan);
            }

            // SIMD Normalization: _gradient[j] *= scale
            TensorPrimitives.Multiply(gradientSpan, _gradientScale, gradientSpan);

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

        private void ApplySgdStep()
        {
            // SIMD: _mu[j] = (_gradient[j] * _learningRate) + _mu[j]
            TensorPrimitives.MultiplyAdd(_gradient, _learningRate, _mu, _mu);
        }

        private void ApplyAdamStep()
        {
            _adamStep++;

            var m = _m!;
            var v = _v!;
            var beta1 = _beta1;
            var beta2 = _beta2;
            var epsilon = _epsilon;
            var paramCount = ParameterCount;

            var biasCorrection1 = 1f - MathF.Pow(beta1, _adamStep);
            var biasCorrection2 = 1f - MathF.Pow(beta2, _adamStep);

            var stepSize = _learningRate / biasCorrection1;
            var invSqrtBc2 = 1f / MathF.Sqrt(biasCorrection2);

            var j = 0;

            // Single-Pass SIMD Kernel
            if (Vector.IsHardwareAccelerated)
            {
                var vBeta1 = new Vector<float>(beta1);
                var vOneMinusBeta1 = new Vector<float>(1f - beta1);
                var vBeta2 = new Vector<float>(beta2);
                var vOneMinusBeta2 = new Vector<float>(1f - beta2);
                var vStepSize = new Vector<float>(stepSize);
                var vInvSqrtBc2 = new Vector<float>(invSqrtBc2);
                var vEpsilon = new Vector<float>(epsilon);

                var limit = paramCount - Vector<float>.Count;

                for (; j <= limit; j += Vector<float>.Count)
                {
                    var g = new Vector<float>(_gradient, j);
                    var mVec = new Vector<float>(m, j);
                    var vVec = new Vector<float>(v, j);
                    var muVec = new Vector<float>(_mu, j);

                    mVec = (vBeta1 * mVec) + (vOneMinusBeta1 * g);
                    vVec = (vBeta2 * vVec) + (vOneMinusBeta2 * g * g);

                    var denom = (Vector.SquareRoot(vVec) * vInvSqrtBc2) + vEpsilon;
                    muVec += (vStepSize * mVec) / denom;

                    mVec.CopyTo(m, j);
                    vVec.CopyTo(v, j);
                    muVec.CopyTo(_mu, j);
                }
            }

            // Scalar fallback 
            for (; j < paramCount; j++)
            {
                var g = _gradient[j];
                m[j] = (beta1 * m[j]) + ((1f - beta1) * g);
                v[j] = (beta2 * v[j]) + ((1f - beta2) * g * g);

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
                return;
            }
            
            if (bestLocalFitness.CompareTo(_bestFitness) <= 0)
            {
                return;
            }

            var paramCount = ParameterCount;
            var pairIndex = bestLocalIndex / 2;
            var isNegative = (bestLocalIndex & 1) == 1;
            var noise = _noiseTable.GetSlice(_noiseOffsets[pairIndex], paramCount);

            if (isNegative)
            {
                TensorPrimitives.MultiplyAdd(noise, -_sigma, _mu, _bestParameters);
            }
            else
            {
                TensorPrimitives.MultiplyAdd(noise, _sigma, _mu, _bestParameters);
            }

            _bestFitness = bestLocalFitness;
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
                throw new InvalidDataException($"Expected magic 0x{CheckpointMagic:X8}, found 0x{magic:X8}. Stream was not produced by OpenAiEsStrategy.");
            }

            var schemaVersion = reader.ReadInt32();
            if (schemaVersion != CheckpointSchemaVersion)
            {
                throw new InvalidDataException($"Unsupported schema version {schemaVersion}; this build supports {CheckpointSchemaVersion}.");
            }

            var populationSize = reader.ReadInt32();
            var parameterCount = reader.ReadInt32();

            if (populationSize != PopulationSize || parameterCount != ParameterCount)
            {
                throw new InvalidDataException($"Checkpoint was produced for ({populationSize}, {parameterCount}); current instance is ({PopulationSize}, {ParameterCount}).");
            }

            Generation = reader.ReadInt32();
            _hasFitness = reader.ReadBoolean();
            _bestFitness = reader.ReadSingle();

            ReadFloats(reader, _mu);
            ReadFloats(reader, _bestParameters);

            var streamUsedAdam = reader.ReadBoolean();
            if (streamUsedAdam != _useAdam)
            {
                throw new InvalidDataException($"Checkpoint optimizer mode (useAdam={streamUsedAdam}) does not match this instance.");
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