// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;
using System.Numerics.Tensors;
using DevOnBike.Overfit.Evolutionary.Abstractions;
using DevOnBike.Overfit.Evolutionary.Fitness;

namespace DevOnBike.Overfit.Evolutionary.Strategies
{
    public sealed class OpenAiEsStrategy : IEvolutionAlgorithm
    {
        private const int CheckpointMagic = 0x4F414553;
        private const int CheckpointSchemaVersion = 3;
        private const uint DefaultNonZeroSeed = 0x6D2B79F5u;

        private readonly INoiseTable _noiseTable;
        private readonly IFitnessShaper _fitnessShaper;

        private readonly float[] _mu;
        private readonly float[] _gradient;
        private readonly float[] _shapedFitness;
        private readonly float[] _bestParameters;
        private readonly int[] _noiseOffsets;

        private readonly int _pairCount;
        private readonly float _sigma;
        private readonly float _learningRate;
        private readonly float _gradientScale;

        private readonly bool _useAdam;
        private readonly float _beta1;
        private readonly float _beta2;
        private readonly float _epsilon;

        private readonly float[]? _m;
        private readonly float[]? _v;

        private uint _rngState;
        private readonly int _initialSeed;

        private int _adamStep;
        private float _bestFitness;
        private bool _disposed;
        private bool _hasFitness;
        private bool _hasPendingPopulation;

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
                throw new ArgumentException("Noise table length is smaller than parameterCount.", nameof(noiseTable));
            }

            _noiseTable = noiseTable;
            _fitnessShaper = shaper ?? new CenteredRankFitnessShaper();

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
            _hasPendingPopulation = false;

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

            _initialSeed = seed ?? Random.Shared.Next(int.MinValue, int.MaxValue);
            _rngState = NormalizeSeed(unchecked((uint)_initialSeed));
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

        public bool HasBest
        {
            get
            {
                ThrowIfDisposed();
                return _hasFitness;
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

        public void Initialize(float min = -0.3f, float max = 0.3f)
        {
            ThrowIfDisposed();

            if (min > max)
            {
                throw new ArgumentException("min cannot be greater than max.");
            }

            ResetRngToInitialSeed();

            var range = max - min;
            for (var i = 0; i < _mu.Length; i++)
            {
                _mu[i] = min + (range * NextUnitFloat());
            }

            Array.Clear(_gradient, 0, _gradient.Length);
            Array.Clear(_shapedFitness, 0, _shapedFitness.Length);
            Array.Clear(_bestParameters, 0, _bestParameters.Length);
            Array.Clear(_noiseOffsets, 0, _noiseOffsets.Length);

            _hasFitness = false;
            _hasPendingPopulation = false;
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
                var offset = NextNoiseOffset(paramCount);
                _noiseOffsets[p] = offset;

                var noise = _noiseTable.GetSlice(offset, paramCount);
                var positiveChild = populationMatrix.Slice((2 * p) * paramCount, paramCount);
                var negativeChild = populationMatrix.Slice(((2 * p) + 1) * paramCount, paramCount);

                TensorPrimitives.MultiplyAdd(noise, sigma, muSpan, positiveChild);
                TensorPrimitives.MultiplyAdd(noise, -sigma, muSpan, negativeChild);
            }

            _hasPendingPopulation = true;
        }

        public void Tell(ReadOnlySpan<float> fitness)
        {
            ThrowIfDisposed();

            if (fitness.Length != PopulationSize)
            {
                throw new ArgumentException($"fitness length must be {PopulationSize}.", nameof(fitness));
            }

            if (!_hasPendingPopulation)
            {
                throw new InvalidOperationException("Tell() was called without a matching Ask().");
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
                TensorPrimitives.MultiplyAdd(noise, weight, gradientSpan, gradientSpan);
            }

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
            _hasPendingPopulation = false;
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

        private void ApplySgdStep()
        {
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

            for (; j < paramCount; j++)
            {
                var g = _gradient[j];
                m[j] = (beta1 * m[j]) + ((1f - beta1) * g);
                v[j] = (beta2 * v[j]) + ((1f - beta2) * g * g);

                var denom = (MathF.Sqrt(v[j]) * invSqrtBc2) + epsilon;
                _mu[j] += stepSize * m[j] / denom;
            }
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

            if (_hasFitness && bestLocalFitness.CompareTo(_bestFitness) <= 0)
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

            writer.Write(_initialSeed);
            writer.Write(_rngState);
            writer.Write(_hasPendingPopulation);

            WriteFloats(writer, _mu);
            WriteFloats(writer, _bestParameters);

            writer.Write(_noiseOffsets.Length);
            
            for (var i = 0; i < _noiseOffsets.Length; i++)
            {
                writer.Write(_noiseOffsets[i]);
            }

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
            if (schemaVersion is not 2 and not 3)
            {
                throw new InvalidDataException($"Unsupported schema version {schemaVersion}; this build supports 2 and {CheckpointSchemaVersion}.");
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

            if (schemaVersion >= 3)
            {
                _ = reader.ReadInt32(); // initial seed from checkpoint; current instance keeps constructor seed
                _rngState = NormalizeSeed(reader.ReadUInt32());
                _hasPendingPopulation = reader.ReadBoolean();
            }
            else
            {
                // Legacy schema v2 had no RNG state and no pending Ask/Tell state.
                // We restore a deterministic fallback state so the strategy remains usable,
                // but exact replay is only guaranteed for schema v3.
                _rngState = NormalizeSeed(unchecked((uint)(Generation ^ PopulationSize ^ ParameterCount ^ 0x9E3779B9)));
                _hasPendingPopulation = false;
            }

            ReadFloats(reader, _mu);
            ReadFloats(reader, _bestParameters);

            if (schemaVersion >= 3)
            {
                var offsetCount = reader.ReadInt32();
                if (offsetCount != _noiseOffsets.Length)
                {
                    throw new InvalidDataException(
                        $"Checkpoint noise-offset count {offsetCount} does not match this instance {_noiseOffsets.Length}.");
                }

                for (var i = 0; i < _noiseOffsets.Length; i++)
                {
                    _noiseOffsets[i] = reader.ReadInt32();
                }
            }
            else
            {
                Array.Clear(_noiseOffsets, 0, _noiseOffsets.Length);
            }

            var streamUsedAdam = reader.ReadBoolean();
            if (streamUsedAdam != _useAdam)
            {
                throw new InvalidDataException(
                    $"Checkpoint optimizer mode (useAdam={streamUsedAdam}) does not match this instance.");
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

        private void ResetRngToInitialSeed()
        {
            _rngState = NormalizeSeed(unchecked((uint)_initialSeed));
        }

        private int NextNoiseOffset(int sliceLength)
        {
            if (sliceLength <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(sliceLength));
            }

            var exclusiveUpper = _noiseTable.Length - sliceLength + 1;
            if (exclusiveUpper <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(sliceLength));
            }

            return (int)NextUInt32Below((uint)exclusiveUpper);
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

        private uint NextUInt32Below(uint maxExclusive)
        {
            if (maxExclusive == 0u)
            {
                throw new ArgumentOutOfRangeException(nameof(maxExclusive));
            }

            var product = (ulong)NextUInt32() * maxExclusive;
            var low = (uint)product;

            if (low < maxExclusive)
            {
                var threshold = unchecked((uint)(0 - maxExclusive)) % maxExclusive;
                while (low < threshold)
                {
                    product = (ulong)NextUInt32() * maxExclusive;
                    low = (uint)product;
                }
            }

            return (uint)(product >> 32);
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