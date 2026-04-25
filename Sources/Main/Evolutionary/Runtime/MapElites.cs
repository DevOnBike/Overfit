// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Diagnostics.Contracts;
using DevOnBike.Overfit.Evolutionary.Abstractions;
using DevOnBike.Overfit.Evolutionary.Storage;

namespace DevOnBike.Overfit.Evolutionary.Runtime
{
    /// <summary>
    /// Minimal MAP-Elites loop.
    /// Uses a fixed-grid archive and a descriptor evaluator that returns fitness and fills descriptor.
    /// </summary>
    public sealed class MapElites<TContext> : IDisposable
    {
        private const uint DefaultNonZeroSeed = 0x6D2B79F5u;

        private readonly GridEliteArchive _archive;
        private readonly IBehaviorDescriptorEvaluator<TContext> _evaluator;

        private readonly float[] _candidateParameters;
        private readonly float[] _candidateDescriptors;
        private readonly float[] _candidateFitness;
        private readonly float[] _bestParameters;

        private readonly float _initialMin;
        private readonly float _initialMax;
        private readonly float _mutationSigma;
        private readonly float _randomInjectionProbability;

        private readonly int _initialSeed;
        private uint _rngState;
        private bool _disposed;
        private bool _hasBest;
        private float _bestFitness;

        private bool _hasSpareGaussian;
        private float _spareGaussian;

        public MapElites(
            int parameterCount,
            int batchSize,
            GridEliteArchive archive,
            IBehaviorDescriptorEvaluator<TContext> evaluator,
            int seed,
            float mutationSigma = 0.02f,
            float initialMin = -0.3f,
            float initialMax = 0.3f,
            float randomInjectionProbability = 0.05f)
        {
            if (parameterCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(parameterCount));
            }

            if (batchSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(batchSize));
            }

            if (mutationSigma <= 0f)
            {
                throw new ArgumentOutOfRangeException(nameof(mutationSigma));
            }

            if (initialMin > initialMax)
            {
                throw new ArgumentException("initialMin cannot be greater than initialMax.");
            }

            if (randomInjectionProbability is < 0f or > 1f)
            {
                throw new ArgumentOutOfRangeException(nameof(randomInjectionProbability));
            }

            ArgumentNullException.ThrowIfNull(archive);
            ArgumentNullException.ThrowIfNull(evaluator);

            if (archive.ParameterCount != parameterCount)
            {
                throw new ArgumentException(
                    $"Archive parameter count {archive.ParameterCount} does not match {parameterCount}.",
                    nameof(archive));
            }

            ParameterCount = parameterCount;
            BatchSize = batchSize;
            DescriptorDimensions = archive.DescriptorDimensions;

            _archive = archive;
            _evaluator = evaluator;

            _candidateParameters = new float[batchSize * parameterCount];
            _candidateDescriptors = new float[batchSize * DescriptorDimensions];
            _candidateFitness = new float[batchSize];
            _bestParameters = new float[parameterCount];

            _initialMin = initialMin;
            _initialMax = initialMax;
            _mutationSigma = mutationSigma;
            _randomInjectionProbability = randomInjectionProbability;

            _initialSeed = seed;
            _rngState = NormalizeSeed(unchecked((uint)seed));

            _bestFitness = float.NaN;
            _hasBest = false;
            Iteration = 0;
        }

        public int ParameterCount { get; }
        public int BatchSize { get; }
        public int DescriptorDimensions { get; }
        public int Iteration { get; private set; }

        public GridEliteArchive Archive
        {
            get
            {
                ThrowIfDisposed();
                return _archive;
            }
        }

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
                return _hasBest;
            }
        }

        public Span<float> CandidateParameters
        {
            get
            {
                ThrowIfDisposed();
                return _candidateParameters;
            }
        }

        public Span<float> CandidateDescriptors
        {
            get
            {
                ThrowIfDisposed();
                return _candidateDescriptors;
            }
        }

        public Span<float> CandidateFitness
        {
            get
            {
                ThrowIfDisposed();
                return _candidateFitness;
            }
        }

        public ReadOnlySpan<float> GetBestParameters()
        {
            ThrowIfDisposed();
            return _hasBest ? _bestParameters : ReadOnlySpan<float>.Empty;
        }

        public void Reset()
        {
            ThrowIfDisposed();

            _rngState = NormalizeSeed(unchecked((uint)_initialSeed));
            Iteration = 0;
            _hasBest = false;
            _bestFitness = float.NaN;
            _hasSpareGaussian = false;
            _spareGaussian = 0f;

            Array.Clear(_candidateParameters, 0, _candidateParameters.Length);
            Array.Clear(_candidateDescriptors, 0, _candidateDescriptors.Length);
            Array.Clear(_candidateFitness, 0, _candidateFitness.Length);
            Array.Clear(_bestParameters, 0, _bestParameters.Length);
        }

        public void Ask(Span<float> populationMatrix)
        {
            ThrowIfDisposed();

            var expectedLength = BatchSize * ParameterCount;
            if (populationMatrix.Length != expectedLength)
            {
                throw new ArgumentException(
                    $"populationMatrix length must be {expectedLength}.",
                    nameof(populationMatrix));
            }

            if (_archive.OccupiedCount == 0)
            {
                FillWithRandomCandidates(populationMatrix);
                return;
            }

            for (var i = 0; i < BatchSize; i++)
            {
                var candidate = populationMatrix.Slice(i * ParameterCount, ParameterCount);

                if (NextUnitFloat() < _randomInjectionProbability ||
                    !_archive.TrySampleOccupiedCell(ref _rngState, out var cellIndex))
                {
                    FillRandomVector(candidate, _initialMin, _initialMax);
                    continue;
                }

                _archive.GetParameters(cellIndex).CopyTo(candidate);
                MutateGaussianInPlace(candidate, _mutationSigma);
            }
        }

        public MapElitesIterationMetrics Tell(
            ReadOnlySpan<float> populationMatrix,
            ReadOnlySpan<float> fitness,
            ReadOnlySpan<float> descriptors)
        {
            ThrowIfDisposed();

            var expectedParameterLength = BatchSize * ParameterCount;
            var expectedDescriptorLength = BatchSize * DescriptorDimensions;

            if (populationMatrix.Length != expectedParameterLength)
            {
                throw new ArgumentException(
                    $"populationMatrix length must be {expectedParameterLength}.",
                    nameof(populationMatrix));
            }

            if (fitness.Length != BatchSize)
            {
                throw new ArgumentException(
                    $"fitness length must be {BatchSize}.",
                    nameof(fitness));
            }

            if (descriptors.Length != expectedDescriptorLength)
            {
                throw new ArgumentException(
                    $"descriptors length must be {expectedDescriptorLength}.",
                    nameof(descriptors));
            }

            var insertedNew = 0;
            var replaced = 0;
            var rejected = 0;
            var outOfBounds = 0;

            for (var i = 0; i < BatchSize; i++)
            {
                var candidate = populationMatrix.Slice(i * ParameterCount, ParameterCount);
                var descriptor = descriptors.Slice(i * DescriptorDimensions, DescriptorDimensions);
                var candidateFitness = fitness[i];

                var status = _archive.Insert(candidate, candidateFitness, descriptor);
                switch (status)
                {
                    case EliteInsertStatus.InsertedNewCell:
                        insertedNew++;
                        break;

                    case EliteInsertStatus.ReplacedExistingCell:
                        replaced++;
                        break;

                    case EliteInsertStatus.Rejected:
                        rejected++;
                        break;

                    case EliteInsertStatus.OutOfBounds:
                        outOfBounds++;
                        break;
                }

                if (!_hasBest || candidateFitness > _bestFitness)
                {
                    candidate.CopyTo(_bestParameters);
                    _bestFitness = candidateFitness;
                    _hasBest = true;
                }
            }

            Iteration++;

            return new MapElitesIterationMetrics(
                iteration: Iteration,
                insertedNewCells: insertedNew,
                replacedExistingCells: replaced,
                rejectedCount: rejected,
                outOfBoundsCount: outOfBounds,
                occupiedCells: _archive.OccupiedCount,
                cellCount: _archive.CellCount,
                coverage: _archive.Coverage,
                qdScore: _archive.QdScore,
                bestFitness: _bestFitness);
        }

        public MapElitesIterationMetrics RunIteration(ref TContext context)
        {
            ThrowIfDisposed();

            Ask(_candidateParameters);

            for (var i = 0; i < BatchSize; i++)
            {
                var candidate = _candidateParameters.AsSpan(i * ParameterCount, ParameterCount);
                var descriptor = _candidateDescriptors.AsSpan(i * DescriptorDimensions, DescriptorDimensions);

                var fitness = _evaluator.Evaluate(candidate, ref context, descriptor);
                _candidateFitness[i] = fitness;
            }

            return Tell(_candidateParameters, _candidateFitness, _candidateDescriptors);
        }

        public void Dispose()
        {
            _disposed = true;
        }

        private void FillWithRandomCandidates(Span<float> populationMatrix)
        {
            for (var i = 0; i < BatchSize; i++)
            {
                var candidate = populationMatrix.Slice(i * ParameterCount, ParameterCount);
                FillRandomVector(candidate, _initialMin, _initialMax);
            }
        }

        private void FillRandomVector(Span<float> destination, float min, float max)
        {
            var range = max - min;
            for (var i = 0; i < destination.Length; i++)
            {
                destination[i] = min + (range * NextUnitFloat());
            }
        }

        private void MutateGaussianInPlace(Span<float> candidate, float sigma)
        {
            for (var i = 0; i < candidate.Length; i++)
            {
                candidate[i] += sigma * NextGaussian();
            }
        }

        private float NextUnitFloat()
        {
            return (NextUInt32() >> 8) * (1.0f / (1u << 24));
        }

        private float NextGaussian()
        {
            if (_hasSpareGaussian)
            {
                _hasSpareGaussian = false;
                return _spareGaussian;
            }

            float u1;
            do
            {
                u1 = NextUnitFloat();
            } while (u1 <= float.Epsilon);

            var u2 = NextUnitFloat();

            var radius = MathF.Sqrt(-2f * MathF.Log(u1));
            var theta = 2f * MathF.PI * u2;

            _spareGaussian = radius * MathF.Sin(theta);
            _hasSpareGaussian = true;

            return radius * MathF.Cos(theta);
        }

        private uint NextUInt32()
        {
            var x = NormalizeSeed(_rngState);
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

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }
}