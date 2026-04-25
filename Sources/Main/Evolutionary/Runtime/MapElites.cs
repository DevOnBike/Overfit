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
        private readonly float[] _bestEvaluatedParameters;

        private readonly float _initialMin;
        private readonly float _initialMax;
        private readonly float _mutationSigma;
        private readonly float _randomInjectionProbability;

        private readonly int _initialSeed;
        private uint _rngState;
        private bool _disposed;

        // "Evaluated" tracks the strongest candidate ever produced by the evaluator,
        // regardless of whether the candidate was admitted to the archive. This includes
        // candidates that were rejected, fell out-of-bounds, or were skipped due to
        // non-finite fitness — but only finite fitnesses ever update the value, so the
        // tracker itself never goes NaN. This is the metric to watch for "is the
        // emitter+evaluator combo capable of finding a strong candidate at all?".
        private bool _hasBestEvaluated;
        private float _bestEvaluatedFitness;

        // "Elite" tracks the strongest candidate currently held in the archive. Because
        // Insert only replaces when the new fitness is strictly higher, this value is
        // monotone non-decreasing across the run (it can plateau but never regress).
        // The cell index is cached so BestEliteParameters can read directly without
        // scanning the entire archive.
        private bool _hasBestElite;
        private float _bestEliteFitness;
        private int _bestEliteCellIndex;

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
            _bestEvaluatedParameters = new float[parameterCount];

            _initialMin = initialMin;
            _initialMax = initialMax;
            _mutationSigma = mutationSigma;
            _randomInjectionProbability = randomInjectionProbability;

            _initialSeed = seed;
            _rngState = NormalizeSeed(unchecked((uint)seed));

            _bestEvaluatedFitness = float.NaN;
            _hasBestEvaluated = false;
            _bestEliteFitness = float.NaN;
            _bestEliteCellIndex = -1;
            _hasBestElite = false;
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

        /// <summary>
        ///     Strongest fitness ever produced by the evaluator since the last
        ///     <see cref="Reset"/>, regardless of archive admission. Updated only on finite
        ///     fitness values, so this never goes NaN even if the evaluator occasionally
        ///     emits invalid samples. <c>NaN</c> when no candidate has yet been evaluated;
        ///     check <see cref="HasBestEvaluated"/> first.
        /// </summary>
        public float BestEvaluatedFitness
        {
            get
            {
                ThrowIfDisposed();
                return _bestEvaluatedFitness;
            }
        }

        public bool HasBestEvaluated
        {
            get
            {
                ThrowIfDisposed();
                return _hasBestEvaluated;
            }
        }

        /// <summary>
        ///     Strongest fitness currently held by the archive. Monotone non-decreasing:
        ///     it can plateau but never regress, because <c>Insert</c> only replaces
        ///     when the new fitness is strictly higher than the existing elite. <c>NaN</c>
        ///     when the archive is empty; check <see cref="HasBestElite"/> first.
        /// </summary>
        public float BestEliteFitness
        {
            get
            {
                ThrowIfDisposed();
                return _bestEliteFitness;
            }
        }

        public bool HasBestElite
        {
            get
            {
                ThrowIfDisposed();
                return _hasBestElite;
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

        /// <summary>
        ///     Parameters of the strongest candidate ever evaluated. Returns an empty
        ///     span when no finite-fitness candidate has been seen yet.
        /// </summary>
        public ReadOnlySpan<float> GetBestEvaluatedParameters()
        {
            ThrowIfDisposed();
            return _hasBestEvaluated ? _bestEvaluatedParameters : ReadOnlySpan<float>.Empty;
        }

        /// <summary>
        ///     Parameters of the archive's strongest elite. Reads from the archive's own
        ///     storage (no extra copy); the returned span is invalidated by any subsequent
        ///     <c>Insert</c>/<c>Tell</c> that touches the same cell. Empty when the archive
        ///     has no occupied cells.
        /// </summary>
        public ReadOnlySpan<float> GetBestEliteParameters()
        {
            ThrowIfDisposed();
            if (!_hasBestElite || _bestEliteCellIndex < 0)
            {
                return ReadOnlySpan<float>.Empty;
            }
            return _archive.GetParameters(_bestEliteCellIndex);
        }

        public void Reset()
        {
            ThrowIfDisposed();

            _rngState = NormalizeSeed(unchecked((uint)_initialSeed));
            Iteration = 0;
            _hasBestEvaluated = false;
            _bestEvaluatedFitness = float.NaN;
            _hasBestElite = false;
            _bestEliteFitness = float.NaN;
            _bestEliteCellIndex = -1;
            _hasSpareGaussian = false;
            _spareGaussian = 0f;

            Array.Clear(_candidateParameters, 0, _candidateParameters.Length);
            Array.Clear(_candidateDescriptors, 0, _candidateDescriptors.Length);
            Array.Clear(_candidateFitness, 0, _candidateFitness.Length);
            Array.Clear(_bestEvaluatedParameters, 0, _bestEvaluatedParameters.Length);
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
            var invalidFitness = 0;

            for (var i = 0; i < BatchSize; i++)
            {
                var candidate = populationMatrix.Slice(i * ParameterCount, ParameterCount);
                var descriptor = descriptors.Slice(i * DescriptorDimensions, DescriptorDimensions);
                var candidateFitness = fitness[i];

                var status = _archive.Insert(candidate, candidateFitness, descriptor, out var cellIndex);
                switch (status)
                {
                    case EliteInsertStatus.InsertedNewCell:
                        insertedNew++;
                        // The archive just took ownership of this candidate. If it's
                        // stronger than any previous elite, redirect _bestEliteCellIndex
                        // so GetBestEliteParameters reads the new cell. NaN comparison
                        // semantics ensure the !_hasBestElite guard handles the very
                        // first insertion correctly.
                        if (!_hasBestElite || candidateFitness > _bestEliteFitness)
                        {
                            _bestEliteFitness = candidateFitness;
                            _bestEliteCellIndex = cellIndex;
                            _hasBestElite = true;
                        }
                        break;

                    case EliteInsertStatus.ReplacedExistingCell:
                        replaced++;
                        // The cell we just wrote *might* be the cached best-elite cell,
                        // or it might be a different cell that's now stronger than the
                        // current best. Either way the new fitness is strictly higher
                        // than what was in this cell before — but it may still be lower
                        // than the elite stored in some other cell. So check the global
                        // best, not just the cell we wrote.
                        if (!_hasBestElite || candidateFitness > _bestEliteFitness)
                        {
                            _bestEliteFitness = candidateFitness;
                            _bestEliteCellIndex = cellIndex;
                            _hasBestElite = true;
                        }
                        break;

                    case EliteInsertStatus.Rejected:
                        rejected++;
                        break;

                    case EliteInsertStatus.OutOfBounds:
                        outOfBounds++;
                        break;

                    case EliteInsertStatus.InvalidFitness:
                        invalidFitness++;
                        // Skip the BestEvaluatedFitness update for non-finite fitnesses —
                        // see below. We don't want NaN poisoning the evaluated tracker.
                        continue;
                }

                // Update BestEvaluatedFitness for any candidate with a finite fitness,
                // regardless of archive admission. Out-of-bounds candidates count here
                // because they were validly evaluated; only invalid (NaN/∞) candidates
                // are skipped via the 'continue' above.
                if (!_hasBestEvaluated || candidateFitness > _bestEvaluatedFitness)
                {
                    candidate.CopyTo(_bestEvaluatedParameters);
                    _bestEvaluatedFitness = candidateFitness;
                    _hasBestEvaluated = true;
                }
            }

            Iteration++;

            return new MapElitesIterationMetrics(
                iteration: Iteration,
                insertedNewCells: insertedNew,
                replacedExistingCells: replaced,
                rejectedCount: rejected,
                outOfBoundsCount: outOfBounds,
                invalidFitnessCount: invalidFitness,
                occupiedCells: _archive.OccupiedCount,
                cellCount: _archive.CellCount,
                coverage: _archive.Coverage,
                qdScore: _archive.QdScore,
                bestEvaluatedFitness: _bestEvaluatedFitness,
                bestEliteFitness: _bestEliteFitness);
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