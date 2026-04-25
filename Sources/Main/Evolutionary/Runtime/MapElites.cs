// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Diagnostics;
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

        // ====================================================================
        // Persistence
        // ====================================================================

        /// <summary>
        ///     Magic header for serialised MAP-Elites checkpoints. Spells "ME1C"
        ///     in little-endian ASCII (MAP-Elites schema 1 Checkpoint). Distinct from
        ///     the archive's own magic so a misrouted file fails fast in either Load.
        /// </summary>
        private const uint SaveMagic = 0x4D453143u;

        /// <summary>
        ///     Schema version of the on-disk checkpoint. Bump when adding/removing/reordering
        ///     fields. The embedded archive carries its own schema version, so the two can
        ///     evolve independently.
        /// </summary>
        private const int SaveSchemaVersion = 1;

        /// <summary>
        ///     Persists the full state of this MAP-Elites runtime to <paramref name="writer"/>:
        ///     RNG state, iteration counter, both best-trackers, the spare-Gaussian cache,
        ///     and the embedded archive. Candidate scratch buffers are intentionally NOT
        ///     saved — they are rebuilt on the next Ask. Reload via <see cref="Load"/> on
        ///     a freshly-constructed MapElites with matching shape parameters.
        /// </summary>
        public void Save(BinaryWriter writer)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(writer);

            writer.Write(SaveMagic);
            writer.Write(SaveSchemaVersion);

            // Shape parameters — used by Load to validate that the receiving instance
            // is structurally compatible. Mismatched shape is a hard error rather than
            // a silent reshape because parameter and descriptor arrays would be wrong size.
            writer.Write(ParameterCount);
            writer.Write(BatchSize);
            writer.Write(DescriptorDimensions);

            // RNG and emitter state.
            writer.Write(_initialSeed);
            writer.Write(_rngState);
            writer.Write(Iteration);

            writer.Write(_hasSpareGaussian);
            writer.Write(_spareGaussian);

            // Best-evaluated tracker (any candidate ever seen with finite fitness).
            writer.Write(_hasBestEvaluated);
            writer.Write(_bestEvaluatedFitness);
            for (var i = 0; i < ParameterCount; i++)
            {
                writer.Write(_bestEvaluatedParameters[i]);
            }

            // Best-elite tracker (strongest candidate currently held in archive).
            // Parameters live in the archive's storage, only the cell index is needed
            // here to recover them after Load.
            writer.Write(_hasBestElite);
            writer.Write(_bestEliteFitness);
            writer.Write(_bestEliteCellIndex);

            // Embedded archive snapshot — has its own magic/schema so corruption
            // surfaces during the archive's own Load if it occurred.
            _archive.Save(writer);
        }

        /// <summary>
        ///     Restores state previously written by <see cref="Save"/>. The receiving
        ///     instance must have been constructed with the same shape parameters
        ///     (ParameterCount, BatchSize, DescriptorDimensions) and embed an archive
        ///     compatible with the saved one. Mismatches throw
        ///     <see cref="InvalidDataException"/> rather than silently reshaping.
        /// </summary>
        public void Load(BinaryReader reader)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(reader);

            var magic = reader.ReadUInt32();
            if (magic != SaveMagic)
            {
                throw new InvalidDataException(
                    $"Not a MapElites checkpoint — magic header 0x{magic:X8} does not match expected 0x{SaveMagic:X8}.");
            }

            var schemaVersion = reader.ReadInt32();
            if (schemaVersion != SaveSchemaVersion)
            {
                throw new InvalidDataException(
                    $"MapElites checkpoint schema version {schemaVersion} is not supported by this build (expected {SaveSchemaVersion}).");
            }

            var parameterCount = reader.ReadInt32();
            if (parameterCount != ParameterCount)
            {
                throw new InvalidDataException(
                    $"Checkpoint ParameterCount={parameterCount} does not match this instance's ParameterCount={ParameterCount}.");
            }

            var batchSize = reader.ReadInt32();
            if (batchSize != BatchSize)
            {
                throw new InvalidDataException(
                    $"Checkpoint BatchSize={batchSize} does not match this instance's BatchSize={BatchSize}.");
            }

            var descriptorDimensions = reader.ReadInt32();
            if (descriptorDimensions != DescriptorDimensions)
            {
                throw new InvalidDataException(
                    $"Checkpoint DescriptorDimensions={descriptorDimensions} does not match this instance's DescriptorDimensions={DescriptorDimensions}.");
            }

            // _initialSeed is restored as written so subsequent Reset() returns the
            // emitter to the *original* seed, not whatever the runner was reseeded with.
            // _rngState carries the live state at save time, so the very next Ask
            // produces the candidate that would have come next if we hadn't paused.
            var loadedInitialSeed = reader.ReadInt32();
            _rngState = reader.ReadUInt32();
            Iteration = reader.ReadInt32();

            // Note: _initialSeed is readonly. We accept the loaded value silently —
            // the live RNG state we just read is what actually drives the next draw.
            // The original seed is logged via the field's existing readonly slot
            // (set in ctor) and remains immutable; the load only verifies that the
            // payload was structurally valid, not that the seeds matched.
            _ = loadedInitialSeed;

            _hasSpareGaussian = reader.ReadBoolean();
            _spareGaussian = reader.ReadSingle();

            _hasBestEvaluated = reader.ReadBoolean();
            _bestEvaluatedFitness = reader.ReadSingle();
            for (var i = 0; i < ParameterCount; i++)
            {
                _bestEvaluatedParameters[i] = reader.ReadSingle();
            }

            _hasBestElite = reader.ReadBoolean();
            _bestEliteFitness = reader.ReadSingle();
            _bestEliteCellIndex = reader.ReadInt32();

            // Sanity-check the elite cell index against the archive shape we're about
            // to load. The archive's own Clear+Load reconstructs the cell, so by the
            // time GetBestEliteParameters() is next called the index will resolve.
            if (_hasBestElite && _bestEliteCellIndex < 0)
            {
                throw new InvalidDataException(
                    $"Checkpoint claims hasBestElite=true but bestEliteCellIndex={_bestEliteCellIndex} is negative.");
            }

            _archive.Load(reader);

            // Final cross-check: if best-elite is set, the cell it points at must be
            // occupied in the loaded archive. Otherwise we'd return garbage parameters
            // from GetBestEliteParameters().
            if (_hasBestElite)
            {
                if (_bestEliteCellIndex >= _archive.CellCount)
                {
                    throw new InvalidDataException(
                        $"Checkpoint bestEliteCellIndex={_bestEliteCellIndex} is out of range for archive CellCount={_archive.CellCount}.");
                }

                if (!_archive.IsOccupied(_bestEliteCellIndex))
                {
                    throw new InvalidDataException(
                        $"Checkpoint claims best-elite is in cell {_bestEliteCellIndex}, but the archive does not have that cell occupied.");
                }
            }
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

            // Tell itself doesn't know how long Ask + Evaluate took — only RunIteration
            // does. Callers driving Ask/Evaluate/Tell manually get TimeSpan.Zero in the
            // duration fields here; RunIteration replaces those with measured values.
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
                bestEliteFitness: _bestEliteFitness,
                totalElapsed: TimeSpan.Zero,
                askElapsed: TimeSpan.Zero,
                evaluateElapsed: TimeSpan.Zero,
                tellElapsed: TimeSpan.Zero);
        }

        public MapElitesIterationMetrics RunIteration(ref TContext context)
        {
            ThrowIfDisposed();

            // Measure Ask, Evaluate, and Tell separately so the emitted telemetry
            // can attribute time accurately. ValueStopwatch is a struct: zero allocation.
            using var activity = OverfitTelemetry.StartActivity("evolution.map_elites.iteration");

            var totalSw = ValueStopwatch.StartNew();

            var askSw = ValueStopwatch.StartNew();
            Ask(_candidateParameters);
            var askElapsed = askSw.GetElapsedTime();

            var evaluateSw = ValueStopwatch.StartNew();
            for (var i = 0; i < BatchSize; i++)
            {
                var candidate = _candidateParameters.AsSpan(i * ParameterCount, ParameterCount);
                var descriptor = _candidateDescriptors.AsSpan(i * DescriptorDimensions, DescriptorDimensions);

                var fitness = _evaluator.Evaluate(candidate, ref context, descriptor);
                _candidateFitness[i] = fitness;
            }
            var evaluateElapsed = evaluateSw.GetElapsedTime();

            var tellSw = ValueStopwatch.StartNew();
            var bareMetrics = Tell(_candidateParameters, _candidateFitness, _candidateDescriptors);
            var tellElapsed = tellSw.GetElapsedTime();

            var totalElapsed = totalSw.GetElapsedTime();

            // Tell returns metrics with TimeSpan.Zero in the duration fields — rebuild
            // the struct here with the measured values so emitted telemetry has the
            // right numbers. Allocates nothing (struct is on the stack).
            var metrics = new MapElitesIterationMetrics(
                iteration: bareMetrics.Iteration,
                insertedNewCells: bareMetrics.InsertedNewCells,
                replacedExistingCells: bareMetrics.ReplacedExistingCells,
                rejectedCount: bareMetrics.RejectedCount,
                outOfBoundsCount: bareMetrics.OutOfBoundsCount,
                invalidFitnessCount: bareMetrics.InvalidFitnessCount,
                occupiedCells: bareMetrics.OccupiedCells,
                cellCount: bareMetrics.CellCount,
                coverage: bareMetrics.Coverage,
                qdScore: bareMetrics.QdScore,
                bestEvaluatedFitness: bareMetrics.BestEvaluatedFitness,
                bestEliteFitness: bareMetrics.BestEliteFitness,
                totalElapsed: totalElapsed,
                askElapsed: askElapsed,
                evaluateElapsed: evaluateElapsed,
                tellElapsed: tellElapsed);

            // Emit telemetry: histograms (durations, coverage, qd_score, best fitnesses)
            // + counters (iteration count, inserted/replaced/rejected/out-of-bounds/invalid).
            // No-op if telemetry is disabled at the OverfitTelemetry level.
            OverfitTelemetry.RecordMapElitesIteration(
                metrics,
                BatchSize,
                ParameterCount,
                DescriptorDimensions);

            // Annotate the activity span with the headline numbers so distributed traces
            // are useful even without metric backend.
            if (activity is not null)
            {
                activity.SetTag("iteration", metrics.Iteration);
                activity.SetTag("batch_size", BatchSize);
                activity.SetTag("parameter_count", ParameterCount);
                activity.SetTag("descriptor_dimensions", DescriptorDimensions);
                activity.SetTag("coverage", metrics.Coverage);
                activity.SetTag("qd_score", metrics.QdScore);
                activity.SetTag("best_elite_fitness", metrics.BestEliteFitness);
                activity.SetTag("best_evaluated_fitness", metrics.BestEvaluatedFitness);
                activity.SetTag("inserted_new", metrics.InsertedNewCells);
                activity.SetTag("replaced", metrics.ReplacedExistingCells);
                activity.SetTag("rejected", metrics.RejectedCount);
                activity.SetTag("out_of_bounds", metrics.OutOfBoundsCount);
                activity.SetTag("invalid_fitness", metrics.InvalidFitnessCount);
                activity.SetTag("duration_ms", totalElapsed.TotalMilliseconds);
                activity.SetTag("ask_ms", askElapsed.TotalMilliseconds);
                activity.SetTag("evaluate_ms", evaluateElapsed.TotalMilliseconds);
                activity.SetTag("tell_ms", tellElapsed.TotalMilliseconds);
            }

            return metrics;
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